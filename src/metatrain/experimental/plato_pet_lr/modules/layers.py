"""Frame-equivariant PET blocks. Tokens are ``(atoms, slots, F, channels)``; slot 0
is the centre atom, slots 1.. its neighbours. ``rel`` is ``(F, F, 9)`` and ``w`` is
``(F,)`` (see ``frames.py``)."""

import math
from typing import List, Optional

import torch
import torch.nn.functional as Fn


class FrameLinear(torch.nn.Module):
    """``y_i = sum_j sum_b k_b(R_i^T R_j) W_b x_j``, a continuous kernel on SO(3).

    Two equal contractions: ``factored`` (``F (F + c_out) n_basis c_in`` per token)
    or ``folded`` into one ``(F c_out, F c_in)`` matrix (``F^2 c_in c_out``).
    ``auto`` picks the cheaper; with one frame folding is a plain linear layer.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        n_basis: int,
        hidden: int,
        bias: bool = True,
        mode: str = "auto",
    ):
        super().__init__()
        self.c_in, self.c_out, self.n_basis, self.mode = c_in, c_out, n_basis, mode
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(9, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, n_basis),
        )
        self.W = torch.nn.Parameter(
            torch.randn(n_basis, c_out, c_in) / math.sqrt(c_in * n_basis)
        )
        self.b = torch.nn.Parameter(torch.zeros(c_out)) if bias else None

    def folded(self, F: int) -> bool:
        if self.mode == "auto":
            return F * self.c_out <= self.n_basis * (F + self.c_out)
        return self.mode == "folded"

    def forward(
        self, x: torch.Tensor, rel: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        F = rel.shape[0]
        k = self.phi(rel) * w[None, :, None]
        if self.folded(F):
            W = torch.einsum("ijb,boc->iojc", k, self.W).reshape(
                F * self.c_out, F * self.c_in
            )
            shape: List[int] = list(x.shape[:-2]) + [F, self.c_out]
            y = Fn.linear(x.reshape(-1, F * self.c_in), W).reshape(shape)
        else:
            z = torch.einsum("ijb,...jc->...ibc", k, x)
            y = torch.einsum("...ibc,boc->...io", z, self.W)
        b = self.b
        return y if b is None else y + b


class FrameMLP(torch.nn.Module):
    def __init__(self, d: int, out: int, n_basis: int, hidden: int, mode: str):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d)
        self.l1 = FrameLinear(d, d, n_basis, hidden, mode=mode)
        self.l2 = FrameLinear(d, out, n_basis, hidden, mode=mode)

    def forward(
        self, x: torch.Tensor, rel: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        return self.l2(Fn.silu(self.l1(self.norm(x), rel, w)), rel, w)


def cutoff_fn(d: torch.Tensor, rc: float) -> torch.Tensor:
    x = (d / rc).clamp(max=1.0)
    return (1 - x * x) ** 3


def rbf(d: torch.Tensor, rc: float, n: int) -> torch.Tensor:
    mu = torch.linspace(0, rc, n, dtype=d.dtype, device=d.device)
    return torch.exp(-(((d[..., None] - mu) / (rc / n)) ** 2))


def apply_rope(x: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, -1)
    c, s = ang.cos(), ang.sin()
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1)


class EdgeEmbed(torch.nn.Module):
    def __init__(self, n_species: int, d: int, n_rbf: int, rc: float):
        super().__init__()
        self.rc, self.n_rbf = rc, n_rbf
        self.vec = torch.nn.Linear(3, d, bias=False)
        self.rad = torch.nn.Linear(n_rbf, d)
        self.sp_center = torch.nn.Embedding(n_species, d)
        self.sp_nbr = torch.nn.Embedding(n_species, d)

    def forward(
        self,
        r: torch.Tensor,
        d: torch.Tensor,
        sp_center: torch.Tensor,
        sp_nbr: torch.Tensor,
        R: torch.Tensor,
    ) -> torch.Tensor:
        r_f = torch.einsum("fji,akj->akfi", R, r)  # r in each frame, R_f^T r
        inv = (
            self.rad(rbf(d, self.rc, self.n_rbf))
            + self.sp_nbr(sp_nbr)
            + self.sp_center(sp_center)[:, None]
        )
        return self.vec(r_f) + inv[:, :, None]


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, manual: bool
) -> torch.Tensor:
    """Manual softmax attention supports double backward (force training)."""
    if not manual:
        return Fn.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    a = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]) + bias
    return torch.softmax(a, -1) @ v


class PETLayer(torch.nn.Module):
    """Edge-reversal merge, self-attention over a centre's tokens (RoPE on r_ij,
    ``log f_c`` bias), MLP; every linear map is a ``FrameLinear``."""

    def __init__(self, d: int, heads: int, n_basis: int, hidden: int, mode: str):
        super().__init__()
        self.h = heads
        self.n0 = torch.nn.LayerNorm(d)
        self.n1 = torch.nn.LayerNorm(d)
        self.n2 = torch.nn.LayerNorm(d)
        self.merge = FrameLinear(2 * d, d, n_basis, hidden, mode=mode)
        self.qkv = FrameLinear(d, 3 * d, n_basis, hidden, mode=mode)
        self.proj = FrameLinear(d, d, n_basis, hidden, mode=mode)
        self.mlp1 = FrameLinear(d, 2 * d, n_basis, hidden, mode=mode)
        self.mlp2 = FrameLinear(2 * d, d, n_basis, hidden, mode=mode)

    def forward(
        self,
        tok: torch.Tensor,
        rel: torch.Tensor,
        w: torch.Tensor,
        ang: Optional[torch.Tensor],
        bias: torch.Tensor,
        rev: torch.Tensor,
        manual: bool,
    ) -> torch.Tensor:
        A, K1, F, d = tok.shape
        H = self.h
        Ch = d // H
        t = self.n0(tok)
        t_rev = t.reshape(A * K1, F, d).index_select(0, rev).reshape(A, K1, F, d)
        x = tok + self.merge(torch.cat([t, t_rev], -1), rel, w)
        q, k, v = self.qkv(self.n1(x), rel, w).chunk(3, -1)
        q = q.reshape(A, K1, F, H, Ch)
        k = k.reshape(A, K1, F, H, Ch)
        if ang is not None:
            q, k = apply_rope(q, ang), apply_rope(k, ang)
        q = q.reshape(A, K1, F * H, Ch).transpose(1, 2)
        k = k.reshape(A, K1, F * H, Ch).transpose(1, 2)
        v = v.reshape(A, K1, F * H, Ch).transpose(1, 2)
        o = attention(q, k, v, bias, manual)
        x = x + self.proj(o.transpose(1, 2).reshape(A, K1, F, d), rel, w)
        return x + self.mlp2(Fn.silu(self.mlp1(self.n2(x), rel, w)), rel, w)


def bspline4_stencil(frac: torch.Tensor, mesh: List[int]):
    K = torch.tensor(mesh, device=frac.device)
    u = frac * K.to(frac.dtype)
    fl = torch.floor(u)
    t = u - fl
    w = (
        torch.stack(
            [
                (1 - t) ** 3,
                3 * t**3 - 6 * t**2 + 4,
                -3 * t**3 + 3 * t**2 + 3 * t + 1,
                t**3,
            ],
            -1,
        )
        / 6
    )
    i = (fl.long()[..., None] + torch.arange(-1, 3, device=frac.device)) % K[:, None]
    idx = (i[:, 0, :, None, None] * mesh[1] + i[:, 1, None, :, None]) * mesh[2] + i[
        :, 2, None, None, :
    ]
    wt = w[:, 0, :, None, None] * w[:, 1, None, :, None] * w[:, 2, None, None, :]
    return idx.reshape(-1, 64), wt.reshape(-1, 64)


class MeshCoupling(torch.nn.Module):
    """Learned PME: spread ``c_lr`` channels per frame on a mesh (B-spline order 4),
    filter by ``4 pi/V exp(-sigma^2 k^2/2)/k^2 (1 + f(|k|))``, interpolate back."""

    def __init__(
        self,
        d: int,
        c_lr: int,
        spacing: float,
        n_basis: int,
        hidden: int,
        mode: str,
        sigma: float = 1.5,
    ):
        super().__init__()
        self.c, self.spacing, self.sigma = c_lr, spacing, sigma
        self.norm = torch.nn.LayerNorm(d)
        self.down = FrameLinear(d, c_lr, n_basis, hidden, mode=mode)
        self.up = FrameLinear(c_lr, d, n_basis, hidden, mode=mode)
        self.filt = torch.nn.Sequential(
            torch.nn.Linear(1, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, c_lr)
        )

    def mesh_size(self, cell: torch.Tensor) -> List[int]:
        """``ceil(|a_i| / spacing)``. A strain that crosses an integer changes the
        mesh, and the energy jumps by the (deconvolved) interpolation error: ~1e-7
        relative in the tests. For NPT pick a spacing away from that boundary."""
        lengths = torch.linalg.norm(cell.detach(), dim=-1)
        return [max(8, int(math.ceil(float(x) / self.spacing))) for x in lengths]

    def kfilter(self, cell: torch.Tensor, mesh: List[int]) -> torch.Tensor:
        dt, dev = cell.dtype, cell.device
        ms = [
            torch.fft.fftfreq(mesh[0], 1.0 / mesh[0], dtype=dt, device=dev),
            torch.fft.fftfreq(mesh[1], 1.0 / mesh[1], dtype=dt, device=dev),
            torch.fft.rfftfreq(mesh[2], 1.0 / mesh[2], dtype=dt, device=dev),
        ]
        m = torch.stack(torch.meshgrid(ms, indexing="ij"), -1)
        k = 2 * math.pi * m @ torch.linalg.inv(cell).T
        k2 = (k * k).sum(-1)
        k2s = k2.clamp_min(1e-12)
        V = torch.det(cell).abs()
        base = 4 * math.pi / V * torch.exp(-0.5 * self.sigma**2 * k2) / k2s * (k2 > 0)
        g = base[..., None] * (1 + self.filt(k2s.sqrt()[..., None]))
        deconv = torch.ones_like(k2)
        for a in range(3):
            deconv = (
                deconv
                * ((4 + 2 * torch.cos(2 * math.pi * m[..., a] / mesh[a])) / 6) ** 2
            )
        return (g / deconv[..., None]).permute(3, 0, 1, 2)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        cell: torch.Tensor,
        rel: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        A, F, _ = x.shape
        C = F * self.c
        M = self.mesh_size(cell)
        v = self.down(self.norm(x), rel, w).reshape(A, C)
        frac = pos @ torch.linalg.inv(cell)
        idx, wt = bspline4_stencil(frac, M)
        rho = torch.zeros(
            C, M[0] * M[1] * M[2], dtype=x.dtype, device=x.device
        ).index_add(
            1, idx.reshape(-1), (wt.reshape(-1, 1) * v.repeat_interleave(64, 0)).T
        )
        rk = torch.fft.rfftn(rho.reshape(C, M[0], M[1], M[2]), dim=(1, 2, 3))
        rk = rk * self.kfilter(cell, M).repeat(F, 1, 1, 1)
        phi = torch.fft.irfftn(rk, s=M, dim=(1, 2, 3)).reshape(C, -1)
        m = (phi[:, idx] * wt).sum(-1).T.reshape(A, F, self.c)
        return self.up(m, rel, w)
