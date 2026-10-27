"""Frame-equivariant PET blocks. Tokens are ``(atoms, slots, F, channels)``; slot 0
is the centre atom, slots 1.. its neighbours. ``rel`` is ``(F, F, 9)`` and ``w`` is
``(F,)`` (see ``frames.py``)."""

import math
import os
import warnings
from typing import List, Optional

import torch
import torch.nn.functional as Fn


def rows(x: torch.Tensor) -> torch.Tensor:
    """``x`` flattened to 2-D rows, for a ``Linear`` (or an MLP of them): the force
    graph traced by ``compile`` keeps symbolic sizes only with 2-D inputs."""
    return x.reshape(-1, x.shape[-1])


def unrows(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """The rows ``y`` of :func:`rows` back to the leading shape of ``x``."""
    return y.reshape(list(x.shape[:-1]) + [y.shape[-1]])


class FrameLinear(torch.nn.Module):
    """``y_i = sum_j sum_b k_b(R_i^T R_j) W_b x_j``, a continuous kernel on SO(3).

    Two equal contractions: ``factored`` (``F (F + c_out) n_basis c_in`` per token)
    or ``folded`` into one ``(F c_out, F c_in)`` matrix (``F^2 c_in c_out``).
    ``auto`` picks the cheaper; with one frame folding is a plain linear layer.

    In evaluation mode the folded matrix depends only on the parameters and the
    (fixed) frames, so it is computed once and cached (detached: no parameter
    gradients flow through ``phi`` / ``W`` in evaluation mode). The cache is keyed on
    the number of frames, dtype and device, and in Python also on the parameter
    versions (in-place updates, ``load_state_dict``); a training-mode call clears it.
    """

    _cache: Optional[torch.Tensor]
    _cache_versions: List[int]

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
        self._cache = None
        self._cache_versions: List[int] = []

    def folded(self, F: int) -> bool:
        if self.mode == "auto":
            return F * self.c_out <= self.n_basis * (F + self.c_out)
        return self.mode == "folded"

    def kernel(self, rel: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """``(F, F, n_basis)`` kernel ``k_b(R_i^T R_j) w_j``."""
        k = unrows(self.phi(rows(rel)), rel)
        if rel.shape[0] == 1:  # w == [1]: the product is exact, skip it
            return k
        return k * w[None, :, None]

    def fold(self, rel: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """The folded ``(F c_out, F c_in)`` matrix."""
        F = rel.shape[0]
        k = self.kernel(rel, w)
        if F == 1:
            return (
                k.reshape(1, self.n_basis) @ self.W.reshape(self.n_basis, -1)
            ).reshape(self.c_out, self.c_in)
        return torch.einsum("ijb,boc->iojc", k, self.W).reshape(
            F * self.c_out, F * self.c_in
        )

    @torch.jit.unused
    def _versions(self) -> List[int]:
        return [p._version for p in self.parameters()]

    def clear_cache(self) -> None:
        self._cache = None
        self._cache_versions = []

    @torch.jit.unused
    def _cache_allowed(self) -> bool:
        # inference tensors cannot be reused outside inference mode
        return not torch.is_inference_mode_enabled()

    def folded_weight(self, rel: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._cache = None
            return self.fold(rel, w)
        F = rel.shape[0]
        versions: List[int] = []
        allowed = True
        if not torch.jit.is_scripting():
            allowed = self._cache_allowed()
            versions = self._versions()
        c = self._cache
        if (
            c is not None
            and c.shape[0] == F * self.c_out
            and c.dtype == self.W.dtype
            and c.device == self.W.device
            and versions == self._cache_versions
        ):
            return c
        W = self.fold(rel, w)
        if allowed:
            W = W.detach()
            self._cache = W
            self._cache_versions = versions
        return W

    def forward(
        self, x: torch.Tensor, rel: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        F = rel.shape[0]
        b = self.b
        if self.folded(F):
            W = self.folded_weight(rel, w)
            shape: List[int] = list(x.shape[:-2]) + [F, self.c_out]
            if F == 1:
                return Fn.linear(x.reshape(-1, self.c_in), W, b).reshape(shape)
            y = Fn.linear(x.reshape(-1, F * self.c_in), W).reshape(shape)
        else:
            k = self.kernel(rel, w)
            z = torch.einsum("ijb,...jc->...ibc", k, x)
            y = torch.einsum("...ibc,boc->...io", z, self.W)
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


def rope_cs(ang: torch.Tensor) -> torch.Tensor:
    """``cat([cos, cos], -1)`` and ``cat([-sin, sin], -1)`` stacked on a new leading
    axis, computed once per forward and shared by every layer."""
    c, s = ang.cos(), ang.sin()
    return torch.stack([torch.cat([c, c], -1), torch.cat([-s, s], -1)])


def apply_rope_cs(x: torch.Tensor, cs: torch.Tensor) -> torch.Tensor:
    """:func:`apply_rope` with precomputed :func:`rope_cs`; the same floating point
    operations (``a - b == a + (-b)``, ``+`` commutes), so bit-identical."""
    h = x.shape[-1] // 2
    xr = torch.cat([x[..., h:], x[..., :h]], -1)
    return x * cs[0] + xr * cs[1]


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
        frame_w: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``(A, K1, F, d)`` tokens, or with ``frame_w`` one group-invariant frame
        ``(A, K1, 1, d)``: the direction enters as ``sum_f w_f SiLU(W R_f^T r)``."""
        r_f = torch.einsum("fji,akj->akfi", R, r)  # r in each frame, R_f^T r
        b = rbf(d, self.rc, self.n_rbf)
        inv = (
            unrows(self.rad(rows(b)), b)
            + self.sp_nbr(sp_nbr)
            + self.sp_center(sp_center)[:, None]
        )
        vec = unrows(self.vec(rows(r_f)), r_f)
        if frame_w is not None:
            v = (Fn.silu(vec) * frame_w[:, None]).sum(-2, keepdim=True)
            return v + inv[:, :, None]
        return vec + inv[:, :, None]


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, manual: bool
) -> torch.Tensor:
    """Manual softmax attention supports double backward (force training)."""
    if not manual:
        return Fn.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    a = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]) + bias
    return torch.softmax(a, -1) @ v


def frame_averaged_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ang: torch.Tensor,
    frame_w: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Attention-only frames: ``sum_f w_f softmax(RoPE_f(q) RoPE_f(k)^T + bias) v``.

    ``q, k, v`` are ``(A, K1, H, Ch)``, shared by the frames; ``ang`` is
    ``(A, K1, F, 1, Ch / 2)``. A group element permutes the frames, so the average
    is group invariant. Averaging the attention weights before ``@ v`` equals
    averaging the outputs, and ``@ v`` runs once. Returns ``(A, H, K1, Ch)``.
    """
    q = apply_rope(q[:, :, None], ang).permute(0, 2, 3, 1, 4)  # (A, F, H, K1, Ch)
    k = apply_rope(k[:, :, None], ang).permute(0, 2, 3, 1, 4)
    a = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]) + bias[:, None]
    p = (torch.softmax(a, -1) * frame_w[:, None, None, None]).sum(1)
    return p @ v.transpose(1, 2)


def frame_averaged_attention_qkv(
    qkv: torch.Tensor,
    cs: torch.Tensor,
    frame_w: torch.Tensor,
    bias: torch.Tensor,
    H: int,
) -> torch.Tensor:
    """:func:`frame_averaged_attention` from the fused ``(A, K1, 3 d)`` projection
    and the precomputed :func:`rope_cs` ``(2, A, K1, F, 1, Ch)``; RoPE on ``q`` and
    ``k`` in one pass. Returns ``(A, K1, d)``."""
    A, K1, d3 = qkv.shape
    d = d3 // 3
    Ch = d // H
    qk = qkv[..., : 2 * d].reshape(A, K1, 1, 2 * H, Ch)
    qk = apply_rope_cs(qk, cs).permute(0, 2, 3, 1, 4)  # (A, F, 2H, K1, Ch)
    q, k = qk[:, :, :H], qk[:, :, H:]
    a = q @ k.transpose(-1, -2) / math.sqrt(Ch) + bias[:, None]
    p = (torch.softmax(a, -1) * frame_w[:, None, None, None]).sum(1)
    v = qkv[..., 2 * d :].reshape(A, K1, H, Ch).transpose(1, 2)
    return (p @ v).transpose(1, 2).reshape(A, K1, d)


# head dimensions (d_model / num_heads) of torch.ops.mlipcpp.pet_attention
FUSED_HEAD_DIMS: List[int] = [8, 16, 32, 64]


def fused_attention_op_available() -> bool:
    """Whether ``torch.ops.mlipcpp.pet_attention`` (mlip.cpp's fused attention, see
    :func:`load_fused_attention_op`) is registered in this process."""
    try:
        torch.ops.mlipcpp.pet_attention
    except (AttributeError, RuntimeError):
        return False
    return True


def load_fused_attention_op() -> bool:
    """Register ``torch.ops.mlipcpp.pet_attention`` if needed: from the library named
    by ``MLIPCPP_TORCH_LIB``, or else ``mlipcpp_torch.load()`` (mlip.cpp
    ``torch_ext/``, builds the library on first use). Returns whether it is
    available; a failure is a warning (the model falls back to the other paths)."""
    if fused_attention_op_available():
        return True
    lib = os.environ.get("MLIPCPP_TORCH_LIB")
    try:
        if lib:
            torch.ops.load_library(lib)
        else:
            import mlipcpp_torch  # noqa: PLC0415

            mlipcpp_torch.load()
    except Exception as e:  # noqa: BLE001
        warnings.warn(
            "attention_backend 'fused': could not load torch.ops.mlipcpp."
            f"pet_attention ({type(e).__name__}: {e}); using the default attention",
            stacklevel=2,
        )
        return False
    return fused_attention_op_available()


class FusedAttention(torch.nn.Module):
    """``torch.ops.mlipcpp.pet_attention``: per head ``softmax(scale q k^T + bias) v``
    of a packed ``(N, L, 3, heads * d)`` projection, ``bias`` ``(N, 1, L, L)`` shared
    by the heads; returns ``(N, L, heads * d)``. Exact first and second derivatives
    in C++ (force training, TorchScript), CUDA float32 / float64. Only instantiated
    when the op is registered, so TorchScript sees the op only when it exists."""

    def forward(
        self, qkv: torch.Tensor, bias: torch.Tensor, heads: int, scale: float
    ) -> torch.Tensor:
        return torch.ops.mlipcpp.pet_attention(qkv, bias, heads, scale)[0]


class NoFusedAttention(torch.nn.Module):
    """Stands in for :class:`FusedAttention` when the op is not used."""

    def forward(
        self, qkv: torch.Tensor, bias: torch.Tensor, heads: int, scale: float
    ) -> torch.Tensor:
        raise RuntimeError("torch.ops.mlipcpp.pet_attention is not available")


def pack_frame_heads(
    qkv: torch.Tensor, cs: Optional[torch.Tensor], H: int
) -> torch.Tensor:
    """The fused op's ``(A, K1, 3, F d)`` input, with the ``F`` frames folded into
    ``F H`` heads (head ``f H + h``) that share the ``(A, K1, K1)`` bias.

    ``qkv`` is ``(A, K1, F0, 3 d)``: ``F0 = F`` token frames (``full``), or one
    (attention-only frames: ``q, k`` are rotated by each of the ``F`` frames of
    ``cs`` and ``v`` is shared). Then ``sum_f w_f softmax(RoPE_f(q) RoPE_f(k)^T +
    bias) v`` is the frame average of the ``F`` heads' outputs (the average of the
    weights before ``@ v`` equals the average after)."""
    x = qkv
    A, K1, F0, d3 = x.shape
    d = d3 // 3
    qk = x[..., : 2 * d]
    if cs is not None:
        qk = apply_rope_cs(qk.reshape(A, K1, F0, 2 * H, d // H), cs)
    F = qk.shape[2]
    qk = qk.reshape(A, K1, F, 2, d).transpose(2, 3).reshape(A, K1, 2, F * d)
    v = x[..., 2 * d :].expand(A, K1, F, d).reshape(A, K1, 1, F * d)
    return torch.cat([qk, v], 2)


class PETLayer(torch.nn.Module):
    """Edge-reversal merge, self-attention over a centre's tokens (RoPE on r_ij,
    ``log f_c`` bias), MLP; every linear map is a ``FrameLinear``.

    With ``frame_w`` (attention-only frames) the tokens have one frame, every
    linear map runs once, and the frames enter only through the RoPE angles
    (see :func:`frame_averaged_attention`). ``cs`` is :func:`rope_cs` of the angles,
    computed once per forward."""

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
        self.fused_attention: torch.nn.Module = NoFusedAttention()

    @torch.jit.unused
    def set_fused(self, enabled: bool) -> None:
        self.fused_attention = FusedAttention() if enabled else NoFusedAttention()

    def forward(
        self,
        tok: torch.Tensor,
        rel: torch.Tensor,
        w: torch.Tensor,
        cs: Optional[torch.Tensor],
        bias: torch.Tensor,
        rev: torch.Tensor,
        manual: bool,
        frame_w: Optional[torch.Tensor] = None,
        fused: bool = False,
    ) -> torch.Tensor:
        """``fused``: attention by :class:`FusedAttention` (the model checks that it
        applies: op registered, CUDA, float32 / float64, ``bias`` ``(A, 1, K1,
        K1)``); else ``manual`` softmax or SDPA."""
        A, K1, F, d = tok.shape
        H = self.h
        Ch = d // H
        t = self.n0(tok)
        t_rev = t.reshape(A * K1, F, d).index_select(0, rev).reshape(A, K1, F, d)
        x = tok + self.merge(torch.cat([t, t_rev], -1), rel, w)
        qkv = self.qkv(self.n1(x), rel, w)
        if fused:
            packed = pack_frame_heads(qkv, cs, H)  # (A, K1, 3, Fh d)
            Fh = packed.shape[-1] // d
            o = self.fused_attention(packed, bias, Fh * H, 1.0 / math.sqrt(Ch))
            o = o.reshape(A, K1, Fh, d)
            if frame_w is not None and cs is not None:  # attention-only frames
                o = (o * frame_w[:, None]).sum(2, keepdim=True)
            x = x + self.proj(o, rel, w)
            return x + self.mlp2(Fn.silu(self.mlp1(self.n2(x), rel, w)), rel, w)
        if frame_w is not None and cs is not None:
            o = frame_averaged_attention_qkv(
                qkv.reshape(A, K1, 3 * d), cs, frame_w, bias, H
            )
            x = x + self.proj(o.reshape(A, K1, 1, d), rel, w)
            return x + self.mlp2(Fn.silu(self.mlp1(self.n2(x), rel, w)), rel, w)
        if cs is not None:
            qk = apply_rope_cs(qkv[..., : 2 * d].reshape(A, K1, F, 2 * H, Ch), cs)
            q = qk[:, :, :, :H].reshape(A, K1, F * H, Ch).transpose(1, 2)
            k = qk[:, :, :, H:].reshape(A, K1, F * H, Ch).transpose(1, 2)
        else:
            q = qkv[..., :d].reshape(A, K1, F * H, Ch).transpose(1, 2)
            k = qkv[..., d : 2 * d].reshape(A, K1, F * H, Ch).transpose(1, 2)
        v = qkv[..., 2 * d :].reshape(A, K1, F * H, Ch).transpose(1, 2)
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
    filter by ``4 pi/V exp(-sigma^2 k^2/2)/k^2 (1 + f(|k|))``, interpolate back.

    ``normalization``: ``physical``, the mesh potential is the Ewald reciprocal
    potential of the Gaussian-smeared mesh charges (the filter is multiplied by the
    number of mesh points ``M``, which undoes the ``1 / M`` of ``irfftn``);
    ``per_point``, that potential divided by ``M`` (the coupling of the models
    trained up to 27 Oct 2026, checkpoint v5 and older)."""

    def __init__(
        self,
        d: int,
        c_lr: int,
        spacing: float,
        n_basis: int,
        hidden: int,
        mode: str,
        sigma: float = 1.5,
        normalization: str = "per_point",
    ):
        super().__init__()
        if normalization not in ("physical", "per_point"):
            raise ValueError(f"unknown mesh_normalization '{normalization}'")
        self.c, self.spacing, self.sigma = c_lr, spacing, sigma
        self.physical = normalization == "physical"
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
        if self.physical:  # undo the 1 / M of irfftn
            V = V / float(mesh[0] * mesh[1] * mesh[2])
        base = 4 * math.pi / V * torch.exp(-0.5 * self.sigma**2 * k2) / k2s * (k2 > 0)
        ks = k2s.sqrt()[..., None]
        g = base[..., None] * (1 + unrows(self.filt(rows(ks)), ks))
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

    def kfilter_batched(self, cells: torch.Tensor, mesh: List[int]) -> torch.Tensor:
        """:meth:`kfilter` of ``U`` cells, ``(U, c_lr, M0, M1, M2 // 2 + 1)``."""
        dt, dev = cells.dtype, cells.device
        ms = [
            torch.fft.fftfreq(mesh[0], 1.0 / mesh[0], dtype=dt, device=dev),
            torch.fft.fftfreq(mesh[1], 1.0 / mesh[1], dtype=dt, device=dev),
            torch.fft.rfftfreq(mesh[2], 1.0 / mesh[2], dtype=dt, device=dev),
        ]
        m = torch.stack(torch.meshgrid(ms, indexing="ij"), -1)
        k = 2 * math.pi * m @ torch.linalg.inv(cells).transpose(-1, -2)[:, None, None]
        k2 = (k * k).sum(-1)
        k2s = k2.clamp_min(1e-12)
        V = torch.det(cells).abs()[:, None, None, None]
        if self.physical:  # undo the 1 / M of irfftn
            V = V / float(mesh[0] * mesh[1] * mesh[2])
        base = 4 * math.pi / V * torch.exp(-0.5 * self.sigma**2 * k2) / k2s * (k2 > 0)
        ks = k2s.sqrt()[..., None]
        g = base[..., None] * (1 + unrows(self.filt(rows(ks)), ks))
        deconv = torch.ones_like(m[..., 0])
        for a in range(3):
            deconv = (
                deconv
                * ((4 + 2 * torch.cos(2 * math.pi * m[..., a] / mesh[a])) / 6) ** 2
            )
        return (g / deconv[..., None]).permute(0, 4, 1, 2, 3)

    def mesh_batched(
        self,
        v: torch.Tensor,
        pos: torch.Tensor,
        cells: torch.Tensor,
        umap: Optional[torch.Tensor],
        M: List[int],
        F: int,
    ) -> torch.Tensor:
        """Spread, filter and gather ``S`` structures on one mesh size ``M``.

        ``v`` is ``(S, P, F c_lr)`` (zero on padded slots), ``pos`` ``(S, P, 3)``,
        ``cells`` ``(U, 3, 3)``: one cell for all (``U = 1``), one per structure
        (``umap`` None), or ``umap`` maps the structures to the ``U`` cells. One
        index_add, one batched FFT pair. Returns ``(S, P, F c_lr)``."""
        S, P, C = v.shape
        Mt = M[0] * M[1] * M[2]
        inv = torch.linalg.inv(cells)
        if cells.shape[0] == 1:
            frac = pos @ inv[0]
        else:
            frac = pos @ (inv if umap is None else inv[umap])
        idx, wt = bspline4_stencil(frac.reshape(S * P, 3), M)
        off = torch.arange(S, device=v.device) * Mt
        gidx = (idx.reshape(S, P * 64) + off[:, None]).reshape(-1)
        src = (v.reshape(S * P, 1, C) * wt[:, :, None]).reshape(-1, C).T
        rho = torch.zeros(C, S * Mt, dtype=v.dtype, device=v.device).index_add(
            1, gidx, src
        )
        rk = torch.fft.rfftn(rho.reshape(C, S, M[0], M[1], M[2]), dim=(2, 3, 4))
        filt = self.kfilter_batched(cells, M)  # (U, c, ...)
        if cells.shape[0] > 1 and umap is not None:
            filt = filt[umap]
        rk = rk * filt.transpose(0, 1).repeat(F, 1, 1, 1, 1)
        phi = torch.fft.irfftn(rk, s=M, dim=(2, 3, 4)).reshape(C, -1)
        m = (phi[:, gidx.reshape(S * P, 64)] * wt).sum(-1)
        return m.T.reshape(S, P, C)
