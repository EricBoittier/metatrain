"""Plato-PET-LR components inside PET (``plato_lr`` hypers, all off by default).

Each new term enters through a zero-initialised gate while its own weights get a
normal initialisation, so a converted model is exactly the pretrained PET and every
gate has a nonzero gradient at the start (the value of the ungated term):

- :class:`FrameRoPEAttention` (``frames``): attention-only frame averaging with
  per-frame RoPE, ``o = o_0 + g_h (sum_f w_f softmax(RoPE_f(q) RoPE_f(k)^T + b) v -
  o_0)`` per head ``h``; ``o_0`` is PET's attention. Parameters ``rope_omega`` (per
  layer, initialised as Plato's ``omega``) and ``rope_gate`` ``(H,)``.
- :class:`GroupSumEdgeEmbedder` (``edge_group_sum``): PET's ``edge_embedder``
  ``Linear(4, d_pet)`` plus ``edge_gate * sum_f w_f SiLU(W R_f^T r)``.
- :class:`PETMeshCoupling` (``mesh_coupling``): Plato's learned PME coupling on the
  node features after GNN layer ``couple_at``, ``x += mesh_gate * up(mesh(down(x)))``.
- :class:`PETLES` (``les``): LES energy of latent charges from a PET-style charge
  head, ``E += les_gate * E_LES(q)``.

Plato's modules are imported, not copied.
"""

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as Fn

from metatrain.experimental.plato_pet_lr.modules.ewald import (
    les_energy_finite,
    les_energy_periodic,
)
from metatrain.experimental.plato_pet_lr.modules.frames import (
    platonic_group,
    relative_rotations,
)
from metatrain.experimental.plato_pet_lr.modules.layers import (
    MeshCoupling,
    apply_rope,
)

from .transformer import AttentionBlock, manual_attention


def rope_omega_init(head_dim: int, rope_max_freq: float, seed: int) -> torch.Tensor:
    """Plato's RoPE frequencies ``(head_dim / 2, 3)``: random unit directions times
    ``linspace(0.3, rope_max_freq)``."""
    half = head_dim // 2
    gen = torch.Generator().manual_seed(seed)
    dirs = Fn.normalize(torch.randn(half, 3, generator=gen), dim=-1)
    return dirs * torch.linspace(0.3, rope_max_freq, half)[:, None]


class FrameRoPEAttention(AttentionBlock):
    """:class:`AttentionBlock` with attention-only frame averaging and per-frame
    RoPE on the token positions (see the module docstring).

    The PET path ``o_0`` is computed with the same operations as
    :meth:`AttentionBlock.forward` (SDPA, or the manual attention for double
    backward), so with ``rope_gate == 0`` the output is bit-identical to PET. The
    frame path always uses the manual attention (double backward).

    ``gated=False`` is the small-``omega`` alternative: ``rope_gate`` is a constant
    buffer of ones and ``rope_omega`` starts at ``rope_eps`` times Plato's init."""

    def __init__(
        self,
        block: AttentionBlock,
        frames: torch.Tensor,
        omega: torch.Tensor,
        gated: bool,
    ) -> None:
        super().__init__(
            block.input_linear.in_features,
            block.num_heads,
            block.temperature,
            block.epsilon,
        )
        self.input_linear = block.input_linear
        self.output_linear = block.output_linear
        self.register_buffer("rope_frames", frames.clone())
        self.rope_omega = torch.nn.Parameter(omega.clone())
        if gated:
            self.rope_gate = torch.nn.Parameter(torch.zeros(block.num_heads))
        else:
            self.register_buffer("rope_gate", torch.ones(block.num_heads))

    def forward(
        self,
        x: torch.Tensor,
        cutoff_factors: torch.Tensor,
        use_manual_attention: bool,
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        N, L, D = x.shape
        H, Ch = self.num_heads, self.head_dim
        qkv = self.input_linear(x).reshape(N, L, 3, H, Ch).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        bias = torch.log(torch.clamp(cutoff_factors[:, None, :, :], self.epsilon))
        scale = 1.0 / (self.head_dim**0.5 * self.temperature)
        if use_manual_attention:
            o = manual_attention(q, k, v, bias, self.temperature)
        else:
            o = Fn.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=scale)
        if token_positions is not None:
            o_ref = o
            if not use_manual_attention:
                o_ref = manual_attention(q, k, v, bias, self.temperature)
            o = o + self.rope_gate[None, :, None, None] * (
                self.frame_attention(q, k, v, bias, token_positions) - o_ref
            )
        return self.output_linear(o.transpose(1, 2).reshape(N, L, D))

    def frame_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        """``sum_f w_f softmax(RoPE_f(q) RoPE_f(k)^T / sqrt(Ch) T + bias) v`` with
        uniform ``w_f``; ``q, k, v`` ``(N, H, L, Ch)``, ``pos`` ``(N, L, 3)``."""
        R = self.rope_frames.to(pos.dtype)
        Rw = torch.einsum("fab,cb->fca", R, self.rope_omega.to(pos.dtype))
        ang = torch.einsum("nlx,fcx->nflc", pos, Rw)[:, :, None]  # (N, F, 1, L, C/2)
        qf = apply_rope(q[:, None], ang)  # (N, F, H, L, Ch)
        kf = apply_rope(k[:, None], ang)
        a = (
            qf @ kf.transpose(-1, -2) / (k.shape[-1] ** 0.5 * self.temperature)
            + bias[:, None]
        )
        p = torch.softmax(a, -1).mean(1)
        return p @ v


class GroupSumEdgeEmbedder(torch.nn.Linear):
    """PET's ``edge_embedder`` (``Linear(4, d)`` on ``[r, |r|]``) plus Plato's
    group-sum direction term ``edge_gate * mean_f SiLU(W R_f^T r)``. Keeps the
    ``weight`` / ``bias`` names of the linear layer it replaces."""

    def __init__(self, linear: torch.nn.Linear, frames: torch.Tensor) -> None:
        super().__init__(linear.in_features, linear.out_features)
        self.weight = linear.weight
        self.bias = linear.bias
        self.register_buffer("group_frames", frames.clone())
        self.vec = torch.nn.Linear(3, linear.out_features, bias=False)
        self.edge_gate = torch.nn.Parameter(torch.zeros(linear.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = Fn.linear(x, self.weight, self.bias)
        r_f = torch.einsum("fji,...j->...fi", self.group_frames.to(x.dtype), x[..., :3])
        return y + self.edge_gate * Fn.silu(self.vec(r_f)).mean(-2)


def _system_atoms(system_indices: torch.Tensor, s: int) -> torch.Tensor:
    return torch.nonzero(system_indices == s).reshape(-1)


def _periodic(pbc: torch.Tensor) -> bool:
    n = int(pbc.to(torch.long).sum())
    if n != 0 and n != 3:
        raise ValueError("plato_lr long range needs full or no periodicity")
    return n == 3


class PETMeshCoupling(torch.nn.Module):
    """Plato's :class:`MeshCoupling` on PET node features (one frame).

    Periodic structures: Plato's mesh, times the number of mesh points (Plato's
    ``irfftn`` normalisation leaves the potential divided by it, so its magnitude
    falls with the cell volume; here it is the Ewald-normalised potential of the
    latent channels, which has a large-cell limit).

    Open structures (``mesh_open_boundary: realspace``): the same continuous kernel
    in real space, ``K(r) = (2 / pi) int_0^kmax (1 + f(k)) exp(-sigma^2 k^2 / 2)
    j0(k r) dk`` (the ``f = 0`` part is ``erf(r / (sqrt(2) sigma)) / r`` exactly,
    the learned part by Gauss-Legendre quadrature), summed over all atom pairs
    including ``i = j`` as the mesh does: the open-boundary limit of the periodic
    mesh, smooth in the positions, no box. ``none``: zero for open structures."""

    def __init__(
        self,
        d: int,
        channels: int,
        spacing: float,
        sigma: float,
        n_basis: int,
        hidden: int,
        open_boundary: str,
        n_quad: int = 128,
        kcut: float = 6.0,
    ) -> None:
        super().__init__()
        if open_boundary not in ("realspace", "none"):
            raise ValueError(f"unknown mesh_open_boundary '{open_boundary}'")
        self.coupling = MeshCoupling(
            d, channels, spacing, n_basis, hidden, "auto", sigma=sigma
        )
        self.spacing, self.sigma = float(spacing), float(sigma)
        self.open_boundary = open_boundary
        I3 = platonic_group("identity").to(torch.get_default_dtype())
        self.register_buffer("I_rel", relative_rotations(I3))
        self.register_buffer("I_w", torch.ones(1))
        kmax = kcut / sigma
        x, w = np.polynomial.legendre.leggauss(n_quad)
        kq = torch.tensor(0.5 * kmax * (x + 1))
        wq = torch.tensor(0.5 * kmax * w) * (2 / math.pi)
        wq = wq * torch.exp(-0.5 * sigma**2 * kq**2)
        self.register_buffer("k_quad", kq.to(torch.get_default_dtype()))
        self.register_buffer("w_quad", wq.to(torch.get_default_dtype()))
        self.mesh_gate = torch.nn.Parameter(torch.zeros(d))

    def realspace(self, v: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        P = pos.shape[0]
        eye = torch.eye(P, dtype=torch.bool, device=pos.device)
        off = (~eye).to(pos.dtype)
        rij = pos[:, None] - pos[None]
        r = ((rij * rij).sum(-1) + eye.to(pos.dtype)).sqrt()  # 1 on the diagonal
        s2 = math.sqrt(2.0) * self.sigma
        k0 = math.sqrt(2.0 / math.pi) / self.sigma
        erf_k = torch.erf(r / s2) / r * off + k0 * eye.to(pos.dtype)
        kq = self.k_quad.to(pos.dtype)
        f = self.coupling.filt(kq[:, None])  # (nq, c)
        x = r[:, :, None] * kq  # (P, P, nq)
        j0 = torch.where(eye[:, :, None], torch.ones_like(x), torch.sin(x) / x)
        wf = self.w_quad.to(pos.dtype)[:, None] * f
        return erf_k @ v + torch.einsum("ijn,nc,jc->ic", j0, wf, v)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cells: torch.Tensor,
        pbc: torch.Tensor,
        system_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Gated update ``(A, d)`` of the node features ``x`` ``(A, d)``."""
        c = self.coupling
        rel, w = self.I_rel.to(x.dtype), self.I_w.to(x.dtype)
        v = c.down(c.norm(x[:, None]), rel, w)[:, 0]  # (A, channels)
        m = torch.zeros_like(v)
        for s in range(cells.shape[0]):
            idx = _system_atoms(system_indices, s)
            periodic = _periodic(pbc[s])
            if not periodic and self.open_boundary == "none":
                continue
            if periodic:
                cell = cells[s]
                lengths: List[float] = torch.linalg.norm(cell.detach(), dim=-1).tolist()
                M = [max(8, int(math.ceil(a / self.spacing))) for a in lengths]
                ms = c.mesh_batched(
                    v[idx][None], positions[idx][None], cell[None], None, M, 1
                )[0] * float(M[0] * M[1] * M[2])
            else:
                ms = self.realspace(v[idx], positions[idx])
            m = m.index_copy(0, idx, ms)
        return self.mesh_gate * c.up(m[:, None], rel, w)[:, 0]


class PETLES(torch.nn.Module):
    """LES energy of ``num_charges`` latent charge channels from a PET-style head
    (node head plus cutoff-weighted edge head, summed over the readout layers).

    Units: PET trains on energies divided by the scaler's scale ``s`` and the LES
    kernel uses ``KE`` in eV A / e^2, so the latent charges are ``q_m = q / sqrt(s)``
    with ``q`` in ``e``; ``charge_unit`` stores ``sqrt(s)`` (set by the converter).
    Charges are neutral per structure, or with ``total_charge_constraint`` they sum
    to the structure's charge: ``q_m,i <- q_m,i - (sum q_m - Q / charge_unit) / N``.
    """

    def __init__(
        self,
        d_node: int,
        d_pet: int,
        d_head: int,
        num_readout_layers: int,
        num_charges: int,
        sigma: float,
        kcut: float,
        total_charge_constraint: bool,
    ) -> None:
        super().__init__()

        def head(d: int) -> torch.nn.Module:
            return torch.nn.Sequential(
                torch.nn.Linear(d, d_head),
                torch.nn.SiLU(),
                torch.nn.Linear(d_head, num_charges),
            )

        self.charge_head = torch.nn.ModuleDict(
            {
                "node": torch.nn.ModuleList(
                    [head(d_node) for _ in range(num_readout_layers)]
                ),
                "edge": torch.nn.ModuleList(
                    [head(d_pet) for _ in range(num_readout_layers)]
                ),
            }
        )
        self.sigma, self.kcut = float(sigma), float(kcut)
        self.total_charge_constraint = total_charge_constraint
        self.les_gate = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer("charge_unit", torch.ones(1))

    def charges(
        self,
        node_features_list: List[torch.Tensor],
        edge_features_list: List[torch.Tensor],
        padding_mask: torch.Tensor,
        cutoff_factors: torch.Tensor,
    ) -> torch.Tensor:
        q = torch.zeros(0)
        for i, (nh, eh) in enumerate(
            zip(self.charge_head["node"], self.charge_head["edge"], strict=True)  # type: ignore
        ):
            qe = eh(edge_features_list[i])
            qe = torch.where(padding_mask[..., None], qe, 0.0)
            qi = nh(node_features_list[i]) + (qe * cutoff_factors[:, :, None]).sum(1)
            q = qi if i == 0 else q + qi
        return q

    def forward(
        self,
        node_features_list: List[torch.Tensor],
        edge_features_list: List[torch.Tensor],
        padding_mask: torch.Tensor,
        cutoff_factors: torch.Tensor,
        positions: torch.Tensor,
        cells: torch.Tensor,
        pbc: torch.Tensor,
        system_indices: torch.Tensor,
        total_charge: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Gated per-atom LES energies ``(A,)`` in model units."""
        q = self.charges(
            node_features_list, edge_features_list, padding_mask, cutoff_factors
        )
        e = torch.zeros_like(q[:, 0])
        for s in range(cells.shape[0]):
            idx = _system_atoms(system_indices, s)
            qs = q[idx]
            shift = qs.mean(0)
            if self.total_charge_constraint:
                if total_charge is None:
                    raise ValueError("total_charge_constraint needs the charges")
                Q = total_charge[s].to(qs.dtype) / self.charge_unit.to(qs.dtype)
                shift = shift - Q / qs.shape[0]
            qs = qs - shift
            pos = positions[idx]
            if _periodic(pbc[s]):
                es = les_energy_periodic(qs, pos, cells[s], self.sigma, self.kcut)
            else:
                es = les_energy_finite(qs, pos, self.sigma)
            e = e.index_copy(0, idx, es)
        return self.les_gate * e


def apply_plato_lr(backend: torch.nn.Module, h: dict) -> None:
    """Add the components enabled in the ``plato_lr`` hypers ``h`` to a
    :class:`PETBackend` (called at the end of its ``__init__``)."""
    if h.get("frame_mode", "attention") != "attention":
        raise ValueError("plato_lr supports only frame_mode 'attention'")
    frames_name = str(h.get("frames", "identity"))
    R = platonic_group(frames_name).to(torch.get_default_dtype())
    if frames_name != "identity":
        rope_init = str(h.get("rope_init", "gate"))
        if rope_init not in ("gate", "small"):
            raise ValueError(f"unknown rope_init '{rope_init}'")
        head_dim = backend.d_pet // backend.num_heads
        if head_dim % 2 != 0:
            raise ValueError("d_pet / num_heads must be even for RoPE")
        omega = rope_omega_init(
            head_dim, float(h.get("rope_max_freq", 3.0)), int(h.get("seed", 0))
        ).to(torch.get_default_dtype())
        if rope_init == "small":
            omega = omega * float(h.get("rope_eps", 1e-2))
        for gnn_layer in backend.gnn_layers:
            for layer in gnn_layer.trans.layers:
                layer.attention = FrameRoPEAttention(
                    layer.attention, R, omega, gated=rope_init == "gate"
                )
        backend.frame_rope = True
    if h.get("edge_group_sum", False):
        for gnn_layer in backend.gnn_layers:
            gnn_layer.edge_embedder = GroupSumEdgeEmbedder(gnn_layer.edge_embedder, R)
    if h.get("mesh_coupling", False):
        couple_at = int(h.get("couple_at", 0))
        if not 0 <= couple_at < backend.num_gnn_layers:
            raise ValueError(f"couple_at must be in [0, {backend.num_gnn_layers})")
        backend.couple_at = couple_at
        backend.mesh = PETMeshCoupling(
            backend.d_node,
            int(h.get("mesh_channels", 8)),
            float(h.get("mesh_spacing", 1.0)),
            float(h.get("mesh_sigma", 1.5)),
            int(h.get("n_basis", 8)),
            int(h.get("kernel_hidden", 32)),
            str(h.get("mesh_open_boundary", "realspace")),
        )
    if h.get("les", False):
        backend.les = PETLES(
            backend.d_node,
            backend.d_pet,
            backend.d_head,
            backend.num_readout_layers,
            int(h.get("num_charges", 1)),
            float(h.get("les_sigma", 1.0)),
            float(h.get("les_kcut", 6.0)),
            bool(h.get("total_charge_constraint", False)),
        )
    elif h.get("total_charge_constraint", False):
        raise ValueError("total_charge_constraint needs les: true")
