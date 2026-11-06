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
            # the padded keys (cutoff factor 0) masked exactly: PET's finite
            # log(epsilon) bias is negligible for its own attention, but the per-frame
            # rotations can lift a padded key's score, so the frame path would depend on
            # what fills the padded slots (and on how structures are batched)
            exact = torch.where(
                cutoff_factors[:, None, :, :] > 0,
                bias,
                torch.full_like(bias, float("-inf")),
            )
            o = o + self.rope_gate[None, :, None, None] * (
                self.frame_attention(q, k, v, exact, token_positions) - o_ref
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


def les_damp_screen_factor(d: torch.Tensor, ds: torch.Tensor) -> torch.Tensor:
    """``F(r) = (1 - w exp(-(r / a)^4)) (1 - s + s exp(-kappa r))`` for ``ds = (w, a_raw, s,
    kappa_raw)``, ``a = softplus(a_raw)``, ``kappa = softplus(kappa_raw)``."""
    w, a, s, k = ds[0], Fn.softplus(ds[1]), ds[2], Fn.softplus(ds[3])
    x = d / a
    return (1.0 - w * torch.exp(-(x * x) * (x * x))) * (1.0 + s * (torch.exp(-k * d) - 1.0))


# Free-atom static dipole polarisabilities (bohr^3), Schwerdtfeger & Nagle, "2018 Table of static
# dipole polarizabilities of the neutral elements in the periodic table", Mol. Phys. 117, 1200 (2019)
# (as tabulated by mendeleev 1.3.0, `dipole_polarizability`), Z = 1..103.
FREE_ATOM_ALPHA_BOHR3 = [
    4.50711, 1.38375, 164.113, 37.74, 20.5, 11.3, 7.4, 5.3, 3.74, 2.6611, 162.7, 71.2, 57.8, 37.3, 25, 19.4,
    14.6, 11.083, 289.7, 160.8, 97, 87, 87, 83, 68, 62, 55, 49, 46.5, 38.67, 50, 40, 30, 28.9, 21, 16.78,
    319.8, 197.2, 162, 112, 98, 87, 79, 72, 66, 26.14, 55, 46, 65, 53, 43, 38, 32.9, 27.32, 400.9, 272,
    215, 205, 216, 208, 200, 192, 184, 158, 170, 163, 165, 150, 144, 139, 137, 103, 74, 68, 62, 57, 54,
    48, 36, 33.91, 50, 47, 48, 44, 42, 35, 317.8, 246, 203, 217, 154, 129, 151, 132, 131, 144, 125, 122,
    118, 113, 109, 110, 320,
]
BOHR3_TO_A3 = 0.529177210903**3


def thole_inv_alpha16(atomic_types: List[int]) -> torch.Tensor:
    """``alpha_Z^(-1/6)`` (A^-1/2) per species index, the free-atom polarisabilities in A^3."""
    return torch.tensor([(FREE_ATOM_ALPHA_BOHR3[int(z) - 1] * BOHR3_TO_A3) ** (-1.0 / 6.0) for z in atomic_types])


def les_thole_factor(d: torch.Tensor, wt: torch.Tensor, at_raw: torch.Tensor, ia: torch.Tensor) -> torch.Tensor:
    """Thole charge-charge damping ``1 - w_T (1 + u/2) exp(-u)``, ``u = a_T r / (alpha_i alpha_j)^(1/6)``,
    ``a_T = softplus(at_raw)``, ``ia = alpha^(-1/6)`` per atom."""
    u = Fn.softplus(at_raw) * d * (ia[:, None] * ia[None, :])
    return 1.0 - wt * (1.0 + 0.5 * u) * torch.exp(-u)


def les_energy_finite_ds(q: torch.Tensor, pos: torch.Tensor, sigma: float, ds: torch.Tensor,
                         ia: Optional[torch.Tensor] = None) -> torch.Tensor:
    """:func:`les_energy_finite` with the pair kernel times :func:`les_damp_screen_factor`
    (``ds[:4]``), and with ``ds = (w, a_raw, s, kappa_raw, w_T, a_T_raw)`` and the per-atom
    ``ia = alpha^(-1/6)`` times :func:`les_thole_factor` too."""
    A = pos.shape[0]
    rij = pos[:, None] - pos[None]
    off = ~torch.eye(A, dtype=torch.bool, device=pos.device)
    d = ((rij * rij).sum(-1) + (~off).to(pos.dtype)).sqrt()  # 1 on the diagonal
    pair = torch.erf(d / (math.sqrt(2) * sigma)) / d * off.to(pos.dtype) * les_damp_screen_factor(d, ds)
    if ds.shape[0] == 6:
        if ia is None:
            raise ValueError("les_thole needs the per-atom polarisabilities")
        pair = pair * les_thole_factor(d, ds[4], ds[5], ia.to(pos.dtype))
    return 0.5 * 14.399645 * (q * (pair @ q)).sum(-1)  # KE


class PETLES(torch.nn.Module):
    """LES energy of ``num_charges`` latent charge channels from a PET-style head
    (node head plus cutoff-weighted edge head, summed over the readout layers).

    Units: PET trains on energies divided by the scaler's scale ``s`` and the LES
    kernel uses ``KE`` in eV A / e^2, so the latent charges are ``q_m = q / sqrt(s)``
    with ``q`` in ``e``; ``charge_unit`` stores ``sqrt(s)`` (set by the converter).
    Charges are neutral per structure, or with ``total_charge_constraint`` they sum
    to the structure's charge: ``q_m,i <- q_m,i - (sum q_m - Q / charge_unit) / N``.

    LES tail (``les_tail_sigma`` > 0): the same energy with the Gaussian width
    ``les_tail_sigma`` instead of ``les_sigma`` (periodic: the reciprocal sum, its k
    cutoff ``les_kcut / les_tail_sigma``, minus the self terms; open: the erf pair sum
    over ``i != j``), so that only the long-range part of the Coulomb interaction is
    kept. With ``les_tail_subtract_cutoff`` the part within PET's neighbour list is
    removed too, ``1/2 sum_i q_i sum_{j in nbrs(i)} q_j erf(r_ij / (sqrt 2 sigma)) / r_ij
    f_c(r_ij)`` with PET's cutoff function ``f_c``: LES then adds almost nothing
    inside the cutoff.

    Damping and screening (``les_damp_screen``): the open-system pair kernel is multiplied by
    ``F(r) = (1 - w exp(-(r / a)^4)) (1 - s + s exp(-kappa r))`` with learnable ``w``, ``s``
    (both 0 at start, so ``F = 1`` and the energy is exactly the undamped one), ``a =
    softplus(a_raw)`` (start 2.5 A) and ``kappa = softplus(kappa_raw)`` (start 0.2 / A).
    ``w`` damps the short-range part (contact distances, where fixed charges overcount
    the Coulomb interaction the backbone already describes), ``s`` blends in Yukawa
    screening. Periodic structures are not supported with it (no screened Ewald sum yet):
    they raise.

    Thole damping (``les_thole``, with ``les_damp_screen``): the kernel also times ``1 - w_T
    (1 + u/2) exp(-u)``, ``u = a_T r / (alpha_i alpha_j)^(1/6)``, free-atom polarisabilities
    ``alpha`` (fixed, :data:`FREE_ATOM_ALPHA_BOHR3`), learnable ``w_T`` (start 0) and ``a_T =
    softplus(a_T_raw)`` (start ``les_thole_a``).
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
        tail_sigma: float = 0.0,
        tail_subtract_cutoff: bool = False,
        damp_screen: bool = False,
        thole: bool = False,
        thole_a: float = 0.39,
        atomic_types: Optional[List[int]] = None,
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
        self.tail_sigma = float(tail_sigma)
        self.tail_subtract_cutoff = bool(tail_subtract_cutoff)
        if self.tail_subtract_cutoff and not self.tail_sigma > 0:
            raise ValueError("les_tail_subtract_cutoff needs les_tail_sigma > 0")
        self.total_charge_constraint = total_charge_constraint
        self.les_gate = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer("charge_unit", torch.ones(1))
        self.damp_screen = bool(damp_screen)
        if self.damp_screen:
            if self.tail_subtract_cutoff:
                raise ValueError("les_damp_screen with les_tail_subtract_cutoff is not supported")
            # softplus^-1(x) = x + log(-expm1(-x))
            inv = lambda x: x + math.log(-math.expm1(-x))  # noqa: E731
            self.les_damp_w = torch.nn.Parameter(torch.zeros(1))
            self.les_damp_a_raw = torch.nn.Parameter(torch.full((1,), inv(2.5)))
            self.les_screen_s = torch.nn.Parameter(torch.zeros(1))
            self.les_screen_kappa_raw = torch.nn.Parameter(torch.full((1,), inv(0.2)))
        self.thole = bool(thole)
        if self.thole:
            if not self.damp_screen or atomic_types is None:
                raise ValueError("les_thole needs les_damp_screen and the atomic types")
            self.les_thole_w = torch.nn.Parameter(torch.zeros(1))
            self.les_thole_a_raw = torch.nn.Parameter(torch.full((1,), inv(float(thole_a))))
            self.register_buffer("les_thole_ia", thole_inv_alpha16(atomic_types))

    def damp_screen_params(self) -> torch.Tensor:
        """``(w, a_raw, s, kappa_raw[, w_T, a_T_raw])`` stacked, the PLATO_LES_DS op's third input."""
        p = [self.les_damp_w, self.les_damp_a_raw, self.les_screen_s, self.les_screen_kappa_raw]
        if getattr(self, "thole", False):  # (modules pickled before les_thole lack it)
            p += [self.les_thole_w, self.les_thole_a_raw]
        return torch.cat(p)

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
        edge_distances: Optional[torch.Tensor] = None,
        neighbor_index: Optional[torch.Tensor] = None,
        species: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Gated per-atom LES energies ``(A,)`` in model units; ``edge_distances`` and
        ``neighbor_index`` (NEF, the neighbour atom of each slot) for the cutoff
        subtraction of the LES tail; ``species`` (species indices) for the Thole damping."""
        q = self.charges(
            node_features_list, edge_features_list, padding_mask, cutoff_factors
        )
        sigma = self.tail_sigma if self.tail_sigma > 0 else self.sigma
        e = torch.zeros_like(q[:, 0])
        q_all = torch.zeros_like(q)
        ia_all: Optional[torch.Tensor] = None
        if getattr(self, "thole", False):
            if species is None:
                raise ValueError("les_thole needs the species")
            ia_all = self.les_thole_ia[species]
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
            q_all = q_all.index_copy(0, idx, qs)
            pos = positions[idx]
            if _periodic(pbc[s]):
                if self.damp_screen:
                    raise ValueError("les_damp_screen: periodic structures are not supported")
                es = les_energy_periodic(qs, pos, cells[s], sigma, self.kcut)
            elif self.damp_screen:
                es = les_energy_finite_ds(qs, pos, sigma, self.damp_screen_params(),
                                          None if ia_all is None else ia_all[idx])
            else:
                es = les_energy_finite(qs, pos, sigma)
            e = e.index_copy(0, idx, es)
        if self.tail_subtract_cutoff:
            if edge_distances is None or neighbor_index is None:
                raise ValueError("les_tail_subtract_cutoff needs the edge distances")
            # the part inside the neighbour list, smoothly by PET's cutoff function
            d = edge_distances
            w = torch.erf(d / (math.sqrt(2.0) * sigma)) / d * cutoff_factors
            w = torch.where(padding_mask, w, torch.zeros_like(w))
            qj = q_all[neighbor_index]  # (n, M, Q)
            sub = (q_all[:, None, :] * qj * w[:, :, None]).sum((1, 2))
            e = e - 0.5 * 14.399645 * sub  # KE
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
            float(h.get("les_tail_sigma", 0.0) or 0.0),
            bool(h.get("les_tail_subtract_cutoff", False)),
            bool(h.get("les_damp_screen", False)),
            bool(h.get("les_thole", False)),
            float(h.get("les_thole_a", 0.39)),
            [z for z, i in sorted(((z, int(i)) for z, i in enumerate(backend.species_to_species_index.tolist())
                                   if int(i) >= 0), key=lambda t: t[1])],
        )
    elif h.get("total_charge_constraint", False):
        raise ValueError("total_charge_constraint needs les: true")
    if h.get("frag_charges", False):
        if backend.system_conditioning is None or backend.les is None:
            raise ValueError("frag_charges needs system_conditioning and les (for the energy unit)")
        backend.frag = PETFragCharges(float(h.get("frag_R", 8.0)), float(h.get("frag_w", 2.0)), float(h.get("frag_tau", 0.1)),
                                      float(h.get("frag_sigma", 4.0)), float(h.get("frag_t", 0.5)), float(h.get("frag_scale", 1.0)),
                                      cond_width=int(backend.system_conditioning.project[2].out_features)
                                      if h.get("frag_cond_gate", False) else 0)


def _smootherstep(x: torch.Tensor) -> torch.Tensor:
    """``6x^5 - 15x^4 + 10x^3`` on [0, 1] (C2: first and second derivatives vanish at both ends)."""
    x = x.clamp(0.0, 1.0)
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


class PETFragCharges(torch.nn.Module):
    """Fragment-constrained charges (``frag_charges``): per-fragment charge conditioning of the backbone plus a
    strictly beyond-cutoff Coulomb term between fixed fragment charges.

    Units are the bonded fragments of a structure. Two atoms are bonded when closer than ``f (r_i + r_j)``, with
    ``r`` the ASE covalent radii and ``f = 1.2`` (QCell / fragments.py), or ``f = 1.3`` when either atom is a
    transition, lanthanide or post-transition metal (metal-ligand coordination, e.g. Zn-N 2.32 A). Alkali metals
    (Li-Cs) never bond, so contact ion pairs stay two units. Units are the connected components. Unit ``u``
    carries the integer charge ``Q_u``, the sum of the per-atom system data ``mtt::formal_charges``. Without that
    data (and for periodic structures), a structure is one unit with its total charge and no Coulomb term: the plain
    model, identical to the base.

    - Conditioning: every atom is conditioned on its unit's charge (the system's spin), at every distance, so a
      contact ion pair is +1/-1.
    - Charges: ``Q_u / n_u`` on each atom of the unit.
    - Energy: ``1/2 KE sum_{i != j} q_i q_j erf(r / (sqrt 2 sigma)) / r T(r)``, where ``T`` rises smoothly from 0
      at ``R - t`` to 1 at ``R``: zero inside the receptive field, no free amplitude. Open structures only.

    Bond making and breaking change the units discontinuously; a fade for reactive dynamics is not implemented.

    ``frag_cond_gate``: instead of replacing the structure-charge conditioning, add the unit-charge conditioning as a
    gated correction ``c(Q_s) + g * (c(Q_u) - c(Q_s))`` with a per-channel ``g`` (``cond_gate``) initialised at zero,
    so that the start equals the base model plus the Coulomb term.
    """

    def __init__(self, R: float = 8.0, w: float = 2.0, tau: float = 0.1, sigma: float = 4.0, t: float = 0.5,
                 scale: float = 1.0, cond_width: int = 0) -> None:
        super().__init__()
        # frag_cond_gate: the conditioning stays c(Q_s) and the unit charges enter as c(Q_s) + g (c(Q_u) - c(Q_s)),
        # g per channel and zero-initialised (the start is exactly the base model; the Coulomb term is unchanged)
        self.cond_gate: Optional[torch.nn.Parameter] = (
            torch.nn.Parameter(torch.zeros(cond_width)) if cond_width > 0 else None)
        self.R, self.w, self.tau, self.sigma, self.t, self.scale = float(R), float(w), float(tau), float(sigma), float(t), float(scale)
        from ase.data import covalent_radii
        self.register_buffer("cov", torch.tensor(covalent_radii, dtype=torch.float64), persistent=False)
        fac = torch.full((len(covalent_radii),), 1.2, dtype=torch.float64)
        metal = list(range(21, 31)) + list(range(39, 49)) + list(range(57, 81)) + [13, 31, 49, 50, 81, 82, 83]
        fac[metal] = 1.3
        alk = torch.zeros(len(covalent_radii), dtype=torch.bool)
        alk[[3, 11, 19, 37, 55, 87]] = True
        self.register_buffer("fac", fac, persistent=False)
        self.register_buffer("alkali", alk, persistent=False)

    def bonded_units(self, z: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Unit label per atom (0..U-1) by the bond rule above."""
        n = pos.shape[0]
        with torch.no_grad():
            r = self.cov.to(pos.dtype)[z]; f = self.fac.to(pos.dtype)[z]; al = self.alkali[z]
            thr = torch.maximum(f[:, None], f[None, :]) * (r[:, None] + r[None, :])
            adj = (torch.cdist(pos.detach(), pos.detach()) < thr) & ~(al[:, None] | al[None, :])
            lab = torch.arange(n, device=pos.device)
            for _ in range(n):
                new = torch.minimum(torch.where(adj, lab[None, :], n).min(1).values, lab)
                if torch.equal(new, lab):
                    break
                lab = new
            _, lab = torch.unique(lab, return_inverse=True)
        return lab

    def units(self, z: torch.Tensor, pos: torch.Tensor, formal: Optional[torch.Tensor], Q: torch.Tensor):
        """``(labels (n,), Q_u (U,), n_u (U,), C (U, U) = identity)`` for one structure."""
        n = pos.shape[0]
        if formal is None:
            lab = torch.zeros(n, dtype=torch.long, device=pos.device)
            return lab, Q.reshape(1).to(pos.dtype), torch.full((1,), float(n), dtype=pos.dtype, device=pos.device), \
                torch.ones(1, 1, dtype=pos.dtype, device=pos.device)
        lab = self.bonded_units(z, pos)
        U = int(lab.max()) + 1
        oh = torch.nn.functional.one_hot(lab, U).to(pos.dtype)
        return lab, oh.T @ formal.to(pos.dtype), oh.sum(0), torch.eye(U, dtype=pos.dtype, device=pos.device)

    def charges(self, lab, Qu, nu, C):
        """``(Q_eff per atom, q per atom)``."""
        Qeff = C @ Qu
        Nv = C @ nu
        qv = C @ (Qu / Nv)  # per unit u: sum_v c_uv Q_v / N_v
        return Qeff[lab], qv[lab]

    def energy(self, q: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Per-atom energies (eV) of the beyond-``R`` Coulomb term."""
        A = pos.shape[0]
        off = ~torch.eye(A, dtype=torch.bool, device=pos.device)
        d = torch.sqrt(((pos[:, None] - pos[None]) ** 2).sum(-1) + (~off).to(pos.dtype))
        T = _smootherstep((d - (self.R - self.t)) / self.t)
        k = torch.erf(d / (math.sqrt(2) * self.sigma)) / d * T * off.to(pos.dtype)
        return self.scale * 0.5 * 14.399645 * q * (k @ q)

    @staticmethod
    def conditioning(sc: torch.nn.Module, qeff: torch.Tensor, spin: torch.Tensor) -> torch.Tensor:
        """``SystemConditioningEmbedding`` evaluated per atom at the (possibly fractional) charge ``qeff``."""
        x = (qeff + sc.max_charge).clamp(0.0, 2.0 * sc.max_charge)
        f = x.detach().floor().clamp(max=2 * sc.max_charge - 1)
        tt = (x - f)[:, None]
        Ew = sc.charge_embedding.weight
        c = (1 - tt) * Ew[f.long()] + tt * Ew[f.long() + 1]
        s = sc.spin_multiplicity_embedding(spin - 1)
        return sc.project(torch.cat([c, s], dim=-1))
