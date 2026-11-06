"""Split-charge equilibration (SQE) with physically bounded parameters (plato_lr ``sqe``).

Energy (eV, charges in e, lengths in A, ``KE = 14.399645 eV A``):

    E_SQE(q) = sum_i [chi_i q_i + 1/2 J_i q_i^2] + sum_{i<j} q_i q_j gamma_ij(r_ij)
               + sum_{(i,j) in pairs} 1/2 kappa_ij(r_ij) p_ij^2,
    q_i = q0_i + sum_j p_ij,  p_ij = -p_ji on the pairs closer than ``r_ct``.

``q0`` is the reference distribution: ``Q / N`` (``sqe_q0: uniform``) or per-atom formal charges
(``sqe_q0: formal``, system data ``mtt::formal_charges``, falling back to ``Q / N``). Since the split charges only
move charge between atoms, ``sum q = sum q0 = Q`` exactly, and within a group of atoms connected by pairs only the
group's ``sum q0`` matters. Beyond ``r_ct`` no charge is transferred, so a fragment separated from the rest by more
than ``r_ct`` carries exactly its ``sum q0``: the integer limit holds for integer per-fragment ``q0`` sums, and with
``Q / N`` a separated fragment carries ``n_f Q / N`` (neutral fragments when ``Q = 0``).

The charges minimise ``E_SQE`` at each geometry. With the bond softness ``s_ij = 1 / kappa_ij``, the minimum over
the split charges is ``p_ij = -s_ij (mu_i - mu_j)``, ``mu = dE/dq = chi + A q``, so

    (I + L A) q = q0 - L chi,   L = sum_pairs s_ij (e_i - e_j)(e_i - e_j)^T (a weighted graph Laplacian),
    E = chi.q + 1/2 q.A q + 1/2 sum_pairs s_ij (mu_i - mu_j)^2,   A = diag(J) + gamma.

This module solves it densely in float64 with ``torch.linalg.solve``: autograd then gives the exact total
derivatives (Hellmann-Feynman forces, and the implicit/adjoint parameter gradients of energy and force losses).
mlip.cpp (``src/runtime/sqe.cpp``) solves the same minimum by preconditioned CG on ``u = p / sqrt(s)``.

Physical restrictions (hard, by bounded maps; no free amplitude, no gate):

- ``chi_i = chi0_Z + dchi_max tanh(a_Z + b_Z (sqrt(CN_i + 1) - 1))``. ``chi0_Z = (IP + EA) / 2``, the Mulliken
  electronegativity; ``CN_i`` the D3 coordination number (counting function ``1 / (1 + exp(-16 (4/3 R_ij / r -
  1)))``, ``R_ij`` the sum of covalent radii, switched off smoothly between ``cn_on`` and ``cn_off``), so ``chi``
  depends on the environment as in the D4 EEQ model. ``a = b = 0`` at the start.
- ``J_i = J0_Z exp(m + h tanh(c_Z))`` in ``[J_lo J0_Z, J_hi J0_Z]``, ``J0_Z = IP - EA`` (the chemical hardness
  2 eta), ``c_Z`` started where the factor is 1.
- ``kappa_ij(r) = kappa0_ij exp((r - r0_ij) / w) / f_c(r)``, i.e. ``s_ij = exp(-(r - r0_ij) / w) f_c(r) /
  kappa0_ij``: positive, increasing in ``r``, finite at short range, infinite at ``r_ct`` (``f_c`` the C2
  smootherstep from 1 at ``r_ct - r_ct_width`` to 0 at ``r_ct``; ``s`` is C2 everywhere, so the charges and
  energy are C2 in the positions). ``r0_ij`` the sum of covalent radii (fixed), ``kappa0_ij = sqrt(kappa_i
  kappa_j)``, ``kappa_Z`` log-uniformly bounded in ``[kappa_lo, kappa_hi]``, ``w`` in ``[w_lo, w_hi]``.
- ``gamma_ij = KE erf(r / sqrt(sigma_i^2 + sigma_j^2)) / r`` (Gaussian charges ``exp(-r^2 / sigma^2)``), with
  ``sigma_i = KE sqrt(2 / pi) / (J_lo J0_Z)`` fixed per element: the Gaussian self-interaction is then
  ``J_lo J0_Z <= J_i``, so ``A`` is positive definite and the minimum is unique for every parameter value
  (no polarisation catastrophe).

Periodic structures: the Coulomb matrix is the point-charge Ewald sum with a neutralising background (the
sigma-independent convention of ``fix/les-charged-background``) plus the short-range Gaussian correction
``-sum_n erfc(|r_ij + n| / sigma_ij) / |r_ij + n|``; ``q0 = Q / N``.

Atomic data: first ionisation energies from the NIST Atomic Spectra Database (A. Kramida, Yu. Ralchenko,
J. Reader and NIST ASD Team, NIST ASD ver. 5.x, https://physics.nist.gov/asd), electron affinities from the
compilation in mendeleev 1.3.0 (Rienstra-Kiracofe et al., Chem. Rev. 102, 231 (2002); Andersen, Haugen & Hotop,
J. Phys. Chem. Ref. Data 28, 1511 (1999); CRC Handbook), both via mendeleev 1.3.0. Elements whose anion is
unbound (negative or missing EA: noble gases, Be, N, Mg, Mn, Zn, Cd, Hg, several lanthanides and actinides) use
EA = 0. Covalent radii: ASE (Cordero et al., Dalton Trans. 2008).
"""

import math
from typing import Dict, List, Optional, Tuple

import torch


KE = 14.399645  # eV A / e^2

# first ionisation energies (eV), Z = 1..103 (NIST ASD via mendeleev 1.3.0)
IP_EV = [
    13.5984, 24.5874, 5.3917, 9.3227, 8.2980, 11.2603, 14.5341, 13.6181, 17.4228, 21.5645,
    5.1391, 7.6462, 5.9858, 8.1517, 10.4867, 10.3600, 12.9676, 15.7596, 4.3407, 6.1132,
    6.5615, 6.8281, 6.7462, 6.7665, 7.4340, 7.9025, 7.8810, 7.6399, 7.7264, 9.3942,
    5.9993, 7.8994, 9.7886, 9.7524, 11.8138, 13.9996, 4.1771, 5.6949, 6.2173, 6.6341,
    6.7588, 7.0924, 7.1194, 7.3605, 7.4589, 8.3368, 7.5762, 8.9938, 5.7864, 7.3439,
    8.6084, 9.0098, 10.4512, 12.1298, 3.8939, 5.2117, 5.5769, 5.5386, 5.4702, 5.5250,
    5.5819, 5.6437, 5.6704, 6.1498, 5.8638, 5.9391, 6.0215, 6.1077, 6.1844, 6.2542,
    5.4259, 6.8251, 7.5496, 7.8640, 7.8335, 8.4382, 8.9670, 8.9588, 9.2256, 10.4375,
    6.1083, 7.4167, 7.2855, 8.4181, 9.3175, 10.7485, 4.0727, 5.2784, 5.3802, 6.3067,
    5.8900, 6.1940, 6.2655, 6.0258, 5.9738, 5.9922, 6.1978, 6.2819, 6.3684, 6.5000,
    6.5800, 6.6262, 4.9600,
]
# electron affinities (eV), Z = 1..103 (mendeleev 1.3.0 compilation; unbound anions -> 0)
EA_EV = [
    0.7542, 0.0000, 0.6180, 0.0000, 0.2797, 1.2621, 0.0000, 1.4611, 3.4012, 0.0000,
    0.5479, 0.0000, 0.4328, 1.3895, 0.7466, 2.0771, 3.6127, 0.0000, 0.5015, 0.0245,
    0.1880, 0.0790, 0.5250, 0.6660, 0.0000, 0.1510, 0.6623, 1.1560, 1.2350, 0.0000,
    0.4300, 1.2327, 0.8040, 2.0207, 3.3636, 0.0000, 0.4859, 0.0480, 0.3070, 0.4260,
    0.9174, 0.7480, 0.5500, 1.0500, 1.1370, 0.5620, 1.3020, 0.0000, 0.3000, 1.1121,
    1.0460, 1.9709, 3.0590, 0.0000, 0.4716, 0.1446, 0.4700, 0.6500, 0.9620, 1.9160,
    0.0000, 0.0000, 0.8640, 0.0000, 1.1650, 0.3520, 0.0000, 0.0000, 1.0290, 0.0000,
    0.3400, 0.0140, 0.3220, 0.8163, 0.1500, 1.1000, 1.5638, 2.1280, 2.3086, 0.0000,
    0.3770, 0.3567, 0.9424, 1.9000, 2.8000, 0.0000, 0.4860, 0.1000, 0.3500, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000,
]


def mulliken_chi(z: int) -> float:
    """``(IP + EA) / 2`` (eV)."""
    return 0.5 * (IP_EV[z - 1] + EA_EV[z - 1])


def hardness_J(z: int) -> float:
    """``IP - EA`` (eV)."""
    return IP_EV[z - 1] - EA_EV[z - 1]


def smootherstep(x: torch.Tensor) -> torch.Tensor:
    """``6x^5 - 15x^4 + 10x^3`` on [0, 1], clamped (C2)."""
    x = x.clamp(0.0, 1.0)
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


def switch_off(r: torch.Tensor, r_on: float, r_off: float) -> torch.Tensor:
    """1 below ``r_on``, 0 above ``r_off``, C2 in between."""
    return 1.0 - smootherstep((r - r_on) / (r_off - r_on))


DEFAULTS: Dict[str, float] = dict(
    r_ct=6.0, r_ct_width=1.0, dchi_max=2.0, J_lo=0.75, J_hi=2.0, kappa_lo=1.0, kappa_hi=100.0, kappa_init=10.0,
    w_lo=0.2, w_hi=2.0, w_init=0.75, cn_on=4.0, cn_off=5.0, ewald_alpha=0.0,
)


def _logit(x: float) -> float:
    return math.log(x / (1.0 - x))


class SQEParams(torch.nn.Module):
    """The bounded SQE parameters for a list of atomic numbers (index = species index)."""

    def __init__(self, atomic_numbers: List[int], cfg: Optional[Dict[str, float]] = None) -> None:
        super().__init__()
        c = {**DEFAULTS, **(cfg or {})}
        self.cfg = c
        zs = [int(z) for z in atomic_numbers]
        self.atomic_numbers = zs
        E = len(zs)
        dt = torch.float64
        from ase.data import covalent_radii

        self.register_buffer("chi0", torch.tensor([mulliken_chi(z) for z in zs], dtype=dt))
        self.register_buffer("J0", torch.tensor([hardness_J(z) for z in zs], dtype=dt))
        self.register_buffer("rcov", torch.tensor([covalent_radii[z] for z in zs], dtype=dt))
        self.register_buffer("sigma", KE * math.sqrt(2.0 / math.pi) / (c["J_lo"] * self.J0))
        lJ = (math.log(c["J_hi"]) + math.log(c["J_lo"])) / 2
        hJ = (math.log(c["J_hi"]) - math.log(c["J_lo"])) / 2
        self.lJ, self.hJ = lJ, hJ
        self.chi_a = torch.nn.Parameter(torch.zeros(E, dtype=dt))
        self.chi_b = torch.nn.Parameter(torch.zeros(E, dtype=dt))
        self.j_z = torch.nn.Parameter(torch.full((E,), math.atanh(-lJ / hJ), dtype=dt))
        lk = (math.log(c["kappa_init"]) - math.log(c["kappa_lo"])) / (math.log(c["kappa_hi"]) - math.log(c["kappa_lo"]))
        self.kappa_z = torch.nn.Parameter(torch.full((E,), _logit(lk), dtype=dt))
        self.w_raw = torch.nn.Parameter(torch.full((1,), _logit((c["w_init"] - c["w_lo"]) / (c["w_hi"] - c["w_lo"])),
                                                   dtype=dt))

    def _apply(self, fn, recurse=True):
        """Device moves apply; dtype casts do not: the parameters and tables stay float64 (as mlip.cpp's)."""
        super()._apply(fn, recurse)
        for t in list(self._parameters.values()) + list(self._buffers.values()):
            if t is not None and t.is_floating_point() and t.dtype != torch.float64:
                t.data = t.data.to(torch.float64)
        return self

    # bounded maps --------------------------------------------------------------------------------------------
    def J(self, e: torch.Tensor) -> torch.Tensor:
        return self.J0[e] * torch.exp(self.lJ + self.hJ * torch.tanh(self.j_z[e]))

    def kappa0(self) -> torch.Tensor:
        c = self.cfg
        lo, hi = math.log(c["kappa_lo"]), math.log(c["kappa_hi"])
        return torch.exp(lo + (hi - lo) * torch.sigmoid(self.kappa_z))

    def w(self) -> torch.Tensor:
        c = self.cfg
        return c["w_lo"] + (c["w_hi"] - c["w_lo"]) * torch.sigmoid(self.w_raw[0])

    def chi(self, e: torch.Tensor, cn: torch.Tensor) -> torch.Tensor:
        x = self.chi_a[e] + self.chi_b[e] * (torch.sqrt(cn + 1.0) - 1.0)
        return self.chi0[e] + self.cfg["dchi_max"] * torch.tanh(x)

    def table(self) -> Dict[str, List[float]]:
        """Per element: chi offset (at CN = 0), CN slope (tanh argument), J, kappa0; global w."""
        with torch.no_grad():
            e = torch.arange(len(self.atomic_numbers))
            return dict(Z=self.atomic_numbers, chi0=self.chi0.tolist(), J0=self.J0.tolist(),
                        dchi=(self.cfg["dchi_max"] * torch.tanh(self.chi_a)).tolist(), chi_b=self.chi_b.tolist(),
                        J=self.J(e).tolist(), J_factor=(self.J(e) / self.J0).tolist(), kappa0=self.kappa0().tolist(),
                        w=[float(self.w())], sigma=self.sigma.tolist())


# geometry helpers ----------------------------------------------------------------------------------------------
def _pair_r(pos: torch.Tensor) -> torch.Tensor:
    n = pos.shape[0]
    eye = torch.eye(n, dtype=pos.dtype, device=pos.device)
    d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    return torch.sqrt(d2 + eye)  # 1 on the diagonal (masked by the callers)


def coordination(P: SQEParams, e: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """D3 coordination numbers (open structures; ``r`` the full distance matrix)."""
    n = r.shape[0]
    off = 1.0 - torch.eye(n, dtype=r.dtype, device=r.device)
    R = P.rcov[e][:, None] + P.rcov[e][None, :]
    cnt = torch.sigmoid(16.0 * (4.0 / 3.0 * R / r - 1.0))
    return (cnt * switch_off(r, P.cfg["cn_on"], P.cfg["cn_off"]) * off).sum(1)


def softness(P: SQEParams, e: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """``s_ij = 1 / kappa_ij(r)`` (symmetric, zero on the diagonal and beyond ``r_ct``)."""
    n = r.shape[0]
    off = 1.0 - torch.eye(n, dtype=r.dtype, device=r.device)
    k = P.kappa0()[e]
    k0 = torch.sqrt(k[:, None] * k[None, :])
    r0 = P.rcov[e][:, None] + P.rcov[e][None, :]
    c = P.cfg
    fc = switch_off(r, c["r_ct"] - c["r_ct_width"], c["r_ct"])
    return torch.exp(-(r - r0) / P.w()) / k0 * fc * off


def gaussian_coulomb_open(P: SQEParams, e: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    n = r.shape[0]
    off = 1.0 - torch.eye(n, dtype=r.dtype, device=r.device)
    sij = torch.sqrt(P.sigma[e][:, None] ** 2 + P.sigma[e][None, :] ** 2)
    return KE * torch.erf(r / sij) / r * off


def _images(cell: torch.Tensor, rc: float) -> torch.Tensor:
    """Integer lattice shifts covering a sphere of radius ``rc`` (plus a cell) around any point in the cell."""
    inv = torch.linalg.inv(cell)
    # perpendicular widths: 1 / |row of inv^T|
    nmax = [int(math.ceil(rc * float(torch.linalg.norm(inv[:, a])))) + 1 for a in range(3)]
    rng = [torch.arange(-m, m + 1, dtype=cell.dtype) for m in nmax]
    return torch.cartesian_prod(*rng)


def gaussian_coulomb_ewald(P: SQEParams, e: torch.Tensor, pos: torch.Tensor, cell: torch.Tensor,
                           alpha: float = 0.0, tol: float = 1e-10) -> torch.Tensor:
    """Periodic Coulomb matrix (eV / e^2) of Gaussian charges: point-charge Ewald with a neutralising background
    (``1/2 q^T A q`` then contains ``-pi Q^2 / (2 V alpha^2)``) plus ``-sum_n erfc(|r + n| / sigma_ij) / |r + n|``.
    The ``i = j, n = 0`` term is excluded (on-site energies are ``J``)."""
    n = pos.shape[0]
    dt = pos.dtype
    V = torch.abs(torch.det(cell))
    L = float(V) ** (1.0 / 3.0)
    if alpha <= 0.0:
        alpha = 5.0 / L
    lt = math.sqrt(-math.log(tol))
    rc = lt / alpha
    sij = torch.sqrt(P.sigma[e][:, None] ** 2 + P.sigma[e][None, :] ** 2)
    rc_g = lt * float(sij.max())
    shifts = _images(cell, max(rc, rc_g)) @ cell  # (S, 3)
    d = pos[None, :, None, :] - pos[:, None, None, :] + shifts[None, None, :, :]  # r_j - r_i + n
    is0 = (shifts.abs().sum(-1) == 0)
    self_mask = torch.eye(n, dtype=torch.bool)[:, :, None] & is0[None, None, :]
    dist = torch.sqrt((d ** 2).sum(-1) + self_mask.to(dt))
    keep = (~self_mask).to(dt)
    real = (torch.erfc(alpha * dist) * (dist < rc).to(dt) - torch.erfc(dist / sij[:, :, None])
            * (dist < rc_g).to(dt)) / dist * keep
    A = real.sum(-1)
    # reciprocal
    rec = 2.0 * math.pi * torch.linalg.inv(cell).T  # rows b_a
    kmax = 2.0 * alpha * lt
    nk = [int(math.ceil(kmax / float(torch.linalg.norm(rec[a])))) for a in range(3)]
    grid = torch.cartesian_prod(*[torch.arange(-m, m + 1, dtype=dt) for m in nk])
    k = grid @ rec
    k2 = (k ** 2).sum(-1)
    sel = (k2 > 0) & (k2 < kmax ** 2)
    k, k2 = k[sel], k2[sel]
    w = 4.0 * math.pi / V * torch.exp(-k2 / (4 * alpha * alpha)) / k2  # (K,)
    ph = pos @ k.T  # (n, K)
    c, s = torch.cos(ph), torch.sin(ph)
    A = A + (c * w) @ c.T + (s * w) @ s.T
    A = A - torch.eye(n, dtype=dt) * (2.0 * alpha / math.sqrt(math.pi))
    A = A - math.pi / (V * alpha * alpha)
    return KE * A


def sqe_system(P: SQEParams, e: torch.Tensor, pos: torch.Tensor, Q: torch.Tensor,
               q0: Optional[torch.Tensor] = None, cell: Optional[torch.Tensor] = None,
               periodic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """One structure: ``(per-atom energies (eV, sum E_SQE), charges q (e), extras)``. ``e`` species indices.
    float64 throughout (inputs are promoted); differentiable in ``pos`` and the parameters."""
    out_dtype = pos.dtype
    pos = pos.to(torch.float64)
    n = pos.shape[0]
    Qd = Q.to(torch.float64).reshape(())
    if q0 is None:
        q0 = Qd.expand(n) / n
    else:
        q0 = q0.to(torch.float64)
    r = _pair_r(pos)
    cn = coordination(P, e, r) if not periodic else coordination_periodic(P, e, pos, cell)
    chi = P.chi(e, cn)
    J = P.J(e)
    if periodic:
        assert cell is not None
        gam = gaussian_coulomb_ewald(P, e, pos, cell.to(torch.float64), P.cfg["ewald_alpha"])
        s = softness_periodic(P, e, pos, cell.to(torch.float64))
    else:
        gam = gaussian_coulomb_open(P, e, r)
        s = softness(P, e, r)
    A = gam + torch.diag(J)
    Lm = torch.diag(s.sum(1)) - s
    eye = torch.eye(n, dtype=torch.float64)
    q = torch.linalg.solve(eye + Lm @ A, (q0 - Lm @ chi)[:, None])[:, 0]
    mu = chi + A @ q
    dmu = mu[:, None] - mu[None, :]
    bond = 0.25 * (s * dmu * dmu).sum(1)  # 1/2 sum_pairs, split half-half over the two atoms
    e_atom = chi * q + 0.5 * q * (A @ q) + bond
    p = -s * dmu  # p_ij (antisymmetric)
    return e_atom.to(out_dtype), q.to(out_dtype), dict(chi=chi, J=J, cn=cn, mu=mu, p=p, s=s)


def coordination_periodic(P: SQEParams, e: torch.Tensor, pos: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    d, dist, keep = _periodic_pairs(pos, cell, P.cfg["cn_off"])
    R = (P.rcov[e][:, None] + P.rcov[e][None, :])[:, :, None]
    cnt = torch.sigmoid(16.0 * (4.0 / 3.0 * R / dist - 1.0))
    return (cnt * switch_off(dist, P.cfg["cn_on"], P.cfg["cn_off"]) * keep).sum((1, 2))


def softness_periodic(P: SQEParams, e: torch.Tensor, pos: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Image-summed softness ``sum_n s_ij(|r_ij + n|)``: split charges of all images of a pair act on the same
    two atoms (minimum image when ``r_ct`` is below half the cell)."""
    c = P.cfg
    d, dist, keep = _periodic_pairs(pos, cell, c["r_ct"])
    k = P.kappa0()[e]
    k0 = torch.sqrt(k[:, None] * k[None, :])[:, :, None]
    r0 = (P.rcov[e][:, None] + P.rcov[e][None, :])[:, :, None]
    fc = switch_off(dist, c["r_ct"] - c["r_ct_width"], c["r_ct"])
    s = (torch.exp(-(dist - r0) / P.w()) / k0 * fc * keep).sum(-1)
    return s * (1.0 - torch.eye(pos.shape[0], dtype=pos.dtype))


def _periodic_pairs(pos: torch.Tensor, cell: torch.Tensor, rc: float):
    n = pos.shape[0]
    shifts = _images(cell, rc) @ cell
    d = pos[None, :, None, :] - pos[:, None, None, :] + shifts[None, None, :, :]
    is0 = (shifts.abs().sum(-1) == 0)
    self_mask = torch.eye(n, dtype=torch.bool)[:, :, None] & is0[None, None, :]
    dist = torch.sqrt((d ** 2).sum(-1) + self_mask.to(pos.dtype))
    keep = ((~self_mask) & (dist < rc)).to(pos.dtype)
    return d, dist, keep


class PETSQE(torch.nn.Module):
    """plato_lr ``sqe``: the SQE energy added to the energy targets, charges from the total charge (and optional
    formal charges). ``backbone_charge: neutral`` conditions the backbone on charge 0 (the ablation in which the
    total charge enters only through SQE)."""

    def __init__(self, atomic_numbers: List[int], cfg: Optional[Dict[str, float]] = None, q0: str = "uniform",
                 backbone_charge: str = "input") -> None:
        super().__init__()
        if q0 not in ("uniform", "formal"):
            raise ValueError(f"sqe_q0 must be uniform or formal, not {q0}")
        if backbone_charge not in ("input", "neutral"):
            raise ValueError(f"sqe_backbone_charge must be input or neutral, not {backbone_charge}")
        self.params = SQEParams(atomic_numbers, cfg)
        self.q0_mode = q0
        self.backbone_neutral = backbone_charge == "neutral"

    def forward_system(self, e: torch.Tensor, pos: torch.Tensor, Q: torch.Tensor, formal: Optional[torch.Tensor],
                       cell: Optional[torch.Tensor], periodic: bool):
        q0 = None
        if self.q0_mode == "formal" and formal is not None and not periodic:
            if abs(float(formal.sum()) - float(Q)) > 1e-6:
                raise ValueError("sqe: formal charges must sum to the total charge")
            q0 = formal
        return sqe_system(self.params, e, pos, Q, q0, cell, periodic)


def spec_json(P: SQEParams) -> Dict[str, object]:
    """The parameters and tables in the layout of mlip.cpp's ``scripts/plato_lr/add_sqe_keys.py --spec``."""
    with torch.no_grad():
        return dict(Z=P.atomic_numbers, chi0=P.chi0.tolist(), J0=P.J0.tolist(), rcov=P.rcov.tolist(),
                    sigma=P.sigma.tolist(), chi_a=P.chi_a.tolist(), chi_b=P.chi_b.tolist(), j_z=P.j_z.tolist(),
                    kappa_z=P.kappa_z.tolist(), w_raw=float(P.w_raw[0]), cfg=dict(P.cfg))
