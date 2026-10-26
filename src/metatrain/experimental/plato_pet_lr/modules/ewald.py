"""LES energy of Gaussian latent charges (width ``sigma``), per atom.

Pair kernel ``KE erf(r / (sqrt(2) sigma)) / r``; in Fourier space
``4 pi exp(-sigma^2 k^2 / 2) / k^2``.

Two fixes over the reference ``ewald_energy``:

1. **k-set.** The reference summed a fixed integer cube ``|n_a| <= 4``, converged
   only for cells of a few Angstrom. Here every ``k`` with ``|k| < k_cut = kcut /
   sigma`` is kept, and a smooth window ``w(|k|)`` (cosine taper from ``k_cut - 1 /
   sigma`` to ``k_cut``) weighs it. The integer range ``|n_a| <= k_cut |a_a| / 2 pi``
   is a strict bound, recomputed per call from the (detached) cell, and every term it
   adds or drops has weight 0, so the energy is a C^1 function of the cell under
   strain (NPT) and does not depend on the enumeration. Truncation is
   ``exp(-(kcut - 1)^2 / 2)``: 4e-6 relative at the default ``kcut = 6``.
2. **Self energy.** The reciprocal sum includes each Gaussian with itself,
   ``KE q^2 / (sqrt(2 pi) sigma)``; the finite (erf pair) path does not. It is
   subtracted, so a large periodic box gives the finite energy up to the tin-foil
   dipole term ``-KE 2 pi |M|^2 / (3 V)`` (cubic cell) and ``O(L^-5)``.
"""

import math
from typing import List, Optional

import torch


KE = 14.399645  # eV Angstrom / e^2


def kvectors(cell: torch.Tensor, kcut: float) -> torch.Tensor:
    """Integer ``n`` (half space, no 0) that may have ``|2 pi n cell^-T| < kcut``."""
    lengths = torch.linalg.norm(cell.detach(), dim=-1)
    nmax: List[int] = [
        int(math.floor(kcut * float(x) / (2 * math.pi))) for x in lengths
    ]
    r = [torch.arange(-n, n + 1, device=cell.device) for n in nmax]
    n = torch.cartesian_prod(r[0], r[1], r[2])
    half = (
        (n[:, 0] > 0)
        | ((n[:, 0] == 0) & (n[:, 1] > 0))
        | ((n[:, 0] == 0) & (n[:, 1] == 0) & (n[:, 2] > 0))
    )
    n = n[half].to(cell.dtype)
    k = 2 * math.pi * n @ torch.linalg.inv(cell.detach()).T
    return n[(k * k).sum(-1) < kcut**2]


def window(k: torch.Tensor, kcut: float, width: float) -> torch.Tensor:
    x = ((k - (kcut - width)) / width).clamp(0.0, 1.0)
    return 0.5 * (1 + torch.cos(math.pi * x))


def les_energy_periodic(
    q: torch.Tensor, pos: torch.Tensor, cell: torch.Tensor, sigma: float, kcut: float
) -> torch.Tensor:
    """Per-atom energies, ``q`` is ``(atoms, channels)`` and neutral per channel."""
    kc = kcut / sigma
    n = kvectors(cell, kc)
    k = 2 * math.pi * n @ torch.linalg.inv(cell).T
    k2 = (k * k).sum(-1)
    g = window(k2.sqrt(), kc, 1.0 / sigma) * torch.exp(-0.5 * sigma**2 * k2) / k2
    ph = pos @ k.T
    c, s = torch.cos(ph), torch.sin(ph)
    C, S = q.T @ c, q.T @ s  # structure factor, (channels, nk)
    V = torch.det(cell).abs()
    # half space: 2 pi / V over all k  ->  4 pi / V over k > 0
    e = 4 * math.pi / V * (q * ((c @ (C * g).T) + (s @ (S * g).T))).sum(-1)
    self_e = (q * q).sum(-1) / (math.sqrt(2 * math.pi) * sigma)
    return 14.399645 * (e - self_e)  # KE


def les_energy_finite(q: torch.Tensor, pos: torch.Tensor, sigma: float) -> torch.Tensor:
    A = pos.shape[0]
    rij = pos[:, None] - pos[None]
    off = ~torch.eye(A, dtype=torch.bool, device=pos.device)
    d = ((rij * rij).sum(-1) + (~off).to(pos.dtype)).sqrt()  # 1 on the diagonal
    pair = torch.erf(d / (math.sqrt(2) * sigma)) / d * off.to(pos.dtype)
    return 0.5 * 14.399645 * (q * (pair @ q)).sum(-1)  # KE


def les_energy(
    q: torch.Tensor,
    pos: torch.Tensor,
    cell: torch.Tensor,
    periodic: bool,
    sigma: float,
    kcut: float,
) -> torch.Tensor:
    if periodic:
        return les_energy_periodic(q, pos, cell, sigma, kcut)
    return les_energy_finite(q, pos, sigma)


def kset_host(
    nmax: List[int], cells: List[List[float]], kcut: float, margin: float = 1e-4
) -> torch.Tensor:
    """Integer ``n`` of :func:`kvectors` for a group of cells with the same ``nmax``:
    the union over ``cells`` (rows of 9, float64 on the host) of ``|k| < kcut``, with
    a relative ``margin`` so the set contains every cell's set as computed on the
    device; each structure then masks its own terms exactly. Same order as
    :func:`kvectors`."""
    r = [torch.arange(-n, n + 1) for n in nmax]
    n = torch.cartesian_prod(r[0], r[1], r[2])
    half = (
        (n[:, 0] > 0)
        | ((n[:, 0] == 0) & (n[:, 1] > 0))
        | ((n[:, 0] == 0) & (n[:, 1] == 0) & (n[:, 2] > 0))
    )
    n = n[half]
    nf = n.to(torch.float64)
    keep = torch.zeros(n.shape[0], dtype=torch.bool)
    for c in cells:
        cell = torch.tensor(c, dtype=torch.float64).reshape(3, 3)
        k = 2 * math.pi * nf @ torch.linalg.inv(cell).T
        keep = keep | ((k * k).sum(-1) < kcut**2 * (1 + margin))
    return n[keep]


def les_energy_periodic_batched(
    q: torch.Tensor,
    pos: torch.Tensor,
    cells: torch.Tensor,
    umap: Optional[torch.Tensor],
    n: torch.Tensor,
    sigma: float,
    kcut: float,
) -> torch.Tensor:
    """:func:`les_energy_periodic` of ``S`` structures padded to ``P`` atoms.

    ``q`` ``(S, P, channels)`` (zero on padded slots, neutral per structure),
    ``pos`` ``(S, P, 3)``, ``cells`` ``(U, 3, 3)`` as in
    :meth:`MeshCoupling.mesh_batched`, ``n`` a superset of every cell's k-set
    (:func:`kset_host`); terms outside a structure's own set get weight exactly 0.
    Returns ``(S, P)``."""
    kc = kcut / sigma
    k = 2 * math.pi * n @ torch.linalg.inv(cells).transpose(-1, -2)  # (U, nk, 3)
    kd = k.detach()
    inside = (kd * kd).sum(-1) < kc**2
    k2 = (k * k).sum(-1)
    g = window(k2.sqrt(), kc, 1.0 / sigma) * torch.exp(-0.5 * sigma**2 * k2) / k2
    g = g * inside
    V = torch.det(cells).abs()
    S, P = pos.shape[0], pos.shape[1]
    if cells.shape[0] == 1:
        ph = (pos.reshape(S * P, 3) @ k[0].T).reshape(S, P, -1)
    else:
        if umap is not None:
            k, g, V = k[umap], g[umap], V[umap]
        ph = pos @ k.transpose(1, 2)
    c, s = torch.cos(ph), torch.sin(ph)
    qT = q.transpose(1, 2)
    C, Sk = qT @ c, qT @ s  # structure factor, (S, channels, nk)
    g = g[:, None, :]
    e = (4 * math.pi / V)[:, None] * (
        q * (c @ (C * g).transpose(1, 2) + s @ (Sk * g).transpose(1, 2))
    ).sum(-1)
    self_e = (q * q).sum(-1) / (math.sqrt(2 * math.pi) * sigma)
    return 14.399645 * (e - self_e)  # KE


def les_energy_finite_batched(
    q: torch.Tensor, pos: torch.Tensor, valid: torch.Tensor, sigma: float
) -> torch.Tensor:
    """:func:`les_energy_finite` of ``S`` padded structures; ``valid`` ``(S, P)``."""
    P = pos.shape[1]
    rij = pos[:, :, None] - pos[:, None]
    eye = torch.eye(P, dtype=torch.bool, device=pos.device)
    off = ~eye & valid[:, :, None] & valid[:, None, :]
    d = ((rij * rij).sum(-1) + (~off).to(pos.dtype)).sqrt()
    pair = torch.erf(d / (math.sqrt(2) * sigma)) / d * off.to(pos.dtype)
    return 0.5 * 14.399645 * (q * (pair @ q)).sum(-1)  # KE
