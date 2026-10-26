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


def _reciprocal(
    q: torch.Tensor, pos: torch.Tensor, k: torch.Tensor, g: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:  # fmt: skip
    """``4 pi / V q_i sum_k g_k (cos_ik C_k + sin_ik S_k)``, ``(S, P)``; ``k`` ``(1 or
    S, nk, 3)``, ``g`` ``(1 or S, nk)``, ``V`` ``(1 or S,)``."""
    S, P = pos.shape[0], pos.shape[1]
    if k.shape[0] == 1:
        ph = (pos.reshape(S * P, 3) @ k[0].T).reshape(S, P, -1)
    else:
        ph = pos @ k.transpose(1, 2)
    c, s = torch.cos(ph), torch.sin(ph)
    qT = q.transpose(1, 2)
    C, Sk = qT @ c, qT @ s  # structure factor, (S, channels, nk)
    g = g[:, None, :]
    return (4 * math.pi / V)[:, None] * (
        q * (c @ (C * g).transpose(1, 2) + s @ (Sk * g).transpose(1, 2))
    ).sum(-1)


def _reciprocal_chunked(
    q: torch.Tensor, pos: torch.Tensor, k: torch.Tensor, g: torch.Tensor,
    V: torch.Tensor, chunk: int,
) -> torch.Tensor:  # fmt: skip
    """:func:`_reciprocal` in chunks of k, each differentiated on its own: the
    memory is ``O(S P chunk)`` instead of ``O(S P nk)``. The value is the same sum;
    the gradient of the total (forces, stress, parameter gradients) is exact, but it
    is attached as a first-order term, so there are no second derivatives and the
    split of the gradient between atoms is not that of the per-atom energies."""
    S, P, nk = pos.shape[0], pos.shape[1], k.shape[1]
    step = max(1, chunk // max(1, S * P))
    need = torch.is_grad_enabled() and (
        q.requires_grad or pos.requires_grad or k.requires_grad or g.requires_grad
    )
    cell_grad = need and (k.requires_grad or g.requires_grad or V.requires_grad)
    qd, pd, kd, gd, Vd = q.detach(), pos.detach(), k.detach(), g.detach(), V.detach()
    e = torch.zeros(S, P, dtype=q.dtype, device=q.device)
    if not need:
        for a in range(0, nk, step):
            e = e + _reciprocal(qd, pd, kd[:, a : a + step], gd[:, a : a + step], Vd)
        return e
    gq, gp, gV = torch.zeros_like(q), torch.zeros_like(pos), torch.zeros_like(V)
    gk: List[torch.Tensor] = []
    gg: List[torch.Tensor] = []
    ql, pl = qd.requires_grad_(), pd.requires_grad_()
    for a in range(0, nk, step):
        kl = kd[:, a : a + step].detach().requires_grad_(cell_grad)
        gl = gd[:, a : a + step].detach().requires_grad_(cell_grad)
        Vl = Vd.detach().requires_grad_(cell_grad)
        ec = _reciprocal(ql, pl, kl, gl, Vl)  # grad mode is on (``need``)
        inputs = [ql, pl]
        if cell_grad:
            inputs = [ql, pl, kl, gl, Vl]
        grads = torch.autograd.grad([ec.sum()], inputs)
        e = e + ec.detach()
        d0, d1 = grads[0], grads[1]
        if d0 is not None:
            gq = gq + d0
        if d1 is not None:
            gp = gp + d1
        if cell_grad:
            d2, d3, d4 = grads[2], grads[3], grads[4]
            gk.append(torch.zeros_like(kl) if d2 is None else d2)
            gg.append(torch.zeros_like(gl) if d3 is None else d3)
            if d4 is not None:
                gV = gV + d4
    # value + 0: x - x.detach() is exactly zero, its gradient is the chunked one
    lin = ((q - q.detach()) * gq).sum(-1) + ((pos - pos.detach()) * gp).sum(-1)
    if cell_grad:
        lk = (
            ((k - kd) * torch.cat(gk, 1)).sum([1, 2])
            + ((g - gd) * torch.cat(gg, 1)).sum(1)
            + (V - Vd) * gV
        )  # (1 or S,), onto the first atom of a structure
        first = torch.zeros(S, P, dtype=q.dtype, device=q.device)
        first[: lk.shape[0], 0] = 1.0
        lin = lin + first * lk.sum() if lk.shape[0] == 1 else lin + first * lk[:, None]
    return e + lin


def les_energy_periodic_batched(
    q: torch.Tensor,
    pos: torch.Tensor,
    cells: torch.Tensor,
    umap: Optional[torch.Tensor],
    n: torch.Tensor,
    sigma: float,
    kcut: float,
    chunk: int = 0,
) -> torch.Tensor:
    """:func:`les_energy_periodic` of ``S`` structures padded to ``P`` atoms.

    ``q`` ``(S, P, channels)`` (zero on padded slots, neutral per structure),
    ``pos`` ``(S, P, 3)``, ``cells`` ``(U, 3, 3)`` as in
    :meth:`MeshCoupling.mesh_batched`, ``n`` a superset of every cell's k-set
    (:func:`kset_host`); terms outside a structure's own set get weight exactly 0.
    With ``chunk > 0`` and more than ``4 chunk`` phase elements ``S P nk``, the sum
    runs in chunks of ``chunk`` elements (:func:`_reciprocal_chunked`, first
    derivatives only; for evaluation of large cells).
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
    if cells.shape[0] > 1 and umap is not None:
        k, g, V = k[umap], g[umap], V[umap]
    if chunk > 0 and S * P * n.shape[0] > 4 * chunk:
        e = _reciprocal_chunked(q, pos, k, g, V, chunk)
    else:
        e = _reciprocal(q, pos, k, g, V)
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
