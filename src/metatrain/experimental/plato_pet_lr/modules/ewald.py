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

The direct sum costs ``O(atoms x k vectors)`` and the k-set grows with the cell
volume, i.e. ``O(N^2)`` at a fixed density. :func:`les_energy_pme_batched` is the
same windowed sum by smooth particle-mesh Ewald (``O(N log N)``): order-``p``
B-spline charge assignment on a mesh of spacing ``h``, ``rfftn``, the influence
function, ``irfftn``, interpolation back to the atoms (the potential, so per-atom
energies ``q_i psi_i``). The influence function is the energy-optimal one of
Hockney & Eastwood for B-splines, ``G(m) = g(k) U^2(m) / (sum_n U^2(m + n M))^2``
with ``g(k) = 4 pi w(|k|) exp(-sigma^2 k^2 / 2) / k^2`` the windowed kernel of the
direct sum, ``U = prod_a sinc^p(pi m_a / M_a)`` and the alias sum
``prod_a sum_j B_2p(j) cos(2 pi m_a j / M_a)`` (order ``2p`` B-spline at the
integers); separable, so it is a product of three per-axis tables. With
``h < pi sigma / kcut`` the mesh contains every ``k`` of the direct sum and every
Nyquist bin has ``|k| > kcut / sigma`` (weight 0), so the only error is the
interpolation (aliasing) error: relative, for random neutral charges, order 8,
``h = 0.5 sigma``: energy ~1e-6, forces ~1e-5, stress ~2e-6 (order 6: 1e-5, 2e-4,
2e-5). The mesh is a function of the cell lengths only, so a rotated cell gives
the same energy; it changes (by the interpolation error) where ``|a_i| / h``
crosses a half-integer below an FFT size (:func:`pme_mesh`).
"""

import math
from typing import List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# smooth particle-mesh Ewald (PME) of the same windowed reciprocal sum


def fft_size(n: int) -> int:
    """Smallest ``2^a 3^b 5^c >= n``."""
    m = max(1, n)
    x = m
    while x != 1:
        x = m
        for f in [2, 3, 5]:
            while x % f == 0:
                x = x // f
        if x != 1:
            m += 1
    return m


def pme_mesh(
    lengths: List[float], sigma: float, spacing: float, order: int
) -> List[int]:
    """Mesh of :func:`les_energy_pme_batched`: per cell vector the FFT size at or
    above ``floor(|a| / h + 1/2) + 1``, ``h = spacing sigma`` (at least ``order``
    points). The ``+ 1/2`` puts the size changes at half-integer ``|a| / h``, away
    from the integer box lengths of typical cells and their rotated copies."""
    h = spacing * sigma
    out: List[int] = []
    for x in lengths:
        out.append(fft_size(max(order, int(math.floor(x / h + 0.5)) + 1)))
    return out


def _bspline_values(t: torch.Tensor, order: int) -> List[torch.Tensor]:
    """``c[k] = M_p(t + k)``, ``k = 0 .. p - 1``, ``t`` in ``[0, 1)`` (cardinal
    B-spline on ``[0, p]``, Essmann et al. 1995 recursion)."""
    c: List[torch.Tensor] = [torch.ones_like(t)]
    for n in range(1, order):
        new: List[torch.Tensor] = []
        for k in range(n + 1):
            x = t + float(k)
            v = torch.zeros_like(t)
            if k < n:
                v = v + x * c[k]
            if k >= 1:
                v = v + (float(n + 1) - x) * c[k - 1]
            new.append(v / float(n))
        c = new
    return c


def bspline_stencil(
    frac: torch.Tensor, mesh: List[int], order: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Order-``p`` (even) B-spline assignment of points at ``frac`` ``(A, 3)``: the
    flat mesh indices and weights ``(A, p^3)`` of the points ``floor(u) - p/2 + 1 ..
    floor(u) + p/2`` per axis, ``u = frac M`` (order 4: ``bspline4_stencil``)."""
    K = torch.tensor(mesh, device=frac.device)
    u = frac * K.to(frac.dtype)
    fl = torch.floor(u.detach())
    t = u - fl
    c = _bspline_values(t, order)
    w = torch.stack(c[::-1], -1)  # (A, 3, p): weight of point s is M_p(t + p - 1 - s)
    shift = torch.arange(order, device=frac.device) - (order // 2 - 1)
    i = (fl.long()[..., None] + shift) % K[:, None]
    idx = (i[:, 0, :, None, None] * mesh[1] + i[:, 1, None, :, None]) * mesh[2] + i[
        :, 2, None, None, :
    ]
    wt = w[:, 0, :, None, None] * w[:, 1, None, :, None] * w[:, 2, None, None, :]
    p3 = order * order * order
    return idx.reshape(-1, p3), wt.reshape(-1, p3)


def pme_axis_factor(
    M: int, order: int, dtype: torch.dtype, device: torch.device, half: bool
) -> torch.Tensor:
    """``U^2(m) / (sum_n U^2(m + n M))^2`` along one axis, ``U = sinc^p(pi m / M)``
    and the alias sum ``sum_j B_2p(j) cos(2 pi m j / M)``; 0 on the Nyquist bin
    (even ``M``). ``half``: the ``rfft`` bins ``0 .. M // 2``, else ``fftfreq``."""
    m = torch.fft.fftfreq(M, 1.0 / M, dtype=torch.float64)
    if half:
        m = torch.fft.rfftfreq(M, 1.0 / M, dtype=torch.float64)
    c = _bspline_values(torch.zeros(1, dtype=torch.float64), 2 * order)
    alias = torch.zeros_like(m)
    for k in range(2 * order):
        j = k - order  # M_2p(0 + k) is the centred spline at j = k - p
        alias = alias + c[k] * torch.cos(2 * math.pi * m * float(j) / M)
    x = math.pi * m / M
    sinc = torch.where(
        x == 0,
        torch.ones_like(x),
        torch.sin(x) / torch.where(x == 0, torch.ones_like(x), x),
    )
    f = sinc ** (2 * order) / alias**2
    if M % 2 == 0:
        f = torch.where(m.abs() == M // 2, torch.zeros_like(f), f)
    return f.to(dtype=dtype, device=device)


def pme_influence(
    cells: torch.Tensor, mesh: List[int], sigma: float, kcut: float, order: int
) -> torch.Tensor:
    """``(U, M0, M1, M2 // 2 + 1)``: ``(2 pi N / V) G(m)`` of the ``U`` cells, the
    factor that turns ``rfftn(rho)`` into ``rfftn`` of the mesh potential (``N =
    M0 M1 M2``). ``|k|^2 = 4 pi^2 m^T B B^T m``, ``B = cell^-T``, from the 1-D
    frequencies (no ``(.., 3)`` k vectors on the mesh); differentiable in the cell."""
    dt, dev = cells.dtype, cells.device
    m0 = torch.fft.fftfreq(mesh[0], 1.0 / mesh[0], dtype=dt, device=dev)[:, None, None]
    m1 = torch.fft.fftfreq(mesh[1], 1.0 / mesh[1], dtype=dt, device=dev)[None, :, None]
    m2 = torch.fft.rfftfreq(mesh[2], 1.0 / mesh[2], dtype=dt, device=dev)[None, None, :]
    inv = torch.linalg.inv(cells)  # (U, 3, 3); k = 2 pi m inv^T
    Q = (4 * math.pi**2) * inv.transpose(-1, -2) @ inv  # k^2 = m Q m^T
    Q = Q[:, :, :, None, None, None]
    k2 = (
        Q[:, 0, 0] * m0 * m0 + Q[:, 1, 1] * m1 * m1 + Q[:, 2, 2] * m2 * m2
        + 2 * (Q[:, 0, 1] * m0 * m1 + Q[:, 0, 2] * m0 * m2 + Q[:, 1, 2] * m1 * m2)
    )  # fmt: skip
    nz = k2 > 0
    k2s = torch.where(nz, k2, torch.ones_like(k2))
    kc = kcut / sigma
    g = window(k2s.sqrt(), kc, 1.0 / sigma) * torch.exp(-0.5 * sigma**2 * k2s) / k2s
    g = g * nz.to(dt)
    f = (
        pme_axis_factor(mesh[0], order, dt, dev, False)[:, None, None]
        * pme_axis_factor(mesh[1], order, dt, dev, False)[None, :, None]
        * pme_axis_factor(mesh[2], order, dt, dev, True)[None, None, :]
    )
    N = mesh[0] * mesh[1] * mesh[2]
    V = torch.det(cells).abs()[:, None, None, None]
    return (2 * math.pi * N / V) * g * f


def les_energy_pme_batched(
    q: torch.Tensor,
    pos: torch.Tensor,
    cells: torch.Tensor,
    umap: Optional[torch.Tensor],
    mesh: List[int],
    sigma: float,
    kcut: float,
    order: int,
) -> torch.Tensor:
    """:func:`les_energy_periodic_batched` by PME, ``S`` structures padded to ``P``
    atoms on one mesh ``mesh`` (:func:`pme_mesh`); ``q`` ``(S, P, channels)``
    (zero on padded slots, neutral per structure), ``cells`` and ``umap`` as there.
    Plain autograd (spread, ``rfftn``, ``irfftn``, gather), so second derivatives
    (forces in a training loss) work. Returns ``(S, P)``."""
    S, P, C = q.shape
    Mt = mesh[0] * mesh[1] * mesh[2]
    inv = torch.linalg.inv(cells)
    if cells.shape[0] == 1:
        frac = pos @ inv[0]
    else:
        frac = pos @ (inv if umap is None else inv[umap])
    idx, wt = bspline_stencil(frac.reshape(S * P, 3), mesh, order)
    p3 = idx.shape[1]
    off = torch.arange(S, device=q.device) * Mt
    gidx = (idx.reshape(S, P * p3) + off[:, None]).reshape(-1)
    src = (q.reshape(S * P, 1, C) * wt[:, :, None]).reshape(-1, C).T
    rho = torch.zeros(C, S * Mt, dtype=q.dtype, device=q.device).index_add(1, gidx, src)
    rk = torch.fft.rfftn(rho.reshape(C, S, mesh[0], mesh[1], mesh[2]), dim=(2, 3, 4))
    G = pme_influence(cells, mesh, sigma, kcut, order)  # (U, ...)
    if cells.shape[0] > 1 and umap is not None:
        G = G[umap]
    phi = torch.fft.irfftn(rk * G, s=mesh, dim=(2, 3, 4)).reshape(C, -1)
    psi = (phi[:, gidx.reshape(S * P, p3)] * wt).sum(-1)  # (C, S P)
    e = (q * psi.T.reshape(S, P, C)).sum(-1)
    self_e = (q * q).sum(-1) / (math.sqrt(2 * math.pi) * sigma)
    return 14.399645 * (e - self_e)  # KE


def les_energy_pme(
    q: torch.Tensor,
    pos: torch.Tensor,
    cell: torch.Tensor,
    sigma: float,
    kcut: float,
    spacing: float = 0.5,
    order: int = 8,
) -> torch.Tensor:
    """:func:`les_energy_periodic` by PME: per-atom energies, ``q`` ``(atoms,
    channels)``."""
    lengths = torch.linalg.norm(cell.detach(), dim=-1).tolist()
    mesh = pme_mesh(lengths, sigma, spacing, order)
    e = les_energy_pme_batched(
        q[None], pos[None], cell[None], None, mesh, sigma, kcut, order
    )
    return e[0]
