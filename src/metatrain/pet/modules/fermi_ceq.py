"""Fermi charge equilibration (finite electronic temperature, Mermin free energy) for plato_lr.

A minimal valence tight-binding model with self-consistent charges, solved at a fixed electronic temperature T (eV).

Basis
    One s-like orbital per atom for H, He and groups 1-2; four (one "s" below and three "p" above the atomic level)
    otherwise. The valence electron count makes neutral closed-shell molecules closed-shell in the model.

Levels (species prior plus an environment shift)
    e_i  = -chi0_Z + dZmax tanh(d_Z) + dmax tanh(sum_Z' b_{Z Z'} CN_{i,Z'})
    chi0 = (IP + EA)/2; CN_{i,Z'} is a species-resolved coordination number, cut off smoothly at 4-5 A.
    Two same-species atoms are degenerate only if their coordination environments (and the self-consistent
    electrostatic potentials) are the same.

Hamiltonian
    H0_aa = e_i - Dsp_Z (s) or e_i + Dsp_Z / 3 (p)
    H0_ab = -sqrt(beta_i beta_j) exp(-(r_ij - r0_ij) / w) f_c(r_ij) / sqrt(m_i m_j),  f_c: C2 switch 1 -> 0 over
            [r_ct - width, r_ct]. There is no hopping beyond r_ct, so no electron is shared between groups farther apart
            than r_ct.
    H(q) = H0 - diag((A q)[orbital -> atom]),  U_Z = IP - EA,
    A    = gamma(r) + diag(s_self U + (1 - xi)(1 - s_self) U),  gamma Gaussian-smeared Coulomb whose self-energy is
           s_self U (sigma_Z = KE sqrt(2/pi) / (s_self U_Z)).
    Integer-favouring term: xi in [0, 1] removes the on-site hardness above the Gaussian self-energy, and s_self < 1
    makes the charge more diffuse. Both lower the curvature of a fragment's E(n), i.e. the fractional-charge
    (delocalisation) error of mean-field charge equilibration between separated fragments, while A stays positive
    definite: F is strictly convex in rho, so there is one SCF solution, continuous in the geometry and the parameters
    (no symmetry breaking, no hysteresis). Removing on-site hardness below the self-energy (an indefinite A) was tried
    and rejected: it produces saddles and spurious minima (Li- PF6+).

Occupations and energy
    rho = 2 C f C^T,  f_k = 1 / (1 + exp((eps_k - mu) / T)),  2 sum_k f_k = N_e = sum_i Zval_i - Q
    q_i = Zval_i - sum_{a on i} rho_aa
    E = tr(rho H0) + 1/2 q A q,  S = -2 sum_k [f ln f + (1 - f) ln(1 - f)],  F = E - T S  (Mermin).

Gradients
    F is stationary in rho at fixed N_e, so its first derivatives (forces, parameter gradients of an energy loss) are the
    explicit partial derivatives at fixed rho (Hellmann-Feynman/Mermin, mu included as the Lagrange multiplier).
    `fermi_system` builds F from the detached converged rho.

    The charges q are not variational. Their derivatives come from implicit differentiation of the SCF fixed point
    q = Phi(q):
        dq = (I - R A)^-1 dPhi,  R_im = d pop_i / d(level shift on atom m) <= 0,
    where R comes from the Daleckii-Krein (divided-difference) form of the Fermi operator with the mu constraint. It is
    finite for degenerate levels; its largest element is bounded by 1/(2T) per orbital pair. Since A > 0 and R <= 0,
    I - R A has eigenvalues >= 1, so the screening never amplifies a perturbation. The 1/T scale of R is the only
    source of large gradients; it is reported by `fermi_system` as extras["R_norm"]. The same R gives a Newton SCF.

Integer limit
    Groups separated by more than r_ct share no eigenstates. Each then holds an integer electron count up to Fermi tails
    exp(-gap / T), unless one of its levels lies within a few T of the common mu. With xi < 1 the on-site curvature can
    pin levels of two separated groups at mu (fractional transfer, as in QEq); xi -> 1 removes the on-site part of it.
"""

import math
from typing import Dict, Optional

import torch

from .sqe import EA_EV, IP_EV, KE, switch_off


_GROUP12 = {1: 1, 2: 2, 3: 1, 4: 2, 11: 1, 12: 2, 19: 1, 20: 2, 37: 1, 38: 2, 55: 1, 56: 2}


def n_orb(z: int) -> int:
    return 1 if z in _GROUP12 else 4


def valence(z: int) -> int:
    if z in _GROUP12:
        return _GROUP12[z]
    for lo, hi in ((5, 10), (13, 18), (31, 36), (49, 54), (81, 86)):
        if lo <= z <= hi:
            return z - lo + 3
    raise ValueError(f"fermi_ceq: no valence model for Z = {z} (transition metals not supported)")


DEFAULTS: Dict[str, float] = dict(
    T=0.1, r_ct=6.0, r_ct_width=1.0, w=0.75, beta_init=2.0, beta_lo=0.1, beta_hi=10.0, dmax=1.0, dZmax=3.0,
    cn_on=4.0, cn_off=5.0, xi=1.0, s_self=0.5, maxiter=100, tol=1e-10,
)


class FermiParams(torch.nn.Module):
    """Bounded per-species parameters: level shift d_Z (|.| <= dZmax), environment weights b_{ZZ'} (shift
    |.| <= dmax), hopping beta_Z in [beta_lo, beta_hi] (log-uniform), s-p splitting Dsp_Z in [1, 10] eV."""

    def __init__(self, atomic_numbers, cfg: Optional[Dict[str, float]] = None, env: bool = True):
        super().__init__()
        from ase.data import covalent_radii

        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.atomic_numbers = [int(z) for z in atomic_numbers]
        zs = self.atomic_numbers
        f64 = torch.float64
        self.register_buffer("chi0", torch.tensor([0.5 * (IP_EV[z - 1] + EA_EV[z - 1]) for z in zs], dtype=f64))
        self.register_buffer("U", torch.tensor([IP_EV[z - 1] - EA_EV[z - 1] for z in zs], dtype=f64))
        self.register_buffer("rcov", torch.tensor([covalent_radii[z] for z in zs], dtype=f64))
        # Gaussian charge width: self-energy KE sqrt(2/pi)/sigma = s_self U. The Coulomb matrix with that diagonal is
        # positive definite, so F is strictly convex in rho and the SCF solution is unique for every parameter value.
        self.register_buffer("sigma", KE * math.sqrt(2.0 / math.pi) / (self.cfg["s_self"] * self.U))
        self.register_buffer("nval", torch.tensor([float(valence(z)) for z in zs], dtype=f64))
        self.register_buffer("norb", torch.tensor([n_orb(z) for z in zs]))
        E = len(zs)
        c = self.cfg
        self.d_raw = torch.nn.Parameter(torch.zeros(E, dtype=f64))
        lb = (math.log(c["beta_init"]) - math.log(c["beta_lo"])) / (math.log(c["beta_hi"]) - math.log(c["beta_lo"]))
        self.beta_raw = torch.nn.Parameter(torch.full((E,), math.log(lb / (1 - lb)), dtype=f64))
        self.dsp_raw = torch.nn.Parameter(torch.zeros(E, dtype=f64))
        self.env = env
        self.b = torch.nn.Parameter(torch.zeros(E, E, dtype=f64))

    def dsp(self) -> torch.Tensor:
        return 1.0 + 9.0 * torch.sigmoid(self.dsp_raw)

    def beta(self) -> torch.Tensor:
        lo, hi = math.log(self.cfg["beta_lo"]), math.log(self.cfg["beta_hi"])
        return torch.exp(lo + (hi - lo) * torch.sigmoid(self.beta_raw))

    def levels(self, e: torch.Tensor, cnp: torch.Tensor) -> torch.Tensor:
        lev = -self.chi0[e] + self.cfg["dZmax"] * torch.tanh(self.d_raw[e])
        if self.env:
            lev = lev + self.cfg["dmax"] * torch.tanh((cnp * self.b[e]).sum(-1))
        return lev


def geometry(P: FermiParams, e: torch.Tensor, pos: torch.Tensor):
    """H0 (orbitals), A (atoms), orbital->atom map oa."""
    c = P.cfg
    n = pos.shape[0]
    eye = torch.eye(n, dtype=pos.dtype)
    r = torch.sqrt(((pos[:, None] - pos[None]) ** 2).sum(-1) + eye)
    off = 1.0 - eye
    R0 = P.rcov[e][:, None] + P.rcov[e][None]
    cnt = torch.sigmoid(16.0 * (4.0 / 3.0 * R0 / r - 1.0)) * switch_off(r, c["cn_on"], c["cn_off"]) * off
    cnp = cnt @ torch.nn.functional.one_hot(e, len(P.atomic_numbers)).to(pos.dtype)
    sij = torch.sqrt(P.sigma[e][:, None] ** 2 + P.sigma[e][None] ** 2)
    gam = KE * torch.erf(r / sij) / r * off
    A = gam + torch.diag((c["s_self"] + (1.0 - c["xi"]) * (1.0 - c["s_self"])) * P.U[e])
    fc = switch_off(r, c["r_ct"] - c["r_ct_width"], c["r_ct"])
    bb = P.beta()[e]
    m = P.norb[e]
    hop = -torch.sqrt(bb[:, None] * bb[None]) * torch.exp(-(r - R0) / c["w"]) * fc * off
    hop = hop / torch.sqrt((m[:, None] * m[None]).to(pos.dtype))
    oa = torch.repeat_interleave(torch.arange(n), m)
    ko = torch.cat([torch.arange(int(k)) for k in m])
    lev = P.levels(e, cnp)
    d = P.dsp()[e][oa]
    lo = lev[oa] + torch.where(ko == 0, -d, d / 3) * (m[oa] > 1).to(pos.dtype)
    H0 = hop[oa][:, oa] + torch.diag(lo)
    return H0, A, oa, lev


def _mu(eps: torch.Tensor, Ne: float, T: float) -> float:
    """Chemical potential with 2 sum f = Ne: safeguarded Newton on a bisection bracket (float64 numpy)."""
    import numpy as np

    x = eps.detach().cpu().numpy()
    lo, hi = float(x.min()) - 50 * T - 10, float(x.max()) + 50 * T + 10
    mu = float(x[min(max(int(round(Ne / 2)) - 1, 0), len(x) - 1)]) if Ne > 0 else lo
    for _ in range(100):
        f = 0.5 * (1 + np.tanh(0.5 * (mu - x) / T))
        g = 2 * f.sum() - Ne
        if abs(g) < 1e-12:
            break
        lo, hi = (mu, hi) if g < 0 else (lo, mu)
        d = 2 * (f * (1 - f)).sum() / T
        mn = mu - g / d if d > 1e-14 else 0.5 * (lo + hi)
        mu = mn if lo < mn < hi else 0.5 * (lo + hi)
    return mu


def _lowner(eps: torch.Tensor, f: torch.Tensor, T: float) -> torch.Tensor:
    fp = -f * (1 - f) / T
    de = eps[:, None] - eps[None]
    df = f[:, None] - f[None]
    deg = de.abs() < 1e-9
    return torch.where(deg, 0.5 * (fp[:, None] + fp[None]), df / torch.where(deg, torch.ones_like(de), de)), fp


class FermiRho(torch.autograd.Function):
    """rho = 2 C f(eps; mu) C^T with 2 tr f = Ne. Backward: Daleckii-Krein divided differences + the mu constraint
    (no 1/(eps_k - eps_l) terms, finite for degenerate levels)."""

    @staticmethod
    def forward(ctx, H, Ne: float, T: float):
        eps, C = torch.linalg.eigh(H)
        mu = _mu(eps, Ne, T)
        f = torch.sigmoid(-(eps - mu) / T)
        ctx.save_for_backward(eps, C, f)
        ctx.T = T
        return 2 * (C * f) @ C.T

    @staticmethod
    def backward(ctx, G):
        eps, C, f = ctx.saved_tensors
        L, fp = _lowner(eps, f, ctx.T)
        Gb = C.T @ (0.5 * (G + G.T)) @ C
        Y = 2 * L * Gb
        s = fp.sum()
        if float(s.abs()) > 1e-300:
            Y = Y - 2 * (fp * torch.diagonal(Gb)).sum() / s * torch.diag(fp)
        return C @ Y @ C.T, None, None


def _pop(rho: torch.Tensor, oa: torch.Tensor, n: int) -> torch.Tensor:
    return torch.zeros(n, dtype=rho.dtype).index_add(0, oa, torch.diagonal(rho))


def response(eps, C, f, T, oa, n) -> torch.Tensor:
    """R_im = d pop_i / d(level shift of all orbitals on atom m) (n x n, symmetric, negative semidefinite)."""
    L, fp = _lowner(eps, f, T)
    W = torch.zeros(n, C.shape[0], C.shape[0], dtype=C.dtype).index_add_(0, oa, C[:, :, None] * C[:, None, :]).reshape(n, -1)
    R = 2 * (W * L.reshape(-1)) @ W.T
    p = W[:, :: C.shape[0] + 1] @ fp
    s = fp.sum()
    if float(s.abs()) > 1e-300:
        R = R - 2 * torch.outer(p, p) / s
    return R


def _state(H0, A, q, oa, zval, Ne, T):
    H = H0 - torch.diag((A @ q)[oa])
    eps, C = torch.linalg.eigh(H)
    mu = _mu(eps, Ne, T)
    f = torch.sigmoid(-(eps - mu) / T)
    rho = 2 * (C * f) @ C.T
    return zval - _pop(rho, oa, len(zval)), eps, C, f, mu, rho


def _newton(H0, A, oa, zval, Ne, T, q, tol, maxiter):
    n = len(zval)
    I = torch.eye(n, dtype=H0.dtype)
    r0 = float("inf")
    for it in range(maxiter):
        phi, eps, C, f, mu, rho = _state(H0, A, q, oa, zval, Ne, T)
        res = phi - q
        r0 = float(res.abs().max())
        if r0 < tol:
            return q, it + 1, r0
        dq = torch.linalg.lstsq(I - response(eps, C, f, T, oa, n) @ A, res[:, None]).solution[:, 0]
        a = 1.0
        while a > 1e-4:
            qn = q + a * dq
            if float((_state(H0, A, qn, oa, zval, Ne, T)[0] - qn).abs().max()) < r0:
                break
            a *= 0.5
        if a <= 1e-4:  # Newton stalled: a few steps of damped linear mixing
            for _ in range(50):
                q = q + 0.05 * (_state(H0, A, q, oa, zval, Ne, T)[0] - q)
        else:
            q = qn
    return q, maxiter, r0


def stability(H0, A, oa, zval, Ne, T, q):
    """Eigen-decomposition of the SCF stability operator I - R A (similar to I + |R|^1/2 A |R|^1/2 when A > 0).
    A negative eigenvalue marks a saddle of F (symmetric solution of a symmetry-breaking problem)."""
    _, eps, C, f, _, _ = _state(H0, A, q, oa, zval, Ne, T)
    ev, V = torch.linalg.eig(torch.eye(len(zval), dtype=A.dtype) - response(eps, C, f, T, oa, len(zval)) @ A)
    k = int(torch.argmin(ev.real))
    return float(ev.real[k]), V[:, k].real


@torch.no_grad()
def scf_newton(H0, A, oa, zval, Ne, T, q0=None, tol=1e-10, maxiter=100, kicks=0):
    """Newton iteration on q = Phi(q) (exact Jacobian R A, backtracking), then a stability check: if I - R A has a
    negative eigenvalue the solution is a saddle of F; it is displaced along that mode and re-converged."""
    n = len(zval)
    q = torch.zeros(n, dtype=H0.dtype) if q0 is None else q0.clone()
    q, it, res = _newton(H0, A, oa, zval, Ne, T, q, tol, maxiter)
    nk = 0
    lam = float("nan")
    for nk in range(kicks + 1):
        lam, v = stability(H0, A, oa, zval, Ne, T, q)
        if lam > 0 or nk == kicks:
            break
        q, it2, res = _newton(H0, A, oa, zval, Ne, T, q + 0.1 * v / v.abs().max(), tol, maxiter)
        it += it2
    phi, eps, C, f, mu, rho = _state(H0, A, q, oa, zval, Ne, T)
    return dict(q=q, eps=eps, C=C, f=f, mu=mu, rho=rho, iters=it, residual=float((phi - q).abs().max()),
                stab_min=lam, kicks=nk)


class SCFCharges(torch.autograd.Function):
    """q*(H0, A): forward Newton SCF; backward implicit differentiation, (I - R A)^T u = dL/dq*."""

    @staticmethod
    def forward(ctx, H0, A, oa, zval, Ne: float, T: float, q_star):
        ctx.save_for_backward(H0, A, oa, zval, q_star)
        ctx.Ne, ctx.T = Ne, T
        return q_star.clone()

    @staticmethod
    def backward(ctx, gq):
        H0, A, oa, zval, q = ctx.saved_tensors
        n = len(zval)
        with torch.no_grad():
            _, eps, C, f, _, _ = _state(H0, A, q, oa, zval, ctx.Ne, ctx.T)
            J = response(eps, C, f, ctx.T, oa, n) @ A
            u = torch.linalg.solve((torch.eye(n, dtype=A.dtype) - J).T, gq)
        with torch.enable_grad():
            H0d = H0.detach().requires_grad_(True)
            Ad = A.detach().requires_grad_(True)
            rho = FermiRho.apply(H0d - torch.diag((Ad @ q)[oa]), ctx.Ne, ctx.T)
            phi = zval - _pop(rho, oa, n)
            gH, gA = torch.autograd.grad(phi, (H0d, Ad), u)
        return gH, gA, None, None, None, None, None


def entropy(f: torch.Tensor) -> torch.Tensor:
    fc = f.clamp(1e-300, 1 - 1e-16)
    return -2 * (fc * torch.log(fc) + (1 - fc) * torch.log1p(-fc)).sum()


def fermi_system(P: FermiParams, e: torch.Tensor, pos: torch.Tensor, Q: float, T: Optional[float] = None,
                 q_init: Optional[torch.Tensor] = None, anneal=(1.0, 0.5, 0.25)):
    """(F, q, extras). F: Mermin free energy (eV) with exact first derivatives in pos and the parameters. q: SCF
    charges, differentiable (implicitly) in pos and the parameters. The SCF is annealed from high T to T."""
    T = float(P.cfg["T"] if T is None else T)
    pos = pos.to(torch.float64)
    zval = P.nval[e]
    Ne = float(zval.sum()) - float(Q)
    H0, A, oa, lev = geometry(P, e, pos)
    qi = q_init
    for Ta in [a for a in anneal if a > T] + [T]:
        s = scf_newton(H0.detach(), A.detach(), oa, zval, Ne, Ta, qi, P.cfg["tol"], int(P.cfg["maxiter"]),
                       kicks=4 if Ta == T else 0)
        qi = s["q"]
    rho, qd = s["rho"], s["q"]
    S = entropy(s["f"])
    F = (rho * H0).sum() + 0.5 * qd @ (A @ qd) - T * S
    q = SCFCharges.apply(H0, A, oa, zval, Ne, T, qd)
    n = len(zval)
    with torch.no_grad():
        R = response(s["eps"], s["C"], s["f"], T, oa, n)
    ex = dict(oa=oa, mu=s["mu"], eps=s["eps"], f=s["f"], S=S, iters=s["iters"], residual=s["residual"], levels=lev.detach(),
              R_norm=float(torch.linalg.matrix_norm(R, 2)), stab_min=s["stab_min"], kicks=s["kicks"], n_near_mu=int(((s["eps"] - s["mu"]).abs() < 3 * T).sum()))
    return F, q, ex
