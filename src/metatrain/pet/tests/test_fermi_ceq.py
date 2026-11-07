"""plato_lr Fermi charge equilibration (modules/fermi_ceq.py): float64 checks."""
import pytest
import torch

from metatrain.pet.modules.fermi_ceq import FermiParams, FermiRho, fermi_system


torch.set_default_dtype(torch.float64)
ZS = [1, 3, 6, 7, 8, 9, 11, 15, 17]
SI = {z: i for i, z in enumerate(ZS)}
LI = ([3], [[0.0, 0.0, 0.0]])
WATER = ([8, 1, 1], [[0.0, 0.0, 0.119], [0.0, 0.763, -0.477], [0.0, -0.763, -0.477]])
PF6 = ([15] + [9] * 6, [[0, 0, 0], [1.6, 0, 0], [-1.6, 0, 0], [0, 1.6, 0], [0, -1.6, 0], [0, 0, 1.6], [0, 0, -1.6]])


def pair(a, b, d):
    (za, xa), (zb, xb) = a, b
    xa, xb = torch.tensor(xa, dtype=torch.float64), torch.tensor(xb, dtype=torch.float64)
    xb = xb + torch.tensor([float(xa[:, 0].max() - xb[:, 0].min()) + d, 0.0, 0.0])
    return torch.tensor([SI[z] for z in za + zb]), torch.cat([xa, xb]), len(za)


def perturbed(seed=0, **cfg):
    P = FermiParams(ZS, cfg)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in P.parameters():
            p.add_(0.3 * torch.randn(p.shape, generator=g))
    return P


@pytest.mark.parametrize("degenerate", [False, True])
def test_fermi_rho_gradcheck(degenerate):
    H = torch.diag(torch.tensor([-1.0, -1.0, 0.5, 0.5, 2.0, 3.0])) if degenerate else torch.randn(6, 6)
    H = H.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda X: FermiRho.apply(0.5 * (X + X.T), 3.0, 0.2), (H,))


@pytest.mark.parametrize("xi,s_self", [(0.0, 0.75), (1.0, 0.5)])
def test_conservation_invariance_forces(xi, s_self):
    P = perturbed(xi=xi, s_self=s_self)
    e, x, _ = pair(LI, WATER, 2.0)
    for Q in (0.0, 1.0):
        F, q, _ = fermi_system(P, e, x, Q)
        assert float(q.detach().sum()) == pytest.approx(Q, abs=1e-10)
        Rm = torch.linalg.qr(torch.randn(3, 3))[0]
        F2, q2, _ = fermi_system(P, e, x @ Rm.T + 1.0, Q)
        assert float((F2 - F).detach()) == pytest.approx(0.0, abs=1e-8)
        assert torch.allclose(q.detach(), q2.detach(), atol=1e-8)
    x = x.clone().requires_grad_(True)
    F, q, _ = fermi_system(P, e, x, 1.0)
    gF = torch.autograd.grad(F, x, retain_graph=True)[0]
    gq = torch.autograd.grad(q[0], x)[0]
    h = 1e-5
    for a in range(x.shape[0]):
        for c in range(3):
            xp, xm = x.detach().clone(), x.detach().clone()
            xp[a, c] += h; xm[a, c] -= h
            with torch.no_grad():
                Fp, qp, _ = fermi_system(P, e, xp, 1.0); Fm, qm, _ = fermi_system(P, e, xm, 1.0)
            assert float(gF[a, c].detach()) == pytest.approx(float(Fp - Fm) / (2 * h), abs=1e-6)  # Mermin/HF force
            assert float(gq[a, c].detach()) == pytest.approx(float(qp[0] - qm[0]) / (2 * h), abs=1e-6)  # implicit dq/dR


def test_implicit_parameter_gradient():
    P = perturbed(1)
    e, x, n = pair(LI, WATER, 1.8)
    _, q, _ = fermi_system(P, e, x, 1.0)
    (q[:n].sum() ** 2).backward()
    h = 1e-6
    for name in ("d_raw", "beta_raw", "b"):
        p = getattr(P, name)
        k = (0, SI[8]) if name == "b" else (SI[8],)
        g = float(p.grad[k])
        with torch.no_grad():
            p[k] += h; qp = float(fermi_system(P, e, x, 1.0)[1][:n].sum()) ** 2
            p[k] -= 2 * h; qm = float(fermi_system(P, e, x, 1.0)[1][:n].sum()) ** 2
            p[k] += h
        assert g == pytest.approx(float(qp - qm) / (2 * h), abs=1e-6)


def test_integer_limit_and_unique_solution():
    """Beyond r_ct, Li+ PF6- carries integer fragment charges once the fragment curvature is small (s_self = 0.2;
    with tabulated levels the default 0.5 leaves 0.72-0.89 e). F is convex, so any start gives the same solution."""
    P = FermiParams(ZS, dict(s_self=0.2))
    for d in (7.0, 12.0):
        e, x, n = pair(LI, PF6, d)
        _, q, ex = fermi_system(P, e, x, 0.0)
        assert float(q[:n].detach().sum()) == pytest.approx(1.0, abs=1e-4)
        assert ex["stab_min"] > 0
        _, q2, _ = fermi_system(P, e, x, 0.0, q_init=-q.detach(), anneal=())
        assert torch.allclose(q.detach(), q2.detach(), atol=1e-7)


def test_environment_lifts_degeneracy():
    """Species-only levels are identical for every O; the environment shift separates a bonded from a lone O."""
    e = torch.tensor([SI[8], SI[1], SI[1], SI[8]])
    x = torch.tensor([*WATER[1], [6.0, 0.0, 0.0]])
    P = perturbed(2)
    lev = fermi_system(P, e, x, 0.0)[2]["levels"]
    P.env = False
    lev0 = fermi_system(P, e, x, 0.0)[2]["levels"]
    assert float(lev0[0] - lev0[3]) == pytest.approx(0.0, abs=1e-12)
    assert abs(float(lev[0] - lev[3])) > 1e-3
