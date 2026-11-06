"""plato_lr sqe (modules/sqe.py): float64 checks of the split-charge equilibration term."""
import math

import pytest
import torch

from metatrain.pet.modules.sqe import KE, SQEParams, hardness_J, mulliken_chi, sqe_system


torch.set_default_dtype(torch.float64)
ZS = [1, 3, 6, 7, 8, 9, 11, 15, 17]


def random_params(seed=0, scale=0.5):
    P = SQEParams(ZS)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in P.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g))
    return P


def cluster(n=8, seed=1, box=5.0):
    g = torch.Generator().manual_seed(seed)
    e = torch.randint(0, len(ZS), (n,), generator=g)
    return e, torch.rand(n, 3, generator=g) * box


def test_tables():
    assert mulliken_chi(9) == pytest.approx(0.5 * (17.4228 + 3.4012))
    assert hardness_J(1) == pytest.approx(13.5984 - 0.7542)
    assert all(hardness_J(z) > 0 for z in range(1, 104))


@torch.no_grad()
def test_charge_conservation_and_invariance():
    P = random_params()
    e, x = cluster()
    for Q in (0.0, 1.0, -2.0):
        ea, q, _ = sqe_system(P, e, x, torch.tensor(Q))
        assert float(q.sum()) == pytest.approx(Q, abs=1e-12)
        R = torch.linalg.qr(torch.randn(3, 3))[0]
        ea2, q2, _ = sqe_system(P, e, x @ R.T + torch.tensor([3.0, -1.0, 2.0]), torch.tensor(Q))
        assert float(ea2.sum() - ea.sum()) == pytest.approx(0.0, abs=1e-10)
        assert torch.allclose(q, q2, atol=1e-10)


def test_minimum_and_forces():
    P = random_params(1)
    e, x = cluster(seed=2)
    x.requires_grad_(True)
    ea, q, aux = sqe_system(P, e, x, torch.tensor(1.0))
    F = -torch.autograd.grad(ea.sum(), x)[0]
    h = 1e-5
    for a in range(x.shape[0]):
        for c in range(3):
            xp, xm = x.detach().clone(), x.detach().clone()
            xp[a, c] += h; xm[a, c] -= h
            fd = -(sqe_system(P, e, xp, torch.tensor(1.0))[0].sum() - sqe_system(P, e, xm, torch.tensor(1.0))[0].sum()) / (2 * h)
            assert float(F[a, c]) == pytest.approx(float(fd.detach()), abs=1e-7)
    # stationarity: p_ij = -s_ij (mu_i - mu_j) makes dE/dp = 0, so a perturbed split charge raises E
    mu, s = aux["mu"].detach(), aux["s"].detach()
    i, j = (s > 0).nonzero()[0].tolist()
    assert float(aux["p"][i, j].detach()) == pytest.approx(float(-s[i, j] * (mu[i] - mu[j])), rel=1e-12)


def test_parameter_gradients_with_force_loss():
    P = random_params(2, 0.3)
    e, x = cluster(6, seed=3, box=4.0)
    names = [k for k, _ in P.named_parameters()]

    def f(*vals):
        orig = {k: P._parameters[k] for k in names}
        for k, v in zip(names, vals):
            P._parameters[k] = v
        xx = x.clone().requires_grad_(True)
        E = sqe_system(P, e, xx, torch.tensor(0.0))[0].sum()
        F = torch.autograd.grad(E, xx, create_graph=True)[0]
        for k, v in orig.items():
            P._parameters[k] = v
        return E + 0.1 * (F ** 2).sum()

    vals = tuple(p.detach().clone().requires_grad_(True) for p in P.parameters())
    assert torch.autograd.gradcheck(f, vals, eps=1e-6, atol=1e-6)


@torch.no_grad()
def test_bounds_hold_for_any_raw_value():
    P = SQEParams(ZS)
    with torch.no_grad():
        for p in P.parameters():
            p.fill_(50.0)
    eidx = torch.arange(len(ZS))
    assert torch.all(P.J(eidx) <= 2.0 * P.J0 + 1e-9)
    assert torch.all(P.kappa0() <= 100.0 + 1e-9) and float(P.w()) <= 2.0
    with torch.no_grad():
        for p in P.parameters():
            p.fill_(-50.0)
    assert torch.all(P.J(eidx) >= 0.75 * P.J0 - 1e-9) and torch.all(P.kappa0() >= 1.0 - 1e-9)
    # Gaussian self energy <= J_lo J0 <= J: A positive definite
    assert torch.allclose(KE * math.sqrt(2 / math.pi) / P.sigma, 0.75 * P.J0)


@torch.no_grad()
def test_integer_limit_and_smoothness():
    """Li+ ... F-: beyond r_ct the fragments carry exactly their q0 sums (formal: +-1, uniform with Q = 0: 0)."""
    P = SQEParams(ZS)
    e = torch.tensor([ZS.index(3), ZS.index(9)])
    q_prev = None
    for r in torch.linspace(2.0, 8.0, 121):
        x = torch.tensor([[0.0, 0.0, 0.0], [float(r), 0.0, 0.0]])
        qf = sqe_system(P, e, x, torch.tensor(0.0), torch.tensor([1.0, -1.0]))[1]
        qu = sqe_system(P, e, x, torch.tensor(0.0))[1]
        if r >= P.cfg["r_ct"]:
            assert float(qf[0]) == pytest.approx(1.0, abs=1e-12) and float(qu[0]) == pytest.approx(0.0, abs=1e-12)
        else:
            assert 0.0 < float(qf[0]) <= 1.0 + 1e-9
        if q_prev is not None:
            assert abs(float(qf[0]) - q_prev) < 0.02  # smooth on a 0.05 A grid
        q_prev = float(qf[0])
    assert q_prev == pytest.approx(1.0)


@torch.no_grad()
def test_periodic_large_box_limit():
    """Ewald with a neutralising background -> the open result as the box grows (neutral cluster)."""
    P = random_params(3, 0.2)
    e, x = cluster(4, seed=4, box=3.0)
    ea, q, _ = sqe_system(P, e, x, torch.tensor(0.0))
    L = 60.0
    ep, qp, _ = sqe_system(P, e, x + L / 2 - 1.5, torch.tensor(0.0), cell=torch.eye(3) * L, periodic=True)
    assert float(ep.sum()) == pytest.approx(float(ea.sum()), abs=2e-3)
    assert torch.allclose(q, qp, atol=1e-3)


def learned_params(seed=4, scale=0.5):
    P = random_params(seed, 0.3)
    P.learned_q0 = True
    g = torch.Generator().manual_seed(seed + 100)
    with torch.no_grad():
        P.q0_a.add_(scale * torch.randn(P.q0_a.shape, generator=g))
        P.q0_b.add_(scale * torch.randn(P.q0_b.shape, generator=g))
    return P


@torch.no_grad()
def test_learned_q0_bounds_and_total_charge():
    from metatrain.pet.modules.sqe import _pair_r, learned_q0
    P = learned_params()
    for seed, Q in ((5, 0.0), (6, 1.0), (7, -1.0), (8, 2.0)):
        e, x = cluster(10, seed=seed)
        q0 = learned_q0(P, e, _pair_r(x), torch.tensor(Q))
        assert float(q0.sum()) == pytest.approx(Q, abs=1e-10)
        if float(P.q0_lo[e].sum()) < Q < float(P.q0_hi[e].sum()):
            assert torch.all(q0 >= P.q0_lo[e] - 1e-12) and torch.all(q0 <= P.q0_hi[e] + 1e-12)
        ea, q, aux = sqe_system(P, e, x, torch.tensor(Q))
        assert torch.allclose(aux["q0"], q0) and float(q.sum()) == pytest.approx(Q, abs=1e-10)


def test_learned_q0_forces_and_gradcheck():
    P = learned_params(9)
    e, x = cluster(6, seed=10, box=4.0)
    xx = x.clone().requires_grad_(True)
    F = -torch.autograd.grad(sqe_system(P, e, xx, torch.tensor(1.0))[0].sum(), xx)[0]
    h = 1e-5
    with torch.no_grad():
        for a in range(x.shape[0]):
            for c in range(3):
                xp, xm = x.clone(), x.clone(); xp[a, c] += h; xm[a, c] -= h
                fd = -(sqe_system(P, e, xp, torch.tensor(1.0))[0].sum() - sqe_system(P, e, xm, torch.tensor(1.0))[0].sum()) / (2 * h)
                assert float(F[a, c]) == pytest.approx(float(fd), abs=1e-7)
    names = [k for k, _ in P.named_parameters()]

    def f(*vals):
        orig = {k: P._parameters[k] for k in names}
        for k, v in zip(names, vals):
            P._parameters[k] = v
        x2 = x.clone().requires_grad_(True)
        E = sqe_system(P, e, x2, torch.tensor(1.0))[0].sum()
        F2 = torch.autograd.grad(E, x2, create_graph=True)[0]
        for k, v in orig.items():
            P._parameters[k] = v
        return E + 0.1 * (F2 ** 2).sum()

    vals = tuple(p.detach().clone().requires_grad_(True) for p in P.parameters())
    assert torch.autograd.gradcheck(f, vals, eps=1e-6, atol=1e-6)


@torch.no_grad()
def test_learned_q0_no_formal_input():
    """In learned mode the formal charges are never used."""
    from metatrain.pet.modules.sqe import PETSQE
    m = PETSQE(ZS, None, "learned")
    e, x = cluster(6, seed=11)
    a = m.forward_system(e, x, torch.tensor(0.0), torch.tensor([1.0, -1.0, 0, 0, 0, 0]), None, False)[0].sum()
    b = m.forward_system(e, x, torch.tensor(0.0), None, None, False)[0].sum()
    assert float(a) == float(b)
