"""Symmetry, gradients, and the LES Ewald sum (float64)."""

import copy
import math

import pytest
import torch
from metatomic.torch import System

from metatrain.experimental.plato_pet_lr import PlatoPETLR
from metatrain.experimental.plato_pet_lr.modules.ewald import (
    KE,
    kvectors,
    les_energy_finite,
    les_energy_periodic,
)
from metatrain.experimental.plato_pet_lr.modules.frames import (
    haar_rotations,
    platonic_group,
    sample_frames,
)
from metatrain.experimental.plato_pet_lr.modules.layers import FrameLinear
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


CELL = torch.tensor(
    [[6.0, 0.0, 0.0], [0.3, 6.2, 0.0], [0.1, 0.2, 5.9]], dtype=torch.float64
)


def _model(**kw) -> PlatoPETLR:
    h = copy.deepcopy(get_default_hypers("experimental.plato_pet_lr")["model"])
    h.update(d_model=16, num_heads=2, num_layers=3, cutoff=3.5, num_rbf=8, n_basis=4,
             mesh_channels=2, mesh_spacing=0.45)  # fmt: skip
    h.update(kw)
    info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 8],
        targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
    )
    torch.manual_seed(0)
    return PlatoPETLR(h, info).double().eval()


def _structure(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(8, 3, generator=g, dtype=torch.float64) @ CELL, torch.tensor(
        [1, 8] * 4
    )


def _energy(model, pos, cell, types, periodic=True) -> torch.Tensor:
    s = System(
        types=types,
        positions=pos,
        cell=cell if periodic else torch.zeros_like(cell),
        pbc=torch.tensor([periodic] * 3),
    )
    s = get_system_with_neighbor_lists(s, model.requested_neighbor_lists())
    E = model([s], {"energy": model.outputs["energy"]})["energy"].block().values.sum()
    return E if pos.requires_grad else E.detach()


def _rel(a, b) -> float:
    return abs(float(a) - float(b)) / abs(float(b))


@pytest.mark.parametrize(
    "frames, mode",
    [("T", "full"), ("O", "full"), ("D2", "full"), ("D2", "attention"),
     ("T", "attention"), ("D3", "attention")],
)  # fmt: skip
@pytest.mark.parametrize("framed", [None, 1])
def test_group_invariance_fixed_frames(frames, mode, framed):
    model = _model(frames=frames, frame_mode=mode, num_framed_layers=framed)
    pos, types = _structure()
    E0 = _energy(model, pos, CELL, types)
    for Q in platonic_group(frames)[1:6]:  # (D2 has 3 non-identity elements)
        assert _rel(_energy(model, pos @ Q.T, CELL @ Q.T, types), E0) < 1e-12
    shift = torch.tensor([0.37, -0.21, 0.55], dtype=torch.float64)
    assert (
        _rel(_energy(model, pos + shift, CELL, types), E0) < 1e-8
    )  # PME interpolation


def test_so3_approximate_invariance():
    """Generic rotations: ~1e-4 with the group, O(1) with a single frame; the mean
    over coset frames (equivalent to random global rotations) is invariant."""
    pos, types = _structure()
    Qs = haar_rotations(4, generator=torch.Generator().manual_seed(1))
    errors = {}
    for frames in ("identity", "T", "O"):
        model = _model(frames=frames)
        E0 = _energy(model, pos, CELL, types)
        errors[frames] = max(
            _rel(_energy(model, pos @ Q.T, CELL @ Q.T, types), E0) for Q in Qs
        )
    # at this init: identity 5e-2, T 7e-3, O 1.4e-4
    assert errors["O"] < 1e-3 and errors["T"] < 2e-2
    assert errors["identity"] > 100 * errors["O"]

    model = _model(frames="T")
    for mode in ("coset", "haar"):
        torch.manual_seed(2)
        frames = [sample_frames(mode, model.G_R) for _ in range(8)]

        def mean_energy(p, c, frames=frames):
            out = []
            for R, rel, w in frames:
                model.G_R, model.G_rel, model.G_w = R, rel, w
                out.append(_energy(model, p, c, types))
            return torch.stack(out)

        E = mean_energy(pos, CELL).detach()
        E_rot = mean_energy(pos @ Qs[0].T, CELL @ Qs[0].T).detach()
        if mode == "coset":  # a coset is the group seen from a rotated input
            assert float(E.std() / E.mean().abs()) < 1e-2
        # the frame distribution is rotation invariant: equal means up to sampling
        sem = float(torch.cat([E, E_rot]).std()) / math.sqrt(len(frames))
        assert abs(float(E_rot.mean() - E.mean())) < 4 * sem


def test_attention_mode_not_trivially_invariant():
    """The group average is what makes attention mode invariant: a generic
    rotation changes the energy (the frames do carry directional information)."""
    model = _model(frames="D2", frame_mode="attention")
    pos, types = _structure()
    E0 = _energy(model, pos, CELL, types)
    Q = haar_rotations(1, generator=torch.Generator().manual_seed(3))[0]
    assert _rel(_energy(model, pos @ Q.T, CELL @ Q.T, types), E0) > 1e-6


def test_attention_mode_cheaper_than_full():
    from torch.utils.flop_counter import FlopCounterMode

    pos, types = _structure()
    flops = {}
    for mode in ("full", "attention"):
        model = _model(frames="T", frame_mode=mode)
        with FlopCounterMode(display=False) as fc:
            _energy(model, pos, CELL, types)
        flops[mode] = fc.get_total_flops()
    assert flops["attention"] < flops["full"] / 4


def test_attention_mode_exported_torchscript():
    model = _model(frames="D2", frame_mode="attention", num_framed_layers=2)
    pos, types = _structure()
    E0 = _energy(model, pos, CELL, types)
    exported = model.export()
    scripted = torch.jit.script(exported)
    from metatomic.torch import ModelEvaluationOptions, ModelOutput

    s = System(types=types, positions=pos, cell=CELL, pbc=torch.tensor([True] * 3))
    s = get_system_with_neighbor_lists(s, exported.requested_neighbor_lists())
    opts = ModelEvaluationOptions(
        length_unit="angstrom", outputs={"energy": ModelOutput(sample_kind="system")}
    )
    E = scripted([s], opts, check_consistency=True)["energy"].block().values.sum()
    assert _rel(E, E0) < 1e-12


@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize(
    "variant",
    [dict(frames="T"), dict(frames="D2", frame_mode="attention"),
     dict(frames="T", frame_mode="attention", num_framed_layers=1)],
)  # fmt: skip
def test_forces_and_stress_finite_differences(periodic, variant):
    model = _model(**variant)
    pos, types = _structure()
    p = pos.clone().requires_grad_()
    strain = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
    D = torch.eye(3, dtype=torch.float64) + 0.5 * (strain + strain.T)
    E = _energy(model, p @ D, CELL @ D, types, periodic)
    F, dE_de = torch.autograd.grad(E, [p, strain])
    h = 1e-5
    for a, x in ((0, 0), (3, 1), (6, 2)):
        dp = torch.zeros_like(pos)
        dp[a, x] = h
        fd = (_energy(model, pos + dp, CELL, types, periodic)
              - _energy(model, pos - dp, CELL, types, periodic)) / (2 * h)  # fmt: skip
        assert abs(float(fd) - float(F[a, x])) < 1e-6 * max(1.0, abs(float(fd)))
    if periodic:
        for a, b in ((0, 0), (1, 2)):
            eps = torch.zeros(3, 3, dtype=torch.float64)
            eps[a, b] = eps[b, a] = h / 2 if a != b else h
            Ep = _energy(
                model, pos @ (torch.eye(3) + eps), CELL @ (torch.eye(3) + eps), types
            )
            Em = _energy(
                model, pos @ (torch.eye(3) - eps), CELL @ (torch.eye(3) - eps), types
            )
            fd = (Ep - Em).detach() / (2 * h)
            assert abs(float(fd) - float(dE_de[a, b])) < 1e-6 * max(1.0, abs(float(fd)))


@pytest.mark.parametrize("mode", ["folded", "factored"])
def test_frame_linear_contractions_agree(mode):
    torch.manual_seed(0)
    R, rel, w = sample_frames("fixed", platonic_group("T"))
    lin = FrameLinear(5, 7, n_basis=3, hidden=4).double()
    x = torch.randn(4, 2, 12, 5, dtype=torch.float64)
    ref = FrameLinear.forward(lin, x, rel, w)
    lin.mode = mode
    torch.testing.assert_close(lin(x, rel, w), ref)


def _brute_reciprocal(q, pos, cell, sigma, nmax):
    r = torch.arange(-nmax, nmax + 1, dtype=pos.dtype)
    n = torch.cartesian_prod(r, r, r)
    n = n[(n != 0).any(-1)]
    k = 2 * math.pi * n @ torch.linalg.inv(cell).T
    k2 = (k * k).sum(-1)
    ph = pos @ k.T
    S2 = (q.T @ torch.cos(ph)) ** 2 + (q.T @ torch.sin(ph)) ** 2
    V = torch.det(cell).abs()
    recip = KE * 2 * math.pi / V * (torch.exp(-0.5 * sigma**2 * k2) / k2 * S2).sum()
    return recip - KE * (q * q).sum() / (math.sqrt(2 * math.pi) * sigma)


@pytest.mark.parametrize("L", [6.0, 12.0, 24.0, 48.0])
def test_ewald_converged(L):
    """Against a brute integer cube with ``k_max L / 2 pi = 8 / sigma``: < 3e-6 at
    all sizes (the reference ``nmax = 4`` was 1% off at 12 A, 50% at 48 A)."""
    g = torch.Generator().manual_seed(0)
    cell = torch.eye(3, dtype=torch.float64) * L
    pos = torch.rand(16, 3, generator=g, dtype=torch.float64) * L
    q = torch.randn(16, 1, generator=g, dtype=torch.float64)
    q = q - q.mean()
    ref = _brute_reciprocal(q, pos, cell, 1.0, int(math.ceil(8 * L / (2 * math.pi))))
    assert _rel(les_energy_periodic(q, pos, cell, 1.0, 6.0).sum(), ref) < 3e-6


def test_ewald_smooth_under_strain():
    """Terms the cell-dependent k-set adds or drops have weight zero: the energy
    and stress are continuous while the set changes."""
    g = torch.Generator().manual_seed(0)
    pos = torch.rand(10, 3, generator=g, dtype=torch.float64) * 10
    q = torch.randn(10, 1, generator=g, dtype=torch.float64)
    q = q - q.mean()
    scales = torch.linspace(0.97, 1.03, 61, dtype=torch.float64)
    counts, E, dE = [], [], []
    for s in scales:
        s = s.clone().requires_grad_()
        cell = torch.eye(3, dtype=torch.float64) * 10 * s
        e = les_energy_periodic(q, pos * s, cell, 1.0, 6.0).sum()
        E.append(float(e.detach()))
        dE.append(float(torch.autograd.grad(e, s)[0]))
        counts.append(len(kvectors(cell, 6.0)))
    assert len(set(counts)) > 1  # the k-set did change along the path
    E, dE = torch.tensor(E), torch.tensor(dE)
    h = float(scales[1] - scales[0])
    fd = (E[2:] - E[:-2]) / (2 * h)
    assert (fd - dE[1:-1]).abs().max() < 1e-3 * dE.abs().max()


def test_finite_periodic_consistency():
    """A +-1 pair at 3 A, sigma 1: finite -4.7869 eV. The reference periodic
    energy was +6.70 (Gaussian self energy KE sum q^2 / (sqrt(2 pi) sigma) = 11.49
    eV included). Now periodic - finite = -KE 2 pi |M|^2 / (3 L^3) + O(L^-5)."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64)
    q = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    E_finite = float(les_energy_finite(q, pos, 1.0).sum())
    assert abs(E_finite - KE * -math.erf(3 / math.sqrt(2)) / 3) < 1e-12
    for L, tol in ((20.0, 2e-3), (40.0, 1e-4), (80.0, 1e-5)):
        cell = torch.eye(3, dtype=torch.float64) * L
        E = float(les_energy_periodic(q, pos + L / 2, cell, 1.0, 6.0).sum())
        dipole = -KE * 2 * math.pi * 9.0 / (3 * L**3)
        assert abs(E - E_finite - dipole) < tol
        assert abs(E - E_finite) < 0.04  # was 11.49 eV
