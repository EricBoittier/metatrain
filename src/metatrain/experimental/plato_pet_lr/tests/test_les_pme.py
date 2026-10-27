"""LES by smooth particle-mesh Ewald against the direct sum (float64)."""

import copy
import math

import pytest
import torch
from metatomic.torch import System

from metatrain.experimental.plato_pet_lr import PlatoPETLR
from metatrain.experimental.plato_pet_lr.modules.ewald import (
    bspline_stencil,
    fft_size,
    les_energy_periodic,
    les_energy_pme,
    les_energy_pme_batched,
    pme_mesh,
)
from metatrain.experimental.plato_pet_lr.modules.frames import haar_rotations
from metatrain.experimental.plato_pet_lr.modules.layers import bspline4_stencil
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


D = torch.float64


def _charges(n, cell, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(n, 3, generator=g, dtype=D) @ cell
    q = torch.randn(n, 1, generator=g, dtype=D)
    return pos, q - q.mean()


def _efs(fn, q, pos, cell):
    """Energy, forces and the strain derivative / V."""
    eps = torch.zeros(3, 3, dtype=D, requires_grad=True)
    p = pos.clone().requires_grad_()
    T = torch.eye(3, dtype=D) + eps
    E = fn(q, p @ T, cell @ T).sum()
    gp, ge = torch.autograd.grad(E, [p, eps])
    return E.detach(), -gp, ge / torch.det(cell).abs()


def _relnorm(a, b) -> float:
    return float((a - b).norm() / b.norm())


def test_fft_size_and_mesh():
    assert [fft_size(n) for n in (1, 7, 11, 61, 97)] == [1, 8, 12, 64, 100]
    # 30 A, h = 0.5: floor(60.5) + 1 = 61 -> 64, and the same for 30 +- 1e-6
    for L in (30.0 - 1e-6, 30.0, 30.0 + 1e-6):
        assert pme_mesh([L] * 3, 1.0, 0.5, 8) == [64, 64, 64]
    assert pme_mesh([3.0] * 3, 1.0, 0.5, 8) == [8, 8, 8]  # at least the order


def test_order4_stencil_is_bspline4():
    g = torch.Generator().manual_seed(0)
    frac = torch.rand(7, 3, generator=g, dtype=D) * 1.7 - 0.3
    M = [9, 10, 12]
    i4, w4 = bspline4_stencil(frac, M)
    i, w = bspline_stencil(frac, M, 4)
    assert torch.equal(i, i4)
    torch.testing.assert_close(w, w4, rtol=0, atol=1e-15)
    for p in (4, 6, 8, 10):
        _, w = bspline_stencil(frac, M, p)
        torch.testing.assert_close(
            w.sum(-1), torch.ones(7, dtype=D)
        )  # partition of unity


@pytest.mark.parametrize("triclinic", [False, True])
@pytest.mark.parametrize("L", [6.0, 12.0, 24.0, 48.0])
def test_pme_matches_direct(L, triclinic):
    """Order 8, h = 0.5 sigma (the defaults): energy < 2e-6, forces < 5e-5, stress
    < 1e-5 relative to the direct sum (measured 1e-8..5e-7, 1e-6..2e-5, 4e-7..4e-6)."""
    cell = torch.eye(3, dtype=D) * L
    if triclinic:
        cell = torch.tensor(
            [[L, 0, 0], [0.03 * L, 1.05 * L, 0], [-0.04 * L, 0.07 * L, 0.97 * L]],
            dtype=D,
        )
    pos, q = _charges(int(min(200, max(16, 0.05 * L**3))), cell)
    Ed, Fd, Sd = _efs(
        lambda q, p, c: les_energy_periodic(q, p, c, 1.0, 6.0), q, pos, cell
    )
    E, F, S = _efs(lambda q, p, c: les_energy_pme(q, p, c, 1.0, 6.0), q, pos, cell)
    assert abs(float(E - Ed)) < 2e-6 * abs(float(Ed))
    assert _relnorm(F, Fd) < 5e-5
    assert _relnorm(S, Sd) < 1e-5


def test_pme_converges_with_spacing_and_order():
    cell = torch.eye(3, dtype=D) * 12.0
    pos, q = _charges(64, cell)
    Ed, Fd, _ = _efs(
        lambda q, p, c: les_energy_periodic(q, p, c, 1.0, 6.0), q, pos, cell
    )
    err = {}
    for p, h in ((4, 0.5), (6, 0.5), (8, 0.5), (8, 0.35)):
        _, F, _ = _efs(
            lambda q, x, c, h=h, p=p: les_energy_pme(q, x, c, 1.0, 6.0, h, p),
            q,
            pos,
            cell,
        )
        err[(p, h)] = _relnorm(F, Fd)
    assert err[(4, 0.5)] > err[(6, 0.5)] > err[(8, 0.5)] > err[(8, 0.35)]
    assert err[(8, 0.35)] < 2e-6


def test_pme_per_atom_and_batched():
    """Per-atom energies close to the direct split; a batch of two cells (one
    mesh) equals the two separate calls."""
    cells = torch.stack([torch.eye(3, dtype=D) * 10.0, torch.eye(3, dtype=D) * 10.2])
    pos0, q0 = _charges(12, cells[0], 1)
    pos1, q1 = _charges(9, cells[1], 2)
    e0 = les_energy_pme(q0, pos0, cells[0], 1.0, 6.0)
    torch.testing.assert_close(
        e0, les_energy_periodic(q0, pos0, cells[0], 1.0, 6.0), rtol=0, atol=1e-5
    )
    M = pme_mesh([10.2] * 3, 1.0, 0.5, 8)
    q = torch.zeros(2, 12, 1, dtype=D)
    pos = torch.zeros(2, 12, 3, dtype=D)
    q[0], pos[0] = q0, pos0
    q[1, :9], pos[1, :9] = q1, pos1
    e = les_energy_pme_batched(q, pos, cells, None, M, 1.0, 6.0, 8)
    for s, (qs, ps) in enumerate(((q0, pos0), (q1, pos1))):
        ref = les_energy_pme_batched(qs[None], ps[None], cells[s : s + 1], None, M,
                                     1.0, 6.0, 8)[0]  # fmt: skip
        torch.testing.assert_close(e[s, : len(qs)], ref, rtol=1e-12, atol=1e-12)
    assert float(e[1, 9:].abs().max()) == 0.0


def test_pme_rotation_invariant():
    cell = torch.tensor([[11.0, 0, 0], [0.5, 10.0, 0], [-0.3, 0.4, 12.0]], dtype=D)
    pos, q = _charges(30, cell)
    E0 = les_energy_pme(q, pos, cell, 1.0, 6.0).sum()
    for R in haar_rotations(3, generator=torch.Generator().manual_seed(1)):
        E = les_energy_pme(q, pos @ R.T, cell @ R.T, 1.0, 6.0).sum()
        assert abs(float(E - E0)) < 1e-12 * abs(float(E0))


def test_pme_smooth_under_strain():
    """With a fixed mesh the energy is C^1 in the strain (the window); where the
    mesh size changes it jumps by the interpolation error only."""
    pos, q = _charges(20, torch.eye(3, dtype=D) * 10.0)
    scales = torch.linspace(0.97, 1.03, 61, dtype=D)
    E, dE, meshes = [], [], []
    for s0 in scales:
        s = s0.clone().requires_grad_()
        cell = torch.eye(3, dtype=D) * 10 * s
        e = les_energy_pme(q, pos * s, cell, 1.0, 6.0).sum()
        E.append(float(e.detach()))
        dE.append(float(torch.autograd.grad(e, s)[0]))
        meshes.append(pme_mesh([10 * float(s0)] * 3, 1.0, 0.5, 8)[0])
    assert len(set(meshes)) > 1  # 20 -> 24 points at 10 s = 9.75
    E, dE = torch.tensor(E), torch.tensor(dE)
    h = float(scales[1] - scales[0])
    fd = (E[2:] - E[:-2]) / (2 * h)
    same = torch.tensor([meshes[i] == meshes[i + 2] for i in range(len(meshes) - 2)])
    assert (fd - dE[1:-1])[same].abs().max() < 1e-4 * dE.abs().max()
    # at the mesh change: the two meshes agree to the interpolation error
    k = [i for i in range(len(meshes) - 1) if meshes[i] != meshes[i + 1]][0]
    s = float(scales[k + 1])
    cell = torch.eye(3, dtype=D) * 10 * s
    Ea = les_energy_pme_batched(q[None], (pos * s)[None], cell[None], None,
                                [meshes[k]] * 3, 1.0, 6.0, 8).sum()  # fmt: skip
    Eb = les_energy_pme_batched(q[None], (pos * s)[None], cell[None], None,
                                [meshes[k + 1]] * 3, 1.0, 6.0, 8).sum()  # fmt: skip
    assert abs(float(Ea - Eb)) < 2e-6 * abs(float(Eb))


def test_pme_second_derivatives():
    """A force loss differentiates through PME (double backward), and its parameter
    gradient (here: w.r.t. the charges) agrees with the direct sum."""
    cell = torch.eye(3, dtype=D) * 9.0
    pos, q = _charges(16, cell)

    def grad_of_force_loss(fn):
        qq = q.clone().requires_grad_()
        p = pos.clone().requires_grad_()
        E = fn(qq - qq.mean(), p, cell).sum()
        (F,) = torch.autograd.grad(E, [p], create_graph=True)
        (g,) = torch.autograd.grad((F * F).sum(), [qq])
        return g

    g_d = grad_of_force_loss(lambda q, p, c: les_energy_periodic(q, p, c, 1.0, 6.0))
    g_p = grad_of_force_loss(lambda q, p, c: les_energy_pme(q, p, c, 1.0, 6.0))
    assert _relnorm(g_p, g_d) < 1e-4


# ---------------------------------------------------------------------------
# the model


def _model(**kw) -> PlatoPETLR:
    h = copy.deepcopy(get_default_hypers("experimental.plato_pet_lr")["model"])
    h.update(d_model=16, num_heads=2, num_layers=2, cutoff=3.5, num_rbf=8, n_basis=4,
             mesh_channels=2, mesh_spacing=0.9, frames="D2",
             frame_mode="attention")  # fmt: skip
    h.update(kw)
    info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 8],
        targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
    )
    torch.manual_seed(0)
    model = PlatoPETLR(h, info).double().eval()
    with torch.no_grad():  # charges large enough for the LES to matter
        for p in model.charge_head.parameters():
            p.mul_(3.0)
    return model


def _system(model, pos, cell, types):
    s = System(types=types, positions=pos, cell=cell, pbc=torch.tensor([True] * 3))
    return get_system_with_neighbor_lists(s, model.requested_neighbor_lists())


def _model_efs(model, pos, cell, types):
    p = pos.clone().requires_grad_()
    eps = torch.zeros(3, 3, dtype=D, requires_grad=True)
    T = torch.eye(3, dtype=D) + eps
    s = _system(model, p @ T, cell @ T, types)
    E = model([s], {"energy": model.outputs["energy"]})["energy"].block().values.sum()
    gp, ge = torch.autograd.grad(E, [p, eps])
    return E.detach(), -gp, ge


def _water_box(L, n):
    g = torch.Generator().manual_seed(0)
    pos = torch.rand(n, 3, generator=g, dtype=D) * L
    return pos, torch.tensor([1, 8] * (n // 2)), torch.eye(3, dtype=D) * L


def test_model_pme_matches_ewald():
    pos, types, cell = _water_box(9.0, 40)
    direct = _model(les_method="ewald")
    ref = _model_efs(direct, pos, cell, types)
    pme = _model(les_method="pme")
    plan = pme._long_range_plan(cell[None], [_system(pme, pos, cell, types)])
    assert plan[5] == [pme_mesh([9.0] * 3, 1.0, 0.5, 8)]
    out = _model_efs(pme, pos, cell, types)
    assert abs(float(out[0] - ref[0])) < 1e-6 * abs(float(ref[0]))
    assert _relnorm(out[1], ref[1]) < 1e-5
    assert _relnorm(out[2], ref[2]) < 1e-5


def test_model_auto_switches_by_atom_count():
    pos, types, cell = _water_box(9.0, 40)
    for atoms, expect in ((40, True), (41, False)):
        m = _model(les_method="auto", les_pme_atoms=atoms)
        plan = m._long_range_plan(cell[None], [_system(m, pos, cell, types)])
        assert (len(plan[5][0]) == 3) == expect
        assert (plan[4][0].shape[0] == 0) == expect


def test_model_pme_training_step():
    """Training mode: a force loss through PME has parameter gradients close to
    those through the direct sum."""
    pos, types, cell = _water_box(8.0, 24)
    grads = {}
    for method in ("ewald", "pme"):
        m = _model(les_method=method).train()
        m.frame_sampling = "fixed"
        p = pos.clone().requires_grad_()
        s = _system(m, p, cell, types)
        E = m([s], {"energy": m.outputs["energy"]})["energy"].block().values.sum()
        (F,) = torch.autograd.grad(E, [p], create_graph=True)
        loss = E**2 + (F * F).sum()
        params = list(m.charge_head.parameters())
        grads[method] = torch.cat(
            [g.reshape(-1) for g in torch.autograd.grad(loss, params)]
        )
    assert _relnorm(grads["pme"], grads["ewald"]) < 1e-4


def test_model_pme_exported_torchscript():
    from metatomic.torch import ModelEvaluationOptions, ModelOutput

    pos, types, cell = _water_box(9.0, 40)
    model = _model(les_method="pme")
    E0 = model([_system(model, pos, cell, types)], {"energy": model.outputs["energy"]})
    E0 = E0["energy"].block().values.sum().detach()
    scripted = torch.jit.script(model.export())
    opts = ModelEvaluationOptions(
        length_unit="angstrom", outputs={"energy": ModelOutput(sample_kind="system")}
    )
    s = _system(model, pos, cell, types)
    E = scripted([s], opts, check_consistency=True)["energy"].block().values.sum()
    E = E.detach()
    assert abs(float(E - E0)) < 1e-12 * abs(float(E0))
    assert math.isfinite(float(E))
