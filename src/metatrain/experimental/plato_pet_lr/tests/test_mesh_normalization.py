"""``mesh_normalization``: old checkpoints keep ``per_point`` bit for bit; ``physical``
is the Ewald reciprocal potential, independent of the mesh and of added vacuum."""

import gzip
import json
import math
import os

import pytest
import torch
from metatomic.torch import ModelEvaluationOptions, ModelOutput, System

from metatrain.experimental.plato_pet_lr.modules.layers import MeshCoupling
from metatrain.utils.io import model_from_checkpoint
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from .test_properties import _energy, _model


CHECKPOINTS = os.path.join(os.path.dirname(__file__), "checkpoints")


def _load(name: str):
    with gzip.open(os.path.join(CHECKPOINTS, name), "rb") as fd:
        checkpoint = torch.load(fd, weights_only=False)
    return model_from_checkpoint(checkpoint, "export").double().eval()


def _reference_systems(model):
    """A molecule in a 30 A box, the same box at 30 + 1e-7 (31 mesh points per
    axis), and a dense 7 A cell; the systems of ``model-v5_per_point_outputs.json``."""
    types = model.atomic_types
    g = torch.Generator().manual_seed(1)
    out = []
    for L, n, dense in [(30.0, 10, False), (30.0 + 1e-7, 10, False), (7.0, 12, True)]:
        cell = torch.eye(3, dtype=torch.float64) * L
        cell[1, 0] = 0.2
        if dense:
            pos = torch.rand(n, 3, generator=g, dtype=torch.float64) @ cell
        else:
            pos = 15.0 + 3.0 * torch.rand(n, 3, generator=g, dtype=torch.float64)
        out.append((torch.tensor([types[i % len(types)] for i in range(n)]), pos, cell))
    return out


def _efs(model, items):
    """Energies (per atom), forces and stress, by autograd through a strain."""
    nl = model.requested_neighbor_lists()
    eye = torch.eye(3, dtype=torch.float64)
    pos_l, eps_l, systems = [], [], []
    for t, pos, cell in items:
        p = pos.clone().requires_grad_(True)
        eps = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
        s = get_system_with_neighbor_lists(
            System(t, p, cell.clone(), torch.tensor([True] * 3)), nl
        )
        s2 = System(t, p @ (eye + eps), s.cell @ (eye + eps), s.pbc)
        for o in s.known_neighbor_lists():
            s2.add_neighbor_list(o, s.get_neighbor_list(o))
        pos_l.append(p), eps_l.append(eps), systems.append(s2)
    E = model(systems, {"energy": model.outputs["energy"]})["energy"].block().values
    E = E.flatten()
    grads = torch.autograd.grad(E.sum(), pos_l + eps_l)
    F = torch.cat([-g for g in grads[: len(items)]])
    S = torch.stack(
        [
            g / torch.det(c).abs()
            for g, (_, _, c) in zip(grads[len(items) :], items, strict=True)
        ]
    )
    return E.detach(), F, S


@pytest.mark.parametrize("name", ["model-v4_trainer-v1", "model-v5_trainer-v2"])
def test_old_checkpoints_are_per_point(name):
    model = _load(f"{name}.ckpt.gz")
    assert model.hypers["mesh_normalization"] == "per_point"
    assert not model.coupling.physical


def test_old_checkpoint_bit_identical():
    """The v5 reference checkpoint gives exactly the outputs of the v5 code
    (``model-v5_per_point_outputs.json``, from 0512cf06 on x86-64 CPU, float64):
    single and batched evaluation, forces, stress, and the exported model."""
    torch.set_default_dtype(torch.float64)
    try:
        model = _load("model-v5_trainer-v2.ckpt.gz")
        with open(os.path.join(CHECKPOINTS, "model-v5_per_point_outputs.json")) as fd:
            ref = {
                k: torch.tensor([float.fromhex(x) for x in v["hex"]]).reshape(
                    v["shape"]
                )
                for k, v in json.load(fd).items()
            }
        items = _reference_systems(model)
        Es, Fs, Ss = [], [], []
        for item in items:
            E, F, S = _efs(model, [item])
            Es.append(E), Fs.append(F), Ss.append(S)
        assert torch.equal(torch.cat(Es), ref["single_E"])
        assert torch.equal(torch.cat(Fs), ref["single_F"])
        assert torch.equal(torch.cat(Ss), ref["single_S"])
        E, F, S = _efs(model, items)
        assert torch.equal(E, ref["batched_E"])
        assert torch.equal(F, ref["batched_F"])
        assert torch.equal(S, ref["batched_S"])
        nl = model.requested_neighbor_lists()
        systems = [
            get_system_with_neighbor_lists(
                System(t, p, c, torch.tensor([True] * 3)), nl
            )
            for t, p, c in items
        ]
        opts = ModelEvaluationOptions(
            length_unit="angstrom", outputs={"energy": ModelOutput(per_atom=False)}
        )
        out = model.export()(systems, opts, check_consistency=True)
        assert torch.equal(out["energy"].block().values.flatten(), ref["export_E"])
    finally:
        torch.set_default_dtype(torch.float32)


def test_new_models_are_physical():
    model = _model()
    assert model.hypers["mesh_normalization"] == "physical"
    assert model.coupling.physical


def _coupling(normalization: str, spacing: float = 0.5) -> MeshCoupling:
    """A coupling with the pure Ewald filter (``f = 0``)."""
    torch.manual_seed(0)
    m = MeshCoupling(4, 2, spacing, 1, 4, "auto", normalization=normalization)
    m = m.double()
    with torch.no_grad():
        m.filt[2].weight.zero_()
        m.filt[2].bias.zero_()
    return m


def _mesh_potential(m: MeshCoupling, q, pos, cell) -> torch.Tensor:
    M = m.mesh_size(cell)
    return m.mesh_batched(q[None], pos[None], cell[None], None, M, 1)[0].detach()


def _ewald_reciprocal(q, pos, cell, sigma: float, nmax: int) -> torch.Tensor:
    """``4 pi / V sum_{k != 0} exp(-sigma^2 k^2 / 2) / k^2 Re[S(k) e^{-i k r_i}]``,
    ``S(k) = sum_j q_j e^{i k r_j}`` (self term included)."""
    inv, V = torch.linalg.inv(cell), torch.det(cell).abs()
    r = torch.arange(-nmax, nmax + 1, dtype=cell.dtype)
    m = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    m = m[(m != 0).any(-1)]
    k = 2 * math.pi * m @ inv.T
    k2 = (k * k).sum(-1)
    g = 4 * math.pi / V * torch.exp(-0.5 * sigma**2 * k2) / k2
    c, s = torch.cos(pos @ k.T), torch.sin(pos @ k.T)
    return c @ (g[:, None] * (c.T @ q)) + s @ (g[:, None] * (s.T @ q))


@pytest.mark.parametrize("L", [12.0, 24.0])
def test_physical_is_ewald(L):
    """With ``f = 0`` the physical mesh potential is the Ewald reciprocal sum of
    the same Gaussian charges (to the B-spline interpolation error, ~1e-4 at
    ``spacing = 0.5``, ``sigma = 1.5``); ``per_point`` is it divided by ``M``."""
    g = torch.Generator().manual_seed(0)
    cell = torch.tensor(
        [[L, 0.0, 0.0], [0.3, 1.05 * L, 0.0], [0.2, -0.4, 0.95 * L]],
        dtype=torch.float64,
    )
    pos = torch.rand(10, 3, generator=g, dtype=torch.float64) @ cell
    q = torch.randn(10, 2, generator=g, dtype=torch.float64)
    ref = _ewald_reciprocal(q, pos, cell, 1.5, int(1.2 * L))
    phys = _mesh_potential(_coupling("physical"), q, pos, cell)
    assert float((phys - ref).abs().max() / ref.abs().max()) < 3e-4
    per_point = _mesh_potential(_coupling("per_point"), q, pos, cell)
    M = math.prod(_coupling("per_point").mesh_size(cell))
    torch.testing.assert_close(per_point * M, phys, rtol=1e-12, atol=1e-15)
    # the unbatched path (MeshCoupling.forward) uses the same filter
    m = _coupling("physical")
    assert torch.equal(m.kfilter(cell, m.mesh_size(cell)),
                       m.kfilter_batched(cell[None], m.mesh_size(cell))[0])  # fmt: skip


def test_physical_vacuum_invariance():
    """A neutral cluster in growing cubic boxes: the physical potential converges
    (the periodic-image error falls as ``1 / L^3``: the mean-potential shift of
    the quadrupole), the per-point one falls as ``1 / M``."""
    pos0 = torch.tensor(
        [
            [0, 0, 0],
            [1.0, 0.2, 0],
            [0.1, 1.1, 0.3],
            [-0.9, 0.4, 0.8],
            [0.5, -0.7, -0.6],
        ],
        dtype=torch.float64,
    )
    q = torch.tensor(
        [[1.0, 0.5], [-0.6, -0.3], [0.4, 0.2], [-0.5, 0.1], [-0.3, -0.5]],
        dtype=torch.float64,
    )
    Ls = [16.0, 24.0, 32.0, 48.0]
    phys, pp, Ms = [], [], []
    for L in Ls:
        cell = torch.eye(3, dtype=torch.float64) * L
        phys.append(_mesh_potential(_coupling("physical"), q, pos0 + L / 2, cell))
        pp.append(_mesh_potential(_coupling("per_point"), q, pos0 + L / 2, cell))
        Ms.append(math.prod(_coupling("per_point").mesh_size(cell)))
    scale = float(phys[-1].abs().max())
    d = [float((phys[i + 1] - phys[i]).abs().max()) / scale for i in range(3)]
    assert d[2] < 2e-3  # 32 -> 48 A
    for i in range(3):  # ~ 1 / L^3
        expect = Ls[i] ** -3 - Ls[i + 1] ** -3
        first = d[0] / (Ls[0] ** -3 - Ls[1] ** -3)
        assert d[i] / expect == pytest.approx(first, rel=0.25)
    # per_point: 27 x weaker at 48 A than at 16 A (the 1 / M)
    ratio = float(pp[0].abs().max() / pp[-1].abs().max())
    assert ratio > 20.0
    assert ratio == pytest.approx(
        Ms[-1] / Ms[0] * float(phys[0].abs().max() / phys[-1].abs().max()), rel=1e-10
    )


def test_model_vacuum_invariance():
    """The whole model on a molecule in growing boxes: the mesh coupling's share of
    the energy, ``E - E(no mesh)``, converges with ``physical`` (to the ``1 / L``
    image error of the non-neutral mesh channels), and vanishes as ``1 / M`` with
    ``per_point``."""
    g = torch.Generator().manual_seed(3)
    pos0 = 2.0 * torch.rand(6, 3, generator=g, dtype=torch.float64)
    types = torch.tensor([1, 8] * 3)
    Ls = [20.0, 30.0, 40.0]
    share = {}
    for norm in ["physical", "per_point"]:
        model = _model(mesh_normalization=norm, mesh_spacing=1.0, mesh_channels=2)
        torch.manual_seed(1)
        with torch.no_grad():  # a non-trivial filter, a strong coupling, up(0) = 0
            for name, p in model.coupling.named_parameters():
                if name != "up.b":
                    p.add_(0.3 * torch.randn_like(p))

        def energy(L):
            cell = torch.eye(3, dtype=torch.float64) * L
            return float(_energy(model, pos0 + L / 2, cell, types))  # noqa: B023

        on = [energy(L) for L in Ls]
        model.use_mesh = False
        share[norm] = [a - energy(L) for a, L in zip(on, Ls, strict=True)]
    phys, pp = share["physical"], share["per_point"]
    assert abs(phys[2] - phys[1]) < 0.15 * abs(phys[2])
    assert abs(phys[2] - phys[1]) < 0.7 * abs(phys[1] - phys[0])  # converging
    assert abs(pp[2]) < 0.2 * abs(pp[0])  # (20 / 40)^3 = 1 / 8 of the 20 A share
