"""The linear electronegativity term: sign identifiability, symmetry, gradients,
TorchScript, and conversion of trained models (float64)."""

import gzip
import math

import pytest
import torch
from metatomic.torch import ModelEvaluationOptions, ModelOutput, System

import metatrain.experimental.plato_pet_lr.model as plato_model
from metatrain.experimental.plato_pet_lr.convert import (
    flip_charge_sign,
    with_electronegativity,
)
from metatrain.experimental.plato_pet_lr.modules.electronegativity import QEQ_CHI_EV
from metatrain.experimental.plato_pet_lr.modules.frames import platonic_group
from metatrain.utils.io import model_from_checkpoint
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from .test_properties import CELL, _energy, _model, _rel, _structure


D2_ATT = dict(frames="D2", frame_mode="attention")


def _chi_model(randomise: bool = True, **kw):
    model = _model(electronegativity="linear", **kw)
    if randomise:  # a trained-like dchi head and chi table, charge unit != 1
        g = torch.Generator().manual_seed(5)
        with torch.no_grad():
            chi = model.electronegativity[0]
            chi.head.l2.W.copy_(torch.randn(chi.head.l2.W.shape, generator=g))
            chi.chi0.add_(
                torch.randn(chi.chi0.shape, generator=g, dtype=chi.chi0.dtype)
            )
            chi.charge_unit.fill_(0.37)
    return model


def _charges(model, pos, cell, types):
    """Neutral latent charges (first channel) that enter the LES energy (one
    periodic structure: the padded ``(1, atoms, channels)`` layout)."""
    seen = []
    orig = plato_model.les_energy_periodic_batched

    def spy(q, *args):
        seen.append(q[:, :, 0].detach().reshape(-1))
        return orig(q, *args)

    plato_model.les_energy_periodic_batched = spy
    try:
        _energy(model, pos, cell, types)
    finally:
        plato_model.les_energy_periodic_batched = orig
    return torch.cat(seen)


def test_initial_table_and_zero_correction():
    model = _chi_model(randomise=False)
    chi = model.electronegativity[0]
    assert chi.chi0.detach().tolist() == pytest.approx(
        [QEQ_CHI_EV[1], QEQ_CHI_EV[8]], 1e-6
    )
    assert float(chi.head.l2.W.detach().abs().sum()) == 0.0
    assert set(_model().state_dict()) < set(model.state_dict())  # 'none' unchanged


@pytest.mark.parametrize("variant", [{}, D2_ATT])
def test_charge_sign_identifiable(variant):
    """``q -> -q`` leaves the energy unchanged without chi, changes it with chi."""
    pos, types = _structure()
    plain = _model(**variant)
    assert _rel(_energy(flip_charge_sign(plain), pos, CELL, types),
                _energy(plain, pos, CELL, types)) < 1e-12  # fmt: skip
    q = _charges(plain, pos, CELL, types)
    assert torch.allclose(_charges(flip_charge_sign(plain), pos, CELL, types), -q)

    for randomise in (False, True):
        model = _chi_model(randomise, **variant)
        E, E_flip = (
            _energy(m, pos, CELL, types) for m in (model, flip_charge_sign(model))
        )
        assert _rel(E_flip, E) > 1e-6


def test_conversion_adds_only_chi0_q():
    """A trained model converted to ``linear``: dchi = 0, so the energy changes by
    exactly ``sum_i chi0[Z_i] q_i / charge_unit`` (q in model units)."""
    pos, types = _structure()
    plain = _model(**D2_ATT)
    plain.scaler.model.scales["energy"].block().values[:] = 0.04
    new = with_electronegativity(plain).eval()
    cu = float(new.electronegativity[0].charge_unit)
    assert abs(cu - math.sqrt(0.04)) < 1e-12
    q = _charges(plain, pos, CELL, types)
    chi0 = new.electronegativity[0].chi0.detach()[new.species_to_species_index[types]]
    assert torch.allclose(
        chi0, torch.tensor([QEQ_CHI_EV[int(z)] for z in types]).double()
    )
    # the scaler multiplies the model energy by s at evaluation
    expected = _energy(plain, pos, CELL, types) + 0.04 * (chi0 * q).sum() / cu
    assert _rel(_energy(new, pos, CELL, types), expected) < 1e-12
    torch.testing.assert_close(_charges(new, pos, CELL, types), q)


@pytest.mark.parametrize("variant", [dict(frames="D2"), D2_ATT])
def test_group_invariance_with_chi(variant):
    model = _chi_model(**variant)
    pos, types = _structure()
    E0 = _energy(model, pos, CELL, types)
    for Q in platonic_group("D2")[1:]:
        assert _rel(_energy(model, pos @ Q.T, CELL @ Q.T, types), E0) < 1e-12


@pytest.mark.parametrize("periodic", [True, False])
def test_forces_and_stress_with_chi(periodic):
    model = _chi_model(**D2_ATT)
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
            Ep, Em = (
                _energy(
                    model,
                    pos @ (torch.eye(3) + s * eps),
                    CELL @ (torch.eye(3) + s * eps),
                    types,
                )  # fmt: skip
                for s in (1, -1)
            )
            fd = (Ep - Em) / (2 * h)
            assert abs(float(fd) - float(dE_de[a, b])) < 1e-6 * max(1.0, abs(float(fd)))


def test_exported_torchscript_with_chi():
    model = _chi_model(**D2_ATT)
    pos, types = _structure()
    E0 = _energy(model, pos, CELL, types)
    exported = model.export()
    scripted = torch.jit.script(exported)
    s = System(types=types, positions=pos, cell=CELL, pbc=torch.tensor([True] * 3))
    s = get_system_with_neighbor_lists(s, exported.requested_neighbor_lists())
    opts = ModelEvaluationOptions(
        length_unit="angstrom", outputs={"energy": ModelOutput(sample_kind="system")}
    )
    E = scripted([s], opts, check_consistency=True)["energy"].block().values.sum()
    assert _rel(E, E0) < 1e-12


@pytest.mark.parametrize("version", [1, 2])
def test_old_checkpoints_convert(version):
    """v1/v2 checkpoints load as ``none`` and convert to ``linear``."""
    path = (
        f"{__file__.rsplit('/', 1)[0]}/checkpoints/model-v{version}_trainer-v1.ckpt.gz"
    )
    with gzip.open(path, "rb") as fd:
        checkpoint = torch.load(fd, weights_only=False)
    model = model_from_checkpoint(checkpoint, "export")
    assert model.hypers["electronegativity"] == "none"
    assert len(model.electronegativity) == 0
    new = with_electronegativity(model)
    assert new.hypers["electronegativity"] == "linear"
    assert len(new.electronegativity) == 1


def test_frozen_chi0_and_bounded_dchi():
    """chi0_trainable: false keeps the QEq table (a buffer, no gradient);
    dchi_bound = b keeps |dchi| < b however large the head output."""
    m = _chi_model(chi0_trainable=False, dchi_bound=1.0)
    chi = m.electronegativity[0]
    assert "electronegativity.0.chi0" not in dict(m.named_parameters())
    assert "electronegativity.0.chi0" in m.state_dict()
    with torch.no_grad():
        chi.head.l2.W.normal_(std=100.0)
        chi.head.l2.b.fill_(50.0)
    _, types = _structure()
    h = torch.randn(len(types), 1, m.layers[0].n0.normalized_shape[0], dtype=chi.chi0.dtype)
    dchi = chi.chi(h, torch.zeros(len(types), dtype=torch.long), m.I_rel, m.I_w) - chi.chi0[0]
    assert dchi.abs().max() <= 1.0
