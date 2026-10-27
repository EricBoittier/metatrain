"""``attention_backend: fused`` (mlip.cpp's ``torch.ops.mlipcpp.pet_attention``)
against the manual softmax attention: energies, forces, stress and the parameter
gradients of an energy + force loss, on CUDA in float64 and float32; TorchScript.
Skipped without CUDA or the op (``MLIPCPP_TORCH_LIB`` or ``mlipcpp_torch``), except
the fallback test."""

import copy
import warnings

import pytest
import torch
from metatomic.torch import System

from metatrain.experimental.plato_pet_lr.modules.layers import (
    fused_attention_op_available,
    load_fused_attention_op,
)
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from .test_properties import CELL, _model, _structure


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    HAS_OP = load_fused_attention_op()
needs_op = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_OP),
    reason="needs CUDA and torch.ops.mlipcpp.pet_attention",
)

VARIANTS = [
    dict(frames="D2", frame_mode="attention"),  # the 4,000-epoch configuration
    dict(frames="T", frame_mode="attention", num_framed_layers=2),
    dict(frames="D2", frame_mode="full"),
    dict(frames="identity"),
    dict(frames="D2", frame_mode="full", rope=False),
]


def _systems(model, periodic, dtype, device):
    nl = model.requested_neighbor_lists()
    out = []
    for seed in range(3):
        pos, types = _structure(seed)
        cell = CELL if periodic else torch.zeros_like(CELL)
        s = System(
            types=types, positions=pos, cell=cell, pbc=torch.tensor([periodic] * 3)
        )
        out.append(get_system_with_neighbor_lists(s, nl).to(dtype=dtype, device=device))
    return out


def _run(model, systems, periodic):
    """Eval-mode E, F, virial; train-mode parameter gradients of E^2 + F^2."""
    grads = ["positions"] + (["strain"] if periodic else [])
    info = {
        "energy": get_energy_target_info(
            "energy",
            {"unit": "eV"},
            add_position_gradients=True,
            add_strain_gradients=periodic,
        )
    }
    model.eval()
    b = evaluate_model(model, systems, info, is_training=False)["energy"].block()
    out = [b.values.detach()] + [b.gradient(g).values.detach() for g in grads]
    model.train()
    model.zero_grad()
    b = evaluate_model(model, systems, info, is_training=True)["energy"].block()
    loss = (b.values**2).sum() + (b.gradient("positions").values ** 2).sum()
    loss.backward()
    return out + [p.grad.clone() for p in model.parameters() if p.grad is not None]


@needs_op
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_fused_matches_manual(variant, periodic, dtype):
    base = _model(**variant).to(dtype)  # fixed frames: training is deterministic
    models = {}
    for backend in ("manual", "fused"):
        m = copy.deepcopy(base).to("cuda")
        m.set_attention_backend(backend)
        models[backend] = m
    assert models["fused"].use_fused
    ref, new = (
        _run(m, _systems(m, periodic, dtype, "cuda"), periodic) for m in models.values()
    )
    tol = 1e-10 if dtype == torch.float64 else 2e-4
    assert len(ref) == len(new)
    # relative to each tensor, floored for the ones that vanish (gradients of
    # parameters an untrained model hardly uses are rounding noise)
    floor = 1e-3 * max(b.abs().max().item() for b in ref)
    for a, b in zip(new, ref, strict=True):
        scale = max(b.abs().max().item(), floor)
        assert (a - b).abs().max().item() / scale < tol


@needs_op
def test_fused_torchscript():
    model = _model(frames="D2", frame_mode="attention").to("cuda")
    ref = copy.deepcopy(model)
    ref.set_attention_backend("manual")
    model.set_attention_backend("fused")
    scripted = torch.jit.script(model.eval())
    systems = _systems(model, True, torch.float64, "cuda")
    outputs = {"energy": model.outputs["energy"]}
    e = scripted(systems, outputs)["energy"].block().values
    e_ref = ref.eval()(systems, outputs)["energy"].block().values
    torch.testing.assert_close(e, e_ref, rtol=1e-11, atol=1e-12)


def test_fused_fallback():
    """Without the op, ``fused`` warns and runs as ``auto``; with it, CPU inputs use
    the PyTorch attention. Either way the energies are those of ``auto``."""
    model = _model(frames="D2", frame_mode="attention")
    ref = copy.deepcopy(model)
    model.set_attention_backend("fused")
    assert model.use_fused == fused_attention_op_available()
    assert model.hypers["attention_backend"] == "fused"
    systems = _systems(model, True, torch.float64, "cpu")
    outputs = {"energy": model.outputs["energy"]}
    e = model(systems, outputs)["energy"].block().values
    torch.testing.assert_close(e, ref(systems, outputs)["energy"].block().values)
    scripted = torch.jit.script(model)  # the op is only referenced if registered
    torch.testing.assert_close(e, scripted(systems, outputs)["energy"].block().values)
    with pytest.raises(ValueError, match="unknown attention_backend"):
        model.set_attention_backend("flash")
