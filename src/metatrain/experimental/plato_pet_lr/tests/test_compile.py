"""The traced training graph of ``compile`` (float64, ``eager`` backend: the trace
and the torch.func force gradient, without the compiler)."""

import torch
from metatomic.torch import System

from metatrain.experimental.plato_pet_lr.compile import CompiledTrainingStep
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from .test_properties import CELL, _model, _structure


def test_traced_step_matches_eager():
    """Energies, position gradients and the parameter gradient of an energy +
    force loss, on a batch with several mesh sizes / k ranges (one of them a single
    structure) and one cell per structure."""
    model = _model(frames="D2", frame_mode="attention").train()
    info = {
        "energy": get_energy_target_info(
            "energy", {"unit": "eV"}, add_position_gradients=True
        )
    }
    systems = []
    for i, scale in enumerate([1.0, 1.01, 1.02, 1.25]):
        pos, types = _structure(i)
        s = System(types, pos * scale, CELL * scale, torch.tensor([True] * 3))
        systems.append(
            get_system_with_neighbor_lists(s, model.requested_neighbor_lists())
        )
    step = CompiledTrainingStep(model, backend="eager")

    def run(fn):
        model.zero_grad()
        block = fn()["energy"].block()
        E, G = block.values, block.gradient("positions").values
        ((E**2).sum() + (G**2).sum()).backward()
        return E.detach(), G.detach(), [p.grad.clone() for p in model.parameters()]

    ref = run(lambda: evaluate_model(model, systems, info, is_training=True))
    new = run(lambda: step(systems, info))
    assert step.n_compiled == 1 and len(step.graphs) == 1
    key = next(iter(step.graphs))
    assert len(key[2]) > 1  # several mesh sizes
    torch.testing.assert_close(new[0], ref[0], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(new[1], ref[1], rtol=1e-11, atol=1e-12)
    for a, b in zip(new[2], ref[2], strict=True):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)
