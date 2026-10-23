"""One Trainer epoch on periodic toy data with energy, forces and stress."""

import copy

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from metatrain.experimental.plato_pet_lr import PlatoPETLR, Trainer
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info


def _energy_map(index: int, n_atoms: int, volume: float) -> TensorMap:
    block = TensorBlock(
        values=torch.tensor([[-10.0 - 0.1 * index]], dtype=torch.float64),
        samples=Labels(["system"], torch.tensor([[index]], dtype=torch.int32)),
        components=[],
        properties=Labels("energy", torch.tensor([[0]])),
    )
    block.add_gradient(
        "positions",
        TensorBlock(
            values=0.05 * torch.randn(n_atoms, 3, 1, dtype=torch.float64),
            samples=Labels(
                ["sample", "system", "atom"],
                torch.tensor(
                    [[0, index, a] for a in range(n_atoms)], dtype=torch.int32
                ),
            ),
            components=[Labels(["xyz"], torch.arange(3).reshape(-1, 1))],
            properties=Labels("energy", torch.tensor([[0]])),
        ),
    )
    stress = 0.02 * torch.randn(1, 3, 3, 1, dtype=torch.float64)
    block.add_gradient(
        "strain",
        TensorBlock(
            values=0.5 * (stress + stress.transpose(1, 2)) * volume,
            samples=Labels(["sample"], torch.tensor([[0]], dtype=torch.int32)),
            components=[
                Labels(["xyz_1"], torch.arange(3).reshape(-1, 1)),
                Labels(["xyz_2"], torch.arange(3).reshape(-1, 1)),
            ],
            properties=Labels("energy", torch.tensor([[0]])),
        ),
    )
    return TensorMap(Labels.single(), [block])


@pytest.mark.parametrize(
    "variant",
    [
        dict(frames="T", frame_sampling="coset"),
        dict(frames="T", num_framed_layers=1, frame_sampling="haar"),
        dict(frames="identity", les=False, mesh_coupling=False),
        dict(frames="D2", frame_mode="attention", frame_sampling="coset"),
        dict(frames="T", frame_mode="attention", num_framed_layers=1),
        dict(frames="D2", num_framed_layers=1, n_basis=1),
        dict(frames="D2", frame_mode="attention", electronegativity="linear"),
    ],
)
def test_one_training_step(variant, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    cell = torch.eye(3, dtype=torch.float64) * 7.0
    systems = [
        System(
            types=torch.tensor([8, 1, 1, 8, 1, 1]),
            positions=torch.rand(6, 3, dtype=torch.float64) * 7.0,
            cell=cell,
            pbc=torch.tensor([True, True, True]),
        )
        for _ in range(4)
    ]
    energies = [_energy_map(i, 6, 343.0) for i in range(4)]
    target = get_energy_target_info(
        "energy", {"unit": "eV"}, add_position_gradients=True, add_strain_gradients=True
    )
    info = DatasetInfo(
        length_unit="angstrom", atomic_types=[1, 8], targets={"energy": target}
    )
    hypers = copy.deepcopy(get_default_hypers("experimental.plato_pet_lr"))
    small = dict(d_model=8, num_heads=2, num_layers=2, n_basis=2, kernel_hidden=4,
                 mesh_channels=1, mesh_spacing=1.5)  # fmt: skip
    hypers["model"].update({**small, **variant})
    model = PlatoPETLR(hypers["model"], info)
    before = copy.deepcopy(model.state_dict())

    training = hypers["training"]
    training.update(num_epochs=1, batch_size=2, checkpoint_interval=100)
    gradients = {
        k: {"type": "mse", "weight": 1.0, "reduction": "mean"}
        for k in ("positions", "strain")
    }
    training["loss"] = {
        "energy": {
            "type": "mse",
            "weight": 1.0,
            "reduction": "mean",
            "gradients": gradients,
        }
    }
    dataset = Dataset.from_dict({"system": systems, "energy": energies})
    trainer = Trainer(training)
    trainer.train(
        model, torch.float32, [torch.device("cpu")], [dataset], [dataset], "."
    )

    assert (
        trainer.best_metric is not None and trainer.best_metric == trainer.best_metric
    )
    state = model.state_dict()
    changed = [
        k
        for k, v in state.items()
        if k in before and v.is_floating_point() and not torch.equal(v, before[k].to(v))
    ]
    assert any(k.startswith("layers.0.") for k in changed)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    for chi in model.electronegativity:  # chi0 and the dchi head train
        assert "electronegativity.0.chi0" in changed
        assert "electronegativity.0.head.l2.W" in changed
        assert float(chi.charge_unit) != 1.0  # sqrt of the fitted energy scale
