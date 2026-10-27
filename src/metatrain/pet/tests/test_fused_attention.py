"""Parity of ``attention_backend: fused`` (``torch.ops.mlipcpp.pet_attention``,
mlip.cpp's fused CUDA attention) with the manual and SDPA attention: energies,
forces, stress and the parameter gradients of a force/stress training step, on
periodic and non-periodic systems with padded neighbour lists, and TorchScript
export through metatomic with the op library as an extension. Skipped without
CUDA or without the op (``MLIPCPP_TORCH_LIB`` or the ``mlipcpp_torch``
package)."""

import copy

import ase.build
import numpy as np
import pytest
import torch
from metatomic.torch import ModelOutput, System, systems_to_torch

from metatrain.pet import PET
from metatrain.pet.modules.transformer import (
    AttentionBlock,
    FusedAttention,
    load_fused_attention_op,
)
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import MODEL_HYPERS


pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and load_fused_attention_op()),
    reason="needs CUDA and torch.ops.mlipcpp.pet_attention",
)

DEVICE = "cuda"
# float64: kernels against float64 torch ops; float32: the fast float32 kernels
# against the float32 manual attention (accumulation order differs)
TOLERANCE = {torch.float64: 1e-9, torch.float32: 2e-4}


def small_hypers(backend: str) -> dict:
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers.update(
        d_pet=32,
        d_head=32,
        d_node=64,
        d_feedforward=64,
        num_heads=2,  # head dim 16
        num_gnn_layers=2,
        num_attention_layers=2,
        cutoff=4.0,
        attention_temperature=1.3,
        attention_backend=backend,
    )
    return hypers


def make_models(dtype):
    info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )
    torch.manual_seed(0)
    reference = PET(small_hypers("manual"), info)
    models = {b: PET(small_hypers(b), info) for b in ("fused", "auto")}
    for m in models.values():
        m.load_state_dict(reference.state_dict())
    models["manual"] = reference
    for m in models.values():
        m.to(DEVICE, dtype)
    assert all(
        isinstance(layer.attention.fused_attention, FusedAttention)
        for gnn in models["fused"].backend.gnn_layers
        for layer in gnn.trans.layers
    )
    return models


def random_systems(dtype):
    """Molecules (non-periodic, varied neighbour counts, so padded edges) and
    periodic cells with rattled positions."""
    rng = np.random.default_rng(0)
    atoms = [ase.build.molecule(name) for name in ("CH3CH2OH", "C6H6", "H2O", "NH3")]
    for cell in (ase.build.bulk("C", "diamond", a=3.6, cubic=True),):
        atoms.append(cell * (2, 1, 1))
    atoms.append(ase.build.bulk("C", "diamond", a=3.6, cubic=True))
    atoms[-1].numbers[::3] = 8
    for a in atoms:
        a.positions += 0.15 * rng.standard_normal(a.positions.shape)
    return systems_to_torch(atoms, dtype=dtype, device=DEVICE)


def energy_forces_stress(model, systems, training: bool, create_graph: bool):
    """Energies, forces and stress (virial / volume, periodic systems only) by
    differentiating w.r.t. positions and a strain applied to positions and cell."""
    model.train(training)
    strained = []
    strains = []
    positions = []
    for s in systems:
        r = s.positions.detach().clone().requires_grad_()
        eps = torch.eye(3, dtype=r.dtype, device=r.device).requires_grad_()
        new = System(
            types=s.types,
            positions=r @ eps,
            cell=s.cell.detach() @ eps,
            pbc=s.pbc,
        )
        new = get_system_with_neighbor_lists(new, model.requested_neighbor_lists())
        strained.append(new)
        strains.append(eps)
        positions.append(r)
    out = model(strained, {"energy": ModelOutput(sample_kind="system")})
    energy = out["energy"].block().values.sum(-1)
    grads = torch.autograd.grad(
        energy.sum(), positions + strains, create_graph=create_graph
    )
    forces = [-g for g in grads[: len(systems)]]
    stress = [
        g / torch.abs(torch.det(s.cell.detach()))
        for g, s in zip(grads[len(systems) :], systems, strict=True)
        if bool(s.pbc.all())
    ]
    return energy, forces, stress


def max_diffs(a, b):
    a = torch.cat([t.reshape(-1) for t in a]).double()
    b = torch.cat([t.reshape(-1) for t in b]).double()
    abs_diff = (a - b).abs().max().item()
    return abs_diff, abs_diff / b.abs().max().clamp_min(1e-30).item()


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("training", [False, True])
def test_energy_forces_stress(dtype, training):
    """Eval mode compares with SDPA ("auto"), training mode with the manual
    attention (which "auto" uses there)."""
    models = make_models(dtype)
    systems = random_systems(dtype)
    reference = "manual" if training else "auto"
    ref = energy_forces_stress(models[reference], systems, training, False)
    got = energy_forces_stress(models["fused"], systems, training, False)
    for name, a, b in zip(("energy", "forces", "stress"), got, ref, strict=True):
        a = [a] if isinstance(a, torch.Tensor) else a
        b = [b] if isinstance(b, torch.Tensor) else b
        abs_diff, rel = max_diffs(a, b)
        mode = "train/manual" if training else "eval/sdpa"
        print(f"PARITY {dtype} {mode} {name}: abs {abs_diff:.3e} rel {rel:.3e}")
        assert rel < TOLERANCE[dtype], (name, rel)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_training_step_gradients(dtype):
    """Parameter gradients of an energy + forces + stress loss (double backward
    through the attention)."""
    models = make_models(dtype)
    systems = random_systems(dtype)

    def grads(model):
        model.zero_grad()
        energy, forces, stress = energy_forces_stress(model, systems, True, True)
        loss = (
            (energy**2).sum()
            + sum((f**2).sum() for f in forces)
            + sum((s**2).sum() for s in stress)
        )
        loss.backward()
        return [p.grad.detach().clone() for p in model.parameters()]

    ref = grads(models["manual"])
    got = grads(models["fused"])
    assert len(ref) == len(got)
    abs_diff, rel = max_diffs(got, ref)
    print(f"PARITY {dtype} param grads: abs {abs_diff:.3e} rel {rel:.3e}")
    assert rel < 5 * TOLERANCE[dtype], rel


def test_cpu_fallback():
    """On CPU the fused block falls back to SDPA / manual attention."""
    block = AttentionBlock(32, 2, 1.0, attention_backend="fused").double()
    assert block.use_fused_attention
    x = torch.randn(5, 7, 32, dtype=torch.float64)
    cutoff = torch.rand(5, 7, 7, dtype=torch.float64)
    ref = AttentionBlock(32, 2, 1.0).double()
    ref.load_state_dict(block.state_dict())
    assert torch.allclose(block(x, cutoff, True), ref(x, cutoff, True))
    assert torch.allclose(block(x, cutoff, False), ref(x, cutoff, False))


def test_unsupported_head_dim_falls_back():
    with pytest.warns(UserWarning, match="head dimension"):
        block = AttentionBlock(24, 2, 1.0, attention_backend="fused")
    assert not block.use_fused_attention


CHILD = """
import sys, torch
from metatomic.torch import ModelEvaluationOptions, ModelOutput, load_atomistic_model
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
model = load_atomistic_model(sys.argv[1]).to("cuda")
systems = [
    get_system_with_neighbor_lists(s.to("cuda"), model.requested_neighbor_lists())
    for s in torch.load(sys.argv[2], weights_only=False)
]
for s in systems:
    s.positions.requires_grad_(True)
options = ModelEvaluationOptions(
    length_unit="Angstrom", outputs={"energy": ModelOutput(sample_kind="system")}
)
e = model(systems, options, check_consistency=True)["energy"].block().values
f = torch.autograd.grad(e.sum(), [s.positions for s in systems])
torch.save([e.detach().cpu()] + [g.cpu() for g in f], sys.argv[3])
"""


def test_export(tmp_path):
    """Export with the op library recorded as a metatomic extension, reload in a
    fresh process (the op then comes only from the extension) and evaluate."""
    import os
    import subprocess
    import sys

    dtype = torch.float32
    models = make_models(dtype)
    systems = random_systems(dtype)
    path = str(tmp_path / "pet-fused.pt")
    # the op library usually lives outside site-packages, where metatomic's
    # ``collect_extensions`` refuses to copy it: record its absolute path instead
    models["fused"].to("cpu").export().save(path)
    torch.save([s.to("cpu") for s in systems], tmp_path / "systems.pt")
    env = {k: v for k, v in os.environ.items() if k != "MLIPCPP_TORCH_LIB"}
    out = str(tmp_path / "out.pt")
    subprocess.run(
        [sys.executable, "-c", CHILD, path, str(tmp_path / "systems.pt"), out],
        check=True,
        env=env,
    )
    loaded = torch.load(out)
    ref_e, ref_f, _ = energy_forces_stress(models["auto"], systems, False, False)
    abs_e, rel_e = max_diffs([loaded[0].reshape(-1)], [ref_e.cpu()])
    abs_f, rel_f = max_diffs(loaded[1:], [-f.cpu() for f in ref_f])
    print(f"PARITY export energy: abs {abs_e:.3e} rel {rel_e:.3e}")
    print(f"PARITY export forces: abs {abs_f:.3e} rel {rel_f:.3e}")
    assert rel_e < TOLERANCE[dtype] and rel_f < TOLERANCE[dtype], (rel_e, rel_f)
