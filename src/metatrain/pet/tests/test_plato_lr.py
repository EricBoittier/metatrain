"""PET with Plato-PET-LR components: exact function preservation at conversion,
live gradients, the RoPE alternatives, open-boundary mesh, export round trip."""

import copy
import math

import metatensor.torch as mts
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System

from metatrain.pet import PET
from metatrain.pet.convert_plato_lr import convert_model, pet_to_plato_lr
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


ALL = {
    "frames": "D2",
    "edge_group_sum": True,
    "mesh_coupling": True,
    "couple_at": 0,
    "mesh_spacing": 0.8,
    "les": True,
    "total_charge_constraint": True,
}
SCALE = 2.5  # energy scale of the scaler: latent charge unit sqrt(2.5)


def small_pet(**kwargs) -> PET:
    hypers = copy.deepcopy(get_default_hypers("pet")["model"])
    hypers.update(
        d_pet=16, d_head=16, d_node=24, d_feedforward=16, num_heads=2,
        num_attention_layers=2, num_gnn_layers=2, cutoff=4.0,
        system_conditioning=True, max_charge=3, max_spin_multiplicity=4,
    )  # fmt: skip
    hypers.update(kwargs)
    info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 8],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )
    torch.manual_seed(0)
    model = PET(hypers, info).to(torch.float64)
    with torch.no_grad():  # a "pretrained" model: conditioning gate and scale live
        for p in model.backend.system_conditioning.project[2].parameters():
            p.normal_(0, 0.1)
    sc = model.scaler
    for kind, maps in [
        ("", sc.model.scales),
        ("_per_target", sc.model.per_target_scales),
    ]:
        maps["energy"].block().values.fill_(SCALE)
        sc.register_buffer(
            f"energy{kind}_scaler_buffer",
            mts.save_buffer(mts.make_contiguous(maps["energy"])),
        )
    sc.sync_tensor_maps()
    return model


def scalar(value: float, name: str) -> TensorMap:
    return TensorMap(
        Labels.single(),
        [
            TensorBlock(
                torch.tensor([[value]], dtype=torch.float64),
                Labels("system", torch.tensor([[0]])),
                [],
                Labels(name, torch.tensor([[0]])),
            )
        ],
    )


def systems(model, seed=1):
    g = torch.Generator().manual_seed(seed)
    out = []
    # an open molecule (charge +1, doublet) and a periodic box (neutral singlet)
    for n, types, cell, pbc, q, spin in [
        (6, [8, 6, 1, 1, 1, 1], torch.zeros(3, 3), False, 1.0, 2),
        (8, [8, 8, 1, 1, 1, 1, 6, 6], 6.5 * torch.eye(3), True, 0.0, 1),
    ]:
        pos = torch.rand(n, 3, generator=g, dtype=torch.float64) * 3.5
        s = System(
            types=torch.tensor(types),
            positions=pos.requires_grad_(),
            cell=cell.to(torch.float64),
            pbc=torch.tensor([pbc] * 3),
        )
        s.add_data("charge", scalar(q, "charge"))
        s.add_data("spin_multiplicity", scalar(spin, "spin_multiplicity"))
        out.append(get_system_with_neighbor_lists(s, model.requested_neighbor_lists()))
    return out


def energy_forces(model, syst, create_graph=False):
    e = model(syst, {"energy": ModelOutput(sample_kind="system")})["energy"]
    e = e.block().values.sum(-1)
    f = torch.autograd.grad(
        e.sum(), [s.positions for s in syst], create_graph=create_graph
    )
    return e, torch.cat(f)


def new_params(model, converted):
    old = set(dict(model.named_parameters()))
    return {k: p for k, p in converted.named_parameters() if k not in old}


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("featurizer", ["feedforward", "residual"])
def test_exact_at_conversion(training, featurizer):
    model = small_pet(featurizer_type=featurizer)
    conv = convert_model(model, ALL)
    model.train(training)
    conv.train(training)
    syst = systems(model)
    e0, f0 = energy_forces(model, syst)
    e1, f1 = energy_forces(conv, syst)
    assert torch.equal(e0, e1) and torch.equal(f0, f1)  # bit-identical


def test_gradients_live():
    model = small_pet()
    conv = convert_model(model, ALL)
    conv.train()
    new = new_params(model, conv)
    gates = {k for k in new if k.endswith("_gate")}
    assert len(gates) == 2 * 2 + 2 + 2  # rope per attention layer, edge, mesh, les

    def grads():
        conv.zero_grad()
        e, f = energy_forces(conv, systems(model), create_graph=True)
        ((e - 1.0) ** 2).sum().add((f**2).sum()).backward()
        return {
            k: 0.0 if p.grad is None else float(p.grad.norm()) for k, p in new.items()
        }

    g0 = grads()
    assert all(g0[k] > 0 for k in gates), g0  # first batch: every gate is live
    with torch.no_grad():  # one gate step
        for k in gates:
            new[k].add_(-1e-2 * torch.sign(new[k].grad))
    g1 = grads()
    assert all(v > 0 for v in g1.values()), {k: v for k, v in g1.items() if v == 0}


def test_rope_small_init():
    model = small_pet()
    model.eval()
    syst = systems(model)
    e0, _ = energy_forces(model, syst)
    de = []
    for eps in [1e-2, 1e-3]:
        conv = convert_model(
            model, {"frames": "D2", "rope_init": "small", "rope_eps": eps}
        ).eval()
        de.append(float((energy_forces(conv, syst)[0] - e0).detach().abs().max()))
    # first order cancels over D2: O(eps^2)
    assert 50 < de[0] / de[1] < 200, de
    conv.train()
    conv.zero_grad()
    energy_forces(conv, syst, create_graph=True)[0].sum().backward()
    assert all(
        float(p.grad.norm()) > 0
        for k, p in conv.named_parameters()
        if k.endswith("rope_omega")
    )


def test_identity_frames_first_order():
    """Without a group average (one frame) the small-omega change is O(eps)."""
    model = small_pet()
    model.eval()
    syst = systems(model)
    e0, _ = energy_forces(model, syst)
    de = []
    for eps in [1e-2, 1e-3]:
        conv = convert_model(
            model, {"frames": "C3", "rope_init": "small", "rope_eps": eps}
        ).eval()
        de.append(float((energy_forces(conv, syst)[0] - e0).detach().abs().max()))
    assert 5 < de[0] / de[1] < 20, de  # C3 has an invariant axis: first order


def test_total_charge_constraint():
    model = small_pet()
    conv = convert_model(model, ALL)
    les = conv.backend.les
    assert math.isclose(float(les.charge_unit), math.sqrt(SCALE))
    raw = {}
    charges = les.charges
    les.charges = lambda *a: raw.setdefault("q", charges(*a))
    conv.eval()
    syst = systems(model)
    energy_forces(conv, syst)
    q, n0 = raw["q"], len(syst[0])
    # the shift les.forward applies, and the charge in e it implies
    for sl, Q in [(slice(0, n0), 1.0), (slice(n0, None), 0.0)]:
        qs = q[sl] - (q[sl].sum(0) - Q / les.charge_unit) / q[sl].shape[0]
        assert torch.allclose(qs.sum(0) * les.charge_unit, torch.tensor([Q]).double())
    # the energy is that of the shifted charges
    with torch.no_grad():
        les.les_gate.fill_(1.0)
    from metatrain.experimental.plato_pet_lr.modules.ewald import les_energy_finite

    les.charges = lambda *a: charges(*a)
    sys0 = [syst[0]]
    raw.clear()
    les.charges = lambda *a: raw.setdefault("q", charges(*a))
    conv.les_probe = True
    out = les(*les_inputs(conv, sys0))
    q0 = raw["q"] - (raw["q"].sum(0) - 1.0 / les.charge_unit) / n0
    ref = les_energy_finite(q0, sys0[0].positions, les.sigma)
    assert torch.allclose(out, ref)


def les_inputs(model, syst):
    """The inputs of PETLES.forward for ``syst``, recomputed with the backend."""
    from metatrain.pet.model import _extract_charge_spin_multiplicity
    from metatrain.pet.modules.structures import concatenate_structures

    b = model.backend
    (pos, centers, nbrs, species, cells, shifts, sidx, _) = concatenate_structures(
        syst, model.requested_neighbor_lists()[0]
    )
    batch = b.preprocess(pos, centers, nbrs, species, cells, shifts, sidx, 1.0)
    batch["charge"], batch["spin_multiplicity"] = _extract_charge_spin_multiplicity(
        syst, pos.device
    )
    batch.update(positions=pos, cells=cells, system_indices=sidx,
                 pbc=torch.stack([s.pbc for s in syst]))  # fmt: skip
    nodes, edges = b.calculate_features(batch)
    return (nodes, edges, batch["padding_mask"], batch["cutoff_factors"], pos, cells,
            batch["pbc"], sidx, batch["charge"])  # fmt: skip


def test_mesh_open_limit():
    """The periodic mesh (Ewald-normalised) of a cluster in a growing box tends to
    the real-space open-boundary kernel (neutral channels)."""
    from metatrain.pet.modules.plato_lr import PETMeshCoupling

    torch.manual_seed(0)
    m = PETMeshCoupling(8, 2, 0.5, 1.5, 4, 16, "realspace").to(torch.float64)
    pos = torch.rand(6, 3, dtype=torch.float64) * 4
    v = torch.randn(6, 2, dtype=torch.float64)
    v = v - v.mean(0)
    ref = m.realspace(v, pos)
    err = []
    for L in [16.0, 32.0]:
        p = pos + L / 2 - 2
        M = [int(math.ceil(L / 0.5))] * 3
        mesh = m.coupling.mesh_batched(
            v[None], p[None], (L * torch.eye(3, dtype=torch.float64))[None], None, M, 1
        )[0] * float(M[0] * M[1] * M[2])
        err.append(float(((mesh - ref).abs().max() / ref.abs().max()).detach()))
    print("mesh vs open", err)
    assert err[1] < err[0] and err[1] < 2e-2, err


def test_checkpoint_round_trip_and_export(tmp_path):
    model = small_pet()
    ckpt = model.get_checkpoint()
    ckpt["model_state_dict"] = ckpt["best_model_state_dict"]
    ckpt["epoch"] = ckpt["best_epoch"] = 0
    new_ckpt = pet_to_plato_lr(ckpt, ALL)
    conv = PET.load_checkpoint(copy.deepcopy(new_ckpt), "export")
    conv.eval()
    model.eval()
    syst = systems(model)
    e0, f0 = energy_forces(model, syst)
    e1, f1 = energy_forces(conv, syst)
    assert torch.equal(e0, e1) and torch.equal(f0, f1)

    exported = conv.export()
    path = str(tmp_path / "model.pt")
    exported.save(path)
    from metatomic.torch import ModelEvaluationOptions, load_atomistic_model

    loaded = load_atomistic_model(path)
    syst = systems(model)
    opts = ModelEvaluationOptions(
        length_unit="Angstrom", outputs={"energy": ModelOutput(sample_kind="system")}
    )
    out = loaded(syst, opts, check_consistency=True)
    e2 = out["energy"].block().values.sum(-1)
    assert torch.allclose(e0.detach(), e2.detach(), rtol=0, atol=1e-12)



def test_frame_attention_ignores_padding():
    """The frame path masks padded neighbour slots (cutoff factor 0) exactly: with the
    gate at 1 the attention output of the real tokens does not depend on the padded
    tokens' contents (PET's finite log(1e-15) bias lets large rotated scores through)."""
    from metatrain.experimental.plato_pet_lr.modules.frames import platonic_group
    from metatrain.pet.modules.plato_lr import FrameRoPEAttention, rope_omega_init
    from metatrain.pet.modules.transformer import AttentionBlock

    torch.manual_seed(0)
    N, L, D, H = 3, 6, 16, 2
    block = AttentionBlock(D, H, 1.0).to(torch.float64)
    att = FrameRoPEAttention(
        block,
        platonic_group("D2"),
        rope_omega_init(D // H, 3.0, 0),
        gated=True,
    ).to(torch.float64)
    with torch.no_grad():
        att.rope_gate.fill_(1.0)
    cf = torch.ones(N, L, dtype=torch.float64)
    cf[:, -2:] = 0.0  # two padded slots per sequence
    cf = cf[:, None, :].repeat(1, L, 1)
    x = torch.randn(N, L, D, dtype=torch.float64)
    pos = torch.randn(N, L, 3, dtype=torch.float64) * 2
    pos[:, 0] = 0.0
    out = att(x, cf, True, pos)
    x2, pos2 = x.clone(), pos.clone()
    x2[:, -2:] = 50 * torch.randn(N, 2, D, dtype=torch.float64)
    pos2[:, -2:] = 5 * torch.randn(N, 2, 3, dtype=torch.float64)
    out2 = att(x2, cf, True, pos2)
    assert torch.allclose(out[:, :-2], out2[:, :-2], rtol=0, atol=1e-10)
