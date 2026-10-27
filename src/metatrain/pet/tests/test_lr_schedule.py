"""Per-group learning-rate schedules: golden file, optimizer/scheduler smoke test
(lr per group per step, freezing, resume)."""

import copy
import json
from pathlib import Path

import torch

from metatrain.pet.convert_plato_lr import convert_model
from metatrain.pet.modules import lr_schedule as ls

from .test_plato_lr import ALL, small_pet


GOLDEN = Path(__file__).parents[4] / "tests" / "resources" / "lr_schedule_golden.json"


def test_golden_up_to_date():
    assert json.loads(GOLDEN.read_text()) == json.loads(json.dumps(ls.golden()))


def run(model, spec, steps, spe, epochs, start=None):
    opt = ls.GroupOptimizer(model, spec)
    sched = ls.GroupLRScheduler(opt, spec, spe, epochs)
    if start is not None:
        opt.load_state_dict(start[0])
        sched.load_state_dict(start[1])
    lrs = []
    for _ in range(steps):
        opt.zero_grad()
        loss = sum((p.double() ** 2).sum() for p in model.parameters())
        loss.backward()
        lrs.append({g["name"]: g["lr"] for g in opt.param_groups})
        opt.drop_frozen_grads()
        opt.step()
        sched.step()
    return opt, sched, lrs


def test_smoke_matches_golden_and_resumes():
    model = convert_model(small_pet(), ALL)
    case = json.loads(GOLDEN.read_text())["cases"][0]
    spec, spe, ep = case["spec"], case["steps_per_epoch"], case["num_epochs"]
    names = dict(model.named_parameters())
    groups = ls.GroupOptimizer(model, spec).names
    head0 = {n: names[n].detach().clone() for n in groups["heads"]}

    full = copy.deepcopy(model)
    _, _, lrs = run(full, spec, 14, spe, ep)
    for n, row in enumerate(lrs):
        for g, lr in row.items():
            assert abs(lr - case["lr"][g][n]) < 1e-15, (g, n)

    # frozen (heads, 2 epochs = 10 steps): no update, no optimizer state
    part = copy.deepcopy(model)
    opt, sched, _ = run(part, spec, 7, spe, ep)
    p = dict(part.named_parameters())
    assert all(torch.equal(p[n], head0[n]) for n in head0)
    state = opt.optimizers[0].state
    assert all(len(state[p[n]]) == 0 for n in head0)
    assert any(len(state[p[n]]) > 0 for n in groups["new_lr"])

    # resume from the state dicts: same lrs and parameters as the uninterrupted run
    resumed = copy.deepcopy(part)
    saved = (copy.deepcopy(opt.state_dict()), copy.deepcopy(sched.state_dict()))
    _, _, lrs2 = run(resumed, spec, 7, spe, ep, start=saved)
    assert lrs2 == lrs[7:]
    for (n, a), b in zip(full.named_parameters(), resumed.parameters(), strict=True):
        assert torch.allclose(a, b, rtol=0, atol=1e-14), n
