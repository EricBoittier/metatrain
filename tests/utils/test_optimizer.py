import copy

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from metatrain.utils.optimizer import MuonAdamW, create_optimizer


def _tiny_module() -> nn.Module:
    module = nn.Module()
    module.linear = nn.Linear(3, 4)
    module.vector = nn.Parameter(torch.randn(5))
    module.cube = nn.Parameter(torch.randn(2, 2, 2))
    return module


def _backward(module: nn.Module) -> None:
    features = torch.randn(2, 3)
    loss = module.linear(features).sum() + module.vector.sum() + module.cube.sum()
    loss.backward()


def _param_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {id(param) for group in optimizer.param_groups for param in group["params"]}


def _assert_scheduled_lr(optimizer: MuonAdamW) -> None:
    for group in optimizer.param_groups:
        inner = optimizer._muon if group["role"] == "muon" else optimizer._adamw
        assert inner is not None
        assert inner.param_groups[0]["lr"] == group["lr"]


def test_adam_is_the_default():
    parameter = nn.Parameter(torch.zeros(2))
    optimizer = create_optimizer([parameter], {"learning_rate": 1e-3})
    assert type(optimizer) is torch.optim.Adam

    decayed = create_optimizer(
        [parameter], {"learning_rate": 1e-3, "weight_decay": 0.01}
    )
    assert type(decayed) is torch.optim.AdamW
    assert decayed.param_groups[0]["weight_decay"] == 0.01


def test_unknown_optimizer_name():
    parameter = nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError, match="Unknown optimizer"):
        create_optimizer([parameter], {"optimizer": "sgd", "learning_rate": 1e-3})


def test_muon_requires_pytorch_muon(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delattr(torch.optim, "Muon")
    parameter = nn.Parameter(torch.zeros(2, 2))
    with pytest.raises(RuntimeError, match="PyTorch >= 2.9"):
        create_optimizer([parameter], {"optimizer": "muon", "learning_rate": 1e-3})


def test_muon_splits_parameters_steps_and_roundtrips_state():
    torch.manual_seed(0)
    module = _tiny_module()
    hypers = {"optimizer": "muon", "learning_rate": 0.05, "weight_decay": None}
    optimizer = create_optimizer(module.parameters(), hypers)

    assert isinstance(optimizer, MuonAdamW)
    assert isinstance(optimizer._muon, torch.optim.Muon)
    assert isinstance(optimizer._adamw, torch.optim.AdamW)
    assert optimizer._muon.param_groups[0]["momentum"] == 0.95
    assert optimizer._muon.param_groups[0]["nesterov"] is True
    assert optimizer._muon.param_groups[0]["adjust_lr_fn"] == "match_rms_adamw"
    assert optimizer._muon.param_groups[0]["weight_decay"] == 0.0
    assert optimizer._adamw.param_groups[0]["weight_decay"] == 0.0

    muon_ids = _param_ids(optimizer._muon)
    adamw_ids = _param_ids(optimizer._adamw)
    assert id(module.linear.weight) in muon_ids
    assert id(module.linear.bias) in adamw_ids
    assert id(module.vector) in adamw_ids
    assert id(module.cube) in adamw_ids

    weight_before = module.linear.weight.detach().clone()
    bias_before = module.linear.bias.detach().clone()
    _backward(module)
    optimizer.step()
    assert not torch.equal(weight_before, module.linear.weight)
    assert not torch.equal(bias_before, module.linear.bias)

    restored = create_optimizer(module.parameters(), hypers)
    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    assert isinstance(restored, MuonAdamW)
    for parameter in (module.linear.weight, module.linear.bias, module.vector):
        inner = restored._muon if parameter.ndim == 2 else restored._adamw
        original = optimizer._muon if parameter.ndim == 2 else optimizer._adamw
        assert inner is not None and original is not None
        for key, value in original.state[parameter].items():
            restored_value = inner.state[parameter][key]
            if torch.is_tensor(value):
                torch.testing.assert_close(restored_value, value)
            else:
                assert restored_value == value


def test_muon_weight_decay_is_shared():
    module = _tiny_module()
    optimizer = create_optimizer(
        module.parameters(),
        {"optimizer": "muon", "learning_rate": 1e-3, "weight_decay": 0.2},
    )
    assert isinstance(optimizer, MuonAdamW)
    assert optimizer._muon is not None and optimizer._adamw is not None
    assert optimizer._muon.param_groups[0]["weight_decay"] == 0.2
    assert optimizer._adamw.param_groups[0]["weight_decay"] == 0.2


def test_scheduled_learning_rate_reaches_both_updates():
    torch.manual_seed(0)
    module = _tiny_module()
    base_lr = 0.2
    optimizer = create_optimizer(
        module.parameters(),
        {"optimizer": "muon", "learning_rate": base_lr},
    )
    assert isinstance(optimizer, MuonAdamW)

    def schedule(step: int) -> float:
        return 0.25 if step == 0 else 0.5

    scheduler = LambdaLR(optimizer, schedule)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr * 0.25)

    _backward(module)
    optimizer.step()
    _assert_scheduled_lr(optimizer)

    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr * 0.5)
    optimizer.step()
    _assert_scheduled_lr(optimizer)
