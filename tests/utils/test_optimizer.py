import copy

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from metatrain.utils.optimizer import Muon, MuonAdamW, MuonRMSNorm, create_optimizer


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


def test_amsgrad_uses_the_second_moment_maximum():
    parameter = nn.Parameter(torch.randn(3))
    optimizer = create_optimizer(
        [parameter], {"optimizer": "amsgrad", "learning_rate": 1e-3}
    )
    assert type(optimizer) is torch.optim.Adam
    assert optimizer.param_groups[0]["amsgrad"] is True

    before = parameter.detach().clone()
    parameter.square().sum().backward()
    optimizer.step()
    assert not torch.equal(before, parameter)
    assert "max_exp_avg_sq" in optimizer.state[parameter]

    decayed = create_optimizer(
        [nn.Parameter(torch.zeros(2))],
        {"optimizer": "amsgrad", "learning_rate": 1e-3, "weight_decay": 0.01},
    )
    assert type(decayed) is torch.optim.AdamW
    assert decayed.param_groups[0]["amsgrad"] is True
    assert decayed.param_groups[0]["weight_decay"] == 0.01


def test_rmsprop_steps():
    torch.manual_seed(0)
    parameter = nn.Parameter(torch.randn(4, 3))
    optimizer = create_optimizer(
        [parameter], {"optimizer": "rmsprop", "learning_rate": 1e-2}
    )
    assert type(optimizer) is torch.optim.RMSprop
    assert optimizer.param_groups[0]["weight_decay"] == 0.0
    assert optimizer.param_groups[0]["lr"] == 1e-2

    before = parameter.detach().clone()
    loss = parameter.square().sum()
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, parameter)

    decayed = create_optimizer(
        [nn.Parameter(torch.zeros(2))],
        {"optimizer": "rmsprop", "learning_rate": 1e-2, "weight_decay": 0.01},
    )
    assert type(decayed) is torch.optim.RMSprop
    assert decayed.param_groups[0]["weight_decay"] == 0.01


def test_muon_rms_differs_from_muon_and_roundtrips():
    torch.manual_seed(0)
    module = _tiny_module()
    other = _tiny_module()
    other.load_state_dict(
        {key: value.clone() for key, value in module.state_dict().items()}
    )
    rms_opt = create_optimizer(
        module.parameters(), {"optimizer": "muon_rms", "learning_rate": 0.05}
    )
    muon_opt = create_optimizer(
        other.parameters(), {"optimizer": "muon", "learning_rate": 0.05}
    )
    assert isinstance(rms_opt, MuonAdamW)
    assert isinstance(rms_opt._muon, MuonRMSNorm)
    assert type(muon_opt._muon) is Muon
    assert id(module.linear.weight) in _param_ids(rms_opt._muon)
    assert id(module.linear.bias) in _param_ids(rms_opt._adamw)

    _backward(module)
    for source, dest in zip(module.parameters(), other.parameters(), strict=True):
        assert source.grad is not None
        dest.grad = source.grad.detach().clone()
    rms_opt.step()
    muon_opt.step()
    assert not torch.allclose(module.linear.weight, other.linear.weight)

    restored = create_optimizer(
        module.parameters(), {"optimizer": "muon_rms", "learning_rate": 0.05}
    )
    restored.load_state_dict(copy.deepcopy(rms_opt.state_dict()))
    assert isinstance(restored, MuonAdamW)
    assert isinstance(restored._muon, MuonRMSNorm)
    original_buf = rms_opt._muon.state[module.linear.weight]["momentum_buffer"]
    restored_buf = restored._muon.state[module.linear.weight]["momentum_buffer"]
    torch.testing.assert_close(restored_buf, original_buf)


def test_unknown_optimizer_name():
    parameter = nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError, match="Unknown optimizer"):
        create_optimizer([parameter], {"optimizer": "sgd", "learning_rate": 1e-3})


@pytest.mark.parametrize("name", ["muon", "muon_rms"])
def test_same_shape_matches_a_separate_step(name: str):
    torch.manual_seed(0)
    shapes = [(4, 16), (4, 16), (16, 4)]
    grads = [torch.randn(*shape) for shape in shapes]
    hypers = {"optimizer": name, "learning_rate": 0.05, "weight_decay": 0.0}

    together = [nn.Parameter(torch.randn(*shape)) for shape in shapes]
    separate = [nn.Parameter(param.detach().clone()) for param in together]
    idle = nn.Parameter(torch.randn(4, 16))
    idle_before = idle.detach().clone()
    for param, grad in zip(together, grads, strict=True):
        param.grad = grad
    for param, grad in zip(separate, grads, strict=True):
        param.grad = grad.clone()

    create_optimizer([*together, idle], hypers).step()
    for param in separate:
        create_optimizer([param], hypers).step()

    for batched, alone in zip(together, separate, strict=True):
        torch.testing.assert_close(batched, alone)
    torch.testing.assert_close(idle, idle_before)


def test_muon_splits_parameters_steps_and_roundtrips_state():
    torch.manual_seed(0)
    module = _tiny_module()
    hypers = {"optimizer": "muon", "learning_rate": 0.05, "weight_decay": None}
    optimizer = create_optimizer(module.parameters(), hypers)

    assert isinstance(optimizer, MuonAdamW)
    assert type(optimizer._muon) is Muon
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


def _role_module() -> nn.Module:
    module = nn.Module()
    module.hidden = nn.Linear(8, 4, bias=False)
    module.node_embedders = nn.Linear(3, 4, bias=False)
    module.node_last_layers = nn.Linear(4, 1, bias=False)
    module.system_conditioning = nn.Linear(4, 4, bias=False)
    module.bias = nn.Parameter(torch.randn(4))
    return module


@pytest.mark.parametrize("name", ["muon", "muon_rms"])
def test_named_parameters_keep_readouts_off_muon(name: str):
    module = _role_module()
    optimizer = create_optimizer(
        module.named_parameters(),
        {"optimizer": name, "learning_rate": 1e-3},
    )
    assert isinstance(optimizer, MuonAdamW)
    assert optimizer._muon is not None and optimizer._adamw is not None
    muon_ids = _param_ids(optimizer._muon)
    adamw_ids = _param_ids(optimizer._adamw)
    assert id(module.hidden.weight) in muon_ids
    for parameter in (
        module.node_embedders.weight,
        module.node_last_layers.weight,
        module.system_conditioning.weight,
        module.bias,
    ):
        assert id(parameter) in adamw_ids


@pytest.mark.parametrize("shape", [(4, 16), (16, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_muon_rms_step_matches_adam_scale(shape: tuple[int, int], dtype: torch.dtype):
    torch.manual_seed(0)
    lr = 0.05
    weight = nn.Parameter(torch.randn(*shape, dtype=dtype))
    optimizer = create_optimizer(
        [weight],
        {"optimizer": "muon_rms", "learning_rate": lr, "weight_decay": 0.0},
    )
    before = weight.detach().clone()
    weight.grad = torch.randn_like(weight)
    optimizer.step()

    assert weight.dtype == dtype
    delta = (weight.detach() - before).float()
    row_rms = delta.square().mean(dim=-1).sqrt()
    expected = torch.full_like(row_rms, 0.2 * lr)
    torch.testing.assert_close(row_rms, expected, rtol=1e-4, atol=1e-6)


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
