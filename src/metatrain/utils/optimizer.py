"""Training optimizers.

``adam`` keeps the historical Adam / AdamW choice. ``muon`` updates hidden
matrix weights with Muon, and embeddings, readouts, conditioning parameters,
and every non-matrix tensor with AdamW, as one optimizer so the existing
learning-rate scheduler and checkpoints still apply. ``rmsprop`` is RMSprop,
the root-mean-square gradient method. ``muon_rms`` is that same split with an
RMS-norm on the orthogonalized update. ``amsgrad`` is Adam with the AMSGrad
second-moment maximum.
"""

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal, Optional, get_args

import torch
from torch import Tensor
from torch.nn import functional as F


OptimizerName = Literal["adam", "amsgrad", "muon", "muon_rms", "rmsprop"]
OPTIMIZER_NAMES: tuple[OptimizerName, ...] = get_args(OptimizerName)

_MUON_MOMENTUM = 0.95
_MUON_ADJUST_LR = "match_rms_adamw"
_MUON_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
_MUON_NS_STEPS = 5
_MUON_NS_EPS = 1e-7
# Once each row has unit RMS, 0.2 puts the update on an Adam-like scale.
_MUON_RMS_LR_SCALE = 0.2
_MUON_RMS_EPS = 1e-8
# Hidden matrices use Muon. These substrings stay on AdamW: embedding tables
# (``embedd``), readout heads (``last_layer``), and the conditioning module,
# whose output gate sets a residual scale.
_ADAMW_NAME_MARKERS = ("embedd", "last_layer", "system_conditioning")


def _orthogonalize(update: Tensor) -> Tensor:
    # Quintic Newton-Schulz iteration from Keller Jordan's Muon. Tall matrices
    # are transposed so the Gram matrix is the smaller side, then transposed
    # back. bf16 is used on CUDA only; CPU stays in the update dtype.
    a, b, c = _MUON_NS_COEFFICIENTS
    ortho = update.bfloat16() if update.is_cuda else update
    transposed = ortho.size(0) > ortho.size(1)
    if transposed:
        ortho = ortho.T
    ortho = ortho / ortho.norm().clamp(min=_MUON_NS_EPS)
    for _ in range(_MUON_NS_STEPS):
        gram = ortho @ ortho.T
        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        ortho = torch.addmm(ortho, gram_update, ortho, beta=a)
    if transposed:
        ortho = ortho.T
    return ortho


def _as_float(value: float | Tensor) -> float:
    if isinstance(value, Tensor):
        return float(value.item())
    return float(value)


class MuonRMSNorm(torch.optim.Optimizer):
    """Muon whose orthogonalized update is RMS-normalized over its last axis.

    This follows the variant in https://github.com/xTimeCrystal/Muon-RMS-Norm.
    For an ``nn.Linear`` weight of shape ``(out, in)`` the last axis is the
    input dimension. After the norm, the update is scaled by ``0.2`` instead
    of Muon's shape-based RMS matching. Only 2D parameters are accepted.

    :param params: Matrix parameters to update.
    :param lr: Learning rate before the ``0.2`` scale.
    :param weight_decay: Decoupled weight decay.
    :param momentum: Momentum coefficient.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float,
        weight_decay: float,
        momentum: float = _MUON_MOMENTUM,
    ) -> None:
        param_list = list(params)
        if len(param_list) == 0:
            raise ValueError("optimizer got an empty parameter list")
        for param in param_list:
            if param.ndim != 2:
                raise ValueError(
                    "MuonRMSNorm only supports 2D parameters, got "
                    f"shape {tuple(param.shape)}"
                )
        super().__init__(
            param_list,
            {
                "lr": lr,
                "weight_decay": weight_decay,
                "momentum": momentum,
                "nesterov": True,
            },
        )

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Orthogonalize, RMS-normalize, and apply the Muon update.

        :param closure: Optional closure that reevaluates the model and
            returns the loss.
        :return: The loss returned by ``closure``, if it was provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = _as_float(group["lr"])
            weight_decay = float(group["weight_decay"])
            momentum = float(group["momentum"])
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param)
                buf = state["momentum_buffer"]
                buf.lerp_(grad, 1 - momentum)
                update = grad.lerp(buf, momentum) if group["nesterov"] else buf
                update = _orthogonalize(update)
                update = F.rms_norm(
                    update.float(), (update.shape[-1],), eps=_MUON_RMS_EPS
                )
                param.mul_(1 - lr * weight_decay)
                param.add_(update.to(dtype=param.dtype), alpha=-_MUON_RMS_LR_SCALE * lr)
        return loss


class MuonAdamW(torch.optim.Optimizer):
    """Muon for hidden matrices and AdamW for everything else.

    Hidden 2D weights use Muon. Embeddings, readout heads, conditioning
    parameters, and every non-matrix tensor use AdamW at the same learning
    rate. Name checks look for ``embedd``, ``last_layer``, and
    ``system_conditioning``; unnamed tensors fall back to the rank rule.
    Parameter groups are tagged with ``role`` (``"muon"`` or ``"adamw"``).
    :meth:`step` copies each group's learning rate onto the matching inner
    optimizer, so a scheduler attached to this object covers both updates.

    :param params: Parameters to optimize, or ``(name, parameter)`` pairs
        from ``module.named_parameters()``.
    :param lr: Learning rate applied to both groups before scheduling.
    :param weight_decay: Decoupled weight decay applied to both groups.
    :param rms_norm: When true, matrix updates use :class:`MuonRMSNorm`
        instead of :class:`torch.optim.Muon`.
    """

    def __init__(
        self,
        params: Iterable[Tensor | tuple[str, Tensor]],
        lr: float,
        weight_decay: float,
        rms_norm: bool = False,
    ) -> None:
        named = _as_named_parameters(params)
        muon_params = [
            param for name, param in named if _is_muon_parameter(name, param)
        ]
        adamw_params = [
            param for name, param in named if not _is_muon_parameter(name, param)
        ]
        if len(muon_params) == 0 and len(adamw_params) == 0:
            raise ValueError("optimizer got an empty parameter list")

        groups: list[dict[str, Any]] = []
        if len(muon_params) > 0:
            groups.append({"params": muon_params, "lr": lr, "role": "muon"})
        if len(adamw_params) > 0:
            groups.append({"params": adamw_params, "lr": lr, "role": "adamw"})
        super().__init__(groups, {"lr": lr})

        self._muon: Optional[torch.optim.Optimizer] = None
        self._adamw: Optional[torch.optim.Optimizer] = None
        if len(muon_params) > 0:
            if rms_norm:
                self._muon = MuonRMSNorm(
                    muon_params,
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=_MUON_MOMENTUM,
                )
            else:
                muon_cls = torch.optim.Muon
                self._muon = muon_cls(
                    muon_params,
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=_MUON_MOMENTUM,
                    nesterov=True,
                    adjust_lr_fn=_MUON_ADJUST_LR,
                )
        if len(adamw_params) > 0:
            self._adamw = torch.optim.AdamW(
                adamw_params,
                lr=lr,
                weight_decay=weight_decay,
            )

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform one Muon step and one AdamW step.

        :param closure: Optional closure that reevaluates the model and
            returns the loss. It is called once, before either update.
        :return: The loss returned by ``closure``, if it was provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            inner = self._muon if group["role"] == "muon" else self._adamw
            if inner is not None:
                inner.param_groups[0]["lr"] = group["lr"]
        if self._muon is not None:
            self._muon.step()
        if self._adamw is not None:
            self._adamw.step()
        return loss

    def state_dict(self) -> dict[str, Any]:
        """Return the Muon and AdamW states for checkpointing.

        :return: A dict with ``muon`` and ``adamw`` entries. Each entry is the
            inner optimizer state dict, or ``None`` when that group is empty.
        """
        return {
            "muon": None if self._muon is None else self._muon.state_dict(),
            "adamw": None if self._adamw is None else self._adamw.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load states produced by :meth:`state_dict`.

        :param state_dict: Checkpoint state for this optimizer.
        """
        if "muon" not in state_dict or "adamw" not in state_dict:
            raise ValueError(
                "MuonAdamW checkpoint is missing optimizer state. "
                "This state was not saved with optimizer: muon."
            )
        _restore_inner(self._muon, state_dict["muon"], "muon")
        _restore_inner(self._adamw, state_dict["adamw"], "adamw")


def _as_named_parameters(
    parameters: Iterable[Tensor | tuple[str, Tensor]],
) -> list[tuple[Optional[str], Tensor]]:
    named: list[tuple[Optional[str], Tensor]] = []
    for item in parameters:
        if isinstance(item, Tensor):
            named.append((None, item))
        else:
            name, param = item
            named.append((name, param))
    return named


def _is_muon_parameter(name: Optional[str], param: Tensor) -> bool:
    # Unnamed 2D tensors are treated as hidden weights. Named embeddings,
    # readouts, and conditioning parameters stay with AdamW.
    if param.ndim != 2:
        return False
    if name is None:
        return True
    return not any(marker in name for marker in _ADAMW_NAME_MARKERS)


def _restore_inner(
    inner: Optional[torch.optim.Optimizer],
    saved: Any,
    name: str,
) -> None:
    if inner is None and saved is None:
        return
    if inner is None or saved is None:
        raise ValueError(
            f"MuonAdamW {name} state does not match the current parameters."
        )
    inner.load_state_dict(saved)


def create_optimizer(
    parameters: Iterable[Tensor | tuple[str, Tensor]],
    hypers: Mapping[str, Any],
) -> torch.optim.Optimizer:
    """Create the optimizer selected by the training hyperparameters.

    ``adam`` (the default) matches the historical behavior: Adam, or AdamW
    when ``weight_decay`` is set. ``muon`` and ``muon_rms`` use
    :class:`MuonAdamW`. ``rmsprop`` uses :class:`torch.optim.RMSprop`.
    ``amsgrad`` uses Adam with ``amsgrad=True``, or AdamW when
    ``weight_decay`` is set. A missing ``weight_decay`` means no decay; for
    Muon, Muon RMS-norm, and RMSprop that is ``0.0`` rather than PyTorch's
    own Muon default of ``0.1``.

    :param parameters: Model parameters, or ``(name, parameter)`` pairs from
        ``module.named_parameters()``. Names are used to keep embeddings,
        readouts, and conditioning off Muon.
    :param hypers: Training hyperparameters. Reads ``optimizer``,
        ``learning_rate``, and optional ``weight_decay``.
    :return: The optimizer to step during training.
    """
    name = hypers.get("optimizer", "adam")
    lr = float(hypers["learning_rate"])
    decay = hypers.get("weight_decay")
    weight_decay = None if decay is None else float(decay)
    named = _as_named_parameters(parameters)
    tensors = [param for _, param in named]

    if name == "adam":
        if weight_decay is not None:
            return torch.optim.AdamW(tensors, lr=lr, weight_decay=weight_decay)
        return torch.optim.Adam(tensors, lr=lr)

    if name in ("muon", "muon_rms"):
        if name == "muon" and not hasattr(torch.optim, "Muon"):
            raise RuntimeError(
                "optimizer 'muon' requires PyTorch >= 2.9, "
                "which provides torch.optim.Muon"
            )
        return MuonAdamW(
            named,
            lr=lr,
            weight_decay=0.0 if weight_decay is None else weight_decay,
            rms_norm=name == "muon_rms",
        )

    if name == "rmsprop":
        return torch.optim.RMSprop(
            tensors,
            lr=lr,
            weight_decay=0.0 if weight_decay is None else weight_decay,
        )

    if name == "amsgrad":
        if weight_decay is not None:
            return torch.optim.AdamW(
                tensors, lr=lr, weight_decay=weight_decay, amsgrad=True
            )
        return torch.optim.Adam(tensors, lr=lr, amsgrad=True)

    expected = ", ".join(repr(allowed) for allowed in OPTIMIZER_NAMES)
    raise ValueError(f"Unknown optimizer {name!r}. Expected one of: {expected}.")
