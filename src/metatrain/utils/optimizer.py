"""Training optimizers.

``adam`` keeps the historical Adam / AdamW choice. ``muon`` updates matrix
parameters with Muon and every other parameter with AdamW, as one optimizer
so the existing learning-rate scheduler and checkpoints still apply.
"""

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Optional

import torch
from torch import Tensor


_MUON_MOMENTUM = 0.95
_MUON_ADJUST_LR = "match_rms_adamw"


class MuonAdamW(torch.optim.Optimizer):
    """Muon for 2D parameters and AdamW for every other rank.

    PyTorch's Muon optimizer only accepts matrices. Biases, gains, and
    higher-rank tensors are updated with AdamW at the same learning rate.
    Parameter groups are tagged with ``role`` (``"muon"`` or ``"adamw"``).
    :meth:`step` copies each group's learning rate onto the matching inner
    optimizer, so a scheduler attached to this object covers both updates.

    :param params: Parameters to optimize. Matrices (``ndim == 2``) use Muon;
        every other rank uses AdamW.
    :param lr: Learning rate applied to both groups before scheduling.
    :param weight_decay: Decoupled weight decay applied to both groups.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float,
        weight_decay: float,
    ) -> None:
        param_list = list(params)
        muon_params = [param for param in param_list if param.ndim == 2]
        adamw_params = [param for param in param_list if param.ndim != 2]
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
    parameters: Iterable[Tensor],
    hypers: Mapping[str, Any],
) -> torch.optim.Optimizer:
    """Create the optimizer selected by the training hyperparameters.

    ``adam`` (the default) matches the historical behavior: Adam, or AdamW
    when ``weight_decay`` is set. ``muon`` uses :class:`MuonAdamW`. A missing
    ``weight_decay`` means no decay; for Muon that is ``0.0`` rather than
    PyTorch's own Muon default of ``0.1``.

    :param parameters: Model parameters, as from ``module.parameters()``.
    :param hypers: Training hyperparameters. Reads ``optimizer``,
        ``learning_rate``, and optional ``weight_decay``.
    :return: The optimizer to step during training.
    """
    name = hypers.get("optimizer", "adam")
    lr = float(hypers["learning_rate"])
    decay = hypers.get("weight_decay")
    weight_decay = None if decay is None else float(decay)

    if name == "adam":
        if weight_decay is not None:
            return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
        return torch.optim.Adam(parameters, lr=lr)

    if name == "muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError(
                "optimizer 'muon' requires PyTorch >= 2.9, "
                "which provides torch.optim.Muon"
            )
        return MuonAdamW(
            parameters,
            lr=lr,
            weight_decay=0.0 if weight_decay is None else weight_decay,
        )

    raise ValueError(f"Unknown optimizer {name!r}. Expected 'adam' or 'muon'.")
