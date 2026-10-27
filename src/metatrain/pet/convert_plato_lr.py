"""Turn a pretrained PET checkpoint into a PET with Plato-PET-LR components.

``pet_to_plato_lr(checkpoint, plato_lr)`` builds a PET with the ``plato_lr`` hypers,
copies every weight of the pretrained model (backbone, conditioning, heads,
composition baseline, scaler, metadata, fine-tuning config), sets the LES charge
unit ``sqrt(s)`` from the energy scale and returns a PET checkpoint. The new
components start with zero gates, so the converted model is the pretrained one.
"""

import copy
import math
from typing import Any, Dict

import torch

from .model import PET


def energy_scale(model: PET) -> float:
    """The scaler's scale ``s`` of the first LES (energy) target, 1 without one."""
    scales = model.scaler.model.scales
    for name in model.backend.les_targets:
        if name in scales:
            return float(scales[name].block().values.flatten()[0])
    return 1.0


def set_charge_unit(model: PET) -> None:
    """Store ``sqrt(s)``: latent charges in model units are ``q / sqrt(s)``."""
    if model.backend.les is not None:
        model.backend.les.charge_unit.fill_(math.sqrt(energy_scale(model)))


def load_pet(checkpoint: Dict[str, Any], context: str = "export") -> PET:
    """A PET from a checkpoint of any supported version."""
    checkpoint = PET.upgrade_checkpoint(copy.deepcopy(checkpoint))
    return PET.load_checkpoint(checkpoint, context)  # type: ignore[arg-type]


def convert_model(model: PET, plato_lr: Dict[str, Any]) -> PET:
    """A copy of ``model`` with the ``plato_lr`` components (exact at init)."""
    hypers = copy.deepcopy(dict(model.hypers))
    hypers["plato_lr"] = {**hypers.get("plato_lr", {}), **plato_lr}
    p = next(model.parameters())
    torch.manual_seed(int(hypers["plato_lr"].get("seed", 0)))
    new = PET(hypers, model.dataset_info).to(dtype=p.dtype, device=p.device)
    if model.finetune_config:
        from .modules.finetuning import apply_finetuning_strategy

        new = apply_finetuning_strategy(
            new, model.finetune_config, apply_inherit_heads=False
        )
    old_state = model.state_dict()
    new_state = new.state_dict()
    missing = [k for k in new_state if k not in old_state]
    unexpected = [k for k in old_state if k not in new_state]
    if unexpected:
        raise ValueError(
            f"weights of the pretrained PET not in the new one: {unexpected}"
        )
    new.load_state_dict({**new_state, **old_state})
    new.additive_models[0].sync_tensor_maps()
    new.scaler.sync_tensor_maps()
    new.metadata = model.metadata
    new.finetune_config = copy.deepcopy(model.finetune_config)
    set_charge_unit(new)
    new.plato_lr_new_tensors = missing  # type: ignore[attr-defined]
    return new


def pet_to_plato_lr(
    checkpoint: Dict[str, Any], plato_lr: Dict[str, Any]
) -> Dict[str, Any]:
    """A PET checkpoint with the ``plato_lr`` components added (see module doc).

    :param checkpoint: a PET checkpoint (any supported version).
    :param plato_lr: ``plato_lr`` hypers to set (merged into the defaults).
    :return: the converted checkpoint, loadable with ``PET.load_checkpoint``.
    """
    new = convert_model(load_pet(checkpoint), plato_lr)
    out = new.get_checkpoint()
    for key in ("train_hypers", "trainer_ckpt_version"):
        if key in checkpoint:
            out[key] = copy.deepcopy(checkpoint[key])
    return out
