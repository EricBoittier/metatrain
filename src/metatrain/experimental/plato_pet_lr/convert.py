"""Model surgery on trained Plato-PET-LR models (outside TorchScript).

- ``energy_scale``: the scaler's energy scale ``s``; latent charges in ``e`` are
  ``sqrt(s) q_m``.
- ``set_charge_unit``: store ``sqrt(s)`` in the electronegativity module.
- ``flip_charge_sign``: ``q -> -q`` (negates the last layer of the charge head). The
  LES energy is even in ``q``: without electronegativity, energies and forces are
  unchanged.
- ``with_electronegativity``: a copy with ``electronegativity`` set (``chi0`` from
  the QEq table, ``dchi = 0``) and every other weight carried over; it adds exactly
  ``sum_i chi0[Z_i] q_i`` to the energy.
"""

import copy
import math

import torch

from .model import PlatoPETLR


def energy_scale(model: PlatoPETLR) -> float:
    scales = model.scaler.model.scales
    name = next(k for k in model.layouts if k in scales)
    return float(scales[name].block().values.flatten()[0])


def set_charge_unit(model: PlatoPETLR) -> None:
    for chi in model.electronegativity:
        chi.charge_unit.fill_(math.sqrt(energy_scale(model)))


@torch.no_grad()
def flip_charge_sign(model: PlatoPETLR) -> PlatoPETLR:
    model = copy.deepcopy(model)
    last = model.charge_head.l2
    last.W.neg_()
    if last.b is not None:
        last.b.neg_()
    return model


def with_electronegativity(model: PlatoPETLR, mode: str = "linear") -> PlatoPETLR:
    hypers = dict(model.hypers, electronegativity=mode)
    new = PlatoPETLR(hypers, model.dataset_info)
    dtype = next(model.parameters()).dtype
    new.to(dtype=dtype, device=next(model.parameters()).device)
    missing, unexpected = new.load_state_dict(model.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("electronegativity.") for k in missing)
    # the fitted scales and baselines live in TensorMaps next to the state dict
    new.additive_models = copy.deepcopy(model.additive_models)
    new.scaler = copy.deepcopy(model.scaler)
    new.metadata = model.metadata
    set_charge_unit(new)
    return new
