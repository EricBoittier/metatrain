"""iris-style PET trunk as ``sr``, with a spherical sidecar for paper LR / BEC.

``PETBackend`` supplies invariant node features (iris ``node_embedding``).
``LoremBackbone`` still builds ``spherical_features`` so
``LoremLongRangeFeaturizer`` and ``BornEffectiveChargeHead`` keep the
equivariant charge / APT path. PET's own ``long_range`` stays off — LOREM
owns ``lr``.
"""

from copy import deepcopy
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import urlretrieve

import torch
from metatomic.torch import NeighborListOptions, System

from metatrain.pet.modules.backend import PETBackend
from metatrain.pet.modules.structures import (
    concatenate_structures as concatenate_pet_structures,
)
from metatrain.utils.architectures import get_default_hypers

from .backbone import LoremBackbone


_HEAD_PREFIXES = (
    "node_heads.",
    "edge_heads.",
    "node_last_layers.",
    "edge_last_layers.",
)


def _is_head_key(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in _HEAD_PREFIXES)


def _resolve_checkpoint_path(path: str) -> str:
    """Resolve a local path, HTTP URL, or Hugging Face URL to a local file."""
    url = urlparse(path)
    if url.scheme and len(url.scheme) > 1:
        if url.netloc == "huggingface.co":
            from metatrain.utils.io import _hf_hub_download_url

            return _hf_hub_download_url(url=path)
        local_path, _ = urlretrieve(url=path)
        return local_path
    return path


def _load_pretrained_pet_backend(
    path: str,
) -> Tuple[dict, List[int], Dict[str, torch.Tensor]]:
    """Load PET ``backend`` weights, dropping prediction heads."""
    resolved = _resolve_checkpoint_path(path)
    checkpoint = torch.load(resolved, weights_only=False, map_location="cpu")
    architecture = checkpoint.get("architecture_name")
    if architecture != "pet":
        raise ValueError(
            "pet.pretrained must point to a PET checkpoint "
            f"(got architecture_name={architecture!r} from {path!r})."
        )
    from metatrain.pet import PET

    pet_model = PET.load_checkpoint(checkpoint, context="finetune")
    backend_state = {
        key[len("backend.") :]: value
        for key, value in pet_model.state_dict().items()
        if key.startswith("backend.") and not _is_head_key(key[len("backend.") :])
    }
    return deepcopy(dict(pet_model.hypers)), list(pet_model.atomic_types), backend_state


class PetTrunk(torch.nn.Module):
    """PET node features + LOREM spherical density sidecar."""

    def __init__(
        self,
        hypers: dict,
        atomic_types: List[int],
        neighbor_list_options: NeighborListOptions,
    ) -> None:
        super().__init__()
        self.neighbor_list_options = neighbor_list_options
        self.num_features = int(hypers["num_features"])
        self.sidecar = LoremBackbone(hypers, atomic_types, neighbor_list_options)

        user_pet = dict(hypers["pet"]) if "pet" in hypers else {}
        pretrained_path = user_pet.pop("pretrained", None)
        pretrained_state: Optional[Dict[str, torch.Tensor]] = None
        pet_types = list(atomic_types)

        if pretrained_path:
            pet_hypers, pet_types, pretrained_state = _load_pretrained_pet_backend(
                str(pretrained_path)
            )
            missing_types = [at for at in atomic_types if at not in pet_types]
            if missing_types:
                raise ValueError(
                    "The pretrained PET checkpoint does not support atomic "
                    f"types {missing_types} required by this dataset."
                )
        else:
            pet_hypers = deepcopy(get_default_hypers("pet")["model"])
            pet_hypers.update(user_pet)
            pet_hypers["d_node"] = self.num_features

        pet_hypers["cutoff"] = float(hypers["cutoff"])
        pet_hypers["cutoff_width"] = float(hypers["cutoff_width"])
        pet_hypers["system_conditioning"] = False
        long_range = dict(pet_hypers["long_range"])
        long_range["enable"] = False
        pet_hypers["long_range"] = long_range
        self.cutoff_width_adaptive = float(pet_hypers["cutoff_width_adaptive"])
        self.backend = PETBackend(pet_hypers, pet_types)
        if pretrained_state is not None:
            missing, unexpected = self.backend.load_state_dict(
                pretrained_state, strict=False
            )
            unexpected_core = [key for key in unexpected if not _is_head_key(key)]
            missing_core = [key for key in missing if not _is_head_key(key)]
            if unexpected_core:
                raise ValueError(
                    "Unexpected PET backbone tensors when loading pretrained "
                    f"weights: {unexpected_core}"
                )
            if missing_core:
                raise ValueError(
                    "Missing PET backbone tensors when loading pretrained "
                    f"weights: {missing_core}"
                )

        d_node = int(pet_hypers["d_node"])
        if d_node == self.num_features:
            self.feature_proj: torch.nn.Module = torch.nn.Identity()
        else:
            self.feature_proj = torch.nn.Linear(d_node, self.num_features)

    def forward(
        self, systems: List[System]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return PET node features, sidecar distances, and spherical features."""
        _, distances, spherical_features = self.sidecar(systems)
        (
            positions,
            centers,
            neighbors,
            species,
            cells,
            cell_shifts,
            system_indices,
            _,
        ) = concatenate_pet_structures(systems, self.neighbor_list_options)
        batch_data = self.backend.preprocess(
            positions,
            centers,
            neighbors,
            species,
            cells,
            cell_shifts,
            system_indices,
            self.cutoff_width_adaptive,
        )
        node_features_list, _ = self.backend.calculate_features(batch_data, False)
        features = self.feature_proj(node_features_list[len(node_features_list) - 1])
        return features, distances, spherical_features
