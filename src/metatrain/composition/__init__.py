import logging
from typing import List, Union

import torch
from torch import nn

from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import Dataset
from metatrain.utils.data.dataset import Subset
from metatrain.utils.fit_cache import (
    copy_cached_fit,
    get_fit_cache_path,
    is_cached,
    preserve_rng,
    store_fit,
)
from metatrain.utils.io import load_model

from ._base_composition import FixedCompositionWeights
from .model import CompositionModel
from .trainer import Trainer


__model__ = CompositionModel
__trainer__ = Trainer

__authors__ = [
    ("Paolo Pegolo <paolo.pegolo@epfl.ch>", "@ppegolo"),
]

__maintainers__ = [
    ("Paolo Pegolo <paolo.pegolo@epfl.ch>", "@ppegolo"),
]


def train_or_load_composition_model(
    composition_model: CompositionModel,
    atomic_baseline: FixedCompositionWeights | str,
    train_datasets: List[Union[Dataset, Subset]],
    other_additive_models: List[nn.Module],
    batch_size: int,
    is_distributed: bool,
    checkpoint_dir: str = "",
) -> None:
    """
    Train the composition model from data or load pre-trained weights.

    This is the single source of truth for how to set up a composition model
    for use as an additive baseline by any architecture.

    :param composition_model: The composition model to train or load into
    :param atomic_baseline: Fixed weights dict, or path to a checkpoint
    :param train_datasets: Training datasets
    :param other_additive_models: Other additive models (e.g. ZBL) to
        subtract before fitting
    :param batch_size: Batch size for data loading
    :param is_distributed: Whether training is distributed
    :param checkpoint_dir: Directory to save the composition model checkpoint
    """
    cache = None
    if isinstance(atomic_baseline, dict):
        # The model is hashed before fitting, so that e.g. the weights it
        # inherited for other targets when fine-tuning are part of the key.
        cache = get_fit_cache_path(
            "composition",
            [
                composition_model,
                other_additive_models,
                atomic_baseline,
                train_datasets,
            ],
        )
        if is_cached(cache, composition_model.dummy_buffer.device, is_distributed):
            assert cache is not None
            copy_cached_fit(
                cache, checkpoint_dir, "composition_model.ckpt", is_distributed
            )
            atomic_baseline = str(cache)

    if isinstance(atomic_baseline, str):
        logging.info(f"Loading composition model from {atomic_baseline}")
        # Building the loaded model draws from the global random state.
        with preserve_rng(cache):
            loaded = load_model(atomic_baseline)
        if not isinstance(loaded, CompositionModel):
            raise ValueError(
                f"The model loaded from {atomic_baseline} is a "
                f"{type(loaded).__name__}, not a composition model."
            )
        if loaded.atomic_types != composition_model.atomic_types:
            raise ValueError(
                "Composition checkpoint atomic types "
                f"({loaded.atomic_types}) do not match the current model's "
                f"atomic types ({composition_model.atomic_types})."
            )
        loaded_targets = loaded.dataset_info.targets
        current_targets = composition_model.dataset_info.targets
        if set(loaded_targets) != set(current_targets):
            raise ValueError(
                "Composition checkpoint targets "
                f"({sorted(loaded_targets)}) do not match the current model's "
                f"targets ({sorted(current_targets)})."
            )
        for name, target_info in current_targets.items():
            loaded_info = loaded_targets[name]
            if (loaded_info.quantity, loaded_info.unit) != (
                target_info.quantity,
                target_info.unit,
            ):
                raise ValueError(
                    f"Target '{name}' from the composition checkpoint has "
                    f"quantity '{loaded_info.quantity}' and unit "
                    f"'{loaded_info.unit}', while the current model expects "
                    f"quantity '{target_info.quantity}' and unit "
                    f"'{target_info.unit}'."
                )
        composition_model.load_state_dict(loaded.state_dict())
        composition_model.sync_tensor_maps()
    else:
        assert isinstance(atomic_baseline, dict)
        logging.info("Calculating composition weights")
        # The trainer expects complete hypers: start from the defaults and
        # override the entries controlled by the parent architecture.
        hypers = get_default_hypers("composition")["training"]
        hypers["atomic_baseline"] = atomic_baseline
        hypers["batch_size"] = batch_size
        hypers["distributed"] = is_distributed
        trainer = Trainer(hypers=hypers)
        trainer._additive_models = other_additive_models
        # The trainer fits on devices[0]; pass the model's current device so
        # embedded training stays where the parent architecture put the model.
        with preserve_rng(cache):
            trainer.train(
                model=composition_model,
                dtype=torch.float64,
                devices=[composition_model.dummy_buffer.device],
                train_datasets=train_datasets,
                val_datasets=train_datasets,
                checkpoint_dir=checkpoint_dir,
            )
        store_fit(
            cache,
            lambda path: trainer.save_checkpoint(composition_model, path),
            is_distributed,
        )
