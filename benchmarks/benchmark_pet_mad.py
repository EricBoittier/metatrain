"""Benchmark fine-tuning a pretrained PET-MAD checkpoint, stage by stage.

Same per-stage timing report as ``benchmark_pipeline.py``
(:py:mod:`metatrain.utils.timing`), but the model starts from a pretrained
PET-MAD checkpoint (`lab-cosmo/upet` on HuggingFace) instead of a small,
freshly-initialized PET, and is loaded via the real finetune path
(:py:func:`metatrain.utils.io.model_from_checkpoint` + ``model.restart``,
the same one `mtt train` uses for `training.finetune.read_from`). PET-MAD
is a much larger, richer model (102 atomic types, energy + non-conservative
force + stress heads) than the small energy-only model
``benchmark_pipeline.py`` trains from scratch, so this checks whether the
same data-loading changes help or hurt at that different point in model
size / per-batch compute.

The checkpoint is a local path, not a URL — download it once (e.g. with
``curl -L <HF resolve URL> -o pet-mad-xs-v1.6.0.ckpt``) and pass that path.

Examples::

    python benchmarks/benchmark_pet_mad.py \
        --checkpoint pet-mad-xs-v1.6.0.ckpt --num-workers 0
"""

from __future__ import annotations

import argparse
import random
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from metatrain.pet import Trainer
from metatrain.utils import timing
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import Dataset, DatasetInfo, get_atomic_types
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.io import model_from_checkpoint
from metatrain.utils.logging import MetricLogger
from metatrain.utils.loss import LossSpecification


DEFAULT_DATASET = (
    Path(__file__).parents[1] / "tests/resources/qm9_reduced_100.xyz"
).as_posix()

DEFAULT_SEED = 0


def build_dataset(path: str, key: str) -> Tuple[Dataset, Dict[str, Any]]:
    """Read an ASE-readable file into a single-energy-target dataset.

    :param path: Path to the structure file.
    :param key: Name of the per-structure energy key in that file.
    :return: The dataset and its target information.
    """
    conf = {
        "energy": {
            "quantity": "energy",
            "read_from": path,
            "reader": "ase",
            "key": key,
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, target_info = read_targets(OmegaConf.create(conf))
    systems = read_systems(path)
    dataset = Dataset.from_dict({"system": systems, "energy": targets["energy"]})
    return dataset, target_info


@contextmanager
def capture_epoch_metrics() -> Iterator[List[Dict[str, Any]]]:
    """Record every ``MetricLogger.log`` call made inside this block.

    See ``benchmark_checkpoint_resume.py`` for why this patches the class
    method rather than reading a log line back out.

    :yield: A list this fills in place, one entry per ``log`` call, in the
        order ``Trainer.train`` made them (so ``[0]`` is epoch 1).
    """
    records: List[Dict[str, Any]] = []
    original = MetricLogger.log

    def _capturing_log(
        self: MetricLogger,
        metrics: Any,
        epoch: Any = None,
        rank: Any = None,
        learning_rate: Any = None,
    ) -> None:
        entries = [metrics] if isinstance(metrics, dict) else metrics
        record: Dict[str, Any] = {"epoch": epoch}
        for name, values in zip(self.names, entries):
            record[f"{name}_loss"] = values.get("loss")
        records.append(record)
        return original(
            self, metrics, epoch=epoch, rank=rank, learning_rate=learning_rate
        )

    MetricLogger.log = _capturing_log
    try:
        yield records
    finally:
        MetricLogger.log = original


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments.

    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", required=True, help="local path to a PET-MAD .ckpt"
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--key", default="U0", help="energy key in the dataset")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    """Fine-tune PET-MAD for a few epochs and print the per-stage timing report."""
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    timing.enable()

    dataset, target_info = build_dataset(args.dataset, args.key)
    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=get_atomic_types(dataset),
        targets=target_info,
    )

    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    with warnings.catch_warnings():
        # the checkpoint's own output units, unrelated to our target config
        warnings.simplefilter("ignore", category=UserWarning)
        model = model_from_checkpoint(checkpoint, context="finetune")
    model = model.restart(dataset_info, model_hypers=None)

    loss = OmegaConf.create({"energy": init_with_defaults(LossSpecification)})
    OmegaConf.resolve(loss)
    hypers = get_default_hypers("pet", base_precision=32)
    hypers["training"].update(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        loss=loss,
    )

    # a disjoint split: the validation loop is not part of the timing report,
    # but it is part of an epoch, and it gets its own loader and workers
    val_size = max(1, round(args.val_fraction * len(dataset)))
    split = len(dataset) - val_size
    train_dataset = torch.utils.data.Subset(dataset, range(split))
    val_dataset = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    trainer = Trainer(hypers["training"])
    with TemporaryDirectory() as checkpoint_dir, capture_epoch_metrics() as epoch_metrics:
        start = time.perf_counter()
        trainer.train(
            model=model,
            dtype=torch.float32,
            devices=[torch.device(args.device)],
            train_datasets=[train_dataset],
            val_datasets=[val_dataset],
            checkpoint_dir=checkpoint_dir,
        )
        wall = time.perf_counter() - start

    print(
        f"\nPET, {len(train_dataset)} train + {len(val_dataset)} validation "
        f"structures, batch_size={args.batch_size}, "
        f"num_workers={args.num_workers}, device={args.device}, "
        f"epochs={args.epochs}, {wall:.1f} s wall (incl. validation)\n"
    )
    print(
        f"best_val_metric {trainer.best_metric:.6f} "
        f"({hypers['training']['best_model_metric']}) at epoch {trainer.best_epoch}"
    )
    if epoch_metrics:
        first = epoch_metrics[0]
        print(
            f"epoch1_metrics train_loss={first['training_loss']:.6f} "
            f"val_loss={first['validation_loss']:.6f}"
        )
    print(timing.report())


if __name__ == "__main__":
    main()
