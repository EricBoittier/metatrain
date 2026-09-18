"""Benchmark a real PET training step, stage by stage.

Runs ``Trainer.train`` on a small dataset with :py:mod:`metatrain.utils.timing`
enabled and prints where the time goes: waiting for the input pipeline
(``loader``) versus using the batch (``step``, broken down into unpack, host to
device transfer, forward, loss, backward and optimizer).

The collate stages (``group_and_join``, ``transforms``, ``serialize``) are
recorded in whichever process runs the collate function, so they are only
reported with ``--num-workers 0``. With workers they are part of the
``loader`` wait instead, which is exactly the number to compare against.
Their call count is a few higher than the number of steps, because the
(untimed) validation loop collates through the same code.

Examples::

    python benchmarks/benchmark_pipeline.py --num-workers 0
    python benchmarks/benchmark_pipeline.py --num-workers 4 --batch-size 16
"""

import argparse
import random
import time
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from metatrain.pet import PET, Trainer
from metatrain.utils import timing
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import Dataset, DatasetInfo, get_atomic_types
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.logging import MetricLogger
from metatrain.utils.loss import LossSpecification


DEFAULT_DATASET = (
    Path(__file__).parents[1] / "tests/resources/qm9_reduced_100.xyz"
).as_posix()

# Default only: pipeline-bench sweeps --seed across a few fixed values so a
# cross-variant agreement isn't a coincidence of one particular seed.
DEFAULT_SEED = 0


@contextmanager
def capture_epoch_metrics() -> Iterator[List[Dict[str, Any]]]:
    """Record every ``MetricLogger.log`` call made inside this block.

    ``Trainer.train`` builds its own ``MetricLogger`` internally, so there is
    no instance to attach a listener to from the outside. Patching the class
    method is the only interception point that sees the *raw* metrics dict
    (train/val loss, before formatting to text) rather than a log line that
    would need re-parsing.

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


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments.

    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
    """Train PET for a few epochs and print the per-stage timing report."""
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

    # the per-target loss config is normally expanded from the `loss: mse`
    # shorthand by the yaml layer, which we bypass here
    loss = OmegaConf.create({"energy": init_with_defaults(LossSpecification)})
    OmegaConf.resolve(loss)

    hypers = get_default_hypers("pet", base_precision=32)
    hypers["training"].update(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        loss=loss,
    )
    model = PET(hypers["model"], dataset_info)

    # a disjoint split: the validation loop is not part of the timing report,
    # but it is part of an epoch, and it gets its own loader and workers.
    # Same val_fraction default as the later branches in this stack, so
    # n_train/n_val (and best_val_metric) are comparable across variants.
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
