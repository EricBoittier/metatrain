"""Benchmark a real FlashMD training step, stage by stage.

FlashMD (:py:mod:`metatrain.experimental.flashmd`) predicts positions and
momenta after a time interval, not energy from structure — a different
architecture from PET's, built on the same :py:mod:`metatrain.utils.data`
loading/collate machinery this harness's other benchmarks exercise. None
of the perf/data-loading branches this harness compares touch FlashMD's
code at all (confirmed by diff against the stacked PRs), so running this
across the six variants is not a "does the optimization help FlashMD"
comparison the way ``benchmark_pipeline.py`` is for PET — it's a
baseline/regression snapshot for FlashMD's own performance on this
hardware, and the scaffolding this harness would need if those
optimizations are ever backported to FlashMD's trainer.

Uses the same small trajectory dataset metatrain's own FlashMD test suite
does (``experimental/flashmd/tests/data/flashmd.xyz``, 10 32-atom frames);
there is no larger FlashMD-labeled dataset in this repo, so epoch count is
the knob for getting a meaningful number of timed steps out of it, not
dataset size.

Examples::

    python benchmarks/benchmark_flashmd.py --num-workers 0 --epochs 20
"""

from __future__ import annotations

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

from metatrain.experimental.flashmd import FlashMD, Trainer
from metatrain.utils import timing
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import Dataset, DatasetInfo, get_atomic_types
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.logging import MetricLogger
from metatrain.utils.loss import LossSpecification


DEFAULT_DATASET = (
    Path(__file__).parents[1]
    / "src/metatrain/experimental/flashmd/tests/data/flashmd.xyz"
).as_posix()

DEFAULT_SEED = 0
DEFAULT_TIMESTEP = 30.0


def build_dataset(path: str) -> Tuple[Dataset, Dict[str, Any]]:
    """Read a FlashMD trajectory file into a position+momentum dataset.

    :param path: Path to the structure file, with ``future_positions`` and
        ``future_momenta`` per-atom arrays (see the FlashMD test fixture).
    :return: The dataset and its target information.
    """
    conf = {
        "position": {
            "quantity": "position",
            "read_from": path,
            "reader": "ase",
            "key": "future_positions",
            "unit": "A",
            "type": {"cartesian": {"rank": 1}},
            "sample_kind": "atom",
            "num_subtargets": 1,
        },
        "momentum": {
            "quantity": "momentum",
            "read_from": path,
            "reader": "ase",
            "key": "future_momenta",
            "unit": "(eV*u)^(1/2)",
            "type": {"cartesian": {"rank": 1}},
            "sample_kind": "atom",
            "num_subtargets": 1,
        },
    }
    targets, target_info = read_targets(OmegaConf.create(conf))
    systems = read_systems(path)
    dataset = Dataset.from_dict(
        {"system": systems, "position": targets["position"], "momentum": targets["momentum"]}
    )
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
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--timestep", type=float, default=DEFAULT_TIMESTEP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    """Train FlashMD for a few epochs and print the per-stage timing report."""
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    timing.enable()

    dataset, target_info = build_dataset(args.dataset)
    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=get_atomic_types(dataset),
        targets=target_info,
    )

    loss = OmegaConf.create(
        {
            "position": init_with_defaults(LossSpecification),
            "momentum": init_with_defaults(LossSpecification),
        }
    )
    OmegaConf.resolve(loss)

    hypers = get_default_hypers("experimental.flashmd", base_precision=32)
    hypers["training"].update(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        timestep=args.timestep,
        loss=loss,
    )
    model = FlashMD(hypers["model"], dataset_info)

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
        f"\nFlashMD, {len(train_dataset)} train + {len(val_dataset)} validation "
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
