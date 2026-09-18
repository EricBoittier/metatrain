"""Benchmark & correctness-check checkpoint save/resume.

Trains two ways from the same seed and compares them:

- **continuous**: one ``Trainer.train`` call, straight through all
  ``--epochs`` epochs.
- **resumed**: train through ``--split-epoch``, ``save_checkpoint``, then
  reload via the *exact* path ``mtt train --restart`` uses in production
  (``model_from_checkpoint`` / ``model.restart`` / ``trainer_from_checkpoint``
  from :py:mod:`metatrain.utils.io`, not a hand-rolled shortcut) and continue
  to ``--epochs``.

The point is regression coverage for the stacked data-loading branches this
harness benchmarks: does resume still work at all with persistent workers,
pinned memory, or the batch-transport format in the picture, and does the
resumed run's `best_val_metric` land close to the continuous run's (some
drift is expected — RNG state isn't part of the checkpoint, so the resumed
run's data order/augmentation after the split epoch is not the one the
continuous run would have drawn).

Examples::

    python benchmarks/benchmark_checkpoint_resume.py --epochs 6 --split-epoch 3
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from metatrain.pet import PET, Trainer
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import Dataset, DatasetInfo, get_atomic_types
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.io import model_from_checkpoint, trainer_from_checkpoint
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


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments.

    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--key", default="U0", help="energy key in the dataset")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument(
        "--split-epoch",
        type=int,
        default=None,
        help="checkpoint after this epoch; default epochs // 2",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    """Train once continuously and once through a checkpoint/resume, and compare."""
    args = parse_args()
    split_epoch = args.split_epoch or max(1, args.epochs // 2)
    if not (0 < split_epoch < args.epochs):
        raise SystemExit(
            f"--split-epoch must be strictly between 0 and --epochs "
            f"({split_epoch} vs {args.epochs})"
        )

    seed_all(args.seed)
    dataset, target_info = build_dataset(args.dataset, args.key)
    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=get_atomic_types(dataset),
        targets=target_info,
    )
    loss = OmegaConf.create({"energy": init_with_defaults(LossSpecification)})
    OmegaConf.resolve(loss)

    val_size = max(1, round(args.val_fraction * len(dataset)))
    split = len(dataset) - val_size
    train_dataset = torch.utils.data.Subset(dataset, range(split))
    val_dataset = torch.utils.data.Subset(dataset, range(split, len(dataset)))
    device = torch.device(args.device)

    def base_hypers() -> Dict[str, Any]:
        hypers = get_default_hypers("pet", base_precision=32)
        hypers["training"].update(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            loss=loss,
        )
        return hypers

    # ---- continuous: one Trainer.train call, start to finish ----
    seed_all(args.seed)
    hypers_continuous = base_hypers()
    hypers_continuous["training"]["num_epochs"] = args.epochs
    model_continuous = PET(hypers_continuous["model"], dataset_info)
    trainer_continuous = Trainer(hypers_continuous["training"])
    with TemporaryDirectory() as ckpt_dir:
        start = time.perf_counter()
        trainer_continuous.train(
            model=model_continuous,
            dtype=torch.float32,
            devices=[device],
            train_datasets=[train_dataset],
            val_datasets=[val_dataset],
            checkpoint_dir=ckpt_dir,
        )
        continuous_wall = time.perf_counter() - start
    assert trainer_continuous.epoch == args.epochs - 1, (
        f"continuous run should end at epoch {args.epochs - 1} (0-indexed), "
        f"landed at {trainer_continuous.epoch}"
    )

    # ---- resumed: train to split_epoch, checkpoint, restart to epochs ----
    seed_all(args.seed)
    hypers_phase1 = base_hypers()
    hypers_phase1["training"]["num_epochs"] = split_epoch
    model_phase1 = PET(hypers_phase1["model"], dataset_info)
    trainer_phase1 = Trainer(hypers_phase1["training"])
    with TemporaryDirectory() as ckpt_dir:
        start = time.perf_counter()
        trainer_phase1.train(
            model=model_phase1,
            dtype=torch.float32,
            devices=[device],
            train_datasets=[train_dataset],
            val_datasets=[val_dataset],
            checkpoint_dir=ckpt_dir,
        )
        phase1_wall = time.perf_counter() - start
        assert trainer_phase1.epoch == split_epoch - 1, (
            f"phase 1 should stop at epoch {split_epoch - 1} (0-indexed), "
            f"landed at {trainer_phase1.epoch}"
        )
        ckpt_path = Path(ckpt_dir) / "resume.ckpt"
        trainer_phase1.save_checkpoint(model_phase1, ckpt_path)
        # read back into memory now: ckpt_dir (and the file in it) goes away
        # when this `with` block exits
        checkpoint = torch.load(ckpt_path, weights_only=False, map_location="cpu")

    # exactly what `mtt train --restart` does internally (see
    # metatrain/cli/train.py) — not Trainer.load_checkpoint /
    # PET.load_checkpoint directly, so this exercises the same path a real
    # user hits, not a shortcut around it
    model_resumed = model_from_checkpoint(checkpoint, context="restart")
    model_resumed = model_resumed.restart(dataset_info, model_hypers=None)
    hypers_phase2 = base_hypers()
    hypers_phase2["training"]["num_epochs"] = args.epochs
    trainer_resumed = trainer_from_checkpoint(
        checkpoint=checkpoint, hypers=hypers_phase2["training"], context="restart"
    )
    assert trainer_resumed.epoch == split_epoch - 1, (
        f"reloaded trainer should resume from epoch {split_epoch - 1} (0-indexed), "
        f"landed at {trainer_resumed.epoch}"
    )
    with TemporaryDirectory() as ckpt_dir:
        start = time.perf_counter()
        trainer_resumed.train(
            model=model_resumed,
            dtype=torch.float32,
            devices=[device],
            train_datasets=[train_dataset],
            val_datasets=[val_dataset],
            checkpoint_dir=ckpt_dir,
        )
        phase2_wall = time.perf_counter() - start
    assert trainer_resumed.epoch == args.epochs - 1, (
        f"resumed run should end at epoch {args.epochs - 1} (0-indexed), "
        f"landed at {trainer_resumed.epoch}"
    )
    resumed_wall = phase1_wall + phase2_wall

    print(
        "resume_check status=ok "
        f"split_epoch={split_epoch} total_epochs={args.epochs} "
        f"continuous_best={trainer_continuous.best_metric:.6f} "
        f"continuous_best_epoch={trainer_continuous.best_epoch} "
        f"continuous_wall_s={continuous_wall:.3f} "
        f"resumed_best={trainer_resumed.best_metric:.6f} "
        f"resumed_best_epoch={trainer_resumed.best_epoch} "
        f"resumed_wall_s={resumed_wall:.3f} "
        f"resume_overhead_s={resumed_wall - continuous_wall:.3f}"
    )


if __name__ == "__main__":
    main()
