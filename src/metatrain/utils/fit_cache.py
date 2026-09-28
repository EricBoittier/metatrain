"""
Content-addressed cache for the composition and scaler fits.

Both fits are deterministic statistics of the training set: they do not depend
on the batch size, the number of workers or anything else about the run that
consumes them. When the ``METATRAIN_FIT_CACHE_DIR`` environment variable is
set, a fit is stored there under a key derived from everything it does depend
on, and later runs with the same key load it instead of iterating over the
training set again.

The key covers the model before fitting, its dataset info, the additive models
(with their weights, so that a scaler fitted on top of a different composition
baseline is never reused), the fixed weights and a fingerprint of the training
datasets. On-disk datasets are fingerprinted by their files' path, size and
modification time, in-memory ones by their contents, and subsets additionally by
their indices.
"""

import hashlib
import logging
import os
import shutil
import time
import weakref
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, ContextManager, List, Optional

import metatensor.torch as mts
import numpy as np
import torch

import metatrain

from .data.dataset import DatasetInfo, DiskDataset, MemmapDataset, Subset
from .data.target_info import TargetInfo


CACHE_DIR_ENV = "METATRAIN_FIT_CACHE_DIR"
# Bump when the key or the stored checkpoints change meaning.
_FORMAT_VERSION = 1

# Hashing an in-memory dataset takes ~0.1 ms per sample, and both fits hash the
# same datasets: remember their fingerprints for the lifetime of the objects.
_FINGERPRINTS: "weakref.WeakKeyDictionary[Any, str]" = weakref.WeakKeyDictionary()


def _script_type(obj: Any) -> Optional[str]:
    # TensorMap and System are TorchScript classes, which isinstance rejects.
    return obj._type().name() if isinstance(obj, torch.ScriptObject) else None


def _update(h: "hashlib._Hash", obj: Any) -> None:
    """
    Feed ``obj`` into ``h``, refusing types that have no stable encoding.

    :param h: The hash to update.
    :param obj: The object to hash.
    """
    # Every value is prefixed by its kind, so that e.g. ``"1"`` and ``1`` or
    # ``[a, b]`` and ``[[a], b]`` cannot collide.
    if obj is None or isinstance(obj, (bool, int, float, str)):
        h.update(f"{type(obj).__name__}:{obj!r};".encode())
    elif isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().contiguous()
        h.update(f"tensor:{obj.dtype}:{tuple(obj.shape)};".encode())
        h.update(obj.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(obj, np.ndarray):
        h.update(f"array:{obj.dtype}:{obj.shape};".encode())
        h.update(np.ascontiguousarray(obj).tobytes())
    elif _script_type(obj) == "TensorMap":
        _update(h, mts.save_buffer(obj.to("cpu")))
    elif _script_type(obj) == "System":
        _update(h, ["system", obj.types, obj.positions, obj.cell, obj.pbc])
    elif isinstance(obj, TargetInfo):
        _update(h, ["target", obj.quantity, obj.unit, obj.layout])
    elif isinstance(obj, DatasetInfo):
        _update(
            h,
            ["dataset_info", obj.length_unit, obj.atomic_types]
            + [dict(obj.targets), dict(obj.extra_data)],
        )
    elif isinstance(obj, torch.nn.Module):
        # ``dummy_buffer`` is random, it only tracks the device and dtype.
        state = {
            name: value
            for name, value in obj.state_dict().items()
            if not name.endswith("dummy_buffer")
        }
        _update(
            h,
            [type(obj).__qualname__, state, getattr(obj, "dataset_info", None)],
        )
    elif isinstance(obj, Mapping):
        h.update(f"dict:{len(obj)};".encode())
        for key in sorted(obj, key=str):
            _update(h, [key, obj[key]])
    elif isinstance(obj, Sequence):
        h.update(f"list:{len(obj)};".encode())
        for item in obj:
            _update(h, item)
    elif isinstance(obj, torch.utils.data.Dataset):
        h.update(f"dataset:{dataset_fingerprint(obj)};".encode())
    else:
        raise TypeError(f"can not fingerprint an object of type {type(obj)}")


def _file_signature(path: Path) -> list:
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]


def dataset_fingerprint(dataset: Any) -> str:
    """
    Fingerprint a training dataset.

    :param dataset: A metatrain dataset, or a :py:class:`Subset` of one.
    :return: A hex digest that changes whenever the samples of the dataset may
        have changed.
    """
    if dataset in _FINGERPRINTS:
        return _FINGERPRINTS[dataset]
    h = hashlib.sha256()
    if isinstance(dataset, Subset):
        _update(h, ["subset", dataset.dataset, np.asarray(dataset.indices, np.int64)])
    elif isinstance(dataset, DiskDataset):
        path = Path(dataset.zip_file_path)
        _update(h, ["disk", _file_signature(path), dataset._fields_to_read])
    elif isinstance(dataset, MemmapDataset):
        path = Path(dataset.path)
        files = sorted(p for p in path.iterdir() if p.is_file())
        _update(h, ["memmap", [_file_signature(p) for p in files]])
        _update(h, [dataset.target_config, dataset.extra_data_config])
    else:
        # In-memory datasets have no source file to stat, but reading them is
        # cheap: hash every sample.
        h.update(f"memory:{len(dataset)};".encode())
        for sample in dataset:
            _update(h, dict(zip(sample._fields, sample, strict=True)))
    _FINGERPRINTS[dataset] = h.hexdigest()
    return _FINGERPRINTS[dataset]


def get_fit_cache_path(kind: str, parts: List[Any]) -> Optional[Path]:
    """
    Path of the cached ``kind`` fit determined by ``parts``.

    :param kind: Name of the fit, used as the prefix of the file name.
    :param parts: Everything the fit depends on.
    :return: The path, or ``None`` if caching is disabled.
    """
    cache_dir = os.environ.get(CACHE_DIR_ENV)
    if not cache_dir:
        return None
    start = time.perf_counter()
    h = hashlib.sha256()
    _update(h, [kind, _FORMAT_VERSION, metatrain.__version__, parts])
    logging.info(
        f"Computed the {kind} fit cache key in {time.perf_counter() - start:.1f} s"
    )
    return Path(cache_dir) / f"{kind}-{h.hexdigest()[:32]}.ckpt"


def is_cached(path: Optional[Path], device: torch.device, is_distributed: bool) -> bool:
    """
    Whether ``path`` holds a cached fit, agreed upon by all ranks.

    Ranks must agree, since fitting involves collectives: another job could
    populate the cache between the checks of two ranks.

    :param path: Path returned by :py:func:`get_fit_cache_path`.
    :param device: Device on which the fit runs, used for the collective.
    :param is_distributed: Whether training is distributed.
    :return: ``True`` if the fit can be loaded on every rank.
    """
    hit = path is not None and path.is_file()
    if is_distributed and torch.distributed.is_initialized():
        flag = torch.tensor([int(hit)], device=device)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
        hit = bool(flag.item())
    return hit


def preserve_rng(path: Optional[Path]) -> ContextManager:
    """
    Keep a fit from consuming the global random state when caching is enabled.

    Fitting (whose dataloaders draw their base seed from the global CPU
    generator) and loading (which builds a model with a random buffer) consume
    different amounts of the random state: without this, the rest of training
    would differ between a run that hit the cache and one that missed it. When
    caching is disabled, the random state is consumed as it always was.

    :param path: Path returned by :py:func:`get_fit_cache_path`.
    :return: A context manager to run the fit in.
    """
    return nullcontext() if path is None else torch.random.fork_rng(devices=[])


def _is_main_process(is_distributed: bool) -> bool:
    return not (
        is_distributed
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() != 0
    )


def store_fit(
    path: Optional[Path], save: Callable[[Path], None], is_distributed: bool
) -> None:
    """
    Store a fit in the cache, atomically so that concurrent runs never read a
    partial file.

    :param path: Path returned by :py:func:`get_fit_cache_path`.
    :param save: Function saving the fitted checkpoint to a given path.
    :param is_distributed: Whether training is distributed; only rank 0 writes.
    """
    if path is None or not _is_main_process(is_distributed):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.ckpt")
    save(tmp)
    os.replace(tmp, path)
    logging.info(f"Stored the fit in {path}")


def copy_cached_fit(
    path: Path, checkpoint_dir: str, filename: str, is_distributed: bool
) -> None:
    """
    Copy a cached fit into ``checkpoint_dir``, where the trainer would have
    saved it, so that a run's outputs do not depend on whether it hit the cache.

    :param path: Path of the cached fit.
    :param checkpoint_dir: Checkpoint directory of the run, may be empty.
    :param filename: Name the trainer gives to the checkpoint.
    :param is_distributed: Whether training is distributed; only rank 0 writes.
    """
    if checkpoint_dir and _is_main_process(is_distributed):
        shutil.copyfile(path, Path(checkpoint_dir) / filename)
