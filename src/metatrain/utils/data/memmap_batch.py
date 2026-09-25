"""Batched loading of :py:class:`MemmapDataset` structures.

``DataLoader`` fetches a whole batch at once through ``__getitems__`` when a
dataset defines it. :py:class:`MemmapDataset` uses that to gather the rows of
every structure of the batch from the memory-mapped arrays in one pass, and to
build each target as a single :py:class:`TensorMap` for the batch, instead of
one ``TensorMap`` per structure that ``group_and_join`` then joins.

The work is done by a small C++ extension (``memmap_batch.cpp``), compiled on
first use with :py:func:`torch.utils.cpp_extension.load` and cached by torch.
When it cannot be built (this logs a warning), on Windows, or when
``METATRAIN_MEMMAP_EXTENSION=0``, an equivalent vectorized Python
implementation is used instead.
"""

import logging
import os
import shutil
import sys
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System


logger = logging.getLogger(__name__)

SOURCE = Path(__file__).with_name("memmap_batch.cpp")


@dataclass
class JoinedSamples(SequenceABC):
    """Samples of a batch, already joined along the samples axis.

    This is what :py:meth:`MemmapDataset.__getitems__` returns, and what
    :py:func:`collate_batch` accepts in place of a list of samples: the same
    content as ``group_and_join`` would produce from them.

    It is also a sequence of the individual samples, loaded one by one on
    access, for collate functions that expect the list ``DataLoader`` would
    otherwise pass them.

    :param systems: One system per structure of the batch.
    :param fields: The targets and extra data, one ``TensorMap`` each.
    :param sample: Loads the individual sample at a position in the batch.
    """

    systems: List[System]
    fields: Dict[str, TensorMap]
    sample: Callable[[int], Any] = dataclass_field(repr=False)

    def __len__(self) -> int:
        return len(self.systems)

    def __getitem__(self, index: int) -> Any:  # type: ignore[override]
        if not -len(self) <= index < len(self):
            raise IndexError(index)
        return self.sample(index % len(self))


@dataclass
class Field:
    """How to read one target or extra data field of a batch.

    :param name: Name of the field in the samples.
    :param array: The memory-mapped values, of shape ``(rows, ..., properties)``
        with a row per atom or per structure.
    :param per_atom: Whether there is a row per atom rather than per structure.
    :param property_name: Name of the properties dimension.
    :param forces: The memory-mapped forces, for energies that have them.
    :param stress: The memory-mapped stresses, for energies that have them.
    """

    name: str
    array: torch.Tensor
    per_atom: bool
    property_name: str
    forces: Optional[torch.Tensor] = None
    stress: Optional[torch.Tensor] = None


def _library_directory(cmake_prefix_path: str) -> Path:
    return Path(cmake_prefix_path).parent


def _include_directory(cmake_prefix_path: str) -> Path:
    return Path(cmake_prefix_path).parent.parent / "include"


@lru_cache(maxsize=1)
def extension_available() -> bool:
    """Build and load the C++ extension, once per process.

    :return: Whether ``torch.ops.metatrain_memmap.load_batch`` can be used.
    """
    if os.environ.get("METATRAIN_MEMMAP_EXTENSION", "1") == "0":
        return False
    if sys.platform == "win32":
        # the link flags below are for GCC / Clang toolchains
        return False

    try:
        import metatensor
        import metatensor.torch
        import metatomic.torch
        from torch.utils import cpp_extension

        if shutil.which("ninja") is None:
            # the pip package, installed without its bin/ directory on the PATH
            import ninja

            os.environ["PATH"] = f"{ninja.BIN_DIR}{os.pathsep}{os.environ['PATH']}"

        prefixes = [
            metatensor.utils.cmake_prefix_path,
            metatensor.torch.utils.cmake_prefix_path,
            metatomic.torch.utils.cmake_prefix_path,
        ]
        libraries = [_library_directory(prefix) for prefix in prefixes]
        ldflags = [f"-L{directory}" for directory in libraries]
        ldflags += [f"-Wl,-rpath,{directory}" for directory in libraries]
        ldflags += ["-lmetatensor", "-lmetatensor_torch", "-lmetatomic_torch"]

        cpp_extension.load(
            name="metatrain_memmap_batch",
            sources=[str(SOURCE)],
            extra_include_paths=[str(_include_directory(p)) for p in prefixes],
            extra_cflags=["-O3"],
            extra_ldflags=ldflags,
            # see the end of memmap_batch.cpp: a Python module keeps the library
            # out of what metatomic collects when exporting models
            is_python_module=True,
            verbose=False,
        )
    except Exception as error:
        logger.debug("building the MemmapDataset C++ extension failed", exc_info=True)
        first_line = str(error).strip().splitlines()[0] if str(error) else repr(error)
        # a log message rather than a warning: the fallback gives the same
        # results, so this must not fail code that turns warnings into errors
        logger.warning(
            "could not build the MemmapDataset C++ extension, falling back to "
            f"the slower Python implementation: {first_line[:300]}"
        )
        return False
    return True


def load_batch(
    na: np.ndarray,
    positions: torch.Tensor,
    types: torch.Tensor,
    cells: Optional[torch.Tensor],
    indices: Sequence[int],
    fields: Sequence[Field],
    sample: Callable[[int], Any],
    use_extension: Optional[bool] = None,
) -> JoinedSamples:
    """Load structures ``indices`` and their ``fields`` as a joined batch.

    :param na: Cumulative atom counts of the dataset.
    :param positions: Positions of every atom of the dataset, float32.
    :param types: Types of every atom of the dataset, int32.
    :param cells: Cell of every structure of the dataset, float32, if any.
    :param indices: The structures to load, without duplicates.
    :param fields: The targets and extra data to load.
    :param sample: Loads the individual sample of structure ``indices[k]``,
        given ``k``.
    :param use_extension: Force the C++ (``True``) or Python (``False``)
        implementation; by default the extension is used when available.
    :return: The systems and the joined fields.
    """
    if use_extension is None:
        use_extension = extension_available()

    if use_extension:
        systems, tensors = torch.ops.metatrain_memmap.load_batch(
            torch.from_numpy(na),
            positions,
            types,
            cells,
            [int(i) for i in indices],
            [field.array for field in fields],
            [field.per_atom for field in fields],
            [field.property_name for field in fields],
            [field.forces for field in fields],
            [field.stress for field in fields],
        )
    else:
        systems, tensors = _load_batch_python(
            na, positions, types, cells, indices, fields
        )
    return JoinedSamples(
        systems=systems,
        fields={f.name: tensor for f, tensor in zip(fields, tensors, strict=True)},
        sample=sample,
    )


@lru_cache(maxsize=None)
def _range(name: str, end: int) -> Labels:
    return Labels.range(name, end)


def _components(dim: int) -> List[Labels]:
    rank = dim - 2
    if rank == 1:
        return [_range("xyz", 3)]
    return [_range(f"xyz_{d + 1}", 3) for d in range(rank)]


def _load_batch_python(
    na: np.ndarray,
    positions: torch.Tensor,
    types: torch.Tensor,
    cells: Optional[torch.Tensor],
    indices: Sequence[int],
    fields: Sequence[Field],
) -> Tuple[List[System], List[TensorMap]]:
    """The same as the C++ ``load_batch``, with numpy and torch.

    :param na: Cumulative atom counts of the dataset.
    :param positions: Positions of every atom of the dataset, float32.
    :param types: Types of every atom of the dataset, int32.
    :param cells: Cell of every structure of the dataset, float32, if any.
    :param indices: The structures to load, without duplicates.
    :param fields: The targets and extra data to load.
    :return: The systems, and one ``TensorMap`` per field.
    """
    structures = torch.as_tensor(np.asarray(indices, dtype=np.int64))
    starts = na[structures.numpy()]
    counts = na[structures.numpy() + 1] - starts
    total = int(counts.sum())
    batch_size = len(structures)

    # index of every atom of the batch in the per-atom arrays
    offsets = np.cumsum(counts) - counts
    atoms = torch.from_numpy(np.repeat(starts - offsets, counts) + np.arange(total))
    atom_in_structure = torch.from_numpy(np.arange(total) - np.repeat(offsets, counts))
    row = torch.repeat_interleave(torch.arange(batch_size), torch.from_numpy(counts))

    all_positions = positions[atoms].to(torch.float64)
    all_types = types[atoms]
    if cells is not None:
        all_cells = cells[structures].to(torch.float64)
    else:
        all_cells = torch.zeros((batch_size, 3, 3), dtype=torch.float64)
    all_pbcs = torch.any(all_cells != 0.0, dim=2)

    split = counts.tolist()
    systems = [
        System(types=t, positions=x, cell=c, pbc=p)
        for t, x, c, p in zip(
            torch.split(all_types, split),
            torch.split(all_positions, split),
            all_cells,
            all_pbcs,
            strict=True,
        )
    ]

    int32 = torch.int32
    per_atom_samples = Labels(
        ["system", "atom"],
        torch.stack([structures[row], atom_in_structure], dim=1).to(int32),
    )
    per_structure_samples = Labels(["system"], structures.reshape(-1, 1).to(int32))

    tensors = []
    for field in fields:
        properties = _range(field.property_name, field.array.shape[-1])
        if field.per_atom:
            values = field.array[atoms].to(torch.float64)
            samples = per_atom_samples
        else:
            values = field.array[structures].to(torch.float64)
            samples = per_structure_samples
        block = TensorBlock(
            values=values,
            samples=samples,
            components=_components(field.array.dim()),
            properties=properties,
        )
        if field.forces is not None:
            block.add_gradient(
                "positions",
                TensorBlock(
                    values=-field.forces[atoms].to(torch.float64),
                    samples=Labels(
                        ["sample", "atom"],
                        torch.stack([row, atom_in_structure], dim=1).to(int32),
                    ),
                    components=_components(field.forces.dim()),
                    properties=properties,
                ),
            )
        if field.stress is not None:
            volumes = torch.abs(torch.det(all_cells)).reshape(-1, 1, 1, 1)
            block.add_gradient(
                "strain",
                TensorBlock(
                    values=field.stress[structures].to(torch.float64) * volumes,
                    samples=Labels(
                        ["sample"], torch.arange(batch_size, dtype=int32).reshape(-1, 1)
                    ),
                    components=_components(field.stress.dim()),
                    properties=properties,
                ),
            )
        tensors.append(TensorMap(keys=Labels.single(), blocks=[block]))

    return systems, tensors
