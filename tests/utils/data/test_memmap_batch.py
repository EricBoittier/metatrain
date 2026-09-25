import metatensor.torch as mts
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset

from metatrain.utils.data import CollateFn, collate_batch, memmap_batch, unpack_batch
from metatrain.utils.data.dataset import MemmapDataset
from metatrain.utils.data.memmap_batch import JoinedSamples, extension_available


ATOMS = [3, 1, 5, 2, 4, 6, 1, 2]


def write_dataset(path, atoms=ATOMS):
    """A dataset with every kind of field ``__getitems__`` handles."""
    rng = np.random.default_rng(0)
    ns, total = len(atoms), sum(atoms)
    np.save(path / "ns.npy", ns)
    np.save(path / "na.npy", np.concatenate([[0], np.cumsum(atoms)]).astype(np.int64))

    def write(name, shape, dtype="float32"):
        rng.normal(size=shape).astype(dtype).tofile(path / f"{name}.bin")

    write("x", (total, 3))
    rng.integers(1, 10, total).astype("int32").tofile(path / "a.bin")
    # periodic, non-periodic, and periodic along one direction only
    cells = np.zeros((ns, 3, 3), dtype="float32")
    cells[0::3] = 10 * np.eye(3) + rng.normal(size=(3, 3))
    cells[2::3, 0] = [7.0, 0.0, 0.0]
    cells.tofile(path / "c.bin")
    write("energy", (ns, 1))
    write("forces", (total, 3, 1))
    write("stress", (ns, 3, 3, 1))
    write("charges", (total, 2))
    write("dipole", (ns, 3, 1))
    write("polarizability", (total, 3, 3, 4))
    write("spin", (ns, 1))
    write("atomic_spin", (total, 1))

    targets = {
        "energy": {
            "key": "energy",
            "quantity": "energy",
            "unit": "eV",
            "sample_kind": "system",
            "num_subtargets": 1,
            "type": "scalar",
            "forces": {"key": "forces"},
            "stress": {"key": "stress"},
            "virial": False,
        },
        "mtt::charges": {
            "key": "charges",
            "quantity": "",
            "sample_kind": "atom",
            "num_subtargets": 2,
            "type": "scalar",
        },
        "mtt::dipole": {
            "key": "dipole",
            "quantity": "",
            "sample_kind": "system",
            "num_subtargets": 1,
            "type": {"cartesian": {"rank": 1}},
        },
        "mtt::polarizability": {
            "key": "polarizability",
            "quantity": "",
            "sample_kind": "atom",
            "num_subtargets": 4,
            "type": {"cartesian": {"rank": 2}},
        },
    }
    extra = {
        "mtt::spin": {
            "key": "spin",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
        },
        "mtt::atomic_spin": {
            "key": "atomic_spin",
            "type": "scalar",
            "sample_kind": "atom",
            "num_subtargets": 1,
        },
    }
    return MemmapDataset(path, targets, extra), list(targets)


def assert_same_batch(expected, actual):
    assert len(expected.systems) == len(actual.systems)
    for a, b in zip(expected.systems, actual.systems, strict=True):
        for name in ("types", "positions", "cell", "pbc"):
            x, y = getattr(a, name), getattr(b, name)
            assert x.dtype == y.dtype, name
            assert torch.equal(x, y), name
    for fields in ("targets", "extra_data"):
        a, b = getattr(expected, fields), getattr(actual, fields)
        assert list(a) == list(b)
        for name in a:
            assert mts.equal(a[name], b[name]), name


IMPLEMENTATIONS = [
    pytest.param(False, id="python"),
    pytest.param(
        True,
        id="extension",
        marks=pytest.mark.skipif(
            not extension_available(), reason="C++ extension unavailable"
        ),
    ),
]


@pytest.mark.parametrize("use_extension", IMPLEMENTATIONS)
@pytest.mark.parametrize("indices", [[5, 0, 2, 7], [3], list(range(len(ATOMS)))])
def test_same_as_joined_samples(tmp_path, monkeypatch, use_extension, indices):
    monkeypatch.setattr(memmap_batch, "extension_available", lambda: use_extension)
    dataset, target_keys = write_dataset(tmp_path)

    joined = dataset.__getitems__(indices)
    assert isinstance(joined, JoinedSamples)

    expected = collate_batch([dataset[i] for i in indices], target_keys)
    assert_same_batch(expected, collate_batch(joined, target_keys))


def test_repeated_structures_fall_back(tmp_path):
    dataset, target_keys = write_dataset(tmp_path)
    samples = dataset.__getitems__([1, 1])
    assert isinstance(samples, list) and len(samples) == 2


@pytest.mark.parametrize("use_extension", IMPLEMENTATIONS)
def test_dataloader(tmp_path, monkeypatch, use_extension):
    """DataLoader goes through ``__getitems__``, also inside a ``Subset``."""
    monkeypatch.setattr(memmap_batch, "extension_available", lambda: use_extension)
    dataset, target_keys = write_dataset(tmp_path)
    subset = Subset(dataset, [6, 1, 4, 3, 0])

    loader = DataLoader(subset, batch_size=2, collate_fn=CollateFn(target_keys))
    for batch, start in zip(loader, [0, 2, 4], strict=True):
        indices = subset.indices[start : start + 2]
        expected = collate_batch([dataset[i] for i in indices], target_keys)
        systems, targets, extra_data = unpack_batch(batch)
        assert len(systems) == len(indices)
        for name, tensor in targets.items():
            assert mts.equal(expected.targets[name], tensor), name
        for name, tensor in extra_data.items():
            assert mts.equal(expected.extra_data[name], tensor), name


def test_sequence_of_samples(tmp_path):
    """Collate functions that iterate over the samples still get them."""
    dataset, target_keys = write_dataset(tmp_path)
    joined = dataset.__getitems__([4, 1, 6])
    assert len(joined) == 3
    for sample, i in zip(joined, [4, 1, 6], strict=True):
        assert torch.equal(sample.system.positions, dataset[i].system.positions)
        assert mts.equal(sample.energy, dataset[i].energy)
    assert mts.equal(joined[-1].energy, dataset[6].energy)
    with pytest.raises(IndexError):
        joined[3]


@pytest.mark.skipif(not extension_available(), reason="C++ extension unavailable")
def test_extension_not_collected_on_export(tmp_path):
    """Models exported after training on a MemmapDataset do not depend on it."""
    from metatomic.torch._extensions import _collect_extensions

    assert not any("metatrain_memmap" in path for path in torch.ops.loaded_libraries)
    extensions, _ = _collect_extensions(str(tmp_path / "extensions"))
    assert not any("metatrain_memmap" in e["name"] for e in extensions)
