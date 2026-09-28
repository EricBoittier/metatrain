from pathlib import Path

import metatensor.torch as mts
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import Subset

from metatrain.composition import CompositionModel, train_or_load_composition_model
from metatrain.composition.trainer import Trainer as CompositionTrainer
from metatrain.scaler import Scaler, train_or_load_scaler
from metatrain.scaler.trainer import Trainer as ScalerTrainer
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.fit_cache import (
    CACHE_DIR_ENV,
    dataset_fingerprint,
    get_fit_cache_path,
)


RESOURCES_PATH = Path(__file__).parents[1] / "resources"


@pytest.fixture
def qm9():
    path = RESOURCES_PATH / "qm9_reduced_100.xyz"
    conf = {
        "energy": {
            "quantity": "energy",
            "read_from": path,
            "reader": "ase",
            "key": "U0",
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
    dataset = Dataset.from_dict({"system": read_systems(path), **targets})
    return dataset, DatasetInfo("angstrom", [1, 6, 7, 8], target_info)


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path / "cache"))
    return tmp_path / "cache"


def fit_composition(dataset, dataset_info, baseline=None, checkpoint_dir=""):
    model = CompositionModel(hypers={}, dataset_info=dataset_info).to(torch.float64)
    train_or_load_composition_model(
        model, baseline or {}, [dataset], [], 10, False, checkpoint_dir
    )
    return model


def fit_scaler(dataset, dataset_info, composition, checkpoint_dir=""):
    scaler = Scaler(hypers={}, dataset_info=dataset_info).to(torch.float64)
    train_or_load_scaler(
        scaler, [dataset], [composition], 10, False, checkpoint_dir=checkpoint_dir
    )
    return scaler


def forbid(monkeypatch, trainer):
    def train(*args, **kwargs):
        raise AssertionError("the fit should have been loaded from the cache")

    monkeypatch.setattr(trainer, "train", train)


def test_disabled_without_env(qm9, tmp_path, monkeypatch):
    monkeypatch.delenv(CACHE_DIR_ENV, raising=False)
    assert get_fit_cache_path("composition", [qm9[0]]) is None


def test_composition_cache_hit(qm9, cache_dir, tmp_path, monkeypatch):
    fitted = fit_composition(*qm9)
    assert len(list(cache_dir.glob("composition-*.ckpt"))) == 1

    forbid(monkeypatch, CompositionTrainer)
    (tmp_path / "run").mkdir()
    loaded = fit_composition(*qm9, checkpoint_dir=str(tmp_path / "run"))

    assert mts.equal(fitted.model.weights["energy"], loaded.model.weights["energy"])
    # The run's outputs are the same as if it had fitted.
    assert (tmp_path / "run" / "composition_model.ckpt").is_file()


def test_composition_cache_miss_on_different_baseline(qm9, cache_dir):
    fit_composition(*qm9)
    fit_composition(*qm9, baseline={"energy": -1.0})
    assert len(list(cache_dir.glob("composition-*.ckpt"))) == 2


def test_scaler_cache_depends_on_composition(qm9, cache_dir, monkeypatch):
    composition = fit_composition(*qm9)
    fitted = fit_scaler(*qm9, composition)
    assert len(list(cache_dir.glob("scaler-*.ckpt"))) == 1

    # Same baseline: hit.
    with monkeypatch.context() as patch:
        forbid(patch, ScalerTrainer)
        loaded = fit_scaler(*qm9, composition)
    assert mts.equal(fitted.model.scales["energy"], loaded.model.scales["energy"])

    # Different baseline, so different residuals to scale: miss.
    fit_scaler(*qm9, fit_composition(*qm9, baseline={"energy": 0.0}))
    assert len(list(cache_dir.glob("scaler-*.ckpt"))) == 2


def test_dataset_fingerprint(qm9):
    dataset = qm9[0]
    assert dataset_fingerprint(dataset) == dataset_fingerprint(dataset)

    first, second = Subset(dataset, range(50)), Subset(dataset, range(50, 100))
    assert dataset_fingerprint(first) != dataset_fingerprint(second)

    modified = Dataset.from_dict(
        {
            "system": [sample.system for sample in dataset],
            "energy": [
                mts.multiply(sample.energy, 1.0 + (i == 0))
                for i, sample in enumerate(dataset)
            ],
        }
    )
    assert dataset_fingerprint(modified) != dataset_fingerprint(dataset)


def test_omegaconf_parts_hash_like_plain_ones(cache_dir):
    plain = {"energy": {1: -0.5}, "targets": ["energy"]}
    assert get_fit_cache_path("scaler", [plain]) == get_fit_cache_path(
        "scaler", [OmegaConf.create(plain)]
    )


def test_unsupported_part(cache_dir):
    with pytest.raises(TypeError, match="can not fingerprint"):
        get_fit_cache_path("composition", [object()])


def test_hit_and_miss_leave_the_same_random_state(qm9, cache_dir):
    # Otherwise, whatever is trained after these fits would depend on whether
    # they were loaded from the cache.
    states = []
    for _ in range(2):  # miss, then hit
        torch.manual_seed(0)
        fit_scaler(*qm9, fit_composition(*qm9))
        states.append(torch.random.get_rng_state())
    assert len(list(cache_dir.glob("*.ckpt"))) == 2
    assert torch.equal(*states)
