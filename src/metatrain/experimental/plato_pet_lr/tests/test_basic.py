"""metatrain's shared architecture suite."""

import copy

import pytest

from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.testing import (
    ArchitectureTests,
    AutogradTests,
    CheckpointTests,
    ExportedTests,
    InputTests,
    OutputTests,
    TorchscriptTests,
    TrainingTests,
)


def _minimal_hypers(arch: str) -> dict:
    hypers = copy.deepcopy(get_default_hypers(arch)["model"])
    hypers.update(
        cutoff=3.0, d_model=4, num_heads=1, num_layers=1, couple_at=0, num_rbf=2,
        n_basis=1, kernel_hidden=2, mesh_channels=1, mesh_spacing=2.0,
    )  # fmt: skip
    return hypers


class PlatoTests(ArchitectureTests):
    architecture = "experimental.plato_pet_lr"

    @pytest.fixture
    def model_hypers(self) -> dict:
        return _minimal_hypers(self.architecture)

    @pytest.fixture
    def minimal_model_hypers(self) -> dict:
        return _minimal_hypers(self.architecture)


class TestInput(InputTests, PlatoTests): ...


class TestOutput(OutputTests, PlatoTests):
    is_equivariant_rotations = False  # exact only under the frame group
    supports_multiscalar_outputs = False
    supports_spherical_outputs = False
    supports_spherical_rank2_outputs = False
    supports_spherical_atomic_basis_outputs = False
    supports_vector_outputs = False
    supports_features = False
    supports_last_layer_features = False

    def test_prediction_energy_subset_atoms(self, *args, **kwargs) -> None:
        pytest.skip("LES makes the interaction range infinite")


class TestAutograd(AutogradTests, PlatoTests):
    cuda_nondet_tolerance = 1e-12


class TestTorchscript(TorchscriptTests, PlatoTests):
    float_hypers = ["cutoff"]
    supports_spherical_outputs = False


class TestExported(ExportedTests, PlatoTests): ...


class TestTraining(TrainingTests, PlatoTests):
    supports_atomic_basis = False


class TestCheckpoints(CheckpointTests, PlatoTests):
    incompatible_trainer_checkpoints = []
