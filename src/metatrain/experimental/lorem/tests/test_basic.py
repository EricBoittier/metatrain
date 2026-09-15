import copy
import glob
import gzip
from typing import Literal

import pytest
import torch

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
    hypers["cutoff"] = 3.0
    hypers["cutoff_width"] = 0.5
    hypers["max_degree"] = 1
    hypers["max_degree_lr"] = 0
    hypers["num_features"] = 4
    hypers["num_spherical_features"] = 1
    hypers["num_radial"] = 2
    hypers["num_message_passing"] = 0
    hypers["long_range"]["enable"] = False
    return hypers


class LoremTests(ArchitectureTests):
    architecture = "experimental.lorem"

    @pytest.fixture
    def model_hypers(self) -> dict:
        return _minimal_hypers(self.architecture)

    @pytest.fixture
    def minimal_model_hypers(self) -> dict:
        return _minimal_hypers(self.architecture)


class TestInput(InputTests, LoremTests): ...


class TestOutput(OutputTests, LoremTests):
    supports_multiscalar_outputs = False
    supports_spherical_outputs = False
    supports_spherical_rank2_outputs = False
    supports_spherical_atomic_basis_outputs = False
    supports_vector_outputs = True
    supports_features = False
    supports_last_layer_features = False


class TestAutograd(AutogradTests, LoremTests):
    cuda_nondet_tolerance = 1e-12


class TestTorchscript(TorchscriptTests, LoremTests):
    float_hypers = ["cutoff", "cutoff_width"]
    supports_spherical_outputs = False


class TestExported(ExportedTests, LoremTests): ...


class TestTraining(TrainingTests, LoremTests):
    supports_atomic_basis = False


class TestCheckpoints(CheckpointTests, LoremTests):
    incompatible_trainer_checkpoints = []

    # Checkpoints saved before the equivariant-message-passing / CG
    # TensorDense-TensorProduct refactor (checkpoint version < 4) introduced
    # trained parameters with no version-3 counterpart to copy weights from,
    # so they cannot be upgraded (see checkpoints.model_update_v3_v4) -- unlike
    # incompatible_trainer_checkpoints, this holds for every context, not just
    # "restart". These fixtures are kept for test_pre_v4_checkpoints_raise_
    # clear_error below, which pins the resulting error, not for "old
    # checkpoints still load" coverage.
    unloadable_model_checkpoints = [
        "checkpoints/model-v1_trainer-v1.ckpt.gz",
        "checkpoints/model-v2_trainer-v1.ckpt.gz",
        "checkpoints/model-v3_trainer-v1.ckpt.gz",
    ]

    @pytest.mark.parametrize("context", ["restart", "finetune", "export"])
    def test_loading_old_checkpoints(
        self,
        default_hypers: dict,
        model_trainer: tuple,
        context: Literal["restart", "finetune", "export"],
    ) -> None:
        """Same as ``CheckpointTests.test_loading_old_checkpoints``, minus the
        pre-version-4 fixtures that are unloadable by design -- see
        ``unloadable_model_checkpoints`` above."""
        model, trainer = model_trainer

        for path in glob.glob("checkpoints/*.ckpt.gz"):
            if path in self.unloadable_model_checkpoints:
                continue
            if path in self.incompatible_trainer_checkpoints and context == "restart":
                continue

            with gzip.open(path, "rb") as fd:
                checkpoint = torch.load(fd, weights_only=False)

            if checkpoint["model_ckpt_version"] != model.__checkpoint_version__:
                checkpoint = model.__class__.upgrade_checkpoint(checkpoint)
            model.load_checkpoint(checkpoint, context)

            if context == "restart":
                if checkpoint["trainer_ckpt_version"] != trainer.__checkpoint_version__:
                    checkpoint = trainer.__class__.upgrade_checkpoint(checkpoint)
                trainer.load_checkpoint(checkpoint, default_hypers, context)

    def test_pre_v4_checkpoints_raise_clear_error(self) -> None:
        """Loading a version 1-3 LOREM checkpoint must fail loudly (a clear
        ``RuntimeError`` naming the refactor and telling the caller to
        retrain), not silently drop the new equivariant-message-passing /
        TensorDense-CG parameters to a random initialization -- the bug
        ``load_state_dict(..., strict=False)`` used to hide."""
        for path in self.unloadable_model_checkpoints:
            with gzip.open(path, "rb") as fd:
                checkpoint = torch.load(fd, weights_only=False)
            with pytest.raises(RuntimeError, match="cannot be automatically upgraded"):
                self.model_cls.upgrade_checkpoint(checkpoint)
