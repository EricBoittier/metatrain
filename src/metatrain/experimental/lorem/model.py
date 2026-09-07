import logging
from typing import Any, Dict, List, Literal, Optional, Union

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)

from metatrain.composition import CompositionModel
from metatrain.scaler import Scaler
from metatrain.utils.abc import ModelInterface
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import TargetInfo
from metatrain.utils.data.atom_pair_helpers import check_no_atom_pair_targets
from metatrain.utils.data.dataset import DatasetInfo
from metatrain.utils.dtype import dtype_to_str
from metatrain.utils.hypers import raise_if_hypers_mismatch
from metatrain.utils.metadata import merge_metadata
from metatrain.utils.sum_over_atoms import sum_over_atoms

from . import checkpoints
from .documentation import ModelHypers
from .modules.backbone import LoremBackbone
from .modules.long_range import (
    DummyLoremLongRangeFeaturizer,
    LoremLongRangeFeaturizer,
)


class LOREM(ModelInterface[ModelHypers]):
    __checkpoint_version__ = 1
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float32, torch.float64]
    __default_metadata__ = ModelMetadata(
        references={
            "implementation": [
                "lorem-jax: https://github.com/lab-cosmo/lorem-jax",
                "torch-pme: https://github.com/lab-cosmo/torch-pme",
            ],
            "architecture": [
                "LOREM: https://arxiv.org/abs/2507.19382",
            ],
        }
    )

    component_labels: Dict[str, List[List[Labels]]]

    def __init__(self, hypers: ModelHypers, dataset_info: DatasetInfo) -> None:
        super().__init__(hypers, dataset_info, self.__default_metadata__)
        check_no_atom_pair_targets(dataset_info.targets, self.__class__.__name__)
        self.atomic_types = dataset_info.atomic_types
        self.has_new_targets = False

        self.requested_nl = NeighborListOptions(
            cutoff=self.hypers["cutoff"],
            full_list=True,
            strict=True,
        )
        self.backbone = LoremBackbone(
            dict(self.hypers),
            self.atomic_types,
            self.requested_nl,
        )
        self.num_features = int(self.hypers["num_features"])
        self.max_degree_lr = int(self.hypers["max_degree_lr"])
        if self.max_degree_lr > int(self.hypers["max_degree"]):
            raise ValueError(
                f"max_degree_lr ({self.max_degree_lr}) cannot exceed "
                f"max_degree ({self.hypers['max_degree']})."
            )

        if self.hypers["long_range"]["enable"]:
            self.long_range = True
            self.long_range_featurizer = LoremLongRangeFeaturizer(
                self.hypers["long_range"],
                self.num_features,
                int(self.hypers["num_radial"]),
                int(self.hypers["num_spherical_features"]),
                self.max_degree_lr,
                self.requested_nl,
            )
        else:
            self.long_range = False
            self.long_range_featurizer = DummyLoremLongRangeFeaturizer()

        self.outputs: Dict[str, ModelOutput] = {}
        self.readouts = torch.nn.ModuleDict({})
        self.single_label = Labels.single()
        self.num_properties: Dict[str, Dict[str, int]] = {}
        self.key_labels: Dict[str, Labels] = {}
        self.component_labels = {}
        self.property_labels: Dict[str, List[Labels]] = {}
        for target_name, target in dataset_info.targets.items():
            self._add_output(target_name, target)

        composition_model = CompositionModel.from_valid_targets(
            dataset_info, self.atomic_types
        )
        self.additive_models = torch.nn.ModuleList([composition_model])

        scaler_hypers = get_default_hypers("scaler")["model"]
        self.scaler = Scaler(hypers=scaler_hypers, dataset_info=dataset_info)

    def _add_output(self, target_name: str, target: TargetInfo) -> None:
        if not target.is_scalar:
            raise ValueError("The LOREM architecture can only predict scalars.")
        n_properties = len(target.layout.block().properties)
        self.num_properties[target_name] = {"default": n_properties}
        self.key_labels[target_name] = target.layout.keys
        self.component_labels[target_name] = [
            block.components for block in target.layout.blocks()
        ]
        self.property_labels[target_name] = [
            block.properties for block in target.layout.blocks()
        ]
        self.readouts[target_name] = torch.nn.Linear(self.num_features, n_properties)
        self.outputs[target_name] = ModelOutput(
            unit=target.unit,
            sample_kind="atom",
        )

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [self.requested_nl]

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs

    def get_fixed_composition_weights(self) -> dict[str, dict[int, float]]:
        return {}

    def get_fixed_scaling_weights(self) -> dict[str, Union[float, dict[int, float]]]:
        return {}

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if len(outputs) == 0:
            return {}

        device = systems[0].positions.device
        if self.single_label.values.device != device:
            self.single_label = self.single_label.to(device)
            self.key_labels = {
                name: label.to(device) for name, label in self.key_labels.items()
            }
            self.component_labels = {
                name: [
                    [labels.to(device) for labels in components_block]
                    for components_block in components_tmap
                ]
                for name, components_tmap in self.component_labels.items()
            }
            self.property_labels = {
                name: [labels.to(device) for labels in properties_tmap]
                for name, properties_tmap in self.property_labels.items()
            }

        features, neighbor_distances, spherical_features = self.backbone(systems)
        if self.long_range:
            if self.training:
                self.long_range_featurizer.use_ewald = True
            features = self.long_range_featurizer(
                systems, features, neighbor_distances, spherical_features
            )

        system_sizes = [len(system) for system in systems]
        system_sizes_tensor = torch.tensor(system_sizes, device=device)
        system_indices = torch.repeat_interleave(
            torch.arange(len(systems), device=device), system_sizes_tensor
        )
        atom_indices = torch.cat(
            [torch.arange(size, device=device) for size in system_sizes]
        )
        sample_values = torch.stack([system_indices, atom_indices], dim=1)
        samples = Labels(names=["system", "atom"], values=sample_values.to(torch.int32))

        return_dict: Dict[str, TensorMap] = {}
        # Enumerate ModuleDict so TorchScript can compile (no variable-key
        # indexing, and no ``continue`` inside the unrolled loop).
        for readout_name, readout in self.readouts.items():
            if readout_name in outputs:
                output = outputs[readout_name]
                atomic_values = readout(features)
                atomic_property = TensorMap(
                    self.key_labels[readout_name],
                    [
                        TensorBlock(
                            values=atomic_values,
                            samples=samples,
                            components=self.component_labels[readout_name][0],
                            properties=self.property_labels[readout_name][0],
                        )
                    ],
                )
                if selected_atoms is not None:
                    atomic_property = mts.slice(
                        atomic_property, axis="samples", selection=selected_atoms
                    )
                if output.sample_kind == "atom":
                    return_dict[readout_name] = atomic_property
                else:
                    return_dict[readout_name] = sum_over_atoms(atomic_property)

        if not self.training:
            return_dict = self.scaler.apply_scales(
                systems,
                return_dict,
                selected_atoms=selected_atoms,
                use_per_target_scales=True,
                use_per_property_scales=True,
            )
            for additive_model in self.additive_models:
                outputs_for_additive: Dict[str, ModelOutput] = {}
                for name, output in outputs.items():
                    if name in additive_model.outputs:
                        outputs_for_additive[name] = output
                additive_contributions = additive_model(
                    systems,
                    outputs_for_additive,
                    selected_atoms,
                )
                for name in additive_contributions:
                    return_dict[name] = mts.add(
                        return_dict[name], additive_contributions[name]
                    )

        return return_dict

    def restart(
        self,
        dataset_info: DatasetInfo,
        model_hypers: Optional[Dict[str, Any]] = None,
    ) -> "LOREM":
        if model_hypers is not None:
            raise_if_hypers_mismatch(
                self.hypers,
                model_hypers,
                default_hypers=get_default_hypers("experimental.lorem")["model"],
            )

        merged_info = self.dataset_info.union(dataset_info)
        new_atomic_types = [
            at for at in merged_info.atomic_types if at not in self.atomic_types
        ]
        new_targets = {
            key: value
            for key, value in merged_info.targets.items()
            if key not in self.dataset_info.targets
        }
        self.has_new_targets = len(new_targets) > 0

        if len(new_atomic_types) > 0:
            raise ValueError(
                f"New atomic types found in the dataset: {new_atomic_types}. "
                "The LOREM model does not support adding new atomic types."
            )

        for target_name, target in new_targets.items():
            self._add_output(target_name, target)

        self.dataset_info = merged_info
        self.additive_models[0].restart(
            dataset_info=DatasetInfo(
                length_unit=dataset_info.length_unit,
                atomic_types=self.atomic_types,
                targets={
                    target_name: target_info
                    for target_name, target_info in dataset_info.targets.items()
                    if CompositionModel.is_valid_target(target_name, target_info)
                },
            ),
        )
        self.scaler.restart(dataset_info)
        return self

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> "LOREM":
        model_data = checkpoint["model_data"]

        if context == "restart":
            logging.info(f"Using latest model from epoch {checkpoint['epoch']}")
            model_state_dict = checkpoint["model_state_dict"]
        elif context in {"finetune", "export"}:
            logging.info(f"Using best model from epoch {checkpoint['best_epoch']}")
            model_state_dict = checkpoint["best_model_state_dict"]
            if model_state_dict is None:
                model_state_dict = checkpoint["model_state_dict"]
        else:
            raise ValueError("Unknown context tag for checkpoint loading!")

        model = cls(
            hypers=model_data["model_hypers"],
            dataset_info=model_data["dataset_info"],
        )
        dtype = next(p.dtype for p in model.parameters() if p.is_floating_point())
        model.to(dtype).load_state_dict(model_state_dict)
        model.additive_models[0].sync_tensor_maps()
        model.scaler.sync_tensor_maps()

        metadata = checkpoint.get("metadata", None)
        if metadata is not None:
            model.metadata = merge_metadata(model.metadata, metadata)

        return model

    def export(self, metadata: Optional[ModelMetadata] = None) -> AtomisticModel:
        dtype = next(self.parameters()).dtype
        if dtype not in self.__supported_dtypes__:
            raise ValueError(f"unsupported dtype {dtype} for LOREM")

        self.to(dtype)
        self.additive_models[0].weights_to(torch.device("cpu"), torch.float64)

        interaction_ranges = [self.hypers["cutoff"]]
        for additive_model in self.additive_models:
            if hasattr(additive_model, "cutoff_radius"):
                interaction_ranges.append(additive_model.cutoff_radius)
        if self.long_range:
            interaction_ranges.append(torch.inf)
        interaction_range = max(interaction_ranges)

        capabilities = ModelCapabilities(
            outputs=self.outputs,
            atomic_types=self.atomic_types,
            interaction_range=interaction_range,
            length_unit=self.dataset_info.length_unit,
            supported_devices=self.__supported_devices__,
            dtype=dtype_to_str(dtype),
        )
        if metadata is None:
            metadata = self.__default_metadata__
        else:
            metadata = merge_metadata(self.__default_metadata__, metadata)

        return AtomisticModel(self.eval(), metadata, capabilities)

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict) -> Dict:
        for v in range(1, cls.__checkpoint_version__):
            if checkpoint["model_ckpt_version"] == v:
                update = getattr(checkpoints, f"model_update_v{v}_v{v + 1}")
                update(checkpoint)
                checkpoint["model_ckpt_version"] = v + 1

        if checkpoint["model_ckpt_version"] != cls.__checkpoint_version__:
            raise RuntimeError(
                f"Unable to upgrade the checkpoint: the checkpoint is using model "
                f"version {checkpoint['model_ckpt_version']}, while the current model "
                f"version is {cls.__checkpoint_version__}."
            )
        return checkpoint

    def get_checkpoint(self) -> Dict[str, Any]:
        return {
            "architecture_name": "experimental.lorem",
            "model_ckpt_version": self.__checkpoint_version__,
            "metadata": self.metadata,
            "model_data": {
                "model_hypers": dict(self.hypers),
                "dataset_info": self.dataset_info,
            },
            "epoch": None,
            "best_epoch": None,
            "model_state_dict": self.state_dict(),
            "best_model_state_dict": None,
        }
