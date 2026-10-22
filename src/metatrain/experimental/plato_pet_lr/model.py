import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

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
from metatrain.pet.modules.structures import (
    compute_batch_tensors,
    concatenate_structures,
)
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

from .documentation import ModelHypers
from .modules.ewald import les_energy
from .modules.frames import platonic_group, relative_rotations, sample_frames
from .modules.layers import EdgeEmbed, FrameMLP, MeshCoupling, PETLayer, cutoff_fn


class PlatoPETLR(ModelInterface[ModelHypers]):
    __checkpoint_version__ = 1
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float32, torch.float64]
    __default_metadata__ = ModelMetadata(
        references={"architecture": ["PET: https://arxiv.org/abs/2305.19302"]}
    )

    layouts: Dict[str, TensorMap]

    def __init__(self, hypers: ModelHypers, dataset_info: DatasetInfo) -> None:
        super().__init__(hypers, dataset_info, self.__default_metadata__)
        check_no_atom_pair_targets(dataset_info.targets, self.__class__.__name__)
        h = self.hypers
        self.atomic_types = dataset_info.atomic_types
        self.has_new_targets = False
        self.cutoff = float(h["cutoff"])
        self.requested_nl = NeighborListOptions(
            cutoff=self.cutoff, full_list=True, strict=True
        )
        d, heads, L = int(h["d_model"]), int(h["num_heads"]), int(h["num_layers"])
        if (d // heads) % 2 != 0 or d % heads != 0:
            raise ValueError("d_model / num_heads must be an even integer (RoPE)")
        self.num_layers = L
        nf = h["num_framed_layers"]
        self.n_framed = L if nf is None else int(nf)
        self.couple_at = int(h["couple_at"])
        self.use_mesh, self.use_les = bool(h["mesh_coupling"]), bool(h["les"])
        if self.use_mesh and not 0 <= self.couple_at < L:
            raise ValueError(f"couple_at must be in [0, {L})")
        self.use_rope = bool(h["rope"])
        self.frame_sampling = str(h["frame_sampling"])
        self.les_sigma, self.les_kcut = float(h["les_sigma"]), float(h["les_kcut"])
        nb, hid, mode = (
            int(h["n_basis"]),
            int(h["kernel_hidden"]),
            str(h["frame_linear_mode"]),
        )

        mapping = torch.full((max(self.atomic_types) + 1,), -1, dtype=torch.long)
        for i, z in enumerate(self.atomic_types):
            mapping[z] = i
        self.register_buffer("species_to_species_index", mapping)

        dt = torch.get_default_dtype()  # buffers follow the parameters
        G = platonic_group(str(h["frames"])).to(dt)
        I3 = platonic_group("identity").to(dt)
        self.register_buffer("G_R", G)
        self.register_buffer("G_rel", relative_rotations(G))
        self.register_buffer(
            "G_w", torch.full((G.shape[0],), 1.0 / G.shape[0], dtype=G.dtype)
        )
        self.register_buffer("I_rel", relative_rotations(I3))
        self.register_buffer("I_w", torch.ones(1, dtype=G.dtype))
        gen = torch.Generator().manual_seed(int(h["seed"]))
        half = (d // heads) // 2
        dirs = torch.nn.functional.normalize(
            torch.randn(half, 3, generator=gen), dim=-1
        )
        self.register_buffer(
            "omega",
            dirs * torch.linspace(0.3, float(h["rope_max_freq"]), half)[:, None],
        )

        self.embed = EdgeEmbed(
            len(self.atomic_types), d, int(h["num_rbf"]), self.cutoff
        )
        self.layers = torch.nn.ModuleList(
            [PETLayer(d, heads, nb, hid, mode) for _ in range(L)]
        )
        self.coupling = MeshCoupling(
            d, int(h["mesh_channels"]), float(h["mesh_spacing"]), nb, hid, mode,
            sigma=float(h["mesh_sigma"]),
        )  # fmt: skip
        self.energy_head = FrameMLP(d, 1, nb, hid, mode)
        self.charge_head = FrameMLP(d, int(h["num_charges"]), nb, hid, mode)

        self.outputs: Dict[str, ModelOutput] = {}
        self.layouts = {}
        for name, target in dataset_info.targets.items():
            self._add_output(name, target)
        self.additive_models = torch.nn.ModuleList(
            [CompositionModel.from_valid_targets(dataset_info, self.atomic_types)]
        )
        self.scaler = Scaler(
            hypers=get_default_hypers("scaler")["model"], dataset_info=dataset_info
        )

    def _add_output(self, name: str, target: TargetInfo) -> None:
        if not (target.is_scalar and len(target.layout.block().properties) == 1):
            raise ValueError(
                f"Plato-PET-LR predicts one scalar (energy), not target '{name}'"
            )
        self.layouts[name] = target.layout
        self.outputs[name] = ModelOutput(
            unit=target.unit, sample_kind="atom", description=target.description
        )

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [self.requested_nl]

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs

    @torch.jit.unused
    def _random_frames(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return sample_frames(self.frame_sampling, self.G_R)

    def frames(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training and self.frame_sampling != "fixed":
            if not torch.jit.is_scripting():
                return self._random_frames()
        return self.G_R, self.G_rel, self.G_w

    def atomic_energies(
        self,
        positions: torch.Tensor,
        cells: torch.Tensor,
        periodic: List[bool],
        sizes: List[int],
        sp_center: torch.Tensor,
        sp_nbr: torch.Tensor,
        r: torch.Tensor,
        mask: torch.Tensor,
        rev: torch.Tensor,
    ) -> torch.Tensor:
        """Per-atom energies from the padded token layout (slot 0 = centre)."""
        R, rel, w = self.frames()
        d = (r * r).sum(-1).clamp_min(1e-12).sqrt()
        fc = cutoff_fn(d, self.cutoff) * mask.to(r.dtype)
        bias = torch.where(
            fc > 0, torch.log(fc.clamp_min(1e-30)), torch.full_like(fc, -1e9)
        )[:, None, None, :]
        ang: Optional[torch.Tensor] = None
        if self.use_rope:
            Rw = torch.einsum("fab,kb->fka", R, self.omega.to(r.dtype))
            ang = torch.einsum("akx,fcx->akfc", r, Rw)[:, :, :, None, :]
        manual = self.training
        tok = self.embed(r, d, sp_center, sp_nbr, R)
        for i, layer in enumerate(self.layers):
            if i == self.n_framed:  # average over frames, then one frame, no RoPE
                tok = (tok * w[:, None]).sum(-2, keepdim=True)
                rel, w, ang = self.I_rel, self.I_w, None
            tok = layer(tok, rel, w, ang, bias, rev, manual)
            if i == self.couple_at and self.use_mesh:
                deltas: List[torch.Tensor] = []
                start = 0
                for s, n in enumerate(sizes):
                    x = tok[start : start + n, 0]
                    if periodic[s]:
                        pos = positions[start : start + n]
                        deltas.append(self.coupling(x, pos, cells[s], rel, w))
                    else:
                        deltas.append(torch.zeros_like(x))
                    start += n
                tok = torch.cat(
                    [tok[:, :1] + torch.cat(deltas)[:, None], tok[:, 1:]], 1
                )
        h = tok[:, 0] + (fc[:, 1:, None, None] * tok[:, 1:]).sum(1)
        e = (self.energy_head(h, rel, w)[..., 0] * w).sum(-1)
        if self.use_les:
            q_all = (self.charge_head(h, rel, w) * w[:, None]).sum(-2)
            e_lr: List[torch.Tensor] = []
            start = 0
            for s, n in enumerate(sizes):
                q = q_all[start : start + n]
                q = q - q.mean(0, keepdim=True)
                pos = positions[start : start + n]
                e_lr.append(
                    les_energy(
                        q, pos, cells[s], periodic[s], self.les_sigma, self.les_kcut
                    )
                )
                start += n
            e = e + torch.cat(e_lr)
        return e

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if len(outputs) == 0:
            return {}
        device = systems[0].positions.device
        self.layouts = {k: v.to(device) for k, v in self.layouts.items()}

        (positions, centers, neighbors, species, cells, cell_shifts, system_indices,
         samples) = concatenate_structures(systems, self.requested_nl)  # fmt: skip
        batch = compute_batch_tensors(
            positions, centers, neighbors, species, cells, cell_shifts,
            system_indices, self.species_to_species_index, self.cutoff, "cosine", 0.5,
        )  # fmt: skip
        sp_center, sp_nbr, edge_vectors, mask, rev = (
            batch[0],
            batch[1],
            batch[2],
            batch[4],
            batch[5],
        )
        A, K = mask.shape
        K1 = K + 1
        # prepend the centre as slot 0; reversed-edge index into (A * K1)
        slots = torch.arange(A * K1, device=device).reshape(A, K1)
        rev1 = (
            torch.div(rev, max(K, 1), rounding_mode="floor") * K1 + rev % max(K, 1) + 1
        )
        rev1 = torch.cat([slots[:, :1], torch.where(mask, rev1, slots[:, 1:])], 1)
        r = torch.cat([edge_vectors.new_zeros(A, 1, 3), edge_vectors], 1)
        mask1 = torch.cat([torch.ones(A, 1, dtype=torch.bool, device=device), mask], 1)
        sp_nbr1 = torch.cat([sp_center[:, None], torch.where(mask, sp_nbr, 0)], 1)

        periodic: List[bool] = []
        for system in systems:
            p = system.pbc
            if bool(p.any()) and not bool(p.all()) and (self.use_les or self.use_mesh):
                raise ValueError("Plato-PET-LR long range needs full or no PBC")
            periodic.append(bool(p.all()))
        sizes = [len(system) for system in systems]

        e = self.atomic_energies(
            positions, cells, periodic, sizes, sp_center, sp_nbr1, r, mask1,
            rev1.reshape(-1),
        )  # fmt: skip

        return_dict: Dict[str, TensorMap] = {}
        for name, output in outputs.items():
            block = self.layouts[name].block(0)
            tmap = TensorMap(
                keys=self.layouts[name].keys,
                blocks=[
                    TensorBlock(
                        values=e[:, None],
                        samples=samples,
                        components=block.components,
                        properties=block.properties,
                    )
                ],
            )
            if selected_atoms is not None:
                tmap = mts.slice(tmap, axis="samples", selection=selected_atoms)
            return_dict[name] = (
                tmap if output.sample_kind == "atom" else sum_over_atoms(tmap)
            )

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
                contributions = additive_model(
                    systems, outputs_for_additive, selected_atoms
                )
                for name in contributions:
                    return_dict[name] = mts.add(return_dict[name], contributions[name])
        return return_dict

    def restart(
        self, dataset_info: DatasetInfo, model_hypers: Optional[Dict[str, Any]] = None
    ) -> "PlatoPETLR":
        if model_hypers is not None:
            raise_if_hypers_mismatch(
                self.hypers,
                model_hypers,
                default_hypers=get_default_hypers("experimental.plato_pet_lr")["model"],
            )
        merged = self.dataset_info.union(dataset_info)
        new_types = [t for t in merged.atomic_types if t not in self.atomic_types]
        if new_types:
            raise ValueError(f"New atomic types found in the dataset: {new_types}")
        new_targets = {
            k: v
            for k, v in merged.targets.items()
            if k not in self.dataset_info.targets
        }
        self.has_new_targets = len(new_targets) > 0
        for name, target in new_targets.items():
            self._add_output(name, target)
        self.dataset_info = merged
        self.additive_models[0].restart(
            dataset_info=DatasetInfo(
                length_unit=dataset_info.length_unit,
                atomic_types=self.atomic_types,
                targets={
                    k: v
                    for k, v in dataset_info.targets.items()
                    if CompositionModel.is_valid_target(k, v)
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
    ) -> "PlatoPETLR":
        if context == "restart":
            logging.info(f"Using latest model from epoch {checkpoint['epoch']}")
            state = checkpoint["model_state_dict"]
        elif context in {"finetune", "export"}:
            logging.info(f"Using best model from epoch {checkpoint['best_epoch']}")
            state = (
                checkpoint["best_model_state_dict"] or checkpoint["model_state_dict"]
            )
        else:
            raise ValueError("Unknown context tag for checkpoint loading!")
        data = checkpoint["model_data"]
        model = cls(hypers=data["model_hypers"], dataset_info=data["dataset_info"])
        dtype = next(p.dtype for p in model.parameters() if p.is_floating_point())
        model.to(dtype).load_state_dict(state)
        model.additive_models[0].sync_tensor_maps()
        model.scaler.sync_tensor_maps()
        if checkpoint.get("metadata") is not None:
            model.metadata = merge_metadata(model.metadata, checkpoint["metadata"])
        return model

    def export(self, metadata: Optional[ModelMetadata] = None) -> AtomisticModel:
        dtype = next(self.parameters()).dtype
        if dtype not in self.__supported_dtypes__:
            raise ValueError(f"unsupported dtype {dtype} for Plato-PET-LR")
        self.to(dtype)
        self.additive_models[0].weights_to(torch.device("cpu"), torch.float64)
        long_range = self.use_les or self.use_mesh
        capabilities = ModelCapabilities(
            outputs=self.outputs,
            atomic_types=self.atomic_types,
            interaction_range=torch.inf
            if long_range
            else self.num_layers * self.cutoff,
            length_unit=self.dataset_info.length_unit,
            supported_devices=self.__supported_devices__,
            dtype=dtype_to_str(dtype),
        )
        metadata = (
            self.__default_metadata__
            if metadata is None
            else merge_metadata(self.__default_metadata__, metadata)
        )
        return AtomisticModel(self.eval(), metadata, capabilities)

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict) -> Dict:
        if checkpoint["model_ckpt_version"] != cls.__checkpoint_version__:
            raise RuntimeError(
                "Unable to upgrade the checkpoint: the checkpoint is using model "
                f"version {checkpoint['model_ckpt_version']}, while the current model "
                f"version is {cls.__checkpoint_version__}."
            )
        return checkpoint

    def get_checkpoint(self) -> Dict[str, Any]:
        return {
            "architecture_name": "experimental.plato_pet_lr",
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
