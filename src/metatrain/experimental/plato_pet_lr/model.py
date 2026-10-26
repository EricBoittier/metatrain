import logging
import math
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

from . import checkpoints
from .documentation import ModelHypers
from .modules.electronegativity import Electronegativity
from .modules.ewald import (
    kset_host,
    les_energy_finite_batched,
    les_energy_periodic_batched,
)
from .modules.frames import platonic_group, relative_rotations, sample_frames
from .modules.layers import (
    EdgeEmbed,
    FrameLinear,
    FrameMLP,
    MeshCoupling,
    PETLayer,
    cutoff_fn,
    rope_cs,
)


def _pad(x: torch.Tensor, slot: Optional[torch.Tensor], S: int, P: int) -> torch.Tensor:
    """``(A, ...)`` atoms to the ``(S, P, ...)`` padded layout (zeros)."""
    shape = [S, P] + list(x.shape[1:])
    if slot is None:
        return x.reshape(shape)
    out = torch.zeros([S * P] + list(x.shape[1:]), dtype=x.dtype, device=x.device)
    return out.index_copy(0, slot, x).reshape(shape)


def _unpad(x: torch.Tensor, slot: Optional[torch.Tensor]) -> torch.Tensor:
    flat = x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))
    return flat if slot is None else flat.index_select(0, slot)


Plan = Tuple[List[List[int]], List[List[int]], List[List[int]], List[torch.Tensor]]
"""Groups of structures: ``(ids, distinct-cell ids, structure -> distinct cell,
index tensors)``. The index tensors are empty (the lists are used) except in the
traced training graph, where the membership varies from batch to batch."""


def _groups(
    keys: List[List[int]], periodic: List[bool], cells: List[List[float]], grad: bool
) -> Tuple[Plan, List[List[int]], List[List[List[float]]]]:
    """Periodic structures grouped by ``keys``; per group the distinct cells (each
    structure its own with a cell gradient, the stress). Also returns the keys and
    the distinct cells of each group."""
    order: Dict[str, int] = {}
    ids: List[List[int]] = []
    uniq: List[List[int]] = []
    umaps: List[List[int]] = []
    gkeys: List[List[int]] = []
    seen: List[List[List[float]]] = []
    for s in range(len(keys)):
        if not periodic[s]:
            continue
        key = str(keys[s])
        if key not in order:
            order[key] = len(ids)
            ids.append(torch.jit.annotate(List[int], []))
            uniq.append(torch.jit.annotate(List[int], []))
            umaps.append(torch.jit.annotate(List[int], []))
            gkeys.append(keys[s])
            seen.append(torch.jit.annotate(List[List[float]], []))
        g = order[key]
        ids[g].append(s)
        j = -1 if grad else _index_of(seen[g], cells[s])
        if j < 0:
            j = len(seen[g])
            seen[g].append(cells[s])
            uniq[g].append(s)
        umaps[g].append(j)
    plan: Plan = (ids, uniq, umaps, torch.jit.annotate(List[torch.Tensor], []))
    return plan, gkeys, seen


def _empty_plan() -> Plan:
    return (
        torch.jit.annotate(List[List[int]], []),
        torch.jit.annotate(List[List[int]], []),
        torch.jit.annotate(List[List[int]], []),
        torch.jit.annotate(List[torch.Tensor], []),
    )


def _group_index(plan: Plan, g: int) -> Optional[torch.Tensor]:
    return plan[3][g] if len(plan[3]) > 0 else None


def _rows(
    x: torch.Tensor, ids: List[int], S: int, index: Optional[torch.Tensor]
) -> torch.Tensor:
    if index is not None:
        return x.index_select(0, index)
    if len(ids) == S:
        return x
    if ids[-1] - ids[0] + 1 == len(ids):
        return x[ids[0] : ids[-1] + 1]
    return x.index_select(0, torch.tensor(ids, device=x.device))


def _put_rows(
    out: torch.Tensor, x: torch.Tensor, ids: List[int], S: int,
    index: Optional[torch.Tensor],
) -> torch.Tensor:  # fmt: skip
    if index is not None:
        return out.index_add(0, index, x)
    if len(ids) == S:
        return out + x
    return out.index_add(0, torch.tensor(ids, device=x.device), x)


def _group_cells(
    cells: torch.Tensor, plan: Plan, g: int, index: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """The ``(U, 3, 3)`` cells of group ``g`` and the structure -> cell map (None:
    one cell for the group, or one per structure)."""
    ids, uniq, umap = plan[0][g], plan[1][g], plan[2][g]
    if len(uniq) == 1:
        return cells[uniq[0] : uniq[0] + 1], None
    if index is not None:  # traced graph: one cell per structure
        return cells.index_select(0, index), None
    cg = _rows(cells, uniq, cells.shape[0], None)
    if len(uniq) == len(ids):
        return cg, None
    return cg, torch.tensor(umap, device=cells.device)


def token_layout(
    sp_center: torch.Tensor, sp_nbr: torch.Tensor, mask: torch.Tensor, rev: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepend the centre as slot 0 to the NEF tensors: neighbour species, mask and
    the reversed-edge index into ``(A * (K + 1))`` (flattened)."""
    A, K = mask.shape
    K1 = K + 1
    device = mask.device
    slots = torch.arange(A * K1, device=device).reshape(A, K1)
    rev1 = torch.div(rev, max(K, 1), rounding_mode="floor") * K1 + rev % max(K, 1) + 1
    rev1 = torch.cat([slots[:, :1], torch.where(mask, rev1, slots[:, 1:])], 1)
    mask1 = torch.cat([torch.ones(A, 1, dtype=torch.bool, device=device), mask], 1)
    sp_nbr1 = torch.cat([sp_center[:, None], torch.where(mask, sp_nbr, 0)], 1)
    return sp_nbr1, mask1, rev1.reshape(-1)


def _clear_caches(module: torch.nn.Module, *args: Any) -> None:
    for m in module.modules():
        if isinstance(m, FrameLinear):
            m.clear_cache()


def _index_of(rows: List[List[float]], row: List[float]) -> int:
    for i, r in enumerate(rows):
        if r == row:
            return i
    return -1


class PlatoPETLR(ModelInterface[ModelHypers]):
    __checkpoint_version__ = 4
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float32, torch.float64]
    __default_metadata__ = ModelMetadata(
        references={"architecture": ["PET: https://arxiv.org/abs/2305.19302"]}
    )

    layouts: Dict[str, TensorMap]
    _kcache: Dict[str, torch.Tensor]

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
        self.frame_mode = str(h["frame_mode"])
        if self.frame_mode not in ("full", "attention"):
            raise ValueError(f"unknown frame_mode '{self.frame_mode}'")
        if self.frame_mode == "attention" and not self.use_rope:
            raise ValueError("frame_mode 'attention' needs rope: true")
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
        chi_mode = str(h["electronegativity"])
        if chi_mode not in ("none", "linear"):
            raise ValueError(f"unknown electronegativity '{chi_mode}'")
        if chi_mode == "linear" and not self.use_les:
            raise ValueError("electronegativity 'linear' needs les: true")
        # empty unless 'linear', so 'none' models keep their state dict
        self.electronegativity = torch.nn.ModuleList(
            [
                Electronegativity(
                    self.atomic_types, d, nb, hid, mode,
                    chi0_trainable=bool(h["chi0_trainable"]),
                    dchi_bound=float(h["dchi_bound"]),
                )
            ]
            if chi_mode == "linear"
            else []
        )

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
        self._kcache = {}  # long-range k-sets per (nmax, cell metrics, device)
        # evaluation: Ewald sums over more than 4 x this many phase elements
        # (atoms x k vectors) run in chunks of it (bounded memory, first derivatives)
        self.les_chunk = 2**25
        self._register_load_state_dict_pre_hook(_clear_caches, with_module=True)

    @torch.jit.unused
    def clear_caches(self) -> None:
        """Drop the cached folded ``FrameLinear`` weights and k-sets."""
        _clear_caches(self)
        self._kcache = {}

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

    def _long_range_plan(
        self, cells: torch.Tensor, systems: List[System]
    ) -> Tuple[List[bool], Plan, List[List[int]], Plan, List[torch.Tensor]]:
        """Host-side grouping of the structures, from one device-to-host copy.

        Periodic structures are grouped by mesh size for the mesh coupling and by
        the LES integer range ``nmax`` for the Ewald sum (bio_dimers: the 30 A box
        gives a mesh of 30 or 31 points per axis under rotation, and one ``nmax``);
        a group runs as one batched call. Returns ``periodic`` per structure, the
        mesh plan and mesh sizes, the LES plan and the k-sets on the device (see
        :func:`_groups`)."""
        S = len(systems)
        lengths = torch.linalg.norm(cells.detach(), dim=-1)  # as mesh_size/kvectors
        pbc = torch.stack([system.pbc for system in systems]).to(cells.dtype)
        rows: List[List[float]] = (
            torch.cat([cells.detach().reshape(S, 9), lengths, pbc], 1)
            .to(device=torch.device("cpu"), dtype=torch.float64)
            .tolist()
        )
        kc = self.les_kcut / self.les_sigma
        periodic: List[bool] = []
        mkeys: List[List[int]] = []
        lkeys: List[List[int]] = []
        for s in range(S):
            row = rows[s]
            n_pbc = int(row[12] + row[13] + row[14])
            if n_pbc != 0 and n_pbc != 3:
                raise ValueError("Plato-PET-LR long range needs full or no PBC")
            periodic.append(n_pbc == 3)
            M: List[int] = []
            nmax: List[int] = []
            for x in row[9:12]:
                M.append(max(8, int(math.ceil(x / self.coupling.spacing))))
                nmax.append(int(math.floor(kc * x / (2 * math.pi))))
            mkeys.append(M)
            lkeys.append(nmax)
        cell_rows = [row[:9] for row in rows]
        grad = cells.requires_grad
        empty = _empty_plan()
        mesh_plan, meshes = empty, torch.jit.annotate(List[List[int]], [])
        if self.use_mesh:
            mesh_plan, meshes, _ = _groups(mkeys, periodic, cell_rows, grad)
        les_plan, kvecs = empty, torch.jit.annotate(List[torch.Tensor], [])
        if self.use_les:
            les_plan, nmaxs, seen = _groups(lkeys, periodic, cell_rows, grad)
            for g in range(len(nmaxs)):
                # metrics G = cell cell^T rounded: rotated copies share the k-set
                metric_keys: List[str] = []
                mcells: List[List[float]] = []
                for c in seen[g]:
                    cell = torch.tensor(c, dtype=torch.float64).reshape(3, 3)
                    G: List[float] = (cell @ cell.T).reshape(-1).tolist()
                    scale = max([abs(x) for x in G])
                    mk = str([int(math.floor(x / scale * 1e6 + 0.5)) for x in G])
                    if mk not in metric_keys:
                        metric_keys.append(mk)
                        mcells.append(c)
                ck = str(nmaxs[g]) + str(metric_keys) + str(cells.device)
                ck = ck + str(cells.dtype)
                if ck not in self._kcache:
                    if len(self._kcache) > 64:
                        self._kcache.clear()
                    n = kset_host(nmaxs[g], mcells, kc)
                    self._kcache[ck] = n.to(device=cells.device, dtype=cells.dtype)
                kvecs.append(self._kcache[ck])
        return periodic, mesh_plan, meshes, les_plan, kvecs

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
        system_atom: torch.Tensor,
        mesh_plan: Plan,
        meshes: List[List[int]],
        les_plan: Plan,
        kvecs: List[torch.Tensor],
        pad_to: int = 0,
    ) -> torch.Tensor:
        """Per-atom energies from the padded token layout (slot 0 = centre).

        ``system_atom`` ``(A, 2)``: structure and atom index. The long-range parts
        run on all structures at once in a ``(S, P)`` padded layout, per group of
        :meth:`_long_range_plan`; ``P = max(sizes)``, or ``pad_to`` if it is
        positive (then the padded layout is always used and only ``len(sizes)``
        matters: the traced training graph of ``compile``)."""
        R, rel, w = self.frames()
        d = (r * r).sum(-1).clamp_min(1e-12).sqrt()
        fc = cutoff_fn(d, self.cutoff) * mask.to(r.dtype)
        bias = torch.where(
            fc > 0, torch.log(fc.clamp_min(1e-30)), torch.full_like(fc, -1e9)
        )[:, None, None, :]
        cs: Optional[torch.Tensor] = None
        if self.use_rope:
            Rw = torch.einsum("fab,kb->fka", R, self.omega.to(r.dtype))
            ang = torch.einsum("akx,fcx->akfc", r, Rw)[:, :, :, None, :]
            cs = rope_cs(ang)
        manual = self.training
        frame_w: Optional[torch.Tensor] = None
        if self.frame_mode == "attention":  # frames only in RoPE, one token frame
            frame_w = w
            rel, w = self.I_rel, self.I_w
        S, A = len(sizes), positions.shape[0]
        P = max(sizes) if pad_to <= 0 else pad_to
        pad_slot: Optional[torch.Tensor] = None
        if A != S * P or pad_to > 0:  # (S, P) padded layout of the atoms
            pad_slot = system_atom[:, 0] * P + system_atom[:, 1]
        any_periodic = len(mesh_plan[0]) + len(les_plan[0]) > 0
        n_periodic = 0
        for p in periodic:
            n_periodic += int(p)
        all_periodic = n_periodic == S
        need_lr = self.use_les or (self.use_mesh and any_periodic)
        pos_pad = positions
        if need_lr:
            pos_pad = _pad(positions, pad_slot, S, P)
        tok = self.embed(r, d, sp_center, sp_nbr, R, frame_w)
        for i, layer in enumerate(self.layers):
            if i == self.n_framed:  # average over frames, then one frame, no RoPE
                tok = (tok * w[:, None]).sum(-2, keepdim=True)
                rel, w, cs = self.I_rel, self.I_w, None
            tok = layer(tok, rel, w, cs, bias, rev, manual, frame_w)
            if i == self.couple_at and self.use_mesh and any_periodic:
                x = tok[:, 0]
                F = x.shape[1]
                v = self.coupling.down(self.coupling.norm(x), rel, w).reshape(A, -1)
                v_pad = _pad(v, pad_slot, S, P)
                m_pad = torch.zeros_like(v_pad)
                for g in range(len(meshes)):
                    gi = _group_index(mesh_plan, g)
                    ids = mesh_plan[0][g]
                    cg, umap = _group_cells(cells, mesh_plan, g, gi)
                    m = self.coupling.mesh_batched(
                        _rows(v_pad, ids, S, gi), _rows(pos_pad, ids, S, gi),
                        cg, umap, meshes[g], F,
                    )  # fmt: skip
                    m_pad = _put_rows(m_pad, m, ids, S, gi)
                m = _unpad(m_pad, pad_slot).reshape(A, F, -1)
                delta = self.coupling.up(m, rel, w)
                if not all_periodic:
                    pa = torch.tensor(periodic, device=x.device)[system_atom[:, 0]]
                    delta = delta * pa.to(delta.dtype)[:, None, None]
                tok = torch.cat([tok[:, :1] + delta[:, None], tok[:, 1:]], 1)
        h = tok[:, 0] + (fc[:, 1:, None, None] * tok[:, 1:]).sum(1)
        e = (self.energy_head(h, rel, w)[..., 0] * w).sum(-1)
        if self.use_les:
            q_all = (self.charge_head(h, rel, w) * w[:, None]).sum(-2)
            ones = torch.ones_like(q_all[:, :1])
            q_pad = _pad(torch.cat([q_all, ones], 1), pad_slot, S, P)
            count = q_pad[:, :, -1:].sum(1).clamp_min(1.0)  # empty rows: compile
            mean = q_pad[:, :, :-1].sum(1) / count  # (S, ch)
            q_n = q_all - mean[system_atom[:, 0]]
            q_pad = _pad(q_n, pad_slot, S, P)
            e_pad = torch.zeros_like(q_pad[:, :, 0])
            for g in range(len(kvecs)):
                gi = _group_index(les_plan, g)
                ids = les_plan[0][g]
                cg, umap = _group_cells(cells, les_plan, g, gi)
                eg = les_energy_periodic_batched(
                    _rows(q_pad, ids, S, gi), _rows(pos_pad, ids, S, gi), cg, umap,
                    kvecs[g], self.les_sigma, self.les_kcut,
                    0 if self.training else self.les_chunk,
                )  # fmt: skip
                e_pad = _put_rows(e_pad, eg, ids, S, gi)
            open_ids: List[int] = []
            for s in range(S):
                if not periodic[s]:
                    open_ids.append(s)
            if len(open_ids) > 0:
                valid = _pad(ones[:, 0] > 0, pad_slot, S, P)
                eg = les_energy_finite_batched(
                    _rows(q_pad, open_ids, S, None),
                    _rows(pos_pad, open_ids, S, None),
                    _rows(valid, open_ids, S, None), self.les_sigma,
                )  # fmt: skip
                e_pad = _put_rows(e_pad, eg, open_ids, S, None)
            e = e + _unpad(e_pad, pad_slot)
            q0 = q_n[:, 0]
            for chi in self.electronegativity:
                e = e + chi(h, sp_center, q0, rel, w)
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
        sp_nbr1, mask1, rev1 = token_layout(sp_center, sp_nbr, mask, rev)
        A = mask.shape[0]
        r = torch.cat([edge_vectors.new_zeros(A, 1, 3), edge_vectors], 1)

        sizes = [len(system) for system in systems]
        periodic: List[bool] = [False] * len(systems)
        empty = _empty_plan()
        mesh_plan, les_plan = empty, empty
        meshes: List[List[int]] = []
        kvecs: List[torch.Tensor] = []
        if self.use_les or self.use_mesh:
            periodic, mesh_plan, meshes, les_plan, kvecs = self._long_range_plan(
                cells, systems
            )

        e = self.atomic_energies(
            positions, cells, periodic, sizes, sp_center, sp_nbr1, r, mask1, rev1,
            samples.values.to(torch.long), mesh_plan, meshes, les_plan, kvecs,
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
        self.clear_caches()  # computed again by the exported model
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
        for v in range(1, cls.__checkpoint_version__):
            if checkpoint["model_ckpt_version"] == v:
                getattr(checkpoints, f"model_update_v{v}_v{v + 1}")(checkpoint)
                checkpoint["model_ckpt_version"] = v + 1
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
