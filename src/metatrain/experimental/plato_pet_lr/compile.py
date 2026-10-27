"""``torch.compile`` of the Plato-PET-LR training step (the ``compile`` hyper).

The force loss needs the second derivative, and ``torch.compile`` graphs cannot be
differentiated twice. As in SPACE (and NequIP), the energy *and* its position
gradient are traced together into one graph with ``make_fx`` (the backward pass of
the forces becomes ordinary forward ops), and that graph is compiled; the loss
backward is then a first-order backward through the compiled graph.

Everything that is data dependent stays outside the graph, in eager mode: the
neighbour-list layout (``compute_batch_tensors``), the long-range plan (mesh size,
k-set) and the token layout. The graph is specialised on the number of structures,
the mesh size and whether the batch has one cell or one per structure; atoms, edges,
neighbours, padded size and k vectors are symbolic. A batch with another layout
(mixed mesh sizes, mixed periodic and open structures, a stress target, ...) is
evaluated eagerly, the same way as without ``compile``.
"""

import contextlib
import itertools
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from metatrain.pet.modules.nef import get_nef_indices
from metatrain.pet.modules.structures import (
    compute_batch_tensors,
    concatenate_structures,
)
from metatrain.utils.data import TargetInfo
from metatrain.utils.evaluate_model import _position_gradients_to_block

from .model import Plan, PlatoPETLR, token_layout


@contextlib.contextmanager
def _disable_fx_duck_shape():
    """Equal sizes stay distinct symbols (as in SPACE's ``compile_model``)."""
    init = torch.fx.experimental._config.use_duck_shape
    torch.fx.experimental._config.use_duck_shape = False
    try:
        yield
    finally:
        torch.fx.experimental._config.use_duck_shape = init


class CompiledTrainingStep:
    """Energies and position gradients of a batch from a compiled graph.

    :param model: the model, already on its training device and dtype (the traced
        graphs reference its parameters).
    :param backend: ``torch.compile`` backend (``"eager"`` checks the trace alone).
    :param options: inductor options. The default turns off the loop reordering
        after fusion: with it, inductor (torch 2.13, CUDA) fails on this graph with
        ``CantSplit: s0*(64*s1 + 64) not divisible by s0*s1 + s0``.
    """

    def __init__(
        self,
        model: PlatoPETLR,
        backend: str = "inductor",
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.backend = backend
        self.options = (
            {"loop_ordering_after_fusion": False} if options is None else options
        )
        self.graphs: Dict[Tuple, Callable] = {}
        self.n_eager = 0
        self.n_compiled = 0

    def _prepare(self, systems: List[System]) -> Optional[Tuple[Tuple, Dict[str, Any]]]:
        m = self.model
        (positions, centers, neighbors, species, cells, cell_shifts, system_indices,
         samples) = concatenate_structures(systems, m.requested_nl)  # fmt: skip
        batch = compute_batch_tensors(
            positions, centers, neighbors, species, cells, cell_shifts,
            system_indices, m.species_to_species_index, m.cutoff, "cosine", 0.5,
        )  # fmt: skip
        sp_center, sp_nbr, mask, rev = batch[0], batch[1], batch[4], batch[5]
        ctr, nbr, shifts = batch[8], batch[9], batch[11]
        A, K = mask.shape
        num_neighbors = torch.zeros(A, dtype=ctr.dtype, device=ctr.device)
        num_neighbors.scatter_add_(0, ctr, torch.ones_like(ctr))
        nef_indices, _, _ = get_nef_indices(ctr, num_neighbors, K)
        sp_nbr1, mask1, rev1 = token_layout(sp_center, sp_nbr, mask, rev)
        S = len(systems)
        periodic: List[bool] = [False] * S
        empty: Plan = ([], [], [], [])
        mesh_plan, les_plan = empty, empty
        meshes: List[List[int]] = []
        kvecs: List[torch.Tensor] = []
        les_meshes: List[List[int]] = []
        if m.use_les or m.use_mesh:
            periodic, mesh_plan, meshes, les_plan, kvecs, les_meshes = (
                m._long_range_plan(cells, systems)
            )
        if any(len(x) > 0 for x in les_meshes):  # LES by PME: eager
            return None
        n_periodic = sum(periodic)
        if n_periodic not in (0, S):
            return None
        plans = (mesh_plan, les_plan)
        shared = all(len(u) == 1 for p in plans for u in p[1])
        if not shared and not all(
            len(u) == len(i) for p in plans for u, i in zip(p[1], p[0], strict=True)
        ):
            return None
        if shared and (len(mesh_plan[0]) > 1 or len(les_plan[0]) > 1):
            return None
        # Canonical mesh groups: rotations of one box give one of two sizes per
        # axis (30 or 31 points for the 30 A bio_dimers box), so with several mesh
        # sizes every combination lo..lo+1 of each axis gets a group, empty or not,
        # and every batch of the dataset shares one graph.
        by_mesh = {tuple(m): ids for m, ids in zip(meshes, mesh_plan[0], strict=True)}
        canonical = sorted(by_mesh)
        if len(by_mesh) > 1:
            lo = [min(m[a] for m in by_mesh) for a in range(3)]
            hi = [max(m[a] for m in by_mesh) for a in range(3)]
            if all(h - b <= 1 for b, h in zip(lo, hi, strict=True)):
                canonical = list(itertools.product(*[(b, b + 1) for b in lo]))
        lorder = sorted(range(len(kvecs)), key=lambda g: les_plan[0][g][0])

        # The empty row S pads a group of one structure (a size of 1 is not
        # symbolic in the trace, and inductor mis-compiled the mesh spread of a
        # one-structure group) and fills an empty group; it has no atoms and adds 0.
        def members(ids: List[int]) -> torch.Tensor:
            ids = ids + [S] * max(0, 2 - len(ids))
            return torch.tensor(ids, device=cells.device)

        index = {
            f"mesh{j}": members(by_mesh.get(m, [])) for j, m in enumerate(canonical)
        }
        index.update({f"les{j}": members(les_plan[0][g]) for j, g in enumerate(lorder)})
        kv = {f"kvec{j}": kvecs[g] for j, g in enumerate(lorder)}
        sizes = [len(s) for s in systems]
        P = max(sizes)
        key = (
            S,
            n_periodic == S,
            tuple(canonical),
            len(lorder),
            shared,
            positions.dtype,
            positions.device,
        )
        inputs = {
            "positions": positions.detach().requires_grad_(True),
            "cells": torch.cat([cells, cells[:1]]).detach(),  # + the empty row
            "ctr": ctr,
            "nbr": nbr,
            "shifts": shifts.to(cells.dtype),
            "edge_system": system_indices[ctr],
            "nef_indices": nef_indices,
            "sp_center": sp_center,
            "sp_nbr1": sp_nbr1,
            "mask1": mask1,
            "rev1": rev1,
            "system_atom": samples.values.to(torch.long),
            "pad_like": torch.zeros(S, P, device=cells.device),
            **index,
            **kv,
        }
        plan = (periodic, [list(m) for m in canonical], len(lorder), shared)
        return key, {"inputs": inputs, "plan": plan, "sizes": sizes}

    def _function(self, S: int, plan) -> Callable:
        periodic, meshes, n_les, shared = plan
        n_mesh = len(meshes)
        model = self.model
        uniq = [0] if shared else [0, 1]  # one cell, or one per structure

        def traced_plan(index: List[torch.Tensor]) -> Plan:
            n = len(index)
            return ([[0]] * n, [uniq] * n, [[0]] * n, index)

        def energy_and_gradient(
            positions, cells, ctr, nbr, shifts, edge_system, nef_indices, sp_center,
            sp_nbr1, mask1, rev1, system_atom, pad_like, *groups,
        ) -> Tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
            mesh_index = list(groups[:n_mesh])
            les_index = list(groups[n_mesh : n_mesh + n_les])
            kvecs = list(groups[n_mesh + n_les :])
            A = positions.shape[0]

            def energies(positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                # the edge vectors as in compute_batch_tensors, then the NEF layout
                if S == 1:
                    cell_contributions = shifts @ cells[0]
                else:
                    cell_contributions = torch.einsum(
                        "ab, abc -> ac", shifts, cells[edge_system]
                    )
                edges = positions[nbr] - positions[ctr] + cell_contributions
                r = edges[nef_indices]
                r = torch.cat([r.new_zeros(A, 1, 3), r], 1)
                e = model.atomic_energies(
                    positions, cells, periodic + [True], [1] * (S + 1), sp_center,
                    sp_nbr1, r,
                    mask1, rev1, system_atom, traced_plan(mesh_index), meshes,
                    traced_plan(les_index), kvecs, pad_to=pad_like.shape[1],
                )  # fmt: skip
                E = torch.zeros(S, dtype=e.dtype, device=e.device).index_add(
                    0, system_atom[:, 0], e
                )
                return E.sum(), E

            # torch.func, not autograd.grad: make_fx records the backward of ops
            # that save their output (softmax, exp) with a detached copy, which
            # loses the second derivative (the parameter gradient of the forces)
            g, E = torch.func.grad(energies, has_aux=True)(positions)
            return E, g

        return energy_and_gradient

    def _build(self, key: Tuple, prep: Dict[str, Any]) -> Callable:
        from torch.fx.experimental.proxy_tensor import make_fx

        logging.info(f"Plato-PET-LR: tracing and compiling the training graph {key}")
        fn = self._function(key[0], prep["plan"])
        args = list(prep["inputs"].values())
        with _disable_fx_duck_shape():
            fx = make_fx(
                fn,
                tracing_mode="symbolic",
                _allow_non_fake_inputs=True,
                _error_on_data_dependent_ops=True,
            )(*args)
        # dynamo cannot trace aten.sym_numel (make_fx emits it for the sizes of 1-D
        # inputs, here the group index tensors): use sym_size(x, 0)
        for node in fx.graph.nodes:
            if node.target is torch.ops.aten.sym_numel.default:
                x = node.args[0]
                if x.meta["val"].dim() == 1:
                    node.target = torch.ops.aten.sym_size.int
                    node.args = (x, 0)
        # dead size queries on tensors of the trace-time grad transform
        fx.graph.eliminate_dead_code()
        used = {n.target for n in fx.graph.nodes if n.op == "get_attr"}
        for name, _ in list(fx.named_buffers()):
            if name not in used:
                delattr(fx, name)
        fx.recompile()
        return torch.compile(
            fx,
            dynamic=True,
            fullgraph=True,
            backend=self.backend,
            options=self.options if self.backend == "inductor" else None,
        )

    def __call__(
        self, systems: List[System], targets: Dict[str, TargetInfo]
    ) -> Optional[Dict[str, TensorMap]]:
        """Predictions as ``evaluate_model(..., is_training=True)``, or None if this
        batch must be evaluated eagerly."""
        m = self.model
        if not m.training:
            return None
        for info in targets.values():
            if info.quantity != "energy" or set(info.gradients) - {"positions"}:
                self.n_eager += 1
                return None
        prepared = self._prepare(systems)
        if prepared is None:
            self.n_eager += 1
            return None
        key, prep = prepared
        if key not in self.graphs:
            self.graphs[key] = self._build(key, prep)
        with _disable_fx_duck_shape():  # equal sizes in one batch: no recompiles
            E, G = self.graphs[key](*prep["inputs"].values())
        self.n_compiled += 1
        S = len(systems)
        device = E.device
        samples = Labels(["system"], torch.arange(S, device=device).reshape(-1, 1))
        sizes = prep["sizes"]
        grads = list(torch.split(G, sizes))
        out: Dict[str, TensorMap] = {}
        for name, info in targets.items():
            layout = m.layouts[name].to(device)
            lb = layout.block(0)
            block = TensorBlock(
                values=E[:, None],
                samples=samples,
                components=lb.components,
                properties=lb.properties,
            )
            if "positions" in info.gradients:
                block.add_gradient("positions", _position_gradients_to_block(grads))
            out[name] = TensorMap(keys=layout.keys, blocks=[block])
        return out
