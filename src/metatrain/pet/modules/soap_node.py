"""SOAP-PET variant 1: the SOAP power spectrum on PET's node token.

``SoapNodeInjection`` adds, to the node features of the layers in ``layers``,

    x_i <- x_i + g_k * MLP_k(LayerNorm(p_i))

with ``p_i`` the atom's :class:`~metatrain.pet.modules.soap.SoapPowerSpectrum`. The last
Linear of every MLP starts at zero (weight and bias), so the model is the pretrained one
at initialisation, and the gates ``soap_node_gate`` (one per injection, start 1) scale the
branch. Setting them to 0 also recovers the base model exactly. Injection index 0 is the
input node embedding; index ``k >= 1`` is the output of GNN layer ``k - 1`` (after the
conditioning and mesh updates, as the conditioning embedding is added).

Configured by ``plato_lr: {soap_node: {...}}`` (see :func:`apply_soap_node`); the
descriptor is the backend's shared ``soap`` module, built here if absent.

The density of every edge is also weighted by PET's cutoff factor (the core's
``weight=``). Under PET-OMol-S's adaptive cutoff an edge leaves the
neighbour list at its pair cutoff, where the SOAP density is still nonzero, so the
unweighted p (and the energy) jumps there. The two neighbour lists (metatrain's and
mlip.cpp's) also differ only in edges with a zero factor.
"""

from typing import Any, Dict, List

import torch

from .soap import SoapPowerSpectrum


class SoapNodeInjection(torch.nn.Module):
    """Gated, zero-initialised MLP(LayerNorm(p)) added to the node features.

    :param d_soap: SOAP dimension (``SoapPowerSpectrum.out_dim``).
    :param d_node: PET's node feature width.
    :param layers: injection points (0: input embedding, k: after GNN layer k - 1).
    :param d_hidden: hidden width of each MLP.
    """

    def __init__(self, d_soap: int, d_node: int, layers: List[int], d_hidden: int = 256) -> None:
        super().__init__()
        self.layers: List[int] = [int(k) for k in layers]
        self.norm = torch.nn.LayerNorm(d_soap)
        self.mlps = torch.nn.ModuleList()
        for _ in self.layers:
            mlp = torch.nn.Sequential(
                torch.nn.Linear(d_soap, d_hidden), torch.nn.SiLU(), torch.nn.Linear(d_hidden, d_node)
            )
            torch.nn.init.zeros_(mlp[2].weight)
            torch.nn.init.zeros_(mlp[2].bias)
            self.mlps.append(mlp)
        self.soap_node_gate = torch.nn.Parameter(torch.ones(len(self.layers)))

    def features(self, p: torch.Tensor) -> torch.Tensor:
        """The normalised SOAP shared by every injection."""
        return self.norm(p)

    def update(self, x: torch.Tensor, pn: torch.Tensor, point: int) -> torch.Tensor:
        """``x`` plus injection ``point``'s term (``x`` unchanged where none is placed)."""
        for k, (layer, mlp) in enumerate(zip(self.layers, self.mlps)):
            if layer == point:
                x = x + self.soap_node_gate[k : k + 1] * mlp(pn)
        return x


def apply_soap_node(backend: torch.nn.Module, cfg: Dict[str, Any]) -> None:
    """Build ``backend.soap`` (if absent) and ``backend.soap_node`` from ``cfg``:
    ``layers`` ("input", "all" or a list of injection points; default "all"), ``d_hidden``
    (256), and the descriptor's ``cutoff`` (5.0), ``l_max`` (4), ``n_max`` (6), ``n_embed`` (4)."""
    n_layers = int(backend.num_gnn_layers)
    layers = cfg.get("layers", "all")
    if layers == "input":
        points = [0]
    elif layers == "all":
        points = list(range(n_layers))  # the input and after every layer but the last
    else:
        points = [int(k) for k in layers]
    if backend.soap is None:
        backend.soap = SoapPowerSpectrum(
            cutoff=float(cfg.get("cutoff", 5.0)),
            n_species=int(backend.edge_embedder.num_embeddings),
            l_max=int(cfg.get("l_max", 4)),
            n_max=int(cfg.get("n_max", 6)),
            n_embed=int(cfg.get("n_embed", 4)),
        )
    backend.soap_node = SoapNodeInjection(
        backend.soap.out_dim, int(backend.d_node), points, int(cfg.get("d_hidden", 256))
    )
