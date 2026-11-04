"""SOAP power spectrum on PET's padded neighbour tensors (SOAP-PET).

``SoapPowerSpectrum`` computes, per atom, a rotation-invariant SOAP power spectrum from
the same ``(n_atoms, max_neighbors)`` edge layout PET uses (no separate neighbour list):

- neighbour density channels ``a = (n, k)``: Gaussian radial functions ``g_n(r)`` times a
  cosine cutoff, times a learnable species embedding ``e_k(Z_j)``;
- spherical expansion ``c_{a,lm} = sum_j g_n(r_ij) f_c(r_ij) e_k(Z_j) Y_lm(r_ij / r_ij)``,
  real spherical harmonics up to ``l_max = 4`` as explicit polynomials;
- power spectrum ``p_{ab,l} = (2l+1)^{-1/2} sum_m c_{a,lm} c_{b,lm}``, upper triangle
  ``a <= b`` per ``l``, then normalised to unit length per atom.

Only plain torch ops (mul / add / bmm / matmul / sqrt / cos), so the mlip.cpp
GGUF exporter traces it into an ordinary GIR graph. The output dimension is
``out_dim = (l_max + 1) * C (C + 1) / 2`` with ``C = n_max * n_embed`` (1500 for the
defaults).

``soap_on_edges(p, neighbor_index)`` gathers ``p_j`` onto the edge slots;
``soap_on_edges_from_reverse(p, reverse_neighbor_index)`` does the same from PET's
reversed-neighbour index (no neighbour-atom index needed, as in the exported graph).
"""

import math
from typing import List

import torch


def _stack(xs: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t[..., None] for t in xs], -1)


def _real_sh(u: torch.Tensor, l_max: int) -> List[torch.Tensor]:
    """Orthonormal real spherical harmonics of unit vectors ``u`` (..., 3), one tensor
    (..., 2l+1) per l <= l_max (l_max <= 4), m = -l..l."""
    x, y, z = u[..., 0], u[..., 1], u[..., 2]
    out = [torch.ones_like(x)[..., None] * 0.28209479177387814]
    if l_max >= 1:
        c = 0.4886025119029199
        out.append(_stack([c * y, c * z, c * x]))
    if l_max >= 2:
        xx, yy, zz = x * x, y * y, z * z
        out.append(_stack([
            1.0925484305920792 * x * y,
            1.0925484305920792 * y * z,
            0.31539156525252005 * (3.0 * zz - 1.0),
            1.0925484305920792 * x * z,
            0.5462742152960396 * (xx - yy),
        ]))
    if l_max >= 3:
        out.append(_stack([
            0.5900435899266435 * y * (3.0 * xx - yy),
            2.890611442640554 * x * y * z,
            0.4570457994644658 * y * (5.0 * zz - 1.0),
            0.3731763325901154 * z * (5.0 * zz - 3.0),
            0.4570457994644658 * x * (5.0 * zz - 1.0),
            1.445305721320277 * z * (xx - yy),
            0.5900435899266435 * x * (xx - 3.0 * yy),
        ]))
    if l_max >= 4:
        out.append(_stack([
            2.5033429417967046 * x * y * (xx - yy),
            1.7701307697799304 * y * z * (3.0 * xx - yy),
            0.9461746957575601 * x * y * (7.0 * zz - 1.0),
            0.6690465435572892 * y * z * (7.0 * zz - 3.0),
            0.10578554691520431 * (35.0 * zz * zz - 30.0 * zz + 3.0),
            0.6690465435572892 * x * z * (7.0 * zz - 3.0),
            0.47308734787878004 * (xx - yy) * (7.0 * zz - 1.0),
            1.7701307697799304 * x * z * (xx - 3.0 * yy),
            0.6258357354491761 * (xx * (xx - 3.0 * yy) - yy * (3.0 * xx - yy)),
        ]))
    if l_max > 4:
        raise ValueError("SoapPowerSpectrum supports l_max <= 4")
    return out


class SoapPowerSpectrum(torch.nn.Module):
    """Per-atom SOAP power spectrum from PET's padded neighbour tensors.

    :param cutoff: radial cutoff (Å); a cosine cutoff takes the density to 0 there.
    :param n_species: number of species indices (PET's ``element_indices``).
    :param l_max: maximum angular order (<= 4).
    :param n_max: number of Gaussian radial functions.
    :param n_embed: species embedding channels (learnable).
    """

    def __init__(self, cutoff: float, n_species: int, l_max: int = 4, n_max: int = 6,
                 n_embed: int = 4) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        self.l_max = int(l_max)
        self.n_max = int(n_max)
        self.n_embed = int(n_embed)
        self.species_embedding = torch.nn.Embedding(n_species, n_embed)
        with torch.no_grad():  # O(1) channel weights, every species distinct
            self.species_embedding.weight.normal_(0.0, 1.0)
        self.register_buffer("centers", torch.linspace(0.0, self.cutoff, self.n_max))
        self.width = self.cutoff / self.n_max
        C = self.n_max * self.n_embed
        iu = torch.triu_indices(C, C)
        # upper-triangle selection as a float one-hot matmul (C*C, T): no integer buffer
        # (the GGUF export has no int tensors)
        sel = torch.zeros(C * C, iu.shape[1])
        sel[iu[0] * C + iu[1], torch.arange(iu.shape[1])] = 1.0
        self.register_buffer("triu_select", sel)
        self.n_channels = C
        self.out_dim = (self.l_max + 1) * int(iu.shape[1])

    def radial(self, r: torch.Tensor) -> torch.Tensor:
        """Radial basis with the cutoff: g_n(r) f_c(r), (...,) -> (..., n_max)."""
        fc = 0.5 * (1.0 + torch.cos(math.pi * torch.clamp(r / self.cutoff, 0.0, 1.0)))
        d = (r[..., None] - self.centers) / self.width
        return torch.exp(-0.5 * d * d) * fc[..., None]

    def real_sph(self, xyz: torch.Tensor) -> torch.Tensor:
        """Orthonormal real SH of the direction of xyz, (..., 3) -> (..., (l_max+1)^2),
        ordered l = 0..l_max, m = -l..l."""
        r = torch.sqrt((xyz * xyz).sum(-1, keepdim=True) + 1e-12)
        return torch.cat(_real_sh(xyz / r, self.l_max), -1)

    def spherical_expansion(self, edge_vectors: torch.Tensor, padding_mask: torch.Tensor,
                            neighbor_species: torch.Tensor,
                            center_species: torch.Tensor) -> torch.Tensor:
        """Density coefficients c_{a,lm} = sum_j g_n(r_ij) f_c(r_ij) e_k(Z_j) Y_lm(r^_ij),
        (n_atoms, C, (l_max+1)^2), channel a = n * n_embed + k (padded slots excluded)."""
        n_atoms, n_nbr = edge_vectors.shape[0], edge_vectors.shape[1]
        r = torch.sqrt((edge_vectors * edge_vectors).sum(-1) + 1e-12)
        g = self.radial(r)
        g = torch.where(padding_mask[..., None], torch.zeros_like(g), g)
        e = self.species_embedding(neighbor_species)
        a = (g[..., :, None] * e[..., None, :]).reshape(n_atoms, n_nbr, self.n_channels)
        return torch.bmm(a.transpose(1, 2), self.real_sph(edge_vectors))

    def forward(self, edge_vectors: torch.Tensor, padding_mask: torch.Tensor,
                neighbor_species: torch.Tensor, center_species: torch.Tensor) -> torch.Tensor:
        """Unit-normalised power spectrum (n_atoms, out_dim): per l, the upper triangle of
        p_{ab,l} = (2l+1)^{-1/2} sum_m c_{a,lm} c_{b,lm} of :meth:`spherical_expansion`.
        center_species is unused (the power spectrum is centre-agnostic)."""
        c = self.spherical_expansion(edge_vectors, padding_mask, neighbor_species, center_species)
        n_atoms = c.shape[0]
        parts: List[torch.Tensor] = []
        for l in range(self.l_max + 1):
            cl = c[:, :, l * l:(l + 1) * (l + 1)]
            pl = torch.bmm(cl, cl.transpose(1, 2)) * (2 * l + 1) ** -0.5
            parts.append(pl.reshape(n_atoms, -1) @ self.triu_select)
        p = torch.cat(parts, -1)
        return p / torch.sqrt((p * p).sum(-1, keepdim=True) + 1e-12)


def soap_on_edges(p: torch.Tensor, neighbor_index: torch.Tensor) -> torch.Tensor:
    """``p_j`` on every edge slot: p (n_atoms, D), neighbor_index (n_atoms, n_nbr) the
    neighbour atom of each slot -> (n_atoms, n_nbr, D). Padded slots carry whatever atom
    the index names; mask them with the padding mask."""
    n_atoms, n_nbr = neighbor_index.shape[0], neighbor_index.shape[1]
    return p[neighbor_index.reshape(-1)].reshape(n_atoms, n_nbr, p.shape[1])


def soap_on_edges_from_reverse(p: torch.Tensor, reverse_neighbor_index: torch.Tensor,
                               n_nbr: int) -> torch.Tensor:
    """As :func:`soap_on_edges` from PET's flat reversed-neighbour index
    (n_atoms * n_nbr,): slot (i, j)'s reverse slot belongs to atom j, so the per-slot
    copy of p gathered by it is p_j. Plain expand / reshape / index ops (export-safe)."""
    n_atoms = p.shape[0]
    rep = p[:, None, :].expand(n_atoms, n_nbr, p.shape[1]).reshape(n_atoms * n_nbr, p.shape[1])
    return rep[reverse_neighbor_index.reshape(-1)].reshape(n_atoms, n_nbr, p.shape[1])


class PETSoapHead(torch.nn.Module):
    """SOAP-PET variant 3: an additive SOAP-BPNN atomic energy, gate * MLP([p_i, e(Z_i)]),
    the gate a scalar starting at 0 (the model starts as the pretrained PET)."""

    def __init__(self, soap_dim: int, n_species: int, hidden: int = 128, n_center: int = 8) -> None:
        super().__init__()
        self.center_embedding = torch.nn.Embedding(n_species, n_center)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(soap_dim + n_center, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, 1))
        self.head_gate = torch.nn.Parameter(torch.zeros(1))

    def forward(self, p: torch.Tensor, center_species: torch.Tensor) -> torch.Tensor:
        """p (n_atoms, D), center_species (n_atoms,) -> per-atom energy (n_atoms,)."""
        x = torch.cat([p, self.center_embedding(center_species)], -1)
        return self.head_gate * self.mlp(x).squeeze(-1)


def apply_soap(backend: torch.nn.Module, h: dict) -> None:
    """Add the SOAP-PET components of the ``plato_lr`` hypers' ``soap`` dict to a
    PETBackend: the shared ``backend.soap`` power spectrum, and per variant its module
    (``head``: :class:`PETSoapHead` as ``backend.soap_head``)."""
    n_species = backend.edge_embedder.num_embeddings
    backend.soap = SoapPowerSpectrum(float(h.get("cutoff", 5.0)), n_species, int(h.get("l_max", 4)),
                                     int(h.get("n_max", 6)), int(h.get("n_embed", 4)))
    if h.get("head", False):
        backend.soap_head = PETSoapHead(backend.soap.out_dim, n_species, int(h.get("head_hidden", 128)))
