from typing import List

import torch
from metatomic.torch import System

from metatrain.utils.long_range import LongRangeHypers
from metatrain.utils.neighbor_lists import NeighborListOptions


def _safe_vector_norm(
    x: torch.Tensor, dim: int, keepdim: bool = False, eps: float = 1.0e-12
) -> torch.Tensor:
    """``torch.linalg.vector_norm`` with a finite gradient at zero.

    ``vector_norm``'s backward is ``x / ||x||``, which is ``0/0 = nan`` whenever
    ``x`` is exactly zero along ``dim`` -- e.g. a vanishing ``m``-component of a
    spherical harmonic. Adding ``eps`` inside the square root keeps the value
    (and gradient) finite everywhere without materially changing it away from
    zero.
    """
    return torch.sqrt(x.pow(2).sum(dim=dim, keepdim=keepdim) + eps)


def _spherical_norm(values: torch.Tensor, max_degree: int) -> torch.Tensor:
    """Per-degree spherical norm with the LOREM :math:`(2\\ell+1)^{1/4}` factor.

    :param values: Tensor of shape ``(n_atoms, (max_degree + 1) ** 2)``.
    :param max_degree: Maximum angular momentum.
    :return: Tensor of shape ``(n_atoms, max_degree + 1)``.
    """
    parts: List[torch.Tensor] = []
    for ell in range(max_degree + 1):
        start = ell * ell
        end = (ell + 1) * (ell + 1)
        chunk = values[:, start:end]
        factor = (2.0 * float(ell) + 1.0) ** 0.25
        parts.append(factor * _safe_vector_norm(chunk, dim=-1, keepdim=True))
    return torch.cat(parts, dim=-1)


class DummyLoremLongRangeFeaturizer(torch.nn.Module):
    """Placeholder used when long-range interactions are disabled (TorchScript)."""

    def __init__(self) -> None:
        super().__init__()
        self.use_ewald = True

    def forward(
        self,
        systems: List[System],
        features: torch.Tensor,
        neighbor_distances: torch.Tensor,
        spherical_features: torch.Tensor,
    ) -> torch.Tensor:
        return features


class LoremLongRangeFeaturizer(torch.nn.Module):
    """Ewald / P3M / direct Coulomb features from equivariant atomic charges.

    Scalar features are mapped to one charge channel. Spherical node features
    are mapped, independently per :math:`\\ell` with weights shared across
    :math:`m`, to charges up to ``max_degree_lr``. Those channels are evaluated
    with torch-pme (the paper's long-range message) and contracted back to
    invariant feature updates.

    This is the Phase-2 mechanism from LOREM / ``lorem-jax``. A Clebsch-Gordan
    self-product of spherical features (``e3x.nn.TensorDense``) is still a
    later refinement.
    """

    def __init__(
        self,
        hypers: LongRangeHypers,
        feature_dim: int,
        num_radial: int,
        num_spherical_features: int,
        max_degree_lr: int,
        neighbor_list_options: NeighborListOptions,
    ) -> None:
        super().__init__()
        if max_degree_lr < 0:
            raise ValueError(f"max_degree_lr must be >= 0, got {max_degree_lr}")

        try:
            from torchpme import (
                Calculator,
                CoulombPotential,
                EwaldCalculator,
                P3MCalculator,
            )
        except ImportError:
            raise ImportError(
                "`torch-pme` is required for long-range models. "
                "Please install it with `pip install 'torch-pme>=0.3.2'`."
            ) from None

        self.max_degree_lr = int(max_degree_lr)
        self.n_lm_lr = (self.max_degree_lr + 1) * (self.max_degree_lr + 1)
        self.num_radial = int(num_radial)
        self.num_spherical_features = int(num_spherical_features)
        self.feature_dim = int(feature_dim)
        self.use_ewald = bool(hypers["use_ewald"])
        self.neighbor_list_options = neighbor_list_options

        self.ewald_calculator = EwaldCalculator(
            potential=CoulombPotential(
                smearing=float(hypers["smearing"]),
                exclusion_radius=neighbor_list_options.cutoff,
            ),
            full_neighbor_list=neighbor_list_options.full_list,
            lr_wavelength=float(hypers["kspace_resolution"]),
        )
        self.p3m_calculator = P3MCalculator(
            potential=CoulombPotential(
                smearing=float(hypers["smearing"]),
                exclusion_radius=neighbor_list_options.cutoff,
            ),
            interpolation_nodes=hypers["interpolation_nodes"],
            full_neighbor_list=neighbor_list_options.full_list,
            mesh_spacing=float(hypers["kspace_resolution"]),
        )
        self.direct_calculator = Calculator(
            potential=CoulombPotential(
                smearing=None,
                exclusion_radius=neighbor_list_options.cutoff,
            ),
            full_neighbor_list=False,
        )

        # lorem-jax: MLP([2 * d, 1]) on scalar features.
        self.scalar_charge_mlp = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 2 * feature_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(2 * feature_dim, 1),
        )
        # Per-ℓ maps, weights shared across m (equivariant linear). The first
        # layer projects the radial density onto ``num_spherical_features``.
        self.spherical_projections = torch.nn.ModuleList(
            [
                torch.nn.Linear(num_radial, num_spherical_features, bias=False)
                for _ in range(self.max_degree_lr + 1)
            ]
        )
        self.spherical_charge_maps = torch.nn.ModuleList(
            [
                torch.nn.Linear(num_spherical_features, 1, bias=False)
                for _ in range(self.max_degree_lr + 1)
            ]
        )

        n_update = 1 + (self.max_degree_lr + 1) * (1 + num_spherical_features)
        self.update_from_potential = torch.nn.Sequential(
            torch.nn.Linear(n_update, 2 * feature_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(2 * feature_dim, feature_dim),
        )
        self.update_residual = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 2 * feature_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(2 * feature_dim, feature_dim),
        )
        self.norm_after_potential = torch.nn.LayerNorm(feature_dim)
        self.norm_after_residual = torch.nn.LayerNorm(feature_dim)

    def project_spherical(self, spherical_features: torch.Tensor) -> torch.Tensor:
        """Project radial density channels onto ``num_spherical_features``.

        :param spherical_features: Neighbor density
            ``(n_atoms, n_lm, num_radial)``.
        :return: ``(n_atoms, (max_degree_lr + 1) ** 2, num_spherical_features)``.
        """
        parts: List[torch.Tensor] = []
        for ell, projection in enumerate(self.spherical_projections):
            start = ell * ell
            end = (ell + 1) * (ell + 1)
            parts.append(projection(spherical_features[:, start:end, :]))
        return torch.cat(parts, dim=1)

    def map_charges(
        self, features: torch.Tensor, spherical_features: torch.Tensor
    ) -> torch.Tensor:
        """Map scalar and spherical features to concatenated charge channels.

        :param features: Invariant atom features ``(n_atoms, feature_dim)``.
        :param spherical_features: Neighbor density
            ``(n_atoms, n_lm, num_radial)`` with
            ``n_lm >= (max_degree_lr + 1) ** 2``.
        :return: Charges ``(n_atoms, 1 + (max_degree_lr + 1) ** 2)`` — the
            leading channel is the scalar charge, the rest are spherical
            charges ordered ``ℓ = 0..max_degree_lr``, ``m = -ℓ..ℓ``.
        """
        scalar_charges = self.scalar_charge_mlp(features)
        projected = self.project_spherical(spherical_features)
        spherical_parts: List[torch.Tensor] = []
        for ell, charge_map in enumerate(self.spherical_charge_maps):
            start = ell * ell
            end = (ell + 1) * (ell + 1)
            spherical_parts.append(charge_map(projected[:, start:end, :]).squeeze(-1))
        spherical_charges = torch.cat(spherical_parts, dim=-1)
        return torch.cat([scalar_charges, spherical_charges], dim=-1)

    def _potentials_for_system(
        self,
        system: System,
        system_charges: torch.Tensor,
        neighbor_indices_system: torch.Tensor,
        neighbor_distances_system: torch.Tensor,
    ) -> torch.Tensor:
        if system.pbc.any():
            if system.pbc.sum() == 1:
                raise NotImplementedError(
                    "Long-range featurizer does not support 1D systems."
                )
            if self.use_ewald and self.training:
                return self.ewald_calculator.forward(
                    charges=system_charges,
                    cell=system.cell,
                    positions=system.positions,
                    neighbor_indices=neighbor_indices_system,
                    neighbor_distances=neighbor_distances_system,
                    periodic=system.pbc,
                )
            return self.p3m_calculator.forward(
                charges=system_charges,
                cell=system.cell,
                positions=system.positions,
                neighbor_indices=neighbor_indices_system,
                neighbor_distances=neighbor_distances_system,
                periodic=system.pbc,
            )

        neighbor_indices_system = torch.combinations(
            torch.arange(len(system), device=system.positions.device), 2
        )
        neighbor_distances_system = torch.sqrt(
            torch.sum(
                (
                    system.positions[neighbor_indices_system[:, 1]]
                    - system.positions[neighbor_indices_system[:, 0]]
                )
                ** 2,
                dim=1,
            )
        )
        return self.direct_calculator.forward(
            charges=system_charges,
            cell=system.cell,
            positions=system.positions,
            neighbor_indices=neighbor_indices_system,
            neighbor_distances=neighbor_distances_system,
        )

    def _evaluate_potentials(
        self,
        systems: List[System],
        charges: torch.Tensor,
        neighbor_distances: torch.Tensor,
    ) -> torch.Tensor:
        last_len_nodes = 0
        last_len_edges = 0
        potentials: List[torch.Tensor] = []
        for system in systems:
            system_charges = charges[last_len_nodes : last_len_nodes + len(system)]
            last_len_nodes += len(system)

            neighbor_list = system.get_neighbor_list(self.neighbor_list_options)
            neighbor_indices_system = torch.stack(
                [
                    neighbor_list.samples.column("first_atom"),
                    neighbor_list.samples.column("second_atom"),
                ],
                dim=-1,
            )
            neighbor_distances_system = neighbor_distances[
                last_len_edges : last_len_edges + len(neighbor_indices_system)
            ]
            last_len_edges += len(neighbor_indices_system)
            potentials.append(
                self._potentials_for_system(
                    system,
                    system_charges,
                    neighbor_indices_system,
                    neighbor_distances_system,
                )
            )
        return torch.cat(potentials, dim=0)

    def _invariant_updates(
        self, potentials: torch.Tensor, spherical_features: torch.Tensor
    ) -> torch.Tensor:
        """Contract equivariant potentials to rotation-invariant updates."""
        scalar_potential = potentials[:, 0:1]
        spherical_potential = potentials[:, 1:]
        parts: List[torch.Tensor] = [scalar_potential]
        parts.append(_spherical_norm(spherical_potential, self.max_degree_lr))
        for ell in range(self.max_degree_lr + 1):
            start = ell * ell
            end = (ell + 1) * (ell + 1)
            v_ell = spherical_potential[:, start:end]
            s_ell = spherical_features[:, start:end, :]
            parts.append(torch.einsum("nm,nmf->nf", v_ell, s_ell))
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        systems: List[System],
        features: torch.Tensor,
        neighbor_distances: torch.Tensor,
        spherical_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return invariant long-range feature updates.

        :param systems: Systems with the requested neighbor list attached.
        :param features: Short-range invariant features.
        :param neighbor_distances: Neighbor distances matching that list.
        :param spherical_features: Short-range spherical features.
        :return: Updates of shape ``(n_atoms, feature_dim)``.
        """
        charges = self.map_charges(features, spherical_features)
        potentials = self._evaluate_potentials(systems, charges, neighbor_distances)
        projected = self.project_spherical(spherical_features)
        updates = self._invariant_updates(potentials, projected)
        features = features + self.update_from_potential(updates)
        features = self.norm_after_potential(features)
        features = features + self.update_residual(features)
        return self.norm_after_residual(features)
