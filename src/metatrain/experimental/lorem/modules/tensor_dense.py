"""Clebsch-Gordan tensor product, the TorchScript port of ``e3x.nn.TensorDense``.

``lorem-jax`` applies ``e3x.nn.TensorDense`` as a self-product on spherical
node features, and ``e3x.nn.Tensor`` to mix long-range potentials back into
those features. Both are ``a, b → CG(a, b)`` over angular momentum; this
module is the metatrain-native version (same CG convention as SOAP-BPNN).
"""

from typing import List

import torch

from .clebsch_gordan import cg_combine_features, get_cg_coefficients


class _CGBuffer(torch.nn.Module):
    """One ``(2l1+1, 2l2+1, 2L+1)`` Clebsch-Gordan tensor as a buffer."""

    l1: int
    l2: int
    L: int

    def __init__(self, tensor: torch.Tensor, l1: int, l2: int, L: int) -> None:
        super().__init__()
        self.register_buffer("cg", tensor)
        self.l1 = int(l1)
        self.l2 = int(l2)
        self.L = int(L)


class _CGProduct(torch.nn.Module):
    """Shared TorchScript CG loop. Couplings live on ``self``, not as args."""

    def _init_couplings(
        self,
        couplings: List[torch.Tensor],
        l1: List[int],
        l2: List[int],
        L: List[int],
        out_n_lm: int,
    ) -> None:
        self.out_n_lm = int(out_n_lm)
        self.couplings = torch.nn.ModuleList(
            [
                _CGBuffer(tensor, l1[index], l2[index], L[index])
                for index, tensor in enumerate(couplings)
            ]
        )

    def _couple(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        n_atoms = left.shape[0]
        n_features = left.shape[2]
        output = left.new_zeros((n_atoms, self.out_n_lm, n_features))
        for coupling in self.couplings:
            l1 = coupling.l1
            l2 = coupling.l2
            L = coupling.L
            coupled = cg_combine_features(
                left[:, l1 * l1 : (l1 + 1) * (l1 + 1), :],
                right[:, l2 * l2 : (l2 + 1) * (l2 + 1), :],
                coupling.cg.to(dtype=left.dtype),
            )
            output[:, L * L : (L + 1) * (L + 1), :] = (
                output[:, L * L : (L + 1) * (L + 1), :] + coupled
            )
        return output


class TensorDense(_CGProduct):
    """Linear feature projections followed by a CG self-product.

    Input and output layout is ``(n_atoms, (L+1)**2, n_features)``, with
    real spherical harmonics ordered ``m = -ℓ … +ℓ`` inside each ``ℓ``
    (the same order as the LOREM backbone).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        in_max_degree: int,
        out_max_degree: int,
        include_pseudotensors: bool = False,
    ) -> None:
        super().__init__()
        if in_max_degree < 0 or out_max_degree < 0:
            raise ValueError("max_degree must be >= 0")
        if out_max_degree > in_max_degree * 2:
            raise ValueError(
                f"out_max_degree ({out_max_degree}) cannot exceed "
                f"2 * in_max_degree ({in_max_degree})."
            )

        self.in_max_degree = int(in_max_degree)
        self.out_max_degree = int(out_max_degree)
        self.in_n_lm = (self.in_max_degree + 1) * (self.in_max_degree + 1)
        self.out_features = int(out_features)

        self.proj_a = torch.nn.Linear(in_features, out_features, bias=False)
        self.proj_b = torch.nn.Linear(in_features, out_features, bias=False)

        cg = get_cg_coefficients(max(self.in_max_degree, self.out_max_degree))
        l1_list: List[int] = []
        l2_list: List[int] = []
        L_list: List[int] = []
        couplings: List[torch.Tensor] = []
        for l1 in range(self.in_max_degree + 1):
            for l2 in range(self.in_max_degree + 1):
                for L in range(abs(l1 - l2), min(l1 + l2, self.out_max_degree) + 1):
                    if not include_pseudotensors and (l1 + l2 + L) % 2 != 0:
                        continue
                    couplings.append(cg.get((l1, l2, L)).to(torch.float32))
                    l1_list.append(l1)
                    l2_list.append(l2)
                    L_list.append(L)
        self._init_couplings(
            couplings,
            l1_list,
            l2_list,
            L_list,
            (self.out_max_degree + 1) * (self.out_max_degree + 1),
        )

    def forward(self, spherical: torch.Tensor) -> torch.Tensor:
        """:param spherical: ``(n_atoms, in_n_lm, in_features)``."""
        return self._couple(self.proj_a(spherical), self.proj_b(spherical))


class TensorProduct(_CGProduct):
    """CG product of two spherical tensors with a shared feature width.

    Port of ``e3x.nn.Tensor(..., include_pseudotensors=False)``.
    """

    def __init__(
        self,
        left_max_degree: int,
        right_max_degree: int,
        out_max_degree: int,
        include_pseudotensors: bool = False,
    ) -> None:
        super().__init__()
        self.left_max_degree = int(left_max_degree)
        self.right_max_degree = int(right_max_degree)
        self.out_max_degree = int(out_max_degree)

        cg = get_cg_coefficients(
            max(self.left_max_degree, self.right_max_degree, self.out_max_degree)
        )
        l1_list: List[int] = []
        l2_list: List[int] = []
        L_list: List[int] = []
        couplings: List[torch.Tensor] = []
        for l1 in range(self.left_max_degree + 1):
            for l2 in range(self.right_max_degree + 1):
                for L in range(abs(l1 - l2), min(l1 + l2, self.out_max_degree) + 1):
                    if not include_pseudotensors and (l1 + l2 + L) % 2 != 0:
                        continue
                    couplings.append(cg.get((l1, l2, L)).to(torch.float32))
                    l1_list.append(l1)
                    l2_list.append(l2)
                    L_list.append(L)
        self._init_couplings(
            couplings,
            l1_list,
            l2_list,
            L_list,
            (self.out_max_degree + 1) * (self.out_max_degree + 1),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """``left`` / ``right`` are ``(n_atoms, n_lm_*, n_features)``."""
        return self._couple(left, right)
