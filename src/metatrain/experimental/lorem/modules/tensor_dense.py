"""Clebsch-Gordan tensor product, the TorchScript port of ``e3x.nn.TensorDense``.

``lorem-jax`` applies ``e3x.nn.TensorDense`` as a self-product on spherical
node features, and ``e3x.nn.Tensor`` to mix long-range potentials back into
those features. Both are ``a, b → CG(a, b)`` over angular momentum; this
module is the metatrain-native version (same CG convention as SOAP-BPNN).
"""

from typing import List

import torch

from .clebsch_gordan import cg_combine_features, get_cg_coefficients


class TensorDense(torch.nn.Module):
    """Linear feature projections followed by a CG self-product.

    Input and output layout is ``(n_atoms, (L+1)**2, n_features)``, with
    real spherical harmonics ordered ``m = -ℓ … +ℓ`` inside each ``ℓ``
    (the same order as the LOREM backbone).
    """

    l1: List[int]
    l2: List[int]
    L: List[int]

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
        self.out_n_lm = (self.out_max_degree + 1) * (self.out_max_degree + 1)
        self.out_features = int(out_features)

        self.proj_a = torch.nn.Linear(in_features, out_features, bias=False)
        self.proj_b = torch.nn.Linear(in_features, out_features, bias=False)

        cg = get_cg_coefficients(max(self.in_max_degree, self.out_max_degree))
        self.l1 = []
        self.l2 = []
        self.L = []
        couplings: List[torch.Tensor] = []
        for l1 in range(self.in_max_degree + 1):
            for l2 in range(self.in_max_degree + 1):
                for L in range(
                    abs(l1 - l2), min(l1 + l2, self.out_max_degree) + 1
                ):
                    if not include_pseudotensors and (l1 + l2 + L) % 2 != 0:
                        continue
                    couplings.append(cg.get((l1, l2, L)).to(torch.float32))
                    self.l1.append(l1)
                    self.l2.append(l2)
                    self.L.append(L)
        self.couplings = torch.nn.ParameterList(
            [torch.nn.Parameter(tensor, requires_grad=False) for tensor in couplings]
        )

    def forward(self, spherical: torch.Tensor) -> torch.Tensor:
        """:param spherical: ``(n_atoms, in_n_lm, in_features)``."""
        left = self.proj_a(spherical)
        right = self.proj_b(spherical)
        return _couple(left, right, self.couplings, self.l1, self.l2, self.L, self.out_n_lm)


class TensorProduct(torch.nn.Module):
    """CG product of two spherical tensors with a shared feature width.

    Port of ``e3x.nn.Tensor(..., include_pseudotensors=False)``.
    """

    l1: List[int]
    l2: List[int]
    L: List[int]

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
        self.out_n_lm = (self.out_max_degree + 1) * (self.out_max_degree + 1)

        cg = get_cg_coefficients(
            max(self.left_max_degree, self.right_max_degree, self.out_max_degree)
        )
        self.l1 = []
        self.l2 = []
        self.L = []
        couplings: List[torch.Tensor] = []
        for l1 in range(self.left_max_degree + 1):
            for l2 in range(self.right_max_degree + 1):
                for L in range(
                    abs(l1 - l2), min(l1 + l2, self.out_max_degree) + 1
                ):
                    if not include_pseudotensors and (l1 + l2 + L) % 2 != 0:
                        continue
                    couplings.append(cg.get((l1, l2, L)).to(torch.float32))
                    self.l1.append(l1)
                    self.l2.append(l2)
                    self.L.append(L)
        self.couplings = torch.nn.ParameterList(
            [torch.nn.Parameter(tensor, requires_grad=False) for tensor in couplings]
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """``left`` / ``right`` are ``(n_atoms, n_lm_*, n_features)``."""
        return _couple(left, right, self.couplings, self.l1, self.l2, self.L, self.out_n_lm)


def _couple(
    left: torch.Tensor,
    right: torch.Tensor,
    couplings: torch.nn.ParameterList,
    l1_list: List[int],
    l2_list: List[int],
    L_list: List[int],
    out_n_lm: int,
) -> torch.Tensor:
    n_atoms = left.shape[0]
    n_features = left.shape[2]
    output = left.new_zeros((n_atoms, out_n_lm, n_features))
    for index in range(len(couplings)):
        l1 = l1_list[index]
        l2 = l2_list[index]
        L = L_list[index]
        coupled = cg_combine_features(
            left[:, l1 * l1 : (l1 + 1) * (l1 + 1), :],
            right[:, l2 * l2 : (l2 + 1) * (l2 + 1), :],
            couplings[index].to(dtype=left.dtype),
        )
        output[:, L * L : (L + 1) * (L + 1), :] = (
            output[:, L * L : (L + 1) * (L + 1), :] + coupled
        )
    return output
