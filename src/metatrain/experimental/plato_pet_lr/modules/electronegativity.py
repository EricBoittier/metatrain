"""Linear electronegativity energy ``E_chi = sum_i chi_i q_i`` of the latent charges.

The LES energy ``1/2 q^T K q`` is even in ``q``, so ``q`` and ``-q`` give the same
energies and forces and the sign of the latent charges is a gauge. A term linear in
``q`` (as in QEq, Rappe & Goddard 1991, and 4G-HDNNP, Ko et al. 2021) is odd and
breaks it. ``chi_i = chi0[Z_i] + dchi_i``: a learnable per-element table started
from the QEq electronegativities below, plus a per-atom correction from the final
node features whose last layer starts at zero (``dchi = 0`` at initialisation).

Units. The model is trained on targets divided by the energy scale ``s`` of the
scaler, and the LES energy uses ``KE`` in eV A / e^2, so the latent charges ``q_m``
of the network are charges in units of ``1 / sqrt(s)``: ``q = sqrt(s) q_m`` is in
``e`` and ``KE q^2 / r`` in eV. ``charge_unit`` stores ``sqrt(s)`` (set by the
trainer once the scaler is fitted), and ``chi`` is in eV / e, so the energy in
model units is ``sum_i chi_i q_i / s = sum_i chi_i q_m,i / charge_unit``.
Only differences of ``chi0`` matter: the latent charges are neutral per structure.
"""

import logging
from typing import Dict, List

import torch

from .layers import FrameMLP


QEQ_CHI_EV: Dict[int, float] = {
    1: 4.528,  # H
    3: 3.006,  # Li
    5: 5.110,  # B
    6: 5.343,  # C
    7: 6.899,  # N
    8: 8.741,  # O
    9: 10.874,  # F
    11: 2.843,  # Na
    12: 3.951,  # Mg
    14: 4.168,  # Si
    15: 5.463,  # P
    16: 6.928,  # S
    17: 8.564,  # Cl
    19: 2.421,  # K
    20: 3.231,  # Ca
    35: 7.790,  # Br
    53: 6.822,  # I
}
"""QEq electronegativities chi in eV (Rappe & Goddard, J. Phys. Chem. 95, 3358
(1991), Table I; the same values are the ``Xi`` column of UFF, Rappe et al., JACS
114, 10024 (1992), as shipped in RDKit's ``Code/ForceField/UFF/Params.cpp``)."""


def initial_chi(atomic_types: List[int]) -> torch.Tensor:
    known = [QEQ_CHI_EV[z] for z in atomic_types if z in QEQ_CHI_EV]
    fill = sum(known) / len(known) if known else 0.0
    missing = [z for z in atomic_types if z not in QEQ_CHI_EV]
    if missing:
        logging.warning(f"no QEq electronegativity for Z={missing}, using {fill:.3f}")
    return torch.tensor([QEQ_CHI_EV.get(z, fill) for z in atomic_types])


class Electronegativity(torch.nn.Module):
    """``chi0_trainable: false`` keeps ``chi0`` at the QEq table (a buffer of the
    same name); ``dchi_bound = b > 0`` bounds the correction, ``dchi = b tanh(head)``."""

    def __init__(
        self,
        atomic_types: List[int],
        d: int,
        n_basis: int,
        hidden: int,
        mode: str,
        chi0_trainable: bool = True,
        dchi_bound: float = 0.0,
    ):
        super().__init__()
        chi0 = initial_chi(atomic_types).to(torch.get_default_dtype())
        if chi0_trainable:
            self.chi0 = torch.nn.Parameter(chi0)
        else:
            self.register_buffer("chi0", chi0)
        self.dchi_bound = float(dchi_bound)
        self.head = FrameMLP(d, 1, n_basis, hidden, mode)
        with torch.no_grad():  # dchi = 0 at the start
            self.head.l2.W.zero_()
            b = self.head.l2.b
            if b is not None:
                b.zero_()
        self.register_buffer("charge_unit", torch.ones(()))

    def chi(
        self, h: torch.Tensor, species: torch.Tensor, rel: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        """Per-atom ``chi`` in eV / e."""
        dchi = (self.head(h, rel, w)[..., 0] * w).sum(-1)
        if self.dchi_bound > 0:
            dchi = self.dchi_bound * torch.tanh(dchi)
        return self.chi0[species] + dchi

    def forward(
        self,
        h: torch.Tensor,
        species: torch.Tensor,
        q: torch.Tensor,
        rel: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        """Per-atom energy in model units; ``q`` is the first latent channel."""
        return self.chi(h, species, rel, w) * q / self.charge_unit
