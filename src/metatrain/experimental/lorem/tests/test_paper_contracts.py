"""Paper / lorem-jax contracts that ``experimental.lorem`` pins in CI.

This file is the checklist. These tests do **not** import JAX and do **not**
claim bit-exact energies. They check the same equations and knobs as
``lorem.Lorem`` / ``lorem.LoremBEC`` (and iris-style ``sr`` / ``lr`` /
``lr_scale``, which are covered elsewhere).

Already tested (do not duplicate):

- acoustic sum rule — ``test_bec.test_bec_acoustic_sum_rule``
- ``lr_scale == 0`` is a no-op — ``test_long_range.test_lr_scale_zero_is_noop``
- children named ``sr`` / ``lr`` — ``test_long_range.test_sr_lr_module_scopes``
- energy rotation invariance —
  ``test_long_range.test_long_range_energy_rotation_invariant``
- ℓ=1 charges rotate as vectors —
  ``test_long_range.test_spherical_charges_rotate_as_vectors``
- ``TensorDense`` scalar rotation invariance —
  ``test_tensor_dense.test_tensor_dense_scalar_is_rotation_invariant``
"""

import pytest
import torch

from metatrain.experimental.lorem.modules.backbone import (
    _cosine_cutoff,
    _degree_norms,
)
from metatrain.experimental.lorem.modules.clebsch_gordan import (
    ClebschGordanReal,
    cg_combine_features,
)
from metatrain.utils.architectures import get_default_hypers

from . import MODEL_HYPERS


def test_default_hypers_match_paper():
    """Paper / ``documentation.py`` defaults (lorem-jax ``Lorem`` knobs)."""
    hypers = get_default_hypers("experimental.lorem")["model"]
    assert hypers == MODEL_HYPERS
    assert hypers["cutoff"] == 5.0
    assert hypers["max_degree"] == 6
    assert hypers["max_degree_lr"] == 2
    assert hypers["num_features"] == 128
    assert hypers["num_radial"] == 32
    assert hypers["num_spherical_features"] == 8
    assert hypers["num_message_passing"] == 0
    assert hypers["long_range"]["enable"] is True


def test_degree_norm_factor_is_two_ell_plus_one_to_the_quarter():
    """``l_factors = (2ℓ+1)^{1/4}`` as in lorem-jax ``LoremBEC`` / ``Lorem``."""
    max_degree = 2
    spherical = torch.zeros(1, (max_degree + 1) ** 2, 1)
    # Unit mass in the ℓ=1 block so the raw ℓ-norm is 1.
    spherical[0, 2, 0] = 1.0
    norms = _degree_norms(spherical, max_degree)
    assert norms.shape == (1, max_degree + 1)
    expected = torch.tensor(
        [
            0.0,
            (2.0 * 1.0 + 1.0) ** 0.25,
            0.0,
        ]
    )
    torch.testing.assert_close(norms[0], expected, atol=1e-6, rtol=1e-6)


def test_cosine_cutoff_is_one_inside_and_zero_at_cutoff():
    """``e3x.nn.functions.cosine_cutoff``: 1 below onset, 0 at the cutoff."""
    cutoff = 5.0
    width = 0.5
    distances = torch.tensor([0.0, 4.4, 4.5, 5.0, 5.2])
    weights = _cosine_cutoff(distances, cutoff, width)
    torch.testing.assert_close(weights[:3], torch.ones(3), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(weights[3:], torch.zeros(2), atol=1e-6, rtol=1e-6)


def test_charge_layout_is_scalar_plus_spherical_lm():
    """lorem-jax equivariant charges: 1 scalar + ``(max_degree_lr+1)²`` (ℓ, m)."""
    pytest.importorskip("torchpme")

    from metatrain.experimental.lorem import LOREM

    from .test_long_range import _energy_dataset_info, _small_lr_hypers

    hypers = _small_lr_hypers(max_degree=1, max_degree_lr=1)
    model = LOREM(hypers, _energy_dataset_info())
    n_atoms = 3
    features = torch.randn(n_atoms, model.num_features)
    n_lm = (int(model.hypers["max_degree"]) + 1) ** 2
    spherical = torch.randn(n_atoms, n_lm, int(model.hypers["num_spherical_features"]))
    charges = model.lr.map_charges(features, spherical)
    max_degree_lr = int(model.hypers["max_degree_lr"])
    assert charges.shape == (n_atoms, 1 + (max_degree_lr + 1) ** 2)


def test_cg_one_otimes_one_to_scalar_is_dot_product():
    """e3x / lorem-jax ``TensorDense``: ``1 ⊗ 1 → 0`` is the invariant."""
    cg = ClebschGordanReal()
    coeff = cg.get((1, 1, 0)).to(torch.float64)
    vectors = torch.tensor(
        [
            [[1.0], [0.0], [0.0]],
            [[0.0], [2.0], [0.0]],
            [[0.0], [0.0], [3.0]],
            [[1.0], [1.0], [1.0]],
        ],
        dtype=torch.float64,
    )
    coupled = cg_combine_features(vectors, vectors, coeff)[:, 0, 0]
    dots = (vectors[:, :, 0] ** 2).sum(dim=-1)
    ratio = coupled / dots
    torch.testing.assert_close(ratio, ratio[0].expand_as(ratio), atol=1e-8, rtol=1e-8)
    assert torch.all(ratio.abs() > 0)
