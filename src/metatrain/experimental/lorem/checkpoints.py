"""Checkpoint upgrade helpers for the LOREM architecture.

Version 1 is the initial experimental checkpoint format (``backbone`` /
``long_range_featurizer``). Version 2 renames those scopes to ``sr`` / ``lr``
(iris PETLR) and adds the CG ``TensorDense`` self-product, the BEC head, and
``lr_scale``. Version 3 adds Bernstein / e3x SH / optional PET-trunk hypers
and the parameter-free ``bernstein_coeff`` buffer.

Version 4 marks checkpoints saved after the equivariant-message-passing and
TensorDense/TensorProduct-CG refactors (commits ``700c0b73`` and ``ce58eaae``):
``sr.equivariant_mp_*``, ``sr.tensor_dense.*``, ``lr.spherical_charge_dense.*``,
and ``lr.potential_product.*`` gained real, non-recomputable trained parameters
that have no counterpart in a version-3 state dict (the old ``TensorDense`` used
a ``proj_a``/``proj_b`` linear-projection parameterization with no equivalent in
the new CG-coupling one). There is no way to migrate a version-3 state dict's
*weights* into this shape, so ``model_update_v3_v4`` does not attempt one -- it
raises, loudly, telling the caller to retrain. (Model loading used to paper
over exactly this gap with ``load_state_dict(..., strict=False)``, which
silently left every one of those new parameters at a fresh random
initialization instead of raising -- see git history on
``LOREM.load_checkpoint`` for the incident this fixed.) Any checkpoint that
predates this refactor is a version-3 checkpoint that cannot be upgraded, by
design: those files remain on disk for reference, but this architecture will no
longer load them for restart/finetune/export, only retraining produces a
version-4 checkpoint.
"""


def model_update_v1_v2(checkpoint: dict) -> None:
    """Rename ``backbone`` / ``long_range_featurizer`` to ``sr`` / ``lr``."""
    for key in ["model_state_dict", "best_model_state_dict"]:
        state_dict = checkpoint.get(key)
        if state_dict is None:
            continue
        renamed = {}
        for name, value in state_dict.items():
            if name.startswith("backbone."):
                name = "sr." + name[len("backbone.") :]
            elif name.startswith("long_range_featurizer."):
                name = "lr." + name[len("long_range_featurizer.") :]
            renamed[name] = value
        checkpoint[key] = renamed


def model_update_v2_v3(checkpoint: dict) -> None:
    """Add paper-matching radial / SH / trunk hypers; Bernstein buffer inits."""
    model_data = checkpoint.setdefault("model_data", {})
    hypers = model_data.setdefault("model_hypers", {})
    hypers.setdefault("radial_basis", "basic_bernstein")
    hypers.setdefault("sh_convention", "e3x")
    hypers.setdefault("trunk", "spherical")
    hypers.setdefault("pet", {})


def model_update_v3_v4(checkpoint: dict) -> None:
    """Refuse to upgrade: no version-3 state dict can be converted to version 4.

    The equivariant-message-passing layers and the CG-coupling ``TensorDense``/
    ``TensorProduct`` reparameterization (commits ``700c0b73``, ``ce58eaae``)
    introduced trained parameters with no version-3 counterpart to copy from --
    unlike every earlier bump here, this one is new model *capacity*, not a
    rename or an addition that can default to something reasonable. Silently
    leaving those parameters at random initialization (the previous behavior,
    via ``load_state_dict(..., strict=False)``) produced a model that looked
    loaded but was actually part-random and non-reproducible run to run.
    """
    raise RuntimeError(
        "This LOREM checkpoint predates the equivariant-message-passing and "
        "TensorDense/TensorProduct-CG refactors (commits 700c0b73, ce58eaae) and "
        "cannot be automatically upgraded to version 4: the new sr.equivariant_mp_*, "
        "sr.tensor_dense.*, lr.spherical_charge_dense.*, and lr.potential_product.* "
        "parameters have no version-3 counterpart to copy weights from. Retrain "
        "under the current architecture to obtain a version-4 checkpoint. The "
        "already-exported model.pt for this checkpoint (if one exists) is a frozen, "
        "self-contained artifact and is unaffected by this -- it remains usable for "
        "inference, just not reloadable as a checkpoint for restart/finetune/export."
    )
