"""Checkpoint upgrade helpers for the LOREM architecture.

Version 1 is the initial experimental checkpoint format (``backbone`` /
``long_range_featurizer``). Version 2 renames those scopes to ``sr`` / ``lr``
(iris PETLR) and adds the CG ``TensorDense`` self-product, the BEC head, and
``lr_scale``. Version 3 adds Bernstein / e3x SH / optional PET-trunk
hypers and the parameter-free ``bernstein_coeff`` buffer. New parameters
that have no older counterpart are initialised on load
(``load_state_dict(..., strict=False)``).
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
