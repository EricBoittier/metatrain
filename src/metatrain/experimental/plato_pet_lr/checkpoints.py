"""Checkpoint upgrade helpers for Plato-PET-LR:
``model_update_v{n}_v{n + 1}`` / ``trainer_update_v{n}_v{n + 1}``."""


def model_update_v1_v2(checkpoint: dict) -> None:
    """v2 adds the ``frame_mode`` hyper; v1 models are ``full``.

    :param checkpoint: The checkpoint to update in place.
    """
    checkpoint["model_data"]["model_hypers"].setdefault("frame_mode", "full")


def model_update_v2_v3(checkpoint: dict) -> None:
    """v3 adds the ``electronegativity`` hyper; v2 models have none.

    :param checkpoint: The checkpoint to update in place.
    """
    checkpoint["model_data"]["model_hypers"].setdefault("electronegativity", "none")


def model_update_v3_v4(checkpoint: dict) -> None:
    """v4 adds ``chi0_trainable`` (true) and ``dchi_bound`` (0: unbounded).

    :param checkpoint: The checkpoint to update in place.
    """
    hypers = checkpoint["model_data"]["model_hypers"]
    hypers.setdefault("chi0_trainable", True)
    hypers.setdefault("dchi_bound", 0.0)


def model_update_v4_v5(checkpoint: dict) -> None:
    """v5 adds ``attention_backend`` (``auto``: the attention of v4).

    :param checkpoint: The checkpoint to update in place.
    """
    checkpoint["model_data"]["model_hypers"].setdefault("attention_backend", "auto")


def model_update_v5_v6(checkpoint: dict) -> None:
    """v6 adds ``mesh_normalization``; every model up to v5 has the mesh potential
    divided by the number of mesh points, ``per_point``.

    :param checkpoint: The checkpoint to update in place.
    """
    hypers = checkpoint["model_data"]["model_hypers"]
    hypers.setdefault("mesh_normalization", "per_point")


def trainer_update_v1_v2(checkpoint: dict) -> None:
    """v2 adds the ``compile`` hyper; v1 trained eagerly.

    :param checkpoint: The checkpoint to update in place.
    """
    checkpoint["train_hypers"].setdefault("compile", False)
