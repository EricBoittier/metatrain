"""Checkpoint upgrade helpers for Plato-PET-LR:
``model_update_v{n}_v{n + 1}`` / ``trainer_update_v{n}_v{n + 1}``."""


def model_update_v1_v2(checkpoint: dict) -> None:
    """v2 adds the ``frame_mode`` hyper; v1 models are ``full``.

    :param checkpoint: The checkpoint to update in place.
    """
    checkpoint["model_data"]["model_hypers"].setdefault("frame_mode", "full")
