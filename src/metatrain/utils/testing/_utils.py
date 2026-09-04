from collections import namedtuple
from importlib.util import find_spec
from typing import Optional

import pytest

from metatrain.utils.io import download_model_from_hf


DepStatus = namedtuple("DepStatus", ["present", "message"])
if find_spec("wandb"):
    WANDB_AVAILABLE = DepStatus(True, "present")
else:
    WANDB_AVAILABLE = DepStatus(False, "wandb not installed")


def is_huggingface_rate_limit(err: BaseException) -> bool:
    """Return whether ``err`` looks like an HTTP 429 from the Hugging Face Hub.

    :param err: Exception raised while contacting the Hub.
    :return: ``True`` if the error is a rate-limit response.
    """
    status = getattr(err, "status_code", None) or getattr(err, "code", None)
    if status == 429:
        return True
    response = getattr(err, "response", None)
    if getattr(response, "status_code", None) == 429:
        return True
    message = str(err).lower()
    return "too many requests" in message or "http error 429" in message


def download_hf_checkpoint_or_skip(
    repo_id: str,
    filename: str,
    revision: Optional[str] = None,
) -> str:
    """Download a Hub checkpoint, skipping the test if the Hub rate-limits us.

    Tests that used ``urllib.request.urlretrieve`` failed CI on shared GitHub
    Actions IPs (HTTP 429) and leaked ``HTTPError`` tempfiles that pytest then
    treated as unraisable errors in later tests. ``huggingface_hub`` retries and
    caches; remaining 429s are skipped as infrastructure, not product failures.

    :param repo_id: Hub repository ID, for example ``lab-cosmo/upet``.
    :param filename: Path of the file inside the repository.
    :param revision: Branch, tag, or commit to download from.
    :return: Local path of the cached checkpoint.
    """
    try:
        return download_model_from_hf(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
        )
    except Exception as err:
        if is_huggingface_rate_limit(err):
            pytest.skip(f"Hugging Face Hub rate-limited {repo_id}/{filename}: {err}")
        raise
