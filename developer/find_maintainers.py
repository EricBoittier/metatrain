"""Write a comment mentioning the maintainers of the architectures touched by a PR.

Usage: python find_maintainers.py <changed-files.txt> <pr-author-login>

Maintainers are read from the ``__maintainers__`` list of each architecture's
``__init__.py``. The files are parsed rather than imported, so this runs without
installing metatrain and never executes repository code (it is used from a
``pull_request_target`` workflow).
"""

import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ARCHITECTURES = ROOT / "src" / "metatrain"


def maintainers(init: Path) -> list[str]:
    """GitHub handles listed in ``__maintainers__`` of an ``__init__.py``."""
    for node in ast.parse(init.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == "__maintainers__" for target in node.targets
        ):
            return [handle for _, handle in ast.literal_eval(node.value)]
    return []


def comment(changed: list[Path], author: str) -> str:
    """Comment mentioning maintainers of touched architectures, or ``""``."""
    touched = {
        init.parent.relative_to(ROOT): [h for h in handles if h != f"@{author}"]
        for init in sorted(ARCHITECTURES.rglob("__init__.py"))
        if (handles := maintainers(init))
        and any(f.is_relative_to(init.parent.relative_to(ROOT)) for f in changed)
    }
    lines = [f"- `{arch}/`: {' '.join(h)}" for arch, h in touched.items() if h]
    if not lines:
        return ""
    return "\n".join(
        ["This PR modifies the following architectures, pinging their maintainers:", ""]
        + lines
    )


if __name__ == "__main__":
    changed_files, author = sys.argv[1:]
    print(comment([Path(f) for f in Path(changed_files).read_text().split()], author))
