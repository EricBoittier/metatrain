"""Rotation frames: point groups (C3, D2, D3, T, O, I) and Haar samples.

Called once at construction (or per step for random frames). Not scripted.
"""

import math
from typing import Tuple

import torch


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    m = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        -1,
    )  # fmt: skip
    return m.reshape(q.shape[:-1] + (3, 3))


def haar_rotations(
    n: int, dtype=torch.float64, device=None, generator=None
) -> torch.Tensor:
    q = torch.randn(n, 4, dtype=dtype, generator=generator).to(device)
    q = torch.nn.functional.normalize(q, dim=-1)
    return quat_to_matrix(q)


def axis_angle(axis, angle: float) -> torch.Tensor:
    a = torch.tensor(axis, dtype=torch.float64)
    a = a / a.norm()
    K = torch.zeros(3, 3, dtype=torch.float64)
    K[0, 1], K[0, 2], K[1, 2] = -a[2], a[1], -a[0]
    K = K - K.T
    return (
        torch.eye(3, dtype=torch.float64)
        + math.sin(angle) * K
        + (1 - math.cos(angle)) * (K @ K)
    )


def _closure(gens) -> torch.Tensor:
    key = lambda M: tuple(torch.round(M * 1e6).long().flatten().tolist())  # noqa: E731
    elems = [torch.eye(3, dtype=torch.float64)]
    seen, frontier = {key(elems[0])}, list(elems)
    while frontier:
        new = []
        for A in frontier:
            for g in gens:
                B = g @ A
                if key(B) not in seen:
                    seen.add(key(B))
                    elems.append(B)
                    new.append(B)
        frontier = new
    return torch.stack(elems)


def platonic_group(name: str) -> torch.Tensor:
    """Rotation matrices of a point group: identity (1), C3 (3, about [111]), D2 (4,
    the Klein four-group of diagonal sign flips, a subgroup of T), D3 (6), T (12),
    O (24), I (60)."""
    if name == "identity":
        return torch.eye(3, dtype=torch.float64)[None]
    c3 = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    c2 = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64))
    if name == "C3":
        return _closure([c3])
    if name == "D2":
        c2x = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float64))
        return _closure([c2x, c2])
    if name == "D3":
        return _closure([c3, axis_angle([1, -1, 0], math.pi)])
    if name == "T":
        return _closure([c3, c2])
    if name == "O":
        return _closure([c3, axis_angle([0, 0, 1], math.pi / 2)])
    if name == "I":
        phi = (1 + 5**0.5) / 2
        return _closure([c3, c2, axis_angle([0, 1, phi], 2 * math.pi / 5)])
    raise ValueError(f"unknown frame group '{name}'")


def relative_rotations(R: torch.Tensor) -> torch.Tensor:
    """``rel[i, j] = R_i^T R_j`` flattened to 9, the input of the frame kernel."""
    F = R.shape[0]
    return (R.transpose(-1, -2)[:, None] @ R[None]).reshape(F, F, 9)


def sample_frames(
    mode: str, G: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(R, rel, w)`` for ``mode`` in fixed (``G``), coset (``Q G``), haar.

    A coset ``Q G`` leaves ``rel`` unchanged, so it equals a global rotation
    of the input by ``Q^T`` with fixed frames.
    """
    if mode == "fixed":
        R = G
    elif mode == "coset":
        R = haar_rotations(1, G.dtype, G.device)[0] @ G
    elif mode == "haar":
        R = haar_rotations(G.shape[0], G.dtype, G.device)
    else:
        raise ValueError(f"unknown frame sampling '{mode}'")
    F = R.shape[0]
    return (
        R,
        relative_rotations(R),
        torch.full((F,), 1.0 / F, dtype=R.dtype, device=R.device),
    )


GROUP_ORDERS = {"identity": 1, "C3": 3, "D2": 4, "D3": 6, "T": 12, "O": 24, "I": 60}
assert {n: platonic_group(n).shape[0] for n in GROUP_ORDERS} == GROUP_ORDERS
