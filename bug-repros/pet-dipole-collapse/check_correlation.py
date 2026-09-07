#!/usr/bin/env python3
"""Train PET on options.yaml, evaluate on the same data, and report how well
the predicted dipole actually correlates with the reference -- not just its
RMSE, which looks deceptively reasonable on its own.

Expected on a working dipole head: Pearson r close to 1, pred_std close to
ref_std. What this bug reproduces instead: r close to 0, pred_std much
smaller than ref_std -- the model has converged to a near-constant output
regardless of RMSE looking like "an OK fit" in isolation.

    mtt train options.yaml
    python check_correlation.py
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from ase.io import read

HERE = Path(__file__).parent


def main() -> int:
    model = HERE / "model.pt"
    if not model.is_file():
        print("model.pt not found -- run `mtt train options.yaml` first.", file=sys.stderr)
        return 1

    eval_out = HERE / "eval-out.xyz"
    subprocess.run(
        [sys.executable, "-m", "metatrain", "eval", str(model), str(HERE / "eval.yaml"),
         "-o", str(eval_out)],
        check=True,
        cwd=HERE,
    )

    ref_frames = read(HERE / "data" / "sn2-subset.xyz", index=":")
    pred_frames = read(eval_out, index=":")
    assert len(ref_frames) == len(pred_frames)

    ref = np.concatenate(
        [np.asarray(f.info["dipole_moment"], dtype=float).reshape(-1) for f in ref_frames]
    )
    pred = np.concatenate(
        [np.asarray(f.info["mtt::dipole"], dtype=float).reshape(-1) for f in pred_frames]
    )

    r = float(np.corrcoef(ref, pred)[0, 1])
    rmse = float(np.sqrt(np.mean((ref - pred) ** 2)))
    print(f"n_components   = {len(ref)}")
    print(f"ref_std        = {ref.std():.4f} e*A")
    print(f"pred_std       = {pred.std():.4f} e*A")
    print(f"pearson r      = {r:.4f}")
    print(f"raw RMSE       = {rmse:.4f} e*A  (looks deceptively OK vs ref_std alone)")
    print()
    if r < 0.15:
        print("BUG REPRODUCED: near-zero correlation -- the model predicts an")
        print("essentially constant dipole regardless of input structure.")
        return 0
    print("Correlation looks real (r >= 0.15) -- bug did not reproduce this run.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
