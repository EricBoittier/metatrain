# Bug repro: PET's rank-1 Cartesian (dipole) head collapses during training

`PET._add_output` builds its per-target heads generically from
`target_info.layout`, with no special-casing for target rank/shape. For a
rank-1 Cartesian, `sample_kind: system` target (e.g. a molecular dipole
moment), this collapses: predictions converge to a near-constant output
regardless of the input structure, rather than an actual fit.

RMSE alone doesn't make this obvious -- 0.42 e&middot;A looks like a
mediocre-but-real fit against a target with std ~2.5 e&middot;A. It isn't.
Predicted-vs-reference correlation shows the real picture:

| | value |
|---|---|
| reference std | 2.486 e&middot;A |
| predicted std | 0.197 e&middot;A |
| Pearson r (ref, pred) | **-0.012** |

That's a model that learned nothing about the target beyond its (roughly
zero) mean -- the RMSE is close to what you'd get from predicting a
constant. Energy and forces train normally in the same run; only the
rank-1 Cartesian head is affected.

## Reproduce

```bash
cd bug-repros/pet-dipole-collapse
mtt train options.yaml          # ~30 epochs, small PET model, ~1 min on GPU
python check_correlation.py     # trains-then-evals in one go if model.pt already exists
```

`check_correlation.py` trains no further; run `mtt train options.yaml` first,
then the script evaluates and prints reference/predicted std and Pearson r.
Expect `r` close to 0 (bug present) rather than close to 1 (fixed).

`data/sn2-subset.xyz` is a 1200-structure random subsample of the PhysNet
SN2 reactions set (Zenodo 2605341, Unke & Meuwly, arXiv:1902.08408) --
X<sup>-</sup> + CH<sub>3</sub>Y &rarr; XCH<sub>3</sub> + Y<sup>-</sup>, X/Y &isin;
{F, Cl, Br, I}. `mtt::dipole` is the molecular dipole moment (`e*A`), a
rank-1 Cartesian target with `sample_kind: system`.

## Where this was first found, and one working fix

Found while comparing `experimental.lorem`, PET and SOAP-BPNN on this same
dataset with matched parameter budgets. `experimental.lorem`'s dipole head
had the identical failure mode (it directly regressed a per-atom Cartesian
vector via a Clebsch-Gordan self-product) and was fixed by predicting a
per-atom **scalar** partial charge instead (the same well-posed regression
the energy readout already does), then building the dipole as
`charge * position`, summed over atoms -- the same approach the original
PhysNet paper uses. That took the correlation from ~0.02 to 0.999 on the
full dataset. SOAP-BPNN's native rank-1 handling (a per-atom `o3_lambda=1`
spherical component, summed over atoms) doesn't collapse the same way, but
is a comparatively weak fit (r ~ 0.32) next to the charge-based approach.

PET likely needs an analogous charge-based (or otherwise better-conditioned)
path for rank-1 Cartesian targets rather than treating them like any other
generic tensor shape.
