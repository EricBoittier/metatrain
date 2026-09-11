# Comparing long-range implementations

Developer notes for `experimental.lorem`. User-facing hypers and install
instructions stay in [`documentation.py`](documentation.py).

This architecture is a paper-shaped PyTorch / TorchScript port of LOREM
(*Learning Long-Range Representations with Equivariant Messages*,
[arXiv:2507.19382](https://arxiv.org/abs/2507.19382)). JAX references
live as metawork git submodules — **not** inside this repository:

```bash
git submodule update --init   # from the metawork root
```

- [lorem-jax](https://github.com/lab-cosmo/lorem-jax) — official paper
  code (`lorem.Lorem`, `lorem.LoremBEC`, `e3x.nn.TensorDense`)
- [iris-infra](https://github.com/sirmarcel/iris-infra) — PET trunk +
  tiled Ewald (`iris.pet.PETLR`)

A private torch sibling (not a submodule) is
[fahrenheit-dev](https://github.com/sirmarcel/fahrenheit-dev)
(`fahrenheit.model`, `_e3x`): train in JAX, infer in torch, fp64
energy / force / stress parity. It is not a dependency of this package.

The five paths below are related but not interchangeable. **This
package claims no numerical parity with JAX.** Fahrenheit is the stack
that does (inference only).

```mermaid
flowchart TD
  subgraph iris [iris PETLR]
    PetTrunk["PET trunk name=sr"]
    NodeEmb[node embedding]
    ChargeMLP["MLP to num_charges"]
    TiledEwald["jax-pme batched-tiled Ewald"]
    LrHead["residual mix + energy MLP * lr_scale"]
    PetTrunk --> NodeEmb --> ChargeMLP --> TiledEwald --> LrHead
  end
  subgraph loremPt [experimental.lorem]
    Spherical["Bernstein x Racah SH + TensorDense"]
    PetOpt["optional PetTrunk"]
    EqCharges["scalar + CG spherical charges"]
    TorchPme["torch-pme Ewald / P3M / direct"]
    FeatUpdate["CG mix * lr_scale into scalar features"]
    BecHead["LoremBEC 3x3 APT"]
    Spherical --> EqCharges --> TorchPme --> FeatUpdate
    PetOpt --> EqCharges
    Spherical --> BecHead
    PetOpt --> BecHead
  end
  subgraph petLr [metatrain PET long_range]
    PetBackbone[PET backend]
    LinearQ["Linear feature_dim to feature_dim"]
    SharedPme[utils.long_range.LongRangeFeaturizer]
    Mix["add to node features * 0.5**0.5"]
    PetBackbone --> LinearQ --> SharedPme --> Mix
  end
  subgraph fahr [fahrenheit]
    Spex["torch-spex Bernstein + e3x SH + cutoff"]
    E3x["_e3x Dense / TensorDense / safe_norm"]
    Load["loader: flax FrozenDict to state_dict"]
    Spex --> E3x
    Load --> E3x
  end
```

## iris `PETLR` (JAX PET + tiled Ewald)

Source: [`iris/pet/petlr.py`](https://github.com/sirmarcel/iris-infra/blob/main/iris/pet/petlr.py)
(local: `iris-infra/iris/pet/petlr.py`).

`PETLR` mounts `iris.pet.PET` as `name="sr"` and `_LRModuleTiled` as
`name="lr"`. Node embedding → MLP → `num_charges` scalar channels →
`jax.vmap` over `jaxpme.batched_tiled.Ewald.potentials` → residual mix →
per-atom LR energy × `lr_scale`. Conservative forces/stress from
`iris.pet.predict`. `warm_start_sr` copies a pet-jax SR checkpoint into
the `sr` scope.

Metatrain equivalents (same package style, not a JAX clone):

| iris | metatrain |
| --- | --- |
| PET / UPET trunk | [`pet`](../../pet/model.py) architecture |
| `name="sr"` / `name="lr"` | `LOREM.sr` / `LOREM.lr` |
| `lr_scale` (init 0 warm-start) | `long_range.lr_scale` / `lr_scale_init` |
| jax-pme tiled / packed batch | torch-pme Ewald / P3M / direct on `System` lists |
| `warm_start_sr` | PET checkpoint load / finetune on `pet` |

## lorem-jax / paper LOREM

Official JAX package (submodule `lorem-jax/`). Cited in
[`model.py`](model.py) metadata.

Spherical-harmonic neighbor density, `e3x.nn.TensorDense` self-product,
equivariant charges through each \((\ell, m)\), jax-pme Ewald message,
optional `LoremBEC` APT head (acoustic sum rule).

## `experimental.lorem` (this package)

Paper-shaped TorchScript port for `mtt` + metatomic export.

- [`LoremBackbone`](modules/backbone.py) (`sr`): e3x ``basic_bernstein``
  × Racah SH (lorem-jax defaults; ``bessel`` / ``orthonormal`` remain
  selectable), then [`TensorDense`](modules/tensor_dense.py) (CG
  self-product). Optional scalar message passing. ``trunk: pet`` mounts
  [`PetTrunk`](modules/pet_trunk.py) (``PETBackend`` + spherical sidecar).
- [`LoremLongRangeFeaturizer`](modules/long_range.py) (`lr`): scalar
  charge MLP + `TensorDense` spherical charges; torch-pme Ewald / P3M /
  direct; CG [`TensorProduct`](modules/tensor_dense.py) mix; residual
  gated by `lr_scale`.
- Dipole head: PhysNet-style :math:`\mu=\sum_i q_i r_i` from a learned
  per-atom charge times position (system or per-atom Cartesian rank-1).
- [`BornEffectiveChargeHead`](modules/bec.py): `LoremBEC` /
  `PerParticleTensorPredictor` — Dense+SiLU, `TensorDense` to
  \(\ell\le 2\), CG reconstruction to a 3×3, acoustic sum rule. Requires
  `max_degree >= 2` and a per-atom Cartesian rank-2 target.
- Composition + `Scaler` additives on scalar targets.

Implemented versus the paper: CG `TensorDense` self-product, PhysNet-style
dipole, `LoremBEC`.

Implemented versus iris, in metatrain style: `sr` / `lr` scopes,
`lr_scale`, optional ``trunk: pet``, flax leaf transfer
([`flax_io.py`](flax_io.py) / ``LOREM.load_flax_weights``), and
[`pme_batch`](modules/pme_batch.py) padding / PBC-split helpers (iris
``BM`` / mixed batch). torch-pme still evaluates one ``System`` at a
time; k-space tiled Ewald stays in jax-pme.

## fahrenheit (torch inference, JAX-trained)

Private sibling: [fahrenheit-dev](https://github.com/sirmarcel/fahrenheit-dev).
Train with lorem-jax, dump flax weights, load a torch ``nn.Module`` that
mirrors ``lorem.mlip``. Correctness target is fp64 numerical equivalence
on representative inputs (energies, forces, stress). This
``experimental.lorem`` package in the
[EricBoittier/metatrain](https://github.com/EricBoittier/metatrain) fork
is the other way around: ``mtt`` training and metatomic TorchScript
export, same equations and knobs, **no** bit-exact JAX match.

Package layout in that repo:

```
src/fahrenheit/
  _e3x/          # torch port of e3x ops that aren't in torch-spex:
    dense.py     #   Dense (per-ℓ linear, bias on ℓ=0)
    tensor.py    #   Tensor, TensorDense (CG contraction; mask ℓ1+ℓ2+ℓ3 odd)
    safe.py      #   normalize_and_return_norm, safe_norm (custom Function)
  model.py       # Lorem torch nn.Module — mirrors lorem-jax mlip.py
  loader.py      # map flax FrozenDict → torch state_dict
  export/        # JAX-side dumpers (under [export] extras):
    dump_cg.py   #   Clebsch–Gordan LUT safetensors
    dump_lorem.py#   flat params + config.yaml
```

Radial basis, spherical harmonics, and cutoff come from torch-spex
directly; convention drift vs. e3x is resolved upstream there (see
`libs/torch-spex/spex/angular/e3x_spherical_harmonics.py` and
`libs/torch-spex/spex/cutoff/cosine.py`).

How that maps onto this architecture:

| fahrenheit | experimental.lorem |
| --- | --- |
| `_e3x/dense.py` | `torch.nn.Linear` on scalar features (no per-ℓ Dense) |
| `_e3x/tensor.py` | [`tensor_dense.py`](modules/tensor_dense.py) (`TensorDense`, `TensorProduct`; same odd-ℓ mask) + [`clebsch_gordan.py`](modules/clebsch_gordan.py) |
| `_e3x/safe.py` | `torch.linalg.vector_norm` in [`backbone.py`](modules/backbone.py) (`_degree_norms`) |
| `model.py` | [`model.py`](model.py) + [`backbone.py`](modules/backbone.py) + [`long_range.py`](modules/long_range.py) |
| `loader.py` | [`flax_io.py`](flax_io.py) |
| `export/dump_cg.py` | CG from `wigners` at init (no safetensors LUT) |
| `export/dump_lorem.py` | load-side only (`read_flax_msgpack`) |
| torch-spex radial / SH / cutoff | [`radial.py`](modules/radial.py), [`spherical.py`](modules/spherical.py) (`to_racah`), `_cosine_cutoff` in [`backbone.py`](modules/backbone.py) |

The two torch ports are not drop-in replacements. Fahrenheit clones
e3x ops for JAX parity. This package is metatrain-native (`sr` / `lr`,
optional ``PetTrunk``, dipole / BEC heads, TorchScript).

## metatrain PET `long_range`

[`PET._calculate_long_range_features`](../../pet/model.py) plus
[`LongRangeFeaturizer`](../../utils/long_range.py): one
`Linear(feature_dim, feature_dim)` charge map, torch-pme, mix
`(node + lr) * 0.5**0.5`. Off by default. This is the PET trunk +
Coulomb path in this stack.

## Side-by-side

| | iris PETLR | lorem-jax / paper | fahrenheit | experimental.lorem | PET `long_range` |
| --- | --- | --- | --- | --- | --- |
| SR trunk | `petjax.UPET` as `sr` | spherical + `TensorDense` | torch-spex + `_e3x.TensorDense` | Bernstein × Racah SH + `TensorDense`, or `PetTrunk` | metatrain PET backend |
| Charges | `num_charges` scalars | CG spherical \((\ell, m)\) | same as lorem-jax | scalar + CG `TensorDense` | `feature_dim` scalars |
| LR engine | jax-pme **batched-tiled** | jax-pme Ewald | torch-pme vs jax-pme | torch-pme list batch + ``BM`` pad helpers | torch-pme Ewald / P3M / direct |
| How LR enters | energy × `lr_scale` | feature message | feature message (JAX clone) | CG mix × `lr_scale` (`lr`) | `(node + lr) * 0.5**0.5` |
| BEC / APT | no | `LoremBEC` | not in scope | `BornEffectiveChargeHead` | no |
| Forces / stress | conservative `predict` | autograd / BEC field | autograd, fp64 vs JAX | autograd on energy | PET heads |
| Warm start | `warm_start_sr` | flax ``model.msgpack`` | `dump_lorem` → `loader.py` | ``load_flax_weights`` + metatrain ckpt | PET / PET-MAD checkpoint |
| Runtime | flax, `iris-train` | JAX | torch infer only | TorchScript `mtt` | TorchScript `mtt` |
| Default LR | on | on | on (JAX config) | on (`lr_scale_init: 1.0`) | off |
| JAX energy match | no | reference | **yes** (fp64) | **no** | no |

## Checking parity

This is a metatrain-native port. It uses the same *equations and knobs* as
the paper / lorem-jax, plus iris-style `sr` / `lr` / `lr_scale`. Defaults
now use Bernstein + Racah SH; PME is still torch-pme (not jax-pme tiled
k-space). **No bit-exact energy match** with JAX is claimed (RNG, PME
implementation, and e3x cartesian vs ``m = -ℓ … +ℓ`` order). iris PETLR
is available as ``trunk: pet`` (PET node features + spherical sidecar).
For a torch clone that *does* target fp64 JAX parity, see fahrenheit
(previous section) — that is inference-only and is not this package.

Three layers:

1. **In-repo contracts** (this package's tests, no JAX). Checklist:
   [`tests/test_paper_contracts.py`](tests/test_paper_contracts.py).
   ``test_hypers_keys_overlap_lorem_jax`` prints both param-dict key
   tables; ``test_torch_param_keys_follow_sr_lr_scopes`` prints the
   ``named_parameters`` tree (``sr`` / ``lr``).
2. **External train / eval** (optional, not in this repository):
   `etc/lorem-parity/` in [metawork](https://github.com/EricBoittier/metawork)
   after `git submodule update --init`. Run `mtt` there; run lorem-jax
   examples in a separate JAX venv.
3. **Checkpoint-parity spot check** (optional, not in this repository, no
   bit-exact claim revised): [`modules/jax_parity.py`](modules/jax_parity.py)
   is a *separate* module-for-module port of lorem-jax's SR trunk and LR
   block, kept apart from `LoremBackbone`/`LoremLongRangeFeaturizer` above so
   it can target exact 1:1 weight loading from a real
   `lorem-jax` flax checkpoint instead of `experimental.lorem`'s own
   equations-and-knobs equivalence. Checked against two real periodic
   [lorem-tmlr-archive](https://github.com/sirmarcel/lorem-tmlr-archive)
   checkpoints (AuMgO, bio_dimers): total energy and forces agree with the
   JAX reference to ~1e-4 relative or better (float32 noise floor),
   matching the paper's own reported test-set accuracy on both. [`jax_parity_checkpoint.py`](modules/jax_parity_checkpoint.py)
   is the leaf-by-leaf loader (tested against a synthetic checkpoint in
   [`tests/test_jax_parity.py`](tests/test_jax_parity.py), no JAX needed);
   `etc/lorem-parity/jax_checkpoint_parity/` in metawork has the worked
   example against real archive checkpoints, including a warm-started
   fine-tune that reaches within ~15-20% of the paper's reported accuracy
   on cumulene from a short training run.
4. **This README** — what the five stacks are, and what we do not claim.

| Symbol | Test | Source |
| --- | --- | --- |
| paper default hypers | `test_default_hypers_match_paper` | `documentation.py` |
| printed lorem-jax vs torch keys | `test_hypers_keys_overlap_lorem_jax` | `lorem.Lorem` fields |
| printed ``sr`` / ``lr`` param tree | `test_torch_param_keys_follow_sr_lr_scopes` | iris PETLR scopes |
| one train step, no NaNs | `test_one_training_step_is_finite` | energy / forces / stress / dipole / BEC × batch 1 and 4 |
| isolated-atom force step | `test_force_step_with_isolated_atom_is_finite` | empty neighbor list + forces |
| `l_factors = (2ℓ+1)^{1/4}` | `test_degree_norm_factor_is_two_ell_plus_one_to_the_quarter` | lorem-jax `Lorem` / `LoremBEC` |
| cosine cutoff | `test_cosine_cutoff_is_one_inside_and_zero_at_cutoff` | e3x `cosine_cutoff` |
| 1 + `(L_lr+1)²` charges | `test_charge_layout_is_scalar_plus_spherical_lm` | lorem-jax equivariant charges |
| `1 ⊗ 1 → 0` | `test_cg_one_otimes_one_to_scalar_is_dot_product` | e3x `TensorDense` |
| acoustic sum rule | `test_bec_acoustic_sum_rule` | `lorem.LoremBEC` |
| `lr_scale == 0` no-op | `test_lr_scale_zero_is_noop` | iris `PETLR` |
| `sr` / `lr` scopes | `test_sr_lr_module_scopes` | iris `name="sr"` / `name="lr"` |
| energy / ℓ=1 rotation | `test_long_range_energy_rotation_invariant`, `test_spherical_charges_rotate_as_vectors` | paper equivariance |
| `TensorDense` scalars | `test_tensor_dense_scalar_is_rotation_invariant` | e3x `TensorDense` |
| checkpoint loader round-trips a synthetic checkpoint | `test_jax_parity_checkpoint_loader_round_trips` | `jax_parity_checkpoint.load_checkpoint` |

CI does not import JAX, does not store golden JAX energies, and does not
treat the two stacks as interchangeable.

## Stack

iris and lorem-jax are separate JAX installs (see their READMEs).
fahrenheit is a private torch inference install (`uv sync` in that
repo); it is not part of `metatrain[lorem]`.
`experimental.lorem` needs `pip install 'metatrain[lorem]'`
(`torch-pme`, `sphericart-torch`, `wigners`). Do not pip-install the
JAX packages into the shared metatrain venv from `setup-metawork.sh`.
