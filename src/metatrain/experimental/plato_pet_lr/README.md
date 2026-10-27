# experimental.plato_pet_lr

Developer notes. User-facing hypers live in [`documentation.py`](documentation.py).

A PET backbone whose tokens carry `F` rotation frames (Platonic `T`/`O`/`I`, or one
identity frame), every linear map a `FrameLinear` (a learned kernel on
`R_i^T R_j`), RoPE on `r_ij` in attention with a `log f_c` bias, a learned PME
coupling after layer `couple_at`, and an LES energy of latent charges.

- [`modules/frames.py`](modules/frames.py): groups, Haar/coset samples, `rel`.
- [`modules/layers.py`](modules/layers.py): `FrameLinear` (folded or factored
  contraction, whichever is fewer FLOPs), `PETLayer`, `MeshCoupling`.
- [`modules/ewald.py`](modules/ewald.py): LES energy per atom. Two fixes over the
  reference: a converged, windowed k-set (`|k| < les_kcut / les_sigma`), and the
  Gaussian self energy subtracted from the periodic sum, so periodic and finite
  energies agree up to the tin-foil dipole term.
- [`model.py`](model.py): neighbour lists and the NEF layout come from PET's
  `compute_batch_tensors`; slot 0 of every atom's token set is the atom itself.

Frames: exact invariance under the group with fixed frames, approximate under SO(3).
Training uses O(3) augmentation (`rotational_augmentation`) like PET; with
`frame_sampling: coset` a random global rotation is applied through the frames.
`num_framed_layers` averages the tokens over the frames after that many layers; the
rest run with one frame and no RoPE (so the group invariance is kept).

`frame_mode: attention` (`PETLayer` with `frame_w`, `frame_averaged_attention`):
the tokens have one frame and every `FrameLinear` runs with `F = 1` (a plain
linear map); only the RoPE angles `r_ij . R_f omega` use the `F` group frames,
the softmax weights are averaged over frames before `@ v`, and the edge
embedding's direction term is `sum_f w_f SiLU(W R_f^T r)`. Group invariance is
exact because a group element permutes the frames. Smaller groups: `D2` (4, a
subgroup of `T`), `C3`, `D3`.

`attention_backend: fused` (`PETLayer` with `fused`, `pack_frame_heads`): mlip.cpp's
`torch.ops.mlipcpp.pet_attention` (`torch_ext/`, fused CUDA kernels with exact first
and second derivatives in C++, so force training and TorchScript work). Its bias is
one `(L, L)` map shared by the heads; Plato's `log f_c` bias is per key, so the model
expands it once per forward to `(A, 1, K1, K1)` and every layer, head and frame
shares it. The `F` frames are folded into `F H` heads: `full` packs the per-frame
`q, k, v`; attention-only frames rotate the one-frame `q, k` by every frame and
repeat `v`, and the average of the `F` heads' outputs with `w_f` is exactly the
frame-averaged attention (averaging the weights before `@ v` equals averaging
after). No new kernel is needed. CPU / other dtypes, a missing op or an unsupported
head dimension fall back to `auto`; `FusedAttention` is only a submodule when the
op is registered, so a scripted model references it only then. Export: load the
library first (`MLIPCPP_TORCH_LIB` or `mlipcpp_torch.load()`), then export; run with
the library as a metatomic extension. The traced graph of `compile` keeps the
PyTorch attention (`allow_fused=False`): the op's meta kernels are not
symbolic-shape aware and its backward op has no autograd kernel of its own.
Checkpoint v5 adds the hyper (v4 upgrades to `auto`); the state dict does not depend
on it, and `set_attention_backend` switches a loaded model.

`electronegativity: linear` ([`modules/electronegativity.py`](modules/electronegativity.py)):
`E_chi = sum_i chi_i q_i` on the first latent channel, `chi_i = chi0[Z_i] + dchi_i`.
The LES energy `1/2 q^T K q` is even in `q`, so without this term the charge sign is
a gauge (`q` and `-q` fit the same). `chi0` is a learnable per-element table started
from the QEq electronegativities of Rappe & Goddard, J. Phys. Chem. 95, 3358 (1991)
(= UFF `Xi`, as in RDKit `Code/ForceField/UFF/Params.cpp`), in eV / e:

| H | Li | B | C | N | O | F | Na | Mg | Si | P | S | Cl | K | Ca | Br | I |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4.528 | 3.006 | 5.110 | 5.343 | 6.899 | 8.741 | 10.874 | 2.843 | 3.951 | 4.168 | 5.463 | 6.928 | 8.564 | 2.421 | 3.231 | 7.790 | 6.822 |

Elements not in the table start at its mean over the model's types. `dchi_i` is a
`FrameMLP` on the final node features whose last layer starts at zero. Units: the
network trains on energies divided by the scaler's scale `s` and the LES energy uses
`KE` in eV A / e^2, so its latent charges are `q_m = q / sqrt(s)` with `q` in `e`.
The trainer stores `charge_unit = sqrt(s)` when a run starts, and the energy in model
units is `sum_i chi_i q_m,i / charge_unit`, i.e. `sum_i chi_i q_i` in eV. Only
differences of `chi0` matter (the charges are neutral per structure). What breaks the
sign is the anchored, asymmetric start of `chi0`: the joint map `(q, chi) -> (-q,
-chi)` is still an exact symmetry of the loss, so the sign is fixed by the
initialisation and training dynamics, not by the data alone. Checkpoint v3 adds the
hyper (v2 upgrades to `none`, and `none` models have an unchanged state dict).
[`convert.py`](convert.py): `with_electronegativity` turns a trained `none` model
into a `linear` one (energy changes by exactly `sum_i chi0[Z_i] q_i`),
`flip_charge_sign` maps `q -> -q`.

Not here (from the reference): the direct short-range force fork with two-stage
training, Born effective charges, and the MTS integrator.

TorchScript: frames are buffers, the random-frame sampler is `@torch.jit.unused`,
so the exported model always uses the fixed group.

`tests/`: metatrain's shared suite (`test_basic.py`), symmetry, finite-difference
forces and stress, Ewald convergence/smoothness/finite-periodic consistency
(`test_properties.py`), one training step (`test_train_step.py`), electronegativity
(`test_electronegativity.py`: sign identifiability, conversion, invariance, gradients,
TorchScript, old checkpoints).
