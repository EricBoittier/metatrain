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

Not here (from the reference): the direct short-range force fork with two-stage
training, Born effective charges, and the MTS integrator.

TorchScript: frames are buffers, the random-frame sampler is `@torch.jit.unused`,
so the exported model always uses the fixed group.

`tests/`: metatrain's shared suite (`test_basic.py`), symmetry, finite-difference
forces and stress, Ewald convergence/smoothness/finite-periodic consistency
(`test_properties.py`), one training step (`test_train_step.py`).
