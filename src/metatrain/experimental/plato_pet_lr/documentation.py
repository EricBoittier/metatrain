"""
Plato-PET-LR (Experimental)
===========================

A PET backbone whose tokens carry ``F`` rotation frames, from a point group
(``T``, ``O``, ``I``: 12, 24, 60 frames; also ``D2``, ``C3``, ``D3``) or a single
identity frame. A cheaper ``frame_mode: attention`` uses the frames only in the
attention scores (see ``frame_mode``). Every linear map
is a ``FrameLinear``, a learned continuous kernel on SO(3) over frame pairs, so the
model is exactly invariant under the group and approximately under SO(3). Attention
over a centre's neighbours uses rotary encodings of ``r_ij`` in each frame and a
``log f_c`` cutoff bias. Long range: a learned particle-mesh coupling of the tokens
after one layer, and an energy of latent (LES) charges with a converged Ewald sum.
Energies, forces and stress (by autograd).

{{SECTION_INSTALLATION}}

{{SECTION_DEFAULT_HYPERS}}

{{SECTION_MODEL_HYPERS}}

"""

from typing import Literal, Optional

from typing_extensions import NotRequired, TypedDict

from metatrain.composition.documentation import FixedCompositionWeights
from metatrain.scaler.documentation import FixedScalerWeights
from metatrain.utils.loss import LossSpecification


###########################
#  MODEL HYPERPARAMETERS  #
###########################


class ModelHypers(TypedDict):
    """Hyperparameters for the experimental Plato-PET-LR model."""

    cutoff: float = 5.0
    """Neighbour cutoff radius; ``(1 - (r / r_c)^2)^3`` goes to zero there."""
    d_model: int = 64
    """Token width per frame."""
    num_heads: int = 4
    """Attention heads. ``d_model / num_heads`` must be even (RoPE)."""
    num_layers: int = 4
    """Number of PET layers (each one is a message-passing step)."""
    num_rbf: int = 16
    """Gaussian radial basis functions of the edge embedding."""
    frames: Literal["identity", "C3", "D2", "D3", "T", "O", "I"] = "T"
    """Frame group. ``identity`` is one frame: no frame machinery, not invariant.
    ``D2`` (4 frames) is the Klein four-group of diagonal sign flips, a subgroup of
    ``T``; ``C3`` (3) and ``D3`` (6) are cyclic and dihedral groups about [111]."""
    frame_mode: Literal["full", "attention"] = "full"
    """Where the frames act. ``full``: tokens carry ``F`` frames and every linear map
    is a ``FrameLinear`` (cost ~``F n_basis`` times a plain layer). ``attention``:
    tokens carry one frame, every linear map and MLP runs once, and the frames enter
    only through the RoPE rotation of queries and keys (angles ``r_ij . R_f omega``);
    the attention weights are averaged over the frames before the values. The edge
    embedding uses ``sum_f w_f SiLU(W R_f^T r_ij)``. A group element permutes the
    frames, so the model stays exactly group invariant; the extra cost is ``F``
    attention-score evaluations per layer. Needs ``rope``. With one token frame
    ``n_basis`` only reparametrises each linear map (more parameters, same FLOPs);
    the layers after ``num_framed_layers`` run without RoPE as in ``full``."""
    num_framed_layers: Optional[int] = None
    """Layers that carry frames (all when ``None``). After them the tokens are
    averaged over the frames, and the remaining layers run with one frame and
    without RoPE, which keeps the group invariance."""
    frame_sampling: Literal["fixed", "coset", "haar"] = "fixed"
    """Frames during training: the group, a random coset ``Q G`` (a random global
    rotation), or ``F`` Haar-random rotations. Evaluation always uses the group."""
    n_basis: int = 8
    """Basis functions of each ``FrameLinear`` kernel."""
    kernel_hidden: int = 32
    """Hidden width of the kernel MLP on ``R_i^T R_j``."""
    frame_linear_mode: Literal["auto", "folded", "factored"] = "auto"
    """Contraction order of ``FrameLinear``; ``auto`` takes the fewer FLOPs."""
    rope: bool = True
    """Rotary encoding of ``r_ij`` (in each frame) on queries and keys."""
    rope_max_freq: float = 3.0
    """Largest RoPE wavenumber, in inverse length units."""
    mesh_coupling: bool = True
    """Learned PME coupling of the tokens (periodic systems only)."""
    couple_at: int = 1
    """Index of the layer after which the mesh coupling is applied."""
    mesh_channels: int = 8
    """Channels per frame carried to the mesh."""
    mesh_spacing: float = 1.0
    """Mesh spacing; the mesh has ``ceil(|a_i| / spacing)`` points per cell vector."""
    mesh_sigma: float = 1.5
    """Gaussian width of the mesh filter."""
    les: bool = True
    """Energy of learned latent charges (LES)."""
    num_charges: int = 1
    """Latent charge channels."""
    les_sigma: float = 1.0
    """Width of the Gaussian latent charges."""
    electronegativity: Literal["none", "linear"] = "none"
    """Energy linear in the latent charges, ``sum_i chi_i q_i`` (needs ``les``).
    The LES energy is even in ``q``, so without it the sign of the latent charges
    is arbitrary (``q`` and ``-q`` fit equally well). ``linear``: ``chi_i =
    chi0[Z_i] + dchi_i``, a learnable per-element table in eV / e started from the
    QEq electronegativities of Rappe & Goddard (1991) (H 4.528, C 5.343, N 6.899,
    O 8.741, S 6.928, ...; elements without a value start at the mean), plus a
    per-atom correction from the final features that starts at zero. It acts on
    the first charge channel. Charges are in ``e`` with ``q = sqrt(s) q_m``, ``s``
    the energy scale of the scaler (stored as ``charge_unit`` when training
    starts), so ``KE q^2 / r`` and ``chi q`` are both in eV. ``chi0`` is fixed only
    up to a common shift (the latent charges are neutral per structure)."""
    chi0_trainable: bool = True
    """With ``electronegativity: linear``: learn ``chi0`` (``false``: keep the QEq
    table fixed, a buffer, so the per-element anchor that selects the charge sign
    cannot drift)."""
    dchi_bound: float = 0.0
    """With ``electronegativity: linear`` and ``> 0``: ``dchi_i = b tanh(head_i)``
    in eV / e, so the per-atom correction cannot undo the QEq ordering of the
    elements (H 4.528 < C 5.343 < N 6.899 < O 8.741, steps of 0.8-1.8 eV / e)."""
    les_kcut: float = 6.0
    """Reciprocal cutoff in units of ``1 / les_sigma``; a cosine window over the
    last ``1 / les_sigma`` keeps the energy smooth under strain."""
    seed: int = 0
    """Seed of the RoPE directions."""
    attention_backend: Literal["auto", "sdpa", "manual", "fused"] = "auto"
    """Attention implementation; all give the same model. ``auto``: manual softmax
    when training (it has a double backward, for force losses) and
    ``scaled_dot_product_attention`` in evaluation. ``manual`` / ``sdpa``: always
    that one (SDPA cannot train on forces). Without ``fused``, the frame average of
    ``frame_mode: attention`` is always the manual softmax. ``fused``: mlip.cpp's
    fused CUDA kernels (``torch.ops.mlipcpp.pet_attention``: exact first and
    second derivatives, no ``K1 x K1`` matrix per head stored), with the frames
    folded into ``F x num_heads`` heads that share the cutoff bias. Needs
    ``d_model / num_heads`` in 8, 16, 32, 64 and the op: the shared library named
    by ``MLIPCPP_TORCH_LIB``, or the ``mlipcpp_torch`` package (mlip.cpp
    ``torch_ext``). Without them, and for CPU or other dtypes, ``fused`` falls
    back to ``auto``. An exported model
    keeps the backend it was exported with; load the library as a metatomic
    extension to run it. Not used in the traced graph of ``compile``."""


##############################
#  TRAINER HYPERPARAMETERS   #
##############################


class TrainerHypers(TypedDict):
    """Hyperparameters for training Plato-PET-LR models."""

    distributed: NotRequired[bool]
    """Whether to use distributed training. When not set, distributed training
    is enabled automatically when running under more than one SLURM task.
    Setting this option explicitly is deprecated."""
    distributed_port: int = 39591
    """Port for DDP communication."""
    batch_size: int = 8
    """The number of samples to use in each batch of training. This
    hyperparameter controls the tradeoff between training speed and memory usage. In
    general, larger batch sizes will lead to faster training, but might require more
    memory."""
    max_atoms_per_batch: Optional[int] = None
    """If set, use greedy atom-count packing instead of fixed ``batch_size``.
    Structures are accumulated into each batch until adding another would exceed this
    limit, producing variable numbers of structures per batch. Supported with any
    dataset type. When set, ``batch_size`` is ignored for constructing training
    and validation batches (it is still used internally for composition model and
    scaler fitting)."""
    min_atoms_per_batch: int = 0
    """Minimum total number of atoms required to keep a batch when
    ``max_atoms_per_batch`` is set. Batches whose total atom count falls below this
    threshold are discarded during packing. Defaults to ``0`` (no minimum)."""
    num_epochs: int = 100
    """Number of epochs."""
    warmup_fraction: float = 0.01
    """Fraction of training steps used for learning rate warmup, before the
    cosine decay."""
    learning_rate: float = 0.001
    """Learning rate."""
    weight_decay: Optional[float] = None
    """Weight decay. If set, AdamW is used instead of Adam."""

    log_interval: int = 1
    """Interval to log metrics."""
    checkpoint_interval: int = 100
    """Interval to save checkpoints."""
    scale_targets: bool = True
    """
    Normalize targets to unit std during training.

    If true, a single scale is computed for each target, given by the uncentered
    standard deviation across all values in the dataset for that target.

    For targets with more than one property (i.e. > 1 block or >= 1 block with > 1
    property), per-property scales are also computed, and used to re-scale model
    predictions.

    See also :ref:`scale-targets`.
    """
    atomic_baseline: FixedCompositionWeights | str = {}
    """The baselines for each target.

    By default, ``metatrain`` fits a linear model (:class:`CompositionModel
    <metatrain.composition.CompositionModel>`) to compute the least-squares
    baseline for each atomic species for each target.

    This hyperparameter instead lets you provide those baselines, either as a
    dictionary or as a path to a pre-trained composition model checkpoint:

    - a dictionary whose keys are target names, and whose values are either
      one baseline for every atomic type, or a dictionary mapping atomic types
      to baselines.
    - a string path to a ``.ckpt`` file from a pre-trained composition model.

    For example:

    - ``atomic_baseline: {"energy": {1: -0.5, 6: -10.0}}`` fixes the energy
      baseline for hydrogen (Z=1) to -0.5 and for carbon (Z=6) to -10.0, and
      fits the rest.
    - ``atomic_baseline: {"energy": -5.0}`` fixes the energy baseline for
      every atomic type to -5.0.
    - ``atomic_baseline: {"mtt:dos": 0.0}`` sets that target's baseline to
      0.0, which disables the atomic baseline for it.
    - ``atomic_baseline: "/path/to/model.ckpt"`` loads a pre-trained
      composition model checkpoint instead of fitting one.

    The baseline is subtracted from the targets during training and added
    back at evaluation. It is a per-atom contribution, so for a per-structure
    target such as the total energy it is multiplied by the number of atoms
    of that type.
    """
    fixed_scaling_weights: FixedScalerWeights | str = {}
    """Weights for target scaling.

    This is passed to the ``fixed_weights`` argument of
    :meth:`Scaler.train_model <metatrain.scaler.Scaler.train_model>`,
    see its documentation to understand exactly what to pass here.

    A path to a model checkpoint is also accepted. If that checkpoint is a
    Scaler model, the pre-trained scaler is loaded. When passing a checkpoint
    for the scaler, ``atomic_baseline`` must also be a checkpoint for a
    composition model.
    """
    per_structure_targets: list[str] = []
    """Targets to calculate per-structure losses."""
    num_workers: Optional[int] = None
    """Number of workers for data loading. If not provided, it is set
    automatically."""
    log_mae: bool = False
    """Log MAE alongside RMSE."""
    log_separate_blocks: bool = False
    """Log per-block error."""
    best_model_metric: Literal["rmse_prod", "mae_prod", "loss"] = "rmse_prod"
    """Metric used to select best checkpoint (e.g., ``rmse_prod``)."""
    grad_clip_norm: float = 1.0
    """Maximum gradient norm value."""

    rotational_augmentation: bool = True
    """Random O(3) rotations/inversions of each training structure, as PET does.
    Platonic frames make the model exactly invariant only under its group."""

    compile: bool = False
    """Train with ``torch.compile``: the energy and its position gradient are traced
    into one graph (``make_fx``, as in SPACE) and compiled, so the force loss runs
    through compiled code. Batches the graph does not cover (mixed mesh sizes,
    mixed periodic and open structures, stress targets) and validation run
    eagerly. Not used with distributed training. Compiling takes a few minutes per
    distinct batch layout (number of structures, mesh size)."""

    loss: str | dict[str, LossSpecification | str] = "mse"
    """This section describes the loss function to be used. See the
    :ref:`loss-functions` for more details."""
