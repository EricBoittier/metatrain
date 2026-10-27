import os
import warnings
from typing import Final, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .utilities import DummyModule


AVAILABLE_NORMALIZATIONS = ["LayerNorm", "RMSNorm"]
AVAILABLE_TRANSFORMER_TYPES = ["PostLN", "PreLN"]
AVAILABLE_ACTIVATIONS = ["SiLU", "SwiGLU"]
AVAILABLE_ATTENTION_BACKENDS = ["auto", "sdpa", "manual", "fused"]
# head dimensions (d_pet / num_heads) supported by torch.ops.mlipcpp.pet_attention
FUSED_ATTENTION_HEAD_DIMS: List[int] = [8, 16, 32, 64]

# SDPA's CUDA backward grids the attention batch (n_nodes) over gridDim.y/z, which
# overflows above this many nodes. We work around it by looping SDPA over chunks of
# the batch dimension instead.
MAX_CUDA_GRID_DIM: Final[int] = 65535


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, activation: str) -> None:
        super().__init__()

        # Check if activation is "swiglu" string
        if activation.lower() == "swiglu":
            # SwiGLU mode: single projection produces both "value" and "gate"
            self.w_in = nn.Linear(d_model, 2 * dim_feedforward)
            self.w_out = nn.Linear(dim_feedforward, d_model)
            self.activation = torch.nn.Identity()
            self.is_swiglu = True
        else:
            # Standard mode: regular activation function
            self.w_in = nn.Linear(d_model, dim_feedforward)
            self.w_out = nn.Linear(dim_feedforward, d_model)
            self.activation = getattr(F, activation.lower())
            self.is_swiglu = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_swiglu:
            # SwiGLU activation: split into value and gate
            v, g = self.w_in(x).chunk(2, dim=-1)
            x = v * torch.sigmoid(g)
            x = self.w_out(x)
        else:
            # Standard activation
            x = self.w_in(x)
            x = self.activation(x)
            x = self.w_out(x)
        return x


def fused_attention_op_available() -> bool:
    """Whether ``torch.ops.mlipcpp.pet_attention`` (mlip.cpp's fused PET attention,
    see :func:`load_fused_attention_op`) is registered in this process."""
    try:
        torch.ops.mlipcpp.pet_attention
    except (AttributeError, RuntimeError):
        return False
    return True


def load_fused_attention_op() -> bool:
    """Register ``torch.ops.mlipcpp.pet_attention`` if it is not registered yet.

    The op lives in mlip.cpp (``torch_ext/``, a CUDA extension with exact first and
    second derivatives). It is loaded from the shared library named by the
    ``MLIPCPP_TORCH_LIB`` environment variable, or else through the
    ``mlipcpp_torch`` Python package (``mlipcpp_torch.load()``, which builds the
    library on first use).

    :return: whether the op is available.
    """
    if fused_attention_op_available():
        return True
    lib = os.environ.get("MLIPCPP_TORCH_LIB")
    try:
        if lib:
            torch.ops.load_library(lib)
        else:
            import mlipcpp_torch  # noqa: PLC0415

            mlipcpp_torch.load()
    except Exception as e:  # noqa: BLE001
        warnings.warn(
            "attention_backend 'fused': could not load torch.ops.mlipcpp."
            f"pet_attention ({type(e).__name__}: {e}); falling back to the "
            "default attention",
            stacklevel=2,
        )
        return False
    return fused_attention_op_available()


class FusedAttention(nn.Module):
    """``torch.ops.mlipcpp.pet_attention``: fused softmax attention of the packed
    ``qkv`` projection ``(N, L, 3 * H * d)`` (q | k | v, heads contiguous) with a
    bias ``(N, L, L)`` shared by the heads; returns ``(N, L, H * d)``. Exact first
    and second derivatives (force training), CUDA float32 / float64 only.

    Only instantiated when the op is registered, so that TorchScript only sees the
    op when it exists (otherwise :class:`NoFusedAttention` is used)."""

    def forward(
        self, qkv: torch.Tensor, bias: torch.Tensor, heads: int, scale: float
    ) -> torch.Tensor:
        """
        :param qkv: packed projections, ``(N, L, 3 * heads * d)``
        :param bias: additive attention bias ``(N, L, L)`` (query, key)
        :param heads: number of heads
        :param scale: scale of the dot products
        :return: attention output ``(N, L, heads * d)``
        """
        return torch.ops.mlipcpp.pet_attention(qkv, bias, heads, scale)[0]


class NoFusedAttention(nn.Module):
    """Placeholder for :class:`FusedAttention` when the op is not available."""

    def forward(
        self, qkv: torch.Tensor, bias: torch.Tensor, heads: int, scale: float
    ) -> torch.Tensor:
        """
        :param qkv: unused
        :param bias: unused
        :param heads: unused
        :param scale: unused
        :return: never returns
        """
        raise RuntimeError("torch.ops.mlipcpp.pet_attention is not available")


class AttentionBlock(nn.Module):
    """
    Multi-head attention block.

    :param total_dim: The total dimension of the input and output tensors.
    :param num_heads: The number of attention heads.
    :param temperature: An additional scaling factor for attention scores.
           This is combined with the standard scaling by the square root of
           the head dimension.
    :param epsilon: A small value to avoid division by zero.
    :param attention_backend: ``"fused"`` uses ``torch.ops.mlipcpp.pet_attention``
        (see :func:`load_fused_attention_op`) for CUDA float32 / float64 inputs, if
        the op can be loaded and the head dimension is supported; any other value
        keeps the SDPA / manual attention selected by ``use_manual_attention``.
    """

    def __init__(
        self,
        total_dim: int,
        num_heads: int,
        temperature: float,
        epsilon: float = 1e-15,
        attention_backend: str = "auto",
    ) -> None:
        super(AttentionBlock, self).__init__()

        self.input_linear = nn.Linear(total_dim, 3 * total_dim)
        self.output_linear = nn.Linear(total_dim, total_dim)

        self.num_heads = num_heads
        self.epsilon = epsilon
        self.temperature = temperature
        if total_dim % num_heads != 0:
            raise ValueError("total dimension is not divisible by the number of heads")
        self.head_dim = total_dim // num_heads
        self.max_cuda_grid_dim = MAX_CUDA_GRID_DIM
        self._warned_cuda_grid_dim = False

        self.use_fused_attention = False
        self.fused_attention: nn.Module = NoFusedAttention()
        if attention_backend == "fused":
            if self.head_dim not in FUSED_ATTENTION_HEAD_DIMS:
                warnings.warn(
                    f"attention_backend 'fused' needs a head dimension in "
                    f"{FUSED_ATTENTION_HEAD_DIMS}, got {self.head_dim} "
                    "(d_pet / num_heads); falling back to the default attention",
                    stacklevel=2,
                )
            elif load_fused_attention_op():
                self.use_fused_attention = True
                self.fused_attention = FusedAttention()

    def forward(
        self, x: torch.Tensor, cutoff_factors: torch.Tensor, use_manual_attention: bool
    ) -> torch.Tensor:
        """
        Forward pass for the attention block.

        :param x: The input tensor, of shape (batch_size, seq_length, total_dim)
        :param cutoff_factors: The cutoff factors for the edges, of shape
            (batch_size, seq_length, seq_length)
        :param use_manual_attention: Whether to use the manual attention implementation
            (which supports double backward, needed for training with conservative
            forces), or the built-in PyTorch attention (which does not support double
            backward).
        :return: The output tensor, of shape (batch_size, seq_length, total_dim)
        """
        initial_shape = x.shape
        x = self.input_linear(x)
        scale = 1.0 / (self.head_dim**0.5 * self.temperature)
        if (
            self.use_fused_attention
            and x.is_cuda
            and (x.dtype == torch.float32 or x.dtype == torch.float64)
        ):
            # same bias as below, (N, L, L) shared by the heads; the packed
            # (N, L, 3, H, d) projection is the op's qkv layout
            bias = torch.log(torch.clamp(cutoff_factors, self.epsilon))
            x = self.fused_attention(x, bias, self.num_heads, scale)
            return self.output_linear(x)
        x = x.reshape(
            initial_shape[0], initial_shape[1], 3, self.num_heads, self.head_dim
        )
        x = x.permute(2, 0, 3, 1, 4)

        queries, keys, values = x[0], x[1], x[2]
        attn_weights = torch.clamp(cutoff_factors[:, None, :, :], self.epsilon)
        attn_weights = torch.log(attn_weights)
        if use_manual_attention:
            x = manual_attention(queries, keys, values, attn_weights, self.temperature)
        elif x.requires_grad and initial_shape[0] > self.max_cuda_grid_dim:
            if not self._warned_cuda_grid_dim:
                warnings.warn(
                    "The number of nodes exceeds the CUDA grid dimension of "
                    f"{self.max_cuda_grid_dim}, which may cause an overflow "
                    "in the backward pass of the "
                    "torch.nn.functional.scaled_dot_product_attention."
                    "Falling back to a for-loop over batch chunks, which "
                    "may reduce performance. If you are running the model inference, "
                    "please consider using the LAMMPS interface with > 1 MPI rank to "
                    "enable domain decomposition. More details can be found on "
                    "https://lab-cosmo.github.io/upet/latest/usage/lammps.html ",
                    stacklevel=2,
                )
                self._warned_cuda_grid_dim = True
            outputs = []
            for start in range(0, initial_shape[0], self.max_cuda_grid_dim):
                end = start + self.max_cuda_grid_dim
                outputs.append(
                    torch.nn.functional.scaled_dot_product_attention(
                        queries[start:end],
                        keys[start:end],
                        values[start:end],
                        attn_mask=attn_weights[start:end],
                        scale=scale,
                    )
                )
            x = torch.cat(outputs, dim=0)
        else:
            x = torch.nn.functional.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=attn_weights,
                scale=scale,
            )
        x = x.transpose(1, 2).reshape(initial_shape)
        x = self.output_linear(x)
        return x


class TransformerLayer(torch.nn.Module):
    """
    Single layer of a Transformer.

    :param d_model: The dimension of the model.
    :param n_heads: The number of attention heads.
    :param dim_node_features: The dimension of the node features.
    :param dim_feedforward: The dimension of the feedforward network.
    :param norm: The normalization type, either "LayerNorm" or "RMSNorm".
    :param activation: The activation function, either "SiLU" or "SwiGLU".
    :param transformer_type: The type of transformer, either "PostLN" or "PreLN".
    :param temperature: An additional scaling factor for attention scores.
    :param attention_backend: See :class:`AttentionBlock`.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_node_features: int,
        dim_feedforward: int = 512,
        norm: str = "LayerNorm",
        activation: str = "SiLU",
        transformer_type: str = "PostLN",
        temperature: float = 1.0,
        attention_backend: str = "auto",
    ) -> None:
        super(TransformerLayer, self).__init__()
        self.attention = AttentionBlock(
            d_model, n_heads, temperature, attention_backend=attention_backend
        )
        self.transformer_type = transformer_type
        self.d_model = d_model
        norm_class = getattr(nn, norm)
        self.norm_attention = norm_class(d_model)
        self.norm_mlp = norm_class(d_model)
        self.mlp = FeedForward(d_model, dim_feedforward, activation)
        self.expanded_node_features = False
        if dim_node_features != d_model:
            self.expanded_node_features = True
            self.center_contraction = nn.Linear(dim_node_features, d_model)
            self.center_expansion = nn.Linear(d_model, dim_node_features)
            self.norm_center_features = norm_class(dim_node_features)
            self.center_mlp = FeedForward(
                dim_node_features, 2 * dim_node_features, activation
            )
        else:
            self.center_contraction = torch.nn.Identity()
            self.center_expansion = torch.nn.Identity()
            self.norm_center_features = torch.nn.Identity()
            self.center_mlp = torch.nn.Identity()

    def _forward_pre_ln_impl(
        self,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        cutoff_factors: torch.Tensor,
        use_manual_attention: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.expanded_node_features:
            input_node_embeddings = self.center_contraction(node_embeddings)
        else:
            input_node_embeddings = node_embeddings
        tokens = torch.cat([input_node_embeddings, edge_embeddings], dim=1)
        new_tokens = self.attention(
            self.norm_attention(tokens), cutoff_factors, use_manual_attention
        )
        output_node_embeddings, output_edge_embeddings = torch.split(
            new_tokens, [1, new_tokens.shape[1] - 1], dim=1
        )
        if self.expanded_node_features:
            output_node_embeddings = node_embeddings + self.center_expansion(
                output_node_embeddings
            )
            output_node_embeddings = output_node_embeddings + self.center_mlp(
                self.norm_center_features(output_node_embeddings)
            )

        output_edge_embeddings = edge_embeddings + output_edge_embeddings
        output_edge_embeddings = output_edge_embeddings + self.mlp(
            self.norm_mlp(output_edge_embeddings)
        )

        return output_node_embeddings, output_edge_embeddings

    def _forward_post_ln_impl(
        self,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        cutoff_factors: torch.Tensor,
        use_manual_attention: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.expanded_node_features:
            input_node_embeddings = self.center_contraction(node_embeddings)
        else:
            input_node_embeddings = node_embeddings
        tokens = torch.cat([input_node_embeddings, edge_embeddings], dim=1)
        tokens = self.norm_attention(
            tokens + self.attention(tokens, cutoff_factors, use_manual_attention)
        )
        tokens = self.norm_mlp(tokens + self.mlp(tokens))
        output_node_embeddings, output_edge_embeddings = torch.split(
            tokens, [1, tokens.shape[1] - 1], dim=1
        )
        if self.expanded_node_features:
            output_node_embeddings = node_embeddings + self.center_expansion(
                output_node_embeddings
            )
            output_node_embeddings = output_node_embeddings + self.center_mlp(
                self.norm_center_features(output_node_embeddings)
            )
        return output_node_embeddings, output_edge_embeddings

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        cutoff_factors: torch.Tensor,
        use_manual_attention: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single Transformer layer.

        :param node_embeddings: The input node embeddings, of shape
            (batch_size, d_model)
        :param edge_embeddings: The input edge embeddings, of shape
            (batch_size, seq_length, d_model)
        :param cutoff_factors: The cutoff factors for the edges, of shape
            (batch_size, seq_length, seq_length)
        :param use_manual_attention: Whether to use the manual attention implementation
            (which supports double backward, needed for training with conservative
            forces), or the built-in PyTorch attention (which does not support double
            backward).
        :return: A tuple containing:
            - The output node embeddings, of shape (batch_size, d_model)
            - The output edge embeddings, of shape (batch_size, seq_length, d_model)
        """
        if self.transformer_type == "PostLN":
            node_embeddings, edge_embeddings = self._forward_post_ln_impl(
                node_embeddings, edge_embeddings, cutoff_factors, use_manual_attention
            )
        if self.transformer_type == "PreLN":
            node_embeddings, edge_embeddings = self._forward_pre_ln_impl(
                node_embeddings, edge_embeddings, cutoff_factors, use_manual_attention
            )
        return node_embeddings, edge_embeddings


class Transformer(torch.nn.Module):
    """
    Transformer implementation.

    :param d_model: The dimension of the model.
    :param num_layers: The number of transformer layers.
    :param n_heads: The number of attention heads.
    :param dim_node_features: The dimension of the node features.
    :param dim_feedforward: The dimension of the feedforward network.
    :param norm: The normalization type, either "LayerNorm" or "RMSNorm".
    :param activation: The activation function, either "SiLU" or "SwiGLU".
    :param transformer_type: The type of transformer, either "PostLN" or "PreLN".
    :param attention_temperature: The temperature scaling factor for attention
        scores. This is combined with the standard scaling by the square root of
        the head dimension.
    :param attention_backend: The attention implementation, one of
        ``AVAILABLE_ATTENTION_BACKENDS`` (see :class:`AttentionBlock`).
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        n_heads: int,
        dim_node_features: int,
        dim_feedforward: int = 512,
        norm: str = "LayerNorm",
        activation: str = "SiLU",
        transformer_type: str = "PostLN",
        attention_temperature: float = 1.0,
        attention_backend: str = "auto",
    ) -> None:
        super(Transformer, self).__init__()
        if attention_backend not in AVAILABLE_ATTENTION_BACKENDS:
            raise ValueError(
                f"Unknown attention backend: {attention_backend}. "
                f"Please choose from: {AVAILABLE_ATTENTION_BACKENDS}"
            )
        if norm not in AVAILABLE_NORMALIZATIONS:
            raise ValueError(
                f"Unknown normalization flag: {norm}. "
                f"Please choose from: {AVAILABLE_NORMALIZATIONS}"
            )

        if transformer_type not in AVAILABLE_TRANSFORMER_TYPES:
            raise ValueError(
                f"Unknown transformer flag: {transformer_type}. "
                f"Please choose from: {AVAILABLE_TRANSFORMER_TYPES}"
            )
        self.transformer_type = transformer_type

        if activation not in AVAILABLE_ACTIVATIONS:
            raise ValueError(
                f"Unknown activation flag: {activation}. "
                f"Please choose from: {AVAILABLE_ACTIVATIONS}"
            )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_node_features=dim_node_features,
                    dim_feedforward=dim_feedforward,
                    norm=norm,
                    activation=activation,
                    transformer_type=transformer_type,
                    temperature=attention_temperature,
                    attention_backend=attention_backend,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        cutoff_factors: torch.Tensor,
        use_manual_attention: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Transformer.

        :param node_embeddings: The input node embeddings, of shape
            (batch_size, d_model)
        :param edge_embeddings: The input edge embeddings, of shape
            (batch_size, seq_length, d_model)
        :param cutoff_factors: The cutoff factors for the edges, of shape
            (batch_size, seq_length, seq_length)
        :param use_manual_attention: Whether to use the manual attention implementation
            (which supports double backward, needed for training with conservative
            forces), or the built-in PyTorch attention (which does not support double
            backward).
        :return: A tuple containing:
            - The output node embeddings, of shape (batch_size, d_model)
            - The output edge embeddings, of shape (batch_size, seq_length, d_model)
        """
        for layer in self.layers:
            node_embeddings, edge_embeddings = layer(
                node_embeddings, edge_embeddings, cutoff_factors, use_manual_attention
            )
        return node_embeddings, edge_embeddings


class CartesianTransformer(torch.nn.Module):
    """
    Cartesian Transformer implementation for handling 3D coordinates.

    :param cutoff: The cutoff distance for neighbor interactions.
    :param cutoff_width: The width of the cutoff function.
    :param d_model: The dimension of the model.
    :param n_head: The number of attention heads.
    :param dim_node_features: The dimension of the node features.
    :param dim_feedforward: The dimension of the feedforward network.
    :param n_layers: The number of transformer layers.
    :param norm: The normalization type, either "LayerNorm" or "RMSNorm".
    :param activation: The activation function, either "SiLU" or "SwiGLU".
    :param attention_temperature: The temperature scaling factor for attention scores.
    :param transformer_type: The type of transformer, either "PostLN" or "PreLN".
    :param n_atomic_species: The number of atomic species.
    :param is_first: Whether this is the first transformer in the model.
    :param attention_backend: The attention implementation (see
        :class:`AttentionBlock`).
    """

    def __init__(
        self,
        cutoff: float,
        cutoff_width: float,
        d_model: int,
        n_head: int,
        dim_node_features: int,
        dim_feedforward: int,
        n_layers: int,
        norm: str,
        activation: str,
        attention_temperature: float,
        transformer_type: str,
        n_atomic_species: int,
        is_first: bool,
        attention_backend: str = "auto",
    ) -> None:
        super(CartesianTransformer, self).__init__()
        self.is_first = is_first
        self.cutoff = cutoff
        self.cutoff_width = cutoff_width
        self.trans = Transformer(
            d_model=d_model,
            num_layers=n_layers,
            n_heads=n_head,
            dim_node_features=dim_node_features,
            dim_feedforward=dim_feedforward,
            norm=norm,
            activation=activation,
            transformer_type=transformer_type,
            attention_temperature=attention_temperature,
            attention_backend=attention_backend,
        )

        self.edge_embedder = nn.Linear(4, d_model)

        if not is_first:
            n_merge = 3
        else:
            n_merge = 2

        self.compress = nn.Sequential(
            nn.Linear(n_merge * d_model, d_model),
            torch.nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.neighbor_embedder = DummyModule()  # for torchscript
        if not is_first:
            self.neighbor_embedder = nn.Embedding(n_atomic_species, d_model)

    def forward(
        self,
        input_node_embeddings: torch.Tensor,
        input_messages: torch.Tensor,
        element_indices_neighbors: torch.Tensor,
        edge_vectors: torch.Tensor,
        padding_mask: torch.Tensor,
        edge_distances: torch.Tensor,
        cutoff_factors: torch.Tensor,
        use_manual_attention: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the CartesianTransformer.

        :param input_node_embeddings: The input node embeddings, of shape
            (n_nodes, d_model)
        :param input_messages: The input messages to the transformer, of shape
            (n_nodes, max_num_neighbors, d_model)
        :param element_indices_neighbors: The atomic species of the neighboring atoms,
            of shape (n_nodes, max_num_neighbors)
        :param edge_vectors: The cartesian edge vectors between the central atoms and
            their neighbors, of shape (n_nodes, max_num_neighbors, 3)
        :param padding_mask: A padding mask indicating which neighbors are real, and
            which are padded, of shape (n_nodes, max_num_neighbors)
        :param edge_distances: The distances between the central atoms and their
            neighbors, of shape (n_nodes, max_num_neighbors)
        :param cutoff_factors: The cutoff factors for the edges, of shape
            (n_nodes, max_num_neighbors)
        :param use_manual_attention: Whether to use the manual attention implementation
            (which supports double backward, needed for training with conservative
            forces), or the built-in PyTorch attention (which does not support double
            backward).
        :return: A tuple containing:
            - The output node embeddings, of shape (n_nodes, d_model)
            - The output edge embeddings, of shape (n_nodes, max_num_neighbors, d_model)
        """
        node_embeddings = input_node_embeddings
        edge_embeddings = [edge_vectors, edge_distances[:, :, None]]

        # on some systems, on isolated atoms, a torchscript bug concatenates the two
        # (empty) float tensors into an int tensors, causing an error later on
        edge_embeddings = torch.cat(edge_embeddings, dim=2).to(edge_vectors.dtype)

        edge_embeddings = self.edge_embedder(edge_embeddings)

        if not self.is_first:
            neighbor_elements_embeddings = self.neighbor_embedder(
                element_indices_neighbors
            )
            edge_tokens = torch.cat(
                [edge_embeddings, neighbor_elements_embeddings, input_messages], dim=2
            )
        else:
            neighbor_elements_embeddings = torch.empty(
                0, device=edge_vectors.device, dtype=edge_vectors.dtype
            )  # for torch script
            edge_tokens = torch.cat([edge_embeddings, input_messages], dim=2)

        edge_tokens = self.compress(edge_tokens)
        # tokens = torch.cat([node_elements_embedding[:, None, :], tokens], dim=1)

        padding_mask_with_central_token = torch.ones(
            padding_mask.shape[0], dtype=torch.bool, device=padding_mask.device
        )
        total_padding_mask = torch.cat(
            [padding_mask_with_central_token[:, None], padding_mask], dim=1
        )

        cutoff_subfactors = torch.ones(
            padding_mask.shape[0],
            dtype=cutoff_factors.dtype,
            device=padding_mask.device,
        )
        cutoff_factors = torch.cat([cutoff_subfactors[:, None], cutoff_factors], dim=1)
        cutoff_factors[~total_padding_mask] = 0.0

        cutoff_factors = cutoff_factors[:, None, :]
        cutoff_factors = cutoff_factors.repeat(1, cutoff_factors.shape[2], 1)

        initial_num_tokens = edge_vectors.shape[1]
        max_num_tokens = input_messages.shape[1]

        output_node_embeddings, output_edge_embeddings = self.trans(
            node_embeddings[:, None, :],
            edge_tokens[:, :max_num_tokens, :],
            cutoff_factors=cutoff_factors[
                :, : (max_num_tokens + 1), : (max_num_tokens + 1)
            ],
            use_manual_attention=use_manual_attention,
        )
        if max_num_tokens < initial_num_tokens:
            padding = torch.zeros(
                output_edge_embeddings.shape[0],
                initial_num_tokens - max_num_tokens,
                output_edge_embeddings.shape[2],
                device=output_edge_embeddings.device,
            )
            output_edge_embeddings = torch.cat([output_edge_embeddings, padding], dim=1)
        output_node_embeddings = output_node_embeddings.squeeze(1)
        return output_node_embeddings, output_edge_embeddings


def manual_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Implements the attention operation manually, using basic PyTorch operations.
    We need it because the built-in PyTorch attention does not support double backward,
    which is needed when training with conservative forces.

    :param q: The queries
    :param k: The keys
    :param v: The values
    :param attn_mask: The attention mask
    :param temperature: An additional scaling factor for attention scores.
    :return: The result of the attention operation
    """
    attention_weights = (
        torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5 * temperature)
    ) + attn_mask
    attention_weights = attention_weights.softmax(dim=-1)
    attention_output = torch.matmul(attention_weights, v)
    return attention_output
