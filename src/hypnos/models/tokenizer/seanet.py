"""SEANet-style encoder and decoder for physiological signal tokenization.

Based on the SEANet architecture used in SoundStream, EnCodec, and BrainOmni.
Implements dilated residual blocks with configurable receptive field.

Causal alignment note:
    Unlike reference Mimi (which uses ``padding = kernel - stride``), this
    implementation uses ``padding = kernel - 1`` for causal convolutions.
    This enforces strict sample-level causality at every layer — no within-stride
    future visibility. The encoder compensates with a global right-pad + drop-first
    scheme so that each token is right-aligned to its segment boundary.
    See ``docs/models/tokenizer-causal-alignment.md`` for full details.
"""

import math
import typing as tp

from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm
from hypnos.models.attention import gradient_checkpoint
from hypnos.models.utils import get_activation, get_norm


class SConv1d(nn.Module):
    """Streaming-compatible 1D convolution with optional causal padding.

    Following SEANet convention:
    - Causal mode: left-pads to ensure no future information leakage
    - Non-causal mode: symmetric padding
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str | None = None,
        pad_mode: str = 'reflect',
    ):
        super().__init__()
        self.causal = causal
        self.pad_mode = pad_mode
        self.stride = stride

        # Calculate padding
        # Effective kernel size accounting for dilation
        effective_kernel = (kernel_size - 1) * dilation + 1

        if causal:
            # Causal: pad only on the left
            self.padding_left = effective_kernel - 1
            self.padding_right = 0
        else:
            # Non-causal: symmetric padding to maintain size (before stride)
            total_padding = effective_kernel - 1
            self.padding_left = total_padding // 2
            self.padding_right = total_padding - self.padding_left

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,  # We handle padding manually
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        if norm == 'weight':
            self.conv = weight_norm(self.conv)
            self.norm = None
        else:
            self.norm = get_norm(norm, num_features=out_channels) if norm else None

    def forward(self, x: Tensor) -> Tensor:
        # Apply padding
        if self.padding_left > 0 or self.padding_right > 0:
            x = nn.functional.pad(x, (self.padding_left, self.padding_right), mode=self.pad_mode)

        x = self.conv(x)

        if self.norm is not None:
            x = self.norm(x)

        return x


class SConvTranspose1d(nn.Module):
    """Streaming-compatible 1D transposed convolution.

    Handles output trimming for causal mode to maintain proper alignment.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str | None = None,
    ):
        super().__init__()
        self.causal = causal
        self.stride = stride
        self.kernel_size = kernel_size

        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            groups=groups,
            bias=bias,
        )

        if norm == 'weight':
            self.conv = weight_norm(self.conv)
            self.norm = None
        else:
            self.norm = get_norm(norm, num_features=out_channels) if norm else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)

        # Trim output to correct size
        # ConvTranspose1d output size: (L_in - 1) * stride + kernel_size
        # We want: L_in * stride
        # So we need to trim: kernel_size - stride
        trim = self.kernel_size - self.stride
        if trim > 0:
            if self.causal:
                # Causal: trim from the right
                x = x[..., :-trim]
            else:
                # Non-causal: trim symmetrically
                trim_left = trim // 2
                trim_right = trim - trim_left
                if trim_right > 0:
                    x = x[..., trim_left:-trim_right]
                else:
                    x = x[..., trim_left:]

        if self.norm is not None:
            x = self.norm(x)

        return x


class SEANetResnetBlock(nn.Module):
    """Residual block with dilated convolutions following SEANet design.

    Each block consists of:
    - Activation -> Conv (kernel=3, dilated) -> Activation -> Conv (kernel=1)
    - Skip connection (identity or 1x1 conv if channels change)

    The dilation parameter controls the receptive field expansion.
    """

    def __init__(
        self,
        dim: int,
        kernel_sizes: tp.List[int] = [3, 1],
        dilations: tp.List[int] = [1, 1],
        activation: str = 'gelu',
        norm: str | None = 'layer',
        causal: bool = False,
        compress: int = 2,
        pad_mode: str = 'reflect',
    ):
        super().__init__()
        assert len(kernel_sizes) == len(dilations) == 2

        hidden = dim // compress

        self.block = nn.Sequential(
            get_activation(activation),
            SConv1d(
                dim,
                hidden,
                kernel_size=kernel_sizes[0],
                dilation=dilations[0],
                causal=causal,
                norm=norm,
                pad_mode=pad_mode,
            ),
            get_activation(activation),
            SConv1d(
                hidden,
                dim,
                kernel_size=kernel_sizes[1],
                dilation=dilations[1],
                causal=causal,
                norm=norm,
                pad_mode=pad_mode,
            ),
        )

        # Skip connection is identity (same channels)
        self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.shortcut(x) + self.block(x)


class SEANetEncoder(nn.Module):
    """SEANet-style encoder for physiological signals.

    Compresses input signals into embeddings using strided convolutions
    with dilated residual blocks at each scale.

    Args:
        in_channels: Number of input channels (e.g., 12 for ECG)
        embed_dim: Output embedding dimension
        n_filters: Initial number of filters (doubles at each stage)
        ratios: Downsampling ratios for each stage (product = compression factor)
        n_residual_layers: Number of residual blocks per stage
        dilation_base: Base for exponential dilation (1 = no dilation, 2 = 1,2,4,...)
        activation: Activation function name
        norm: Normalization type
        causal: Whether to use causal convolutions
        kernel_size: Kernel size for residual blocks
        last_kernel_size: Kernel size for final projection
        stride_kernel_multiplier: Multiplier for strided conv kernels (kernel = ratio * multiplier).
            Default 2 follows SoundStream/EnCodec convention. Use 1 for smaller receptive field.
        pad_mode: Padding mode for convolutions
        use_activation_checkpointing: Trade compute for memory by recomputing activations during backward.
    """

    def __init__(
        self,
        in_channels: int = 12,
        embed_dim: int = 512,
        n_filters: int = 64,
        ratios: tp.List[int] = [4, 4, 4, 4],
        n_residual_layers: int = 1,
        dilation_base: int = 1,
        activation: str = 'gelu',
        norm: str | None = 'layer',
        causal: bool = False,
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        stride_kernel_multiplier: int = 2,
        pad_mode: str = 'reflect',
        use_activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.ratios = ratios
        self.hop_length = math.prod(ratios)
        self.causal = causal
        self.use_activation_checkpointing = use_activation_checkpointing

        # Initial convolution
        self.initial_conv = SConv1d(
            in_channels,
            n_filters,
            kernel_size=kernel_size,
            causal=causal,
            norm=norm,
            pad_mode=pad_mode,
        )

        # Build encoder stages
        self.stages = nn.ModuleList()
        current_channels = n_filters
        for i, ratio in enumerate(ratios):
            stage = nn.ModuleList()

            # Residual blocks with exponential dilations
            for j in range(n_residual_layers):
                dilation = dilation_base**j if dilation_base > 1 else 1
                stage.append(
                    SEANetResnetBlock(
                        current_channels,
                        kernel_sizes=[3, 1],
                        dilations=[dilation, 1],
                        activation=activation,
                        norm=norm,
                        causal=causal,
                        pad_mode=pad_mode,
                    )
                )

            # Activation before downsampling
            stage.append(get_activation(activation))

            # Strided convolution for downsampling
            next_channels = current_channels * 2
            stage.append(
                SConv1d(
                    current_channels,
                    next_channels,
                    kernel_size=ratio * stride_kernel_multiplier,
                    stride=ratio,
                    causal=causal,
                    norm=norm,
                    pad_mode=pad_mode,
                )
            )

            self.stages.append(stage)
            current_channels = next_channels

        # Final projection to embedding dimension
        self.final_conv = SConv1d(
            current_channels,
            embed_dim,
            kernel_size=last_kernel_size,
            causal=causal,
            norm=norm,
            pad_mode=pad_mode,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode input signal to embeddings.

        In causal mode, applies a right-pad + drop-first correction so that each
        output token is right-aligned to its segment boundary. Token ``t`` encodes
        the signal through sample ``(t+1) * hop_length``. See the module docstring
        and ``docs/models/tokenizer-causal-alignment.md`` for rationale.

        Args:
            x: Input signal (B, C, T)

        Returns:
            Embeddings (B, T', D) where T' = T // hop_length.
        """
        if self.causal:
            # Right-pad by hop_length so the last token's causal RF reaches the
            # end of the actual signal. Without this, the final hop_length samples
            # fall outside any token's receptive field.
            x = nn.functional.pad(x, (0, self.hop_length))

        x = self.initial_conv(x)

        for stage in self.stages:
            if self.use_activation_checkpointing and x.requires_grad:

                def _run_stage(x, _stage=stage):
                    for layer in _stage:
                        x = layer(x)
                    return x

                x = gradient_checkpoint(_run_stage, x, use_reentrant=False)
            else:
                for layer in stage:
                    x = layer(x)

        x = self.final_conv(x)

        # Transpose to (B, T, D) for transformer compatibility
        x = x.transpose(1, 2)

        if self.causal:
            x = x[:, 1:]  # Drop warm-up token (mostly padding, no real signal info)

        return x


class SEANetDecoder(nn.Module):
    """SEANet-style decoder for physiological signals.

    Reconstructs signals from embeddings using transposed convolutions
    with dilated residual blocks at each scale.

    Args:
        out_channels: Number of output channels (e.g., 12 for ECG)
        embed_dim: Input embedding dimension
        n_filters: Initial number of filters in decoder (halves at each stage)
        ratios: Upsampling ratios for each stage (reversed from encoder)
        n_residual_layers: Number of residual blocks per stage
        dilation_base: Base for exponential dilation
        activation: Activation function name
        norm: Normalization type
        causal: Whether to use causal convolutions
        kernel_size: Kernel size for initial and final convolutions
        stride_kernel_multiplier: Multiplier for transposed conv kernels (kernel = ratio * multiplier)
        pad_mode: Padding mode for convolutions
        use_activation_checkpointing: Trade compute for memory by recomputing activations during backward.
    """

    def __init__(
        self,
        out_channels: int = 12,
        embed_dim: int = 512,
        n_filters: int = 64,
        ratios: tp.List[int] = [4, 4, 4, 4],
        n_residual_layers: int = 1,
        dilation_base: int = 1,
        activation: str = 'gelu',
        norm: str | None = 'layer',
        causal: bool = False,
        kernel_size: int = 7,
        stride_kernel_multiplier: int = 2,
        pad_mode: str = 'reflect',
        use_activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.ratios = ratios
        self.hop_length = math.prod(ratios)
        self.use_activation_checkpointing = use_activation_checkpointing

        # Calculate channel progression (reverse of encoder)
        # Encoder: n_filters -> n_filters*2 -> n_filters*4 -> ... -> n_filters * 2^len(ratios)
        # Decoder: n_filters * 2^len(ratios) -> ... -> n_filters
        n_stages = len(ratios)
        initial_channels = n_filters * (2**n_stages)

        # Initial convolution from embedding dimension
        self.initial_conv = SConv1d(
            embed_dim,
            initial_channels,
            kernel_size=kernel_size,
            causal=causal,
            norm=norm,
            pad_mode=pad_mode,
        )

        # Build decoder stages (reverse order of encoder)
        self.stages = nn.ModuleList()
        current_channels = initial_channels
        for i, ratio in enumerate(reversed(ratios)):
            stage = nn.ModuleList()

            # Activation before upsampling
            stage.append(get_activation(activation))

            # Transposed convolution for upsampling
            next_channels = current_channels // 2
            stage.append(
                SConvTranspose1d(
                    current_channels,
                    next_channels,
                    kernel_size=ratio * stride_kernel_multiplier,
                    stride=ratio,
                    causal=causal,
                    norm=norm,
                )
            )

            # Residual blocks with exponential dilations
            for j in range(n_residual_layers):
                dilation = dilation_base**j if dilation_base > 1 else 1
                stage.append(
                    SEANetResnetBlock(
                        next_channels,
                        kernel_sizes=[3, 1],
                        dilations=[dilation, 1],
                        activation=activation,
                        norm=norm,
                        causal=causal,
                        pad_mode=pad_mode,
                    )
                )

            self.stages.append(stage)
            current_channels = next_channels

        # Final convolution to output channels (no activation)
        self.final_conv = SConv1d(
            current_channels,
            out_channels,
            kernel_size=kernel_size,
            causal=causal,
            norm=None,  # No norm on final output
            pad_mode=pad_mode,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Decode embeddings to signal.

        Args:
            x: Embeddings (B, T, embed_dim)

        Returns:
            Reconstructed signal (B, out_channels, T * hop_length)
        """
        x = x.transpose(1, 2)  # (B, T, D) -> (B, D, T)

        x = self.initial_conv(x)

        for stage in self.stages:
            if self.use_activation_checkpointing and x.requires_grad:

                def _run_stage(x, _stage=stage):
                    for layer in _stage:
                        x = layer(x)
                    return x

                x = gradient_checkpoint(_run_stage, x, use_reentrant=False)
            else:
                for layer in stage:
                    x = layer(x)

        x = self.final_conv(x)

        return x
