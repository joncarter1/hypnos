"""Unified signal tokenizer with discrete (RVQ) and continuous (VAE) modes.

Provides a single architecture for tokenizing physiological signals:
- Discrete mode: SEANet → Transformer → RVQ → Transformer → SEANet
- VAE mode: SEANet → Transformer → mu/logvar → sample → Transformer → SEANet

Both modes share the same SEANet encoder/decoder backbone and optional
RoPETransformer attention layers.
"""

import logging
import math
import typing as tp
from pathlib import Path

import torch
from torch import Tensor, nn

from .attention import IdentityAttention, RoPETransformer
from .quantizer import ResidualVectorQuantizer
from .seanet import SEANetDecoder, SEANetEncoder

logger = logging.getLogger(__name__)


class SignalTokenizer(nn.Module):
    """Tokenizer for physiological signals with discrete or VAE modes.

    Pipeline (discrete mode):
        SEANetEncoder → RoPETransformer → project_in → RVQ → project_out → RoPETransformer → SEANetDecoder

    Pipeline (VAE mode):
        SEANetEncoder → RoPETransformer → fc_mu/fc_logvar → reparameterize → fc_decode → RoPETransformer → SEANetDecoder

    Args:
        in_channels: Number of input channels.
        sample_rate: Signal sample rate in Hz.
        token_duration_sec: Duration of each token in seconds.
        embed_dim: Token embedding dimension.
        n_filters: Initial encoder filters.
        ratios: Downsampling ratios (product should equal sample_rate * token_duration_sec).
        n_residual_layers: Residual blocks per encoder stage (controls receptive field).
        dilation_base: Dilation base (1=no dilation, 2=exponential).
        mode: Tokenization mode — 'discrete' (RVQ) or 'vae' (variational autoencoder).
        codebook_size: Number of codebook entries (discrete mode).
        codebook_dim: Dimension for quantization, projected from embed_dim (discrete mode).
        num_quantizers: Number of RVQ levels (discrete mode).
        commitment_cost: VQ commitment loss weight (discrete mode).
        quantization_dropout: Probability of skipping quantizer during training (discrete mode).
        quantizer_dropout: Probability of dropping RVQ levels during training (discrete mode).
        ema_decay: EMA decay rate for codebook updates (discrete mode). 0.99 = ~100 batch memory.
        latent_dim: VAE latent dimension (defaults to embed_dim if None).
        attention_depth: Number of transformer layers in encoder/decoder (0 to disable).
        attention_heads: Number of attention heads.
        window_size: Attention window size in tokens (None for full attention).
        transformer_dim_feedforward: FFN hidden dimension in transformer.
        layer_scale_init: LayerScale initialization value.
        causal: Whether to use causal convolutions and attention.
        activation: Activation function name.
        norm: Normalization type.
        last_kernel_size: Kernel size for encoder's final projection conv.
        stride_kernel_multiplier: Multiplier for strided conv kernels (kernel = ratio * multiplier).
        pad_mode: Padding mode for convolutions ('reflect' or 'constant').
        use_activation_checkpointing: Trade compute for memory by recomputing activations during backward.
        rotation_trick: Use rotation trick (Fifty et al., 2025) instead of STE for VQ gradients.
    """

    VALID_MODES = ("discrete", "vae")

    def __init__(
        self,
        # Signal parameters
        in_channels: int = 12,
        sample_rate: int = 128,
        token_duration_sec: float = 1.0,
        # Encoder parameters
        embed_dim: int = 512,
        n_filters: int = 64,
        ratios: tp.List[int] = [4, 4, 4, 4],
        n_residual_layers: int = 1,
        dilation_base: int = 1,
        # Mode
        mode: str = "discrete",
        # Discrete (RVQ) parameters
        codebook_size: int = 512,
        codebook_dim: int | None = None,
        num_quantizers: int = 4,
        commitment_cost: float = 0.25,
        quantization_dropout: float = 0.0,
        quantizer_dropout: float = 0.0,
        ema_decay: float = 0.99,
        # VAE parameters
        latent_dim: int | None = None,
        # Transformer parameters
        attention_depth: int = 0,
        attention_heads: int = 8,
        window_size: int | None = 250,
        transformer_dim_feedforward: int = 2048,
        layer_scale_init: float = 0.01,
        # Causality
        causal: bool = False,
        # Architecture
        activation: str = "gelu",
        norm: str | None = "layer",
        last_kernel_size: int = 7,
        stride_kernel_multiplier: int = 2,
        pad_mode: str = "reflect",
        # Memory optimization
        use_activation_checkpointing: bool = False,
        # VQ gradient method
        rotation_trick: bool = False,
    ):
        super().__init__()

        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode {mode!r}, must be one of {self.VALID_MODES}")

        # Store configuration
        self.mode = mode
        self.causal = causal
        self.in_channels = in_channels
        self.sample_rate = sample_rate
        self.token_duration_sec = token_duration_sec
        self.embed_dim = embed_dim
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        self.quantization_dropout = quantization_dropout

        # Calculate samples per token
        self.samples_per_token = int(sample_rate * token_duration_sec)
        hop_length = math.prod(ratios)

        if hop_length != self.samples_per_token:
            raise ValueError(
                f"Product of ratios ({hop_length}) must equal "
                f"sample_rate * token_duration_sec ({self.samples_per_token})"
            )

        # Encoder
        self.encoder = SEANetEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            n_filters=n_filters,
            ratios=ratios,
            n_residual_layers=n_residual_layers,
            dilation_base=dilation_base,
            activation=activation,
            norm=norm,
            causal=causal,
            last_kernel_size=last_kernel_size,
            stride_kernel_multiplier=stride_kernel_multiplier,
            pad_mode=pad_mode,
            use_activation_checkpointing=use_activation_checkpointing,
        )

        # Encoder transformer
        if attention_depth > 0:
            self.encoder_transformer = RoPETransformer(
                embed_dim=embed_dim,
                depth=attention_depth,
                num_heads=attention_heads,
                dim_feedforward=transformer_dim_feedforward,
                window_size=window_size,
                causal=causal,
                layer_scale_init=layer_scale_init,
                use_activation_checkpointing=use_activation_checkpointing,
            )
        else:
            self.encoder_transformer = IdentityAttention()

        # Mode-specific bottleneck
        if mode == "discrete":
            codebook_dim = codebook_dim if codebook_dim is not None else embed_dim
            self.codebook_dim = codebook_dim
            self.project_in = nn.Linear(embed_dim, codebook_dim)
            self.project_out = nn.Linear(codebook_dim, embed_dim)
            self.quantizer = ResidualVectorQuantizer(
                dim=codebook_dim,
                codebook_size=codebook_size,
                num_quantizers=num_quantizers,
                commitment_cost=commitment_cost,
                ema_decay=ema_decay,
                quantizer_dropout=quantizer_dropout,
                rotation_trick=rotation_trick,
            )
        else:  # vae
            self.latent_dim = latent_dim if latent_dim is not None else embed_dim
            self.fc_mu = nn.Linear(embed_dim, self.latent_dim)
            self.fc_logvar = nn.Linear(embed_dim, self.latent_dim)
            if self.latent_dim != embed_dim:
                self.fc_decode = nn.Linear(self.latent_dim, embed_dim)
            else:
                self.fc_decode = nn.Identity()

        # Decoder transformer
        if attention_depth > 0:
            self.decoder_transformer = RoPETransformer(
                embed_dim=embed_dim,
                depth=attention_depth,
                num_heads=attention_heads,
                dim_feedforward=transformer_dim_feedforward,
                window_size=window_size,
                causal=causal,
                layer_scale_init=layer_scale_init,
                use_activation_checkpointing=use_activation_checkpointing,
            )
        else:
            self.decoder_transformer = IdentityAttention()

        # Decoder
        self.decoder = SEANetDecoder(
            out_channels=in_channels,
            embed_dim=embed_dim,
            n_filters=n_filters,
            ratios=ratios,
            n_residual_layers=n_residual_layers,
            dilation_base=dilation_base,
            activation=activation,
            norm=norm,
            causal=causal,
            stride_kernel_multiplier=stride_kernel_multiplier,
            pad_mode=pad_mode,
            use_activation_checkpointing=use_activation_checkpointing,
        )

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick: sample z = mu + std * eps."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    @staticmethod
    def _kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
        """KL divergence from N(mu, sigma) to N(0, I)."""
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def encode(self, x: Tensor, return_embeddings: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """Encode signal to tokens (discrete) or continuous embeddings (VAE).

        Args:
            x: Input signal (B, C, T).
            return_embeddings: If True and discrete mode, also return continuous embeddings.

        Returns:
            Discrete mode: indices (B, num_tokens, num_quantizers), optionally with embeddings.
            VAE mode: sampled latents (B, num_tokens, latent_dim).
        """
        z = self.encoder(x)
        z = self.encoder_transformer(z)

        if self.mode == "vae":
            mu = self.fc_mu(z)
            return self._reparameterize(mu, self.fc_logvar(z))

        z_proj = self.project_in(z)
        z_q, indices, _ = self.quantizer(z_proj)

        if return_embeddings:
            return indices, z_q
        return indices

    def decode(self, z: Tensor) -> Tensor:
        """Decode embeddings to signal.

        Args:
            z: Continuous embeddings — quantized (B, T, codebook_dim) or
               sampled latent (B, T, latent_dim).

        Returns:
            Reconstructed signal (B, C, T).
        """
        if self.mode == "discrete":
            z = self.project_out(z)
        else:
            z = self.fc_decode(z)
        z = self.decoder_transformer(z)
        return self.decoder(z)

    @torch.no_grad()
    def decode_tokens(self, indices: Tensor) -> Tensor:
        """Decode discrete tokens to signal (discrete mode only, inference).

        Args:
            indices: Token indices (B, num_tokens, num_quantizers).

        Returns:
            Reconstructed signal (B, C, T).
        """
        if self.mode != "discrete":
            raise RuntimeError('decode_tokens() requires mode="discrete"')
        z_q = self.quantizer.decode(indices)
        return self.decode(z_q)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Full forward pass with reconstruction and losses.

        Args:
            x: Input signal (B, C, T).

        Returns:
            Dict with reconstruction, embeddings, indices, commitment_loss,
            kl_loss, and residual_norm.
        """
        zero = torch.tensor(0.0, device=x.device)
        input_length = x.size(-1)

        # Encode
        z = self.encoder(x)
        z = self.encoder_transformer(z)

        if self.mode == "discrete":
            z_proj = self.project_in(z)

            # Always quantize (EMA codebook updates + commitment loss on every batch)
            z_q, indices, vq_losses = self.quantizer(z_proj)
            commitment_loss = vq_losses["commitment_loss"]
            residual_norm = vq_losses.get("residual_norm", zero)

            # Per-sequence quantization dropout (Défossez et al., 2024):
            # independently bypass VQ for each sequence in the batch
            if self.training and self.quantization_dropout > 0:
                skip_mask = torch.rand(z_proj.size(0), 1, 1, device=z_proj.device) < self.quantization_dropout
                z_q = torch.where(skip_mask, z_proj, z_q)

            # Decode
            z_out = self.project_out(z_q)
            z_out = self.decoder_transformer(z_out)
            x_recon = self.decoder(z_out)
            assert x_recon.size(-1) == input_length, (
                f"Decoder output length {x_recon.size(-1)} != input length {input_length}"
            )

            return {
                "reconstruction": x_recon,
                "embeddings": z_q,
                "indices": indices,
                "commitment_loss": commitment_loss,
                "kl_loss": zero,
                "residual_norm": residual_norm,
            }

        # VAE mode
        mu = self.fc_mu(z)
        logvar = self.fc_logvar(z)
        z_sampled = self._reparameterize(mu, logvar)
        kl_loss = self._kl_divergence(mu, logvar)

        # Decode
        z_out = self.fc_decode(z_sampled)
        z_out = self.decoder_transformer(z_out)
        x_recon = self.decoder(z_out)
        assert x_recon.size(-1) == input_length, (
            f"Decoder output length {x_recon.size(-1)} != input length {input_length}"
        )

        return {
            "reconstruction": x_recon,
            "embeddings": z_sampled,
            "indices": None,
            "commitment_loss": zero,
            "kl_loss": kl_loss,
            "residual_norm": zero,
        }

    def tokenize(self, x: Tensor) -> Tensor:
        """Convert signal to discrete tokens (inference only, discrete mode).

        Args:
            x: Input signal (B, C, T).

        Returns:
            Token indices (B, num_tokens, num_quantizers).

        Raises:
            RuntimeError: If mode is not 'discrete'.
        """
        if self.mode != "discrete":
            raise RuntimeError('tokenize() requires mode="discrete"; use encode() for VAE embeddings')
        with torch.no_grad():
            return self.encode(x)

    def get_num_tokens(self, signal_length: int) -> int:
        """Calculate number of tokens for a given signal length."""
        return signal_length // self.samples_per_token

    def get_codebook_usage(self) -> dict[str, Tensor]:
        """Get codebook usage statistics (discrete mode only)."""
        if self.mode != "discrete":
            return {}
        return self.quantizer.get_codebook_usage()

    def reset_marginal_stats(self) -> None:
        """Reset raw assignment counters before a measurement window."""
        if self.mode != "discrete":
            return
        self.quantizer.reset_marginal_stats()

    def get_marginal_stats(self) -> dict[str, Tensor]:
        """Per-quantizer marginal entropy stats from raw counts (discrete mode only)."""
        if self.mode != "discrete":
            return {}
        return self.quantizer.get_marginal_stats()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, **kwargs) -> "SignalTokenizer":
        """Load a SignalTokenizer from a training checkpoint.

        Constructs the model from the provided kwargs, then loads the tokenizer weights from
        the checkpoint (keys are stored under a ``model.`` prefix). Accepts local paths and
        ``hf://`` / ``s3://`` URIs.

        Args:
            checkpoint_path: Path or URI to the ``.ckpt`` file.
            **kwargs: Arguments to construct the SignalTokenizer.

        Returns:
            SignalTokenizer with loaded weights, in eval mode.
        """
        model = cls(**kwargs)
        checkpoint_str = str(checkpoint_path)
        if checkpoint_str.startswith("hf://") or checkpoint_str.startswith("s3://"):
            import fsspec

            with fsspec.open(checkpoint_str, "rb") as f:
                checkpoint = torch.load(f, map_location="cpu", weights_only=False)
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Extract model weights (strip 'model.' prefix from Lightning module state_dict,
        # and '_orig_mod.' prefix from torch.compile'd models)
        state_dict = {}
        for key, value in checkpoint["state_dict"].items():
            if not key.startswith("model."):
                continue
            clean_key = key.removeprefix("model.").replace("_orig_mod.", "")
            state_dict[clean_key] = value

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys when loading checkpoint: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys when loading checkpoint: {unexpected}")

        model.eval()
        return model
