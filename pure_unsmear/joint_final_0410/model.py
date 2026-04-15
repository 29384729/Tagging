"""
Joint unsmear models.

This directory keeps only the models actually needed for joint training:
- `ParticleTransformerKD`: offline teacher / HLT baseline / HLT+KD
- `SharedEncoderUnsmearClassifier`: joint model with a shared encoder
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(dim), int(dim)),
            nn.LayerNorm(int(dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def _init_linear_layers(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class SharedTokenEncoder(nn.Module):
    """Shared token encoder used by both baseline and joint models."""

    def __init__(
        self,
        *,
        input_dim: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
        add_mask_channel: bool = False,
        use_positional_embedding: bool = False,
        max_seq_len: int = 128,
        mask_encoder_input: bool = False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.add_mask_channel = bool(add_mask_channel)
        self.use_positional_embedding = bool(use_positional_embedding)
        self.max_seq_len = int(max_seq_len)
        self.mask_encoder_input = bool(mask_encoder_input)

        in_dim = self.input_dim + (1 if self.add_mask_channel else 0)
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        if self.use_positional_embedding:
            self.pos_embed = nn.Embedding(self.max_seq_len, self.embed_dim)
        else:
            self.pos_embed = None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self._init_weights()

    def _init_weights(self):
        _init_linear_layers(self.input_proj)
        _init_linear_layers(self.transformer)
        if self.pos_embed is not None:
            nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        if self.pos_embed is not None and seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}")

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        inp = x
        if self.add_mask_channel:
            inp = torch.cat([inp, mask.to(dtype=x.dtype).unsqueeze(-1)], dim=-1)

        h = self.input_proj(inp.reshape(bsz * seq_len, -1)).reshape(bsz, seq_len, self.embed_dim)
        if self.pos_embed is not None:
            pos_idx = torch.arange(seq_len, device=x.device)
            h = h + self.pos_embed(pos_idx).unsqueeze(0)
        if self.mask_encoder_input:
            h = h * mask.to(dtype=h.dtype).unsqueeze(-1)
        h = self.transformer(h, src_key_padding_mask=~mask)
        return h


class SharedJetClassifierHead(nn.Module):
    """Shared jet-level pooling and classifier head."""

    def __init__(self, *, embed_dim: int, dropout: float):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.pool_query = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            self.embed_dim, num_heads=4, dropout=float(dropout), batch_first=True
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            ResidualBlock(128, dropout=float(dropout)),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self):
        _init_linear_layers(self.pool_attn)
        _init_linear_layers(self.classifier)

    def pool_tokens(
        self,
        h: torch.Tensor,
        mask: torch.Tensor,
        *,
        return_attention: bool = False,
    ):
        q = self.pool_query.expand(h.shape[0], -1, -1)
        pooled, attn = self.pool_attn(
            q,
            h,
            h,
            key_padding_mask=~mask,
            need_weights=bool(return_attention),
            average_attn_weights=True,
        )
        z = self.norm(pooled.squeeze(1))
        if return_attention:
            return z, attn.squeeze(1)
        return z

    def classify_from_pooled(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(z)


class ParticleTransformerKD(nn.Module):
    """Downstream classifier."""

    def __init__(
        self,
        *,
        input_dim: int = 7,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 512,
        dropout: float = 0.1,
        add_mask_channel: bool = False,
        use_positional_embedding: bool = False,
        max_seq_len: int = 128,
        mask_encoder_input: bool = False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.token_encoder = SharedTokenEncoder(
            input_dim=int(input_dim),
            embed_dim=int(embed_dim),
            num_heads=int(num_heads),
            num_layers=int(num_layers),
            ff_dim=int(ff_dim),
            dropout=float(dropout),
            add_mask_channel=bool(add_mask_channel),
            use_positional_embedding=bool(use_positional_embedding),
            max_seq_len=int(max_seq_len),
            mask_encoder_input=bool(mask_encoder_input),
        )
        self.cls_head = SharedJetClassifierHead(embed_dim=int(embed_dim), dropout=float(dropout))

    @property
    def input_proj(self):
        return self.token_encoder.input_proj

    @property
    def transformer(self):
        return self.token_encoder.transformer

    @property
    def pool_query(self):
        return self.cls_head.pool_query

    @property
    def pool_attn(self):
        return self.cls_head.pool_attn

    @property
    def norm(self):
        return self.cls_head.norm

    @property
    def classifier(self):
        return self.cls_head.classifier

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        *,
        return_attention: bool = False,
        return_aux: bool = False,
    ):
        h = self.token_encoder(x, mask)
        if return_attention:
            z, attn = self.cls_head.pool_tokens(h, mask, return_attention=True)
        else:
            z = self.cls_head.pool_tokens(h, mask, return_attention=False)
            attn = None
        logits = self.cls_head.classify_from_pooled(z)
        aux = {"z": z}
        if return_attention and return_aux:
            return logits, attn, aux
        if return_attention:
            return logits, attn
        if return_aux:
            return logits, aux
        return logits


class SharedEncoderUnsmearClassifier(nn.Module):
    """Joint model with a shared encoder for unsmear regression and jet classification."""

    def __init__(
        self,
        *,
        input_dim: int = 7,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        unsmear_decoder_layers: int = 2,
        unsmear_decoder_heads: int = 8,
        unsmear_decoder_ff_dim: int = 512,
        unsmear_decoder_dropout: float | None = None,
        return_reco: bool = True,
        add_mask_channel: bool = True,
        mask_output: bool = True,
        use_positional_embedding: bool = True,
        max_seq_len: int = 128,
        mask_encoder_input: bool = False,
        cls_use_delta_fusion: bool = True,
        cls_detach_delta_for_cls: bool = True,
        cls_gate_hidden_dim: int = 128,
        cls_gate_init_bias: float = -2.0,
        cls_alpha_init: float = 0.05,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.return_reco = bool(return_reco)
        self.add_mask_channel = bool(add_mask_channel)
        self.mask_output = bool(mask_output)
        self.use_positional_embedding = bool(use_positional_embedding)
        self.max_seq_len = int(max_seq_len)
        self.mask_encoder_input = bool(mask_encoder_input)
        self.cls_use_delta_fusion = bool(cls_use_delta_fusion)
        self.cls_detach_delta_for_cls = bool(cls_detach_delta_for_cls)
        self.cls_gate_hidden_dim = int(cls_gate_hidden_dim)
        self.cls_gate_init_bias = float(cls_gate_init_bias)
        self.cls_alpha_init = float(cls_alpha_init)
        self.unsmear_decoder_layers = int(unsmear_decoder_layers)
        self.unsmear_decoder_heads = int(unsmear_decoder_heads)
        self.unsmear_decoder_ff_dim = int(unsmear_decoder_ff_dim)
        self.unsmear_decoder_dropout = (
            float(dropout)
            if unsmear_decoder_dropout is None
            else float(unsmear_decoder_dropout)
        )
        self.token_encoder = SharedTokenEncoder(
            input_dim=int(input_dim),
            embed_dim=int(embed_dim),
            num_heads=int(num_heads),
            num_layers=int(num_layers),
            ff_dim=int(ff_dim),
            dropout=float(dropout),
            add_mask_channel=bool(add_mask_channel),
            use_positional_embedding=bool(use_positional_embedding),
            max_seq_len=int(max_seq_len),
            mask_encoder_input=bool(mask_encoder_input),
        )
        self.cls_head = SharedJetClassifierHead(embed_dim=int(embed_dim), dropout=float(dropout))

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=self.unsmear_decoder_heads,
            dim_feedforward=self.unsmear_decoder_ff_dim,
            dropout=self.unsmear_decoder_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.unsmear_decoder = nn.TransformerDecoder(
            dec_layer,
            num_layers=self.unsmear_decoder_layers,
        )
        self.unsmear_decoder_input = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(self.unsmear_decoder_dropout),
        )
        self.unsmear_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(self.unsmear_decoder_dropout),
            nn.Linear(self.embed_dim, self.input_dim),
        )
        self.delta_token_proj = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.delta_summary_norm = nn.LayerNorm(self.embed_dim)
        self.delta_gate = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.cls_gate_hidden_dim),
            nn.LayerNorm(self.cls_gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.cls_gate_hidden_dim, self.embed_dim),
        )
        self.cls_alpha = nn.Parameter(torch.tensor(float(cls_alpha_init), dtype=torch.float32))
        self._init_extra_weights()

    @property
    def input_proj(self):
        return self.token_encoder.input_proj

    @property
    def pos_embed(self):
        return self.token_encoder.pos_embed

    @property
    def encoder(self):
        return self.token_encoder.transformer

    @property
    def pool_query(self):
        return self.cls_head.pool_query

    @property
    def pool_attn(self):
        return self.cls_head.pool_attn

    @property
    def cls_norm(self):
        return self.cls_head.norm

    @property
    def classifier(self):
        return self.cls_head.classifier

    def _init_extra_weights(self):
        _init_linear_layers(self.unsmear_decoder)
        _init_linear_layers(self.unsmear_decoder_input)
        _init_linear_layers(self.unsmear_head)
        _init_linear_layers(self.delta_token_proj)
        _init_linear_layers(self.delta_gate)
        gate_last = self.delta_gate[-1]
        if isinstance(gate_last, nn.Linear) and gate_last.bias is not None:
            nn.init.constant_(gate_last.bias, self.cls_gate_init_bias)

    @staticmethod
    def _masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.to(dtype=x.dtype).unsqueeze(-1)
        num = (x * m).sum(dim=1)
        den = m.sum(dim=1).clamp_min(1.0)
        return num / den

    def _build_delta_summary(self, delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        delta_for_cls = delta.detach() if self.cls_detach_delta_for_cls else delta
        delta_tok = self.delta_token_proj(delta_for_cls)
        delta_tok = delta_tok * mask.to(dtype=delta_tok.dtype).unsqueeze(-1)
        z_delta = self._masked_mean_pool(delta_tok, mask)
        return self.delta_summary_norm(z_delta)

    def get_fusion_alpha(self) -> torch.Tensor:
        return F.softplus(self.cls_alpha)

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.token_encoder(x, mask)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        *,
        return_attention: bool = False,
        return_aux: bool = False,
    ):
        """Return `(reco, logits)` and optionally pooling attention / fusion diagnostics."""
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        h = self.encode(x, mask)

        dec_in = self.unsmear_decoder_input(h)
        dec_in = dec_in * mask.to(dtype=dec_in.dtype).unsqueeze(-1)
        dec_out = self.unsmear_decoder(
            tgt=dec_in,
            memory=h,
            tgt_key_padding_mask=~mask,
            memory_key_padding_mask=~mask,
        )
        dec_out = dec_out * mask.to(dtype=dec_out.dtype).unsqueeze(-1)
        delta = self.unsmear_head(dec_out)
        if self.mask_output:
            delta = delta * mask.to(dtype=delta.dtype).unsqueeze(-1)
        reco = (x + delta) if self.return_reco else delta
        if self.mask_output:
            reco = reco * mask.to(dtype=reco.dtype).unsqueeze(-1)

        if return_attention:
            z_main, attn = self.cls_head.pool_tokens(h, mask, return_attention=True)
        else:
            z_main = self.cls_head.pool_tokens(h, mask, return_attention=False)
            attn = None
        z_delta = torch.zeros_like(z_main)
        gate = torch.zeros_like(z_main)
        z_final = z_main
        if self.cls_use_delta_fusion:
            z_delta = self._build_delta_summary(delta, mask)
            gate_in = torch.cat([z_main, z_delta], dim=-1)
            gate = torch.sigmoid(self.delta_gate(gate_in))
            alpha = self.get_fusion_alpha().to(dtype=z_main.dtype)
            z_final = z_main + alpha * gate * z_delta
        else:
            alpha = self.get_fusion_alpha().to(dtype=z_main.dtype)

        logits = self.cls_head.classify_from_pooled(z_final)
        aux = {
            "z_main": z_main,
            "z_delta": z_delta,
            "z_final": z_final,
            "gate": gate,
            "gate_mean": gate.mean(),
            "gate_std": gate.std(unbiased=False),
            "alpha": alpha,
            "fusion_enabled": bool(self.cls_use_delta_fusion),
            "delta_detached_for_cls": bool(self.cls_detach_delta_for_cls),
        }

        if return_attention and return_aux:
            return reco, logits, attn, aux
        if return_attention:
            return reco, logits, attn
        if return_aux:
            return reco, logits, aux
        return reco, logits
