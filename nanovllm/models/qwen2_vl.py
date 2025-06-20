import torch
from torch import nn
from transformers.models.qwen2_vl.configuration_qwen2_vl import (
    Qwen2VLConfig, Qwen2VLVisionConfig)

from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.models.qwen3 import Qwen3DecoderLayer


class Qwen2VisionPatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 14, in_channels: int = 3, embed_dim: int = 1280) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Qwen2VisionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, act: str) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU() if act == "gelu" else nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen2VisionTransformer(nn.Module):
    def __init__(self, vision_config: Qwen2VLVisionConfig) -> None:
        super().__init__()
        self.patch_embed = Qwen2VisionPatchEmbed(
            patch_size=vision_config.patch_size,
            in_channels=vision_config.in_channels,
            embed_dim=vision_config.embed_dim,
        )
        self.blocks = nn.ModuleList([
            Qwen2VisionBlock(
                vision_config.embed_dim,
                vision_config.num_heads,
                vision_config.mlp_ratio,
                vision_config.hidden_act,
            )
            for _ in range(vision_config.depth)
        ])
        self.norm = nn.LayerNorm(vision_config.embed_dim)
        self.proj = nn.Linear(vision_config.embed_dim, vision_config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.proj(x)
        return x


class Qwen2VLModel(nn.Module):
    def __init__(self, config: Qwen2VLConfig) -> None:
        super().__init__()
        text_config = config.text_config
        self.text_config = text_config
        self.visual = Qwen2VisionTransformer(config.vision_config)
        self.embed_tokens = VocabParallelEmbedding(text_config.vocab_size,
                                                   text_config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(text_config)
            for _ in range(text_config.num_hidden_layers)
        ])
        self.norm = RMSNorm(text_config.hidden_size,
                            eps=text_config.rms_norm_eps)

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                pixel_values: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        if pixel_values is not None:
            vis_embed = self.visual(pixel_values)
            image_mask = (input_ids == self.text_config.image_token_id)
            if image_mask.any():
                vis_embed = vis_embed.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
                hidden_states = torch.where(image_mask.unsqueeze(-1), vis_embed, hidden_states)
        residual = None
        seq_len = hidden_states.size(1)
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        positions = positions.view(-1)
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        hidden_states = hidden_states.view(-1, seq_len, hidden_states.size(-1))
        return hidden_states


class Qwen2VLForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen2VLConfig) -> None:
        super().__init__()
        self.model = Qwen2VLModel(config)
        text_config = config.text_config
        self.lm_head = ParallelLMHead(text_config.vocab_size,
                                      text_config.hidden_size)
        self.tie_word_embeddings = text_config.tie_word_embeddings
        if self.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                pixel_values: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(input_ids, positions, pixel_values)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        logits = self.lm_head(hidden_states)
        return logits.view(-1, logits.size(-1))
