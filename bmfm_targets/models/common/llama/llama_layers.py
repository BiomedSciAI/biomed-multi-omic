from typing import Protocol

import torch
import torch.nn as nn
from pydantic import BaseModel
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from .constants import AttentionKind


class EmbeddingLayer(Protocol):
    def __init__(self):
        ...

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of the embeddings layer.

        Args:
        ----
            input_ids (torch.Tensor): A torch tensor with shape [batch_size, num_fields, seq_length] containing the input ids.
            position_ids (torch.LongTensor | None): position ids. Default, None
            inputs_embeds: (torch.Tensor | None) precalculated embeddings. If supplied,
                bypasses logic of this module. Default, None.

        Returns:
        -------
            torch.Tensor: A torch tensor with shape [batch_size, _, hidden_size] containing the embeddings.

        """
        ...


class LlamaParams(BaseModel):
    n_layer: int
    n_embd: int
    n_head: int
    n_aux_tokens: int = 0
    attention: AttentionKind = AttentionKind.TORCH


def _find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def _calc_hidden_mlp_dim(n_input):
    hidden_dim = 4 * n_input
    n_hidden = int(2 * hidden_dim / 3)
    n_hidden = _find_multiple(n_hidden, 256)
    return n_hidden


class MLP(nn.Module):
    def __init__(self, n_embd: int) -> None:
        super().__init__()
        n_hidden = _calc_hidden_mlp_dim(n_embd)
        self.c_fc1 = nn.Linear(n_embd, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(n_embd, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.register_parameter("scale", param=nn.Parameter(torch.ones(size)))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.pow(2).mean(dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return (self.scale * x_normed).type_as(x)


class LlamaAttentionBlock(nn.Module):
    def __init__(self, n_head, n_embd) -> None:
        super().__init__()

        self.n_embd = n_embd
        self.n_head = n_head

        assert n_embd % n_head == 0

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        (B, T, C) = h.size()
        n_head = self.n_head
        q, k, v = self.c_attn(h).split(self.n_embd, dim=2)

        head_size = C // n_head
        k = k.view(B, T, n_head, head_size).transpose(1, 2)
        q = q.view(B, T, n_head, head_size).transpose(1, 2)
        v = v.view(B, T, n_head, head_size).transpose(1, 2)

        y = LlamaEncoder.compute_attention(q, k, v, attention_mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)

        return y


class LlamaEncoderBlock(nn.Module):
    def __init__(self, pars: LlamaParams, no_first_rms=False) -> None:
        super().__init__()

        n_embd = pars.n_embd
        if not no_first_rms:
            self.rms_1 = RMSNorm(n_embd)
        self.attn = LlamaAttentionBlock(pars.n_head, n_embd)
        self.rms_2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd)
        self.no_first_rms = no_first_rms

    def forward(
        self,
        h: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
    ) -> torch.Tensor:
        if self.no_first_rms:
            h = h + self.attn(h, attention_mask)
        else:
            h = h + self.attn(self.rms_1(h), attention_mask)
        h = h + self.mlp(self.rms_2(h))
        return h


class LlamaEncoder(nn.Module):
    def __init__(self, embedding_layer_class: EmbeddingLayer, pars: LlamaParams):
        super().__init__()
        self.embeddings = embedding_layer_class()
        self.pars = pars
        if pars.n_aux_tokens:
            self.register_parameter(
                "aux_tokens",
                param=nn.Parameter(torch.randn(1, pars.n_aux_tokens, pars.n_embd)),
            )
            self.aux_proj = nn.Linear(pars.n_embd, pars.n_embd, bias=False)

        transformer_blocks = [
            LlamaEncoderBlock(pars, no_first_rms=(index == 0))
            for index in range(pars.n_layer)
        ]
        self.blocks = nn.ModuleList(transformer_blocks)
        if pars.attention == AttentionKind.FLEX:
            LlamaEncoder.flex_attention = torch.compile(flex_attention)

    @staticmethod
    def compute_attention(q, k, v, attention_mask):
        if attention_mask.ndim == 2:

            def padding_mask_mod(score, b, h, q_idx, kv_idx):
                return score + attention_mask[b, kv_idx]

            out = LlamaEncoder.flex_attention(q, k, v, score_mod=padding_mask_mod)
        else:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        return out

    def get_aux_tokens(
        self,
        batch_size,
        input_aux_tokens,
    ) -> tuple[int | None, list[torch.Tensor] | None]:
        """Creates a list of auxiliary (e.g., memory or time) tokens."""
        if not self.pars.n_aux_tokens:
            return None, None

        aux_tokens: torch.Tensor = self.aux_tokens
        aux_tokens = aux_tokens.expand(batch_size, -1, -1)

        if input_aux_tokens is not None:
            aux_tokens = self.aux_proj(aux_tokens) + input_aux_tokens
            aux_tokens = aux_tokens + input_aux_tokens

        return self.pars.n_aux_tokens, [aux_tokens]

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        input_aux_tokens: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
    ):
        h = self.embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )
        batch_size = h.size(0)
        n_aux_tokens, aux_tokens = self.get_aux_tokens(batch_size, input_aux_tokens)

        if n_aux_tokens:
            ones = torch.ones(
                batch_size,
                n_aux_tokens,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([attention_mask, ones], dim=1)
            h = torch.cat([h] + aux_tokens, dim=1)

        attention_mask = (1.0 - attention_mask.to(dtype=h.dtype)) * torch.finfo(
            h.dtype
        ).min
        if attention_bias is not None:
            attention_mask[..., :-n_aux_tokens] += attention_bias
        if self.pars.attention == AttentionKind.TORCH:
            attention_mask = attention_mask[:, None, None, :]

        hidden_states = []
        for block in self.blocks:
            h = checkpoint(block, h, attention_mask, use_reentrant=False)
            if output_hidden_states:
                hidden_states.append(h[:, :-n_aux_tokens, ...] if n_aux_tokens else h)

        if n_aux_tokens:
            h = h[:, :-n_aux_tokens, ...]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=h,
            hidden_states=hidden_states if output_hidden_states else None,
        )
