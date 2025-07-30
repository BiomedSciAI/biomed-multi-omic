import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.import_utils import (
    is_flash_attn_2_available,
    is_triton_available,
)

from bmfm_targets.config import SCModernBertConfig
from bmfm_targets.models.model_utils import (
    MaskedLMOutputWithEmbeddings,
    SequenceClassifierOutputWithEmbeddings,
)
from bmfm_targets.models.predictive.layers import (
    SCEmbeddingsLayer,
    SCMultiTaskHead,
    SCOnlyMLMHead,
    SCPooler,
    SCSequenceLabelingHead,
)
from bmfm_targets.training.serialization import prepare_model_dict_from_checkpoint

if is_flash_attn_2_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    from flash_attn.layers.rotary import RotaryEmbedding
    from flash_attn.ops.triton.rotary import apply_rotary
else:
    RotaryEmbedding = object

logger = logging.get_logger(__name__)


class ApplyRotaryEmbUnpad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cos,
        sin,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ):
        qkv = qkv.contiguous()
        total_nnz, _three, _nheads, headdim = qkv.shape
        qk = qkv[:, :2].view(total_nnz, -1, headdim)
        apply_rotary(
            qk,
            cos,
            sin,
            seqlen_offsets=0,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=False,
            inplace=True,
        )

        ctx.save_for_backward(cos, sin, cu_seqlens)
        ctx.max_seqlen = max_seqlen
        return qkv

    @staticmethod
    def backward(ctx, do):
        cos, sin, cu_seqlens = ctx.saved_tensors
        do = do.contiguous()
        total_nnz, _three, _nheads, headdim = do.shape
        dqk = do[:, :2].view(total_nnz, -1, headdim)
        apply_rotary(
            dqk,
            cos,
            sin,
            seqlen_offsets=0,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=False,
            inplace=True,
            conjugate=True,
        )

        return do, None, None, None, None, None, None


def apply_rotary_unpadded(
    qkv,
    cos,
    sin,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
):
    return ApplyRotaryEmbUnpad.apply(qkv, cos, sin, cu_seqlens, max_seqlen)


class SCModernBertUnpaddedRotaryEmbedding(RotaryEmbedding):
    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        max_seqlen: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(
            dim=dim, base=base, pos_idx_in_fp32=True, device=device, interleaved=False
        )
        self.max_seqlen = max_seqlen

        if max_seqlen is not None and device is not None and dtype is not None:
            self._update_cos_sin_cache(max_seqlen, device=device, dtype=dtype)

    def forward(
        self,
        qkv: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)

        qkv = apply_rotary_unpadded(
            qkv,
            self._cos_cached,
            self._sin_cached,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        return qkv

    def extra_repr(self) -> str:
        return f"dim={self.dim}, base={self.base}, scale_base={self.scale_base}"


class SCModernBertMLP(nn.Module):
    """
    Applies the GLU at the end of each ModernBERT layer.

    Compared to the default BERT architecture, this block replaces :class:`~transformers.model.bert.modeling_bert.BertIntermediate`
    and :class:`~transformers.model.bert.modeling_bert.SelfOutput` with a single module that has similar functionality.
    """

    def __init__(self, config: SCModernBertConfig):
        super().__init__()
        self.config = config
        self.Wi = nn.Linear(
            config.hidden_size, int(config.intermediate_size) * 2, bias=config.mlp_bias
        )
        self.act = ACT2FN[config.hidden_activation]
        self.drop = nn.Dropout(config.mlp_dropout)
        self.Wo = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(input) * gate))


class SCModernBertRotaryEmbedding(nn.Module):
    def __init__(self, config: SCModernBertConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def eager_attention_forward(
    module: "SCModernBertAttention",
    qkv: torch.Tensor,
    attention_mask: torch.Tensor,
    sliding_window_mask: torch.Tensor,
    position_ids: torch.LongTensor | None,
    local_attention: tuple[int, int],
    bs: int,
    dim: int,
    output_attentions: bool | None = False,
    **_kwargs,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
    # qkv: [batch_size, seqlen, 3, nheads, headdim]
    cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    # query, key, value: [batch_size, heads, seq_len, head_dim]
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    scale = module.head_dim**-0.5
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale

    if local_attention != (-1, -1):
        attention_mask = sliding_window_mask

    attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=module.attention_dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bs, -1, dim)
    if output_attentions:
        return (attn_output, attn_weights)
    return (attn_output,)


def flash_attention_forward(
    module: "SCModernBertAttention",
    qkv: torch.Tensor,
    rotary_emb: SCModernBertUnpaddedRotaryEmbedding,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    local_attention: tuple[int, int],
    bs: int,
    dim: int,
    target_dtype: torch.dtype = torch.bfloat16,
    **_kwargs,
) -> tuple[torch.Tensor]:
    # (total_seqlen, 3, nheads, headdim)
    qkv = rotary_emb(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

    convert_dtype = qkv.dtype not in (torch.float16, torch.bfloat16)
    if convert_dtype:
        # FA2 implementation only supports fp16 and bf16. If FA2 is supported,
        # bfloat16 must be supported as of FA2 2.5.7. (Turing GPUs not supported)
        orig_dtype = qkv.dtype
        qkv = qkv.to(target_dtype)

        attn = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=module.attention_dropout if module.training else 0.0,
            deterministic=module.deterministic_flash_attn,
            window_size=local_attention,
        )
        attn = attn.to(orig_dtype)  # type: ignore
    else:
        attn = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=module.attention_dropout if module.training else 0.0,
            deterministic=module.deterministic_flash_attn,
            window_size=local_attention,
        )
    return (attn.view(bs, dim),)


def sdpa_attention_forward(
    module: "SCModernBertAttention",
    qkv: torch.Tensor,
    attention_mask: torch.Tensor,
    sliding_window_mask: torch.Tensor,
    position_ids: torch.LongTensor | None,
    local_attention: tuple[int, int],
    bs: int,
    dim: int,
    **_kwargs,
) -> tuple[torch.Tensor]:
    # qkv: [batch_size, seqlen, 3, nheads, headdim]
    cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    # query, key, value: [batch_size, heads, seq_len, head_dim]
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    if local_attention != (-1, -1):
        attention_mask = sliding_window_mask

    attn_output = (
        F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=module.attention_dropout if module.training else 0.0,
            attn_mask=attention_mask,
        )
        .transpose(1, 2)
        .contiguous()
    )
    attn_output = attn_output.view(bs, -1, dim)
    return (attn_output,)


MODERNBERT_ATTENTION_FUNCTION = {
    "flash_attention_2": flash_attention_forward,
    "eager": eager_attention_forward,
    "sdpa": sdpa_attention_forward,
}


class SCModernBertAttention(nn.Module):
    """
    Performs multi-headed self attention on a batch of unpadded sequences.

    If Flash Attention 2 is installed, this module uses Flash Attention to improve throughput.
    If Flash Attention 2 is not installed, the implementation will use PyTorch's SDPA kernel,
    which requires padding and unpadding inputs, adding some overhead.

    See `forward` method for additional details.
    """

    def __init__(self, config: SCModernBertConfig, layer_id: int | None = None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})"
            )

        self.attention_dropout = config.attention_dropout
        self.deterministic_flash_attn = config.deterministic_flash_attn
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.head_dim * self.num_heads
        self.Wqkv = nn.Linear(
            config.hidden_size, 3 * self.all_head_size, bias=config.attention_bias
        )

        if layer_id % config.global_attn_every_n_layers != 0:
            self.local_attention = (
                config.local_attention // 2,
                config.local_attention // 2,
            )
        else:
            self.local_attention = (-1, -1)

        max_position_embeddings = config.max_position_embeddings
        if self.local_attention != (-1, -1):
            rope_theta = (
                config.global_rope_theta
                if config.local_rope_theta is None
                else config.local_rope_theta
            )
            max_position_embeddings = config.local_attention

        if config._attn_implementation == "flash_attention_2":
            self.rotary_emb = SCModernBertUnpaddedRotaryEmbedding(
                dim=self.head_dim, max_seqlen=max_position_embeddings, base=rope_theta
            )
        else:
            self.rotary_emb = SCModernBertRotaryEmbedding(config=config)

        self.Wo = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )
        self.out_drop = (
            nn.Dropout(config.attention_dropout)
            if config.attention_dropout > 0.0
            else nn.Identity()
        )
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> torch.Tensor:
        qkv = self.Wqkv(hidden_states)

        bs = hidden_states.shape[0]
        if self.config._attn_implementation == "flash_attention_2":
            qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
        else:
            qkv = qkv.view(bs, -1, 3, self.num_heads, self.head_dim)

        attn_outputs = MODERNBERT_ATTENTION_FUNCTION[self.config._attn_implementation](
            self,
            qkv=qkv,
            rotary_emb=self.rotary_emb,
            local_attention=self.local_attention,
            bs=bs,
            dim=self.all_head_size,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = attn_outputs[0]
        hidden_states = self.out_drop(self.Wo(hidden_states))

        return (hidden_states,) + attn_outputs[1:]  # add attentions if outputted


class SCModernBertEncoderLayer(nn.Module):
    def __init__(self, config: SCModernBertConfig, layer_id: int | None = None):
        super().__init__()
        self.config = config
        if layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(
                config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
            )
        self.attn = SCModernBertAttention(config=config, layer_id=layer_id)
        self.mlp_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )
        self.mlp = SCModernBertMLP(config)

    @torch.compile(dynamic=True)
    def compiled_mlp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.mlp_norm(hidden_states))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sliding_window_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        output_attentions: bool | None = False,
    ) -> torch.Tensor:
        attn_outputs = self.attn(
            self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + attn_outputs[0]
        mlp_output = (
            self.compiled_mlp(hidden_states)
            if self.config.reference_compile
            else self.mlp(self.mlp_norm(hidden_states))
        )
        hidden_states = hidden_states + mlp_output

        return (hidden_states,) + attn_outputs[1:]  # add attentions if outputted


class SCModernBertPreTrainedModel(PreTrainedModel):
    config: SCModernBertConfig
    base_model_prefix = "scmodernbert"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SCModernBertEncoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = False

    def _init_weights(self, module: nn.Module):
        cutoff_factor = self.config.initializer_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3

        def init_weight(module: nn.Module, std: float):
            nn.init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=std,
                a=-cutoff_factor * std,
                b=cutoff_factor * std,
            )

            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        stds = {
            "in": self.config.initializer_range,
            "out": self.config.initializer_range
            / math.sqrt(2.0 * self.config.num_hidden_layers),
            "embedding": self.config.initializer_range,
            "final_out": self.config.hidden_size**-0.5,
        }

        if isinstance(module, SCModernBertMLP):
            init_weight(module.Wi, stds["in"])
            init_weight(module.Wo, stds["out"])
        elif isinstance(module, SCModernBertAttention):
            init_weight(module.Wqkv, stds["in"])
            init_weight(module.Wo, stds["out"])
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def set_attention_implementation(self, attn_implementation: dict | str):
        """Checks and dispatches to hhe requested attention implementation."""
        # If the user didn't specify anything, try to use flash_attention_2 if available.
        # Otherwise we fall back to the default SDPA -> Eager from the super() method.
        # ModernBert's FA2 implementation correctly handles non-fp16/bf16 dtypes, we don't
        # need the FA2 warning for non-fp16/bf16 dtypes so we set fp16 for the FA2 check.
        requested_attn_implementation = self._check_attn_implementation(
            attn_implementation
        )
        try:
            attn_implementation = (
                "flash_attention_2"
                if requested_attn_implementation is None
                and self._flash_attn_2_can_dispatch()
                else attn_implementation
            )
        except (ValueError, ImportError):
            pass
        return super().set_attention_implementation(
            attn_implementation=attn_implementation
        )

    def _maybe_set_compile(self):
        if self.config.reference_compile is False:
            return

        if hasattr(self, "hf_device_map") and len(self.hf_device_map) > 1:
            if self.config.reference_compile:
                logger.warning_once(
                    "If `accelerate` split the model across devices, `torch.compile` will not work. "
                    "Falling back to non-compiled mode."
                )
            self.config.reference_compile = False

        if self.device.type == "mps":
            if self.config.reference_compile:
                logger.warning_once(
                    "Compiling the model with `torch.compile` and using a `torch.mps` device is not supported. "
                    "Falling back to non-compiled mode."
                )
            self.config.reference_compile = False

        if self.device.type == "cpu":
            if self.config.reference_compile:
                logger.warning_once(
                    "Compiling the model with `torch.compile` and using a `torch.cpu` device is not supported. "
                    "Falling back to non-compiled mode."
                )
            self.config.reference_compile = False

        if self.config.reference_compile is None:
            self.config.reference_compile = is_triton_available()

    def resize_token_embeddings(self, *args, **kwargs):
        model_embeds = super().resize_token_embeddings(*args, **kwargs)

        if self.config.reference_compile in {True, None}:
            if self.config.reference_compile:
                logger.warning_once(
                    "Resizing token embeddings with `torch.compile` is not supported. Falling back to non-compiled mode."
                )
            self.config.reference_compile = False

        return model_embeds


def _unpad_modernbert_input(
    inputs: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """
    Remove padding from input sequences.

    Args:
    ----
        inputs: (batch, seqlen, ...) or (batch, seqlen)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
        position_ids: (batch, seqlen), int, position ids
        labels: (batch, seqlen), int, labels

    Returns:
    -------
        unpadded_inputs: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask.
        indices: (total_nnz)
        cu_seqlens: (batch + 1), the cumulative sequence lengths
        max_seqlen_in_batch: int
        unpadded_position_ids: (total_nnz) or None
        unpadded_labels: (total_nnz) or None

    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = torch.nn.functional.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
    )

    if inputs.dim() == 2:
        unpadded_inputs = inputs.flatten()[indices]
    else:
        batch, seqlen, *rest = inputs.shape
        shape = batch * seqlen
        unpadded_inputs = inputs.view(shape, *rest)[indices]

    unpadded_position_ids = (
        position_ids.flatten()[indices] if position_ids is not None else None
    )
    unpadded_labels = labels.flatten()[indices] if labels is not None else None

    return (
        unpadded_inputs,
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        unpadded_position_ids,
        unpadded_labels,
    )


def _pad_modernbert_output(
    inputs: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int,
) -> torch.Tensor:
    if inputs.dim() == 1:
        output = torch.zeros(batch * seqlen, dtype=inputs.dtype, device=inputs.device)
        output[indices] = inputs
        padded_inputs = output.view(batch, seqlen)
    else:
        _, *rest = inputs.shape
        output = torch.zeros(
            batch * seqlen, *rest, dtype=inputs.dtype, device=inputs.device
        )
        output[indices] = inputs
        padded_inputs = output.view(batch, seqlen, *rest)

    return padded_inputs


class SCModernBertModel(SCModernBertPreTrainedModel):
    def __init__(self, config: SCModernBertConfig, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings = SCEmbeddingsLayer(config)
        self.layers = nn.ModuleList(
            [
                SCModernBertEncoderLayer(config, layer_id)
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )
        self.gradient_checkpointing = False
        self.pooler = SCPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.genes_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.genes_embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sliding_window_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        batch_size: int | None = None,
        seq_len: int | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        head_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...] | BaseModelOutputWithPoolingAndCrossAttentions:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        original_attention_mask = attention_mask.detach().clone()
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        self._maybe_set_compile()

        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)

        if batch_size is None and seq_len is None:
            if inputs_embeds is not None:
                batch_size, seq_len = inputs_embeds.shape[:2]
            else:
                batch_size, num_fields, seq_len = input_ids.shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_len), device=device, dtype=torch.bool
            )

        repad = False
        if self.config._attn_implementation == "flash_attention_2":
            if indices is None and cu_seqlens is None and max_seqlen is None:
                repad = True
                if inputs_embeds is None:
                    with torch.no_grad():
                        (
                            input_ids,
                            indices,
                            cu_seqlens,
                            max_seqlen,
                            *_,
                        ) = _unpad_modernbert_input(
                            inputs=input_ids, attention_mask=attention_mask
                        )
                else:
                    (
                        inputs_embeds,
                        indices,
                        cu_seqlens,
                        max_seqlen,
                        *_,
                    ) = _unpad_modernbert_input(
                        inputs=inputs_embeds, attention_mask=attention_mask
                    )
        else:
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            attention_mask, sliding_window_mask = self._update_attention_mask(
                attention_mask, output_attentions=output_attentions
            )

        hidden_states = self.embeddings(
            input_ids=input_ids, inputs_embeds=inputs_embeds
        )

        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    sliding_window_mask,
                    position_ids,
                    cu_seqlens,
                    max_seqlen,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    sliding_window_mask=sliding_window_mask,
                    position_ids=position_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    output_attentions=output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions and len(layer_outputs) > 1:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.final_norm(hidden_states)

        if repad:
            hidden_states = _pad_modernbert_output(
                inputs=hidden_states, indices=indices, batch=batch_size, seqlen=seq_len
            )
            if all_hidden_states is not None:
                all_hidden_states = tuple(
                    _pad_modernbert_output(
                        inputs=hs, indices=indices, batch=batch_size, seqlen=seq_len
                    )
                    for hs in all_hidden_states
                )

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )

        pooler_output = (
            self.pooler(hidden_states, original_attention_mask)
            if self.pooler is not None
            else None
        )

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            pooler_output=pooler_output,
        )

    def _update_attention_mask(
        self, attention_mask: torch.Tensor, output_attentions: bool
    ) -> torch.Tensor:
        if output_attentions:
            if self.config._attn_implementation == "sdpa":
                logger.warning_once(
                    "Outputting attentions is only supported with the 'eager' attention implementation, "
                    'not with "sdpa". Falling back to `attn_implementation="eager"`.'
                )
                self.config._attn_implementation = "eager"
            elif self.config._attn_implementation != "eager":
                logger.warning_once(
                    "Outputting attentions is only supported with the eager attention implementation, "
                    f'not with {self.config._attn_implementation}. Consider setting `attn_implementation="eager"`.'
                    " Setting `output_attentions=False`."
                )

        global_attention_mask = _prepare_4d_attention_mask(attention_mask, self.dtype)

        # Create position indices
        rows = torch.arange(global_attention_mask.shape[2]).unsqueeze(0)
        # Calculate distance between positions
        distance = torch.abs(rows - rows.T)

        # Create sliding window mask (1 for positions within window, 0 outside)
        window_mask = (
            (distance <= self.config.local_attention // 2)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(attention_mask.device)
        )
        # Combine with existing mask
        sliding_window_mask = global_attention_mask.masked_fill(
            window_mask.logical_not(), torch.finfo(self.dtype).min
        )

        return global_attention_mask, sliding_window_mask


class SCModernBertPredictionHead(nn.Module):
    def __init__(self, config: SCModernBertConfig):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(
            config.hidden_size, config.hidden_size, config.classifier_bias
        )
        self.act = ACT2FN[config.classifier_activation]
        self.norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.dense(hidden_states)))


class SCModernBertForMaskedLM(SCModernBertPreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.weight"]

    def __init__(self, config: SCModernBertConfig):
        super().__init__(config)
        self.config = config
        self.scmodernbert = SCModernBertModel(config, add_pooling_layer=False)
        self.cls = SCOnlyMLMHead(config)

        self.sparse_prediction = self.config.sparse_prediction
        self.sparse_pred_ignore_index = self.config.sparse_pred_ignore_index

        # Initialize weights and apply final processing
        self.post_init()
        if self.config.checkpoint:
            logger.info("Loading model from checkpoint " + str(self.config.checkpoint))
            model_dict = prepare_model_dict_from_checkpoint(self.config.checkpoint)
            key_report = self.load_state_dict(model_dict, strict=False)
            logger.info(f"Loading complete. {len(model_dict)} layers in ckpt.")
            logger.info(f"Unexpected keys: {key_report.unexpected_keys}")
            logger.info(f"Missing keys: {key_report.missing_keys}")

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.cls.predictions.decoder = new_embeddings

    def tie_weights(self):
        logger.warning("Tie weights not supported for this model")
        return

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sliding_window_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        batch_size: int | None = None,
        seq_len: int | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor] | MaskedLMOutput:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        self._maybe_set_compile()

        if self.config._attn_implementation == "flash_attention_2":
            if indices is None and cu_seqlens is None and max_seqlen is None:
                if batch_size is None and seq_len is None:
                    if inputs_embeds is not None:
                        batch_size, seq_len = inputs_embeds.shape[:2]
                    else:
                        batch_size, seq_len = input_ids.shape[:2]
                device = (
                    input_ids.device if input_ids is not None else inputs_embeds.device
                )

                if attention_mask is None:
                    attention_mask = torch.ones(
                        (batch_size, seq_len), device=device, dtype=torch.bool
                    )

                if inputs_embeds is None:
                    with torch.no_grad():
                        (
                            input_ids,
                            indices,
                            cu_seqlens,
                            max_seqlen,
                            position_ids,
                            labels,
                        ) = _unpad_modernbert_input(
                            inputs=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            labels=labels,
                        )
                else:
                    (
                        inputs_embeds,
                        indices,
                        cu_seqlens,
                        max_seqlen,
                        position_ids,
                        labels,
                    ) = _unpad_modernbert_input(
                        inputs=inputs_embeds,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        labels=labels,
                    )

        outputs = self.scmodernbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]

        if self.sparse_prediction and labels is not None:
            # flatten labels and output first
            labels = labels.view(-1)
            last_hidden_state = last_hidden_state.view(labels.shape[0], -1)

            # then filter out the non-masked tokens
            mask_tokens = labels != self.sparse_pred_ignore_index
            last_hidden_state = last_hidden_state[mask_tokens]
            labels = labels[mask_tokens]

        logits = self.cls(last_hidden_state)

        if self.config._attn_implementation == "flash_attention_2":
            with (
                nullcontext()
                if self.config.repad_logits_with_grad or labels is None
                else torch.no_grad()
            ):
                logits = _pad_modernbert_output(
                    inputs=logits, indices=indices, batch=batch_size, seqlen=seq_len
                )

        if not return_dict:
            return (logits,) + outputs[2:]

        embeddings = outputs["last_hidden_state"][:, 0, :]

        return MaskedLMOutputWithEmbeddings(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            embeddings=embeddings,
        )

    def prepare_inputs_for_generation(
        self, input_ids, attention_mask=None, **model_kwargs
    ):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))],
            dim=-1,
        )
        dummy_token = torch.full(
            (effective_batch_size, 1),
            self.config.pad_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class SCModernBertForSequenceClassification(SCModernBertPreTrainedModel):
    def __init__(self, config: SCModernBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.scmodernbert = SCModernBertModel(config)

        self.head = SCModernBertPredictionHead(config)
        self.label_column = self.config.label_columns[0]
        self.drop = torch.nn.Dropout(config.classifier_dropout)
        self.label_column = self.config.label_columns[0]
        self.classifier = nn.Linear(config.hidden_size, self.label_column.output_size)

        # Initialize weights and apply final processing
        self.post_init()

        if self.config.checkpoint:
            logger.info("Loading model from checkpoint " + str(self.config.checkpoint))
            model_dict = prepare_model_dict_from_checkpoint(self.config.checkpoint)
            key_report = self.load_state_dict(model_dict, strict=False)
            logger.info(f"Loading complete. {len(model_dict)} layers in ckpt.")
            logger.info(f"Unexpected keys: {key_report.unexpected_keys}")
            logger.info(f"Missing keys: {key_report.missing_keys}")

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sliding_window_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        batch_size: int | None = None,
        seq_len: int | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor] | SequenceClassifierOutputWithEmbeddings:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        self._maybe_set_compile()

        outputs = self.scmodernbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = self.head(outputs.pooler_output)
        pooled_output = self.drop(pooled_output)
        logits = {self.label_column.label_column_name: self.classifier(pooled_output)}

        if not return_dict:
            return (logits,) + outputs[2:]

        return SequenceClassifierOutputWithEmbeddings(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            embeddings=pooled_output,
        )

    def prepare_inputs_for_generation(
        self, input_ids, attention_mask=None, **model_kwargs
    ):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))],
            dim=-1,
        )
        dummy_token = torch.full(
            (effective_batch_size, 1),
            self.config.pad_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class SCModernBertForSequenceLabeling(SCModernBertPreTrainedModel):
    def __init__(self, config: SCModernBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.head = SCModernBertPredictionHead(config)
        self.scmodernbert = SCModernBertModel(config)
        self.cls = SCSequenceLabelingHead(config)

        # Initialize weights and apply final processing
        self.post_init()

        if self.config.checkpoint:
            logger.info("Loading model from checkpoint " + str(self.config.checkpoint))
            model_dict = prepare_model_dict_from_checkpoint(
                self.config.checkpoint, self.base_model_prefix
            )
            key_report = self.load_state_dict(model_dict, strict=False)
            logger.info(f"Loading complete. {len(model_dict)} layers in ckpt.")
            logger.info(f"Unexpected keys: {key_report.unexpected_keys}")
            logger.info(f"Missing keys: {key_report.missing_keys}")

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def tie_weights(self):
        logger.warning("Tie weights not supported for this model")
        return

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sliding_window_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        batch_size: int | None = None,
        seq_len: int | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor] | TokenClassifierOutput:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        self._maybe_set_compile()

        outputs = self.scmodernbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]

        last_hidden_state = self.head(last_hidden_state)
        last_hidden_state = self.drop(last_hidden_state)

        logits = self.cls(last_hidden_state)

        if not return_dict:
            return (logits,) + outputs[:2]

        return TokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, attention_mask=None, **model_kwargs
    ):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))],
            dim=-1,
        )
        dummy_token = torch.full(
            (effective_batch_size, 1),
            self.config.pad_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class SCModernBertForMultiTaskModeling(SCModernBertModel):
    _tied_weights_keys = [
        "cls.predictions.decoder.bias",
        "cls.predictions.decoder.weight",
    ]

    def __init__(self, config: SCModernBertConfig):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.scmodernbert = SCModernBertModel(config)
        self.cls = SCMultiTaskHead(config)

        # Initialize weights and apply final processing
        self.post_init()
        if self.config.checkpoint:
            logger.info("Loading model from checkpoint " + str(self.config.checkpoint))
            model_dict = prepare_model_dict_from_checkpoint(self.config.checkpoint)
            key_report = self.load_state_dict(model_dict, strict=False)
            logger.info(f"Loading complete. {len(model_dict)} layers in ckpt.")
            logger.info(f"Unexpected keys: {key_report.unexpected_keys}")
            logger.info(f"Missing keys: {key_report.missing_keys}")

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def tie_weights(self):
        logger.warning("Tie weights not supported for this model")
        return

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sliding_window_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        batch_size: int | None = None,
        seq_len: int | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor] | MaskedLMOutput:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # You can do a for loop over the fields but it's not efficient
        outputs = self.scmodernbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]
        if self.config.classifier_pooling == "cls":
            last_hidden_state = last_hidden_state[:, 0]
        elif self.config.classifier_pooling == "mean":
            last_hidden_state = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(
                dim=1
            ) / attention_mask.sum(dim=1, keepdim=True)

        sequence_output = outputs[0]
        logits = self.cls(sequence_output, last_hidden_state)

        if not return_dict:
            return (logits,) + outputs[2:]

        return SequenceClassifierOutputWithEmbeddings(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            embeddings=last_hidden_state,
        )

    def prepare_inputs_for_generation(
        self, input_ids, attention_mask=None, **model_kwargs
    ):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))],
            dim=-1,
        )
        dummy_token = torch.full(
            (effective_batch_size, 1),
            self.config.pad_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}
