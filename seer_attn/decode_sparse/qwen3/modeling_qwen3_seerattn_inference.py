# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.logits_process import TopPLogitsWarper
from transformers.generation import GenerationMixin

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
# from transformers.processing_utils import Unpack
from transformers.utils import (
    logging,
)
from .configuration_qwen3_seerattn import SeerAttnQwen3Config
from ...utils import BaseModelOutputWithPastAndCache, CausalLMOutputWithPastAndCache
from ..attn_gate_inf import ATTNGATE_CLASSES, compute_oracle_sparse_mask
import copy, math, os
from einops import rearrange
from ..attention_forward_sparse import sparse_flash_attention_forward
from ..attention_forward_dense import dense_flash_attention_forward
from ...modules.layernorm import RMSNorm
from flash_attn.layers.rotary import apply_rotary_emb_func
from ...decode_sparse.cache_utils import KCompressionCache
from ...modules.common import apply_rotary_pos_emb



from huggingface_hub import hf_hub_download

logger = logging.get_logger(__name__)

class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
    


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj



class Qwen3SeerAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: SeerAttnQwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.block_budget = self.config.seerattn_token_budget // self.config.seerattn_gate_block_size
        self.block_sliding_window_size = self.config.seerattn_sliding_window_size // self.config.seerattn_gate_block_size
        
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window
        if not (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            self.sliding_window = None

        self.attn_gate = ATTNGATE_CLASSES[config.seerattn_k_seq_pooling_type](
            config.seerattn_gate_block_size, 
            self.head_dim, 
            config.seerattn_gate_hidden_size,
            num_k_head=config.num_key_value_heads, 
            num_q_head=config.num_attention_heads,
            q_head_pooling_type=config.seerattn_q_head_pooling_type,
            use_flash_rope=config.use_flash_rope,
            use_qk_norm=config.seerattn_use_qk_norm,
        )

        self.mask_loss_func = torch.nn.KLDivLoss()
        self.use_flash_rope = config.use_flash_rope
        self.seerattn_implementation = config.seerattn_implementation
        self.seerattn_output_sparsity = config.seerattn_output_sparsity
        self.block_budget = config.seerattn_token_budget // config.seerattn_gate_block_size
        self.block_sliding_window_size = config.seerattn_sliding_window_size // config.seerattn_gate_block_size
        self.seerattn_start_layer = config.seerattn_start_layer


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        k_compressed_cache: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cache_seqlens: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  
        position_embeddings_gate_q: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        block_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, ## block position embeddings
        block_attention_mask: Optional[torch.Tensor] = None, ## block attention mask
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        input_shape = hidden_states.shape[:-1]
        q_len = hidden_states.shape[1]

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = rearrange(q, '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)


        q_nope, k_nope = q, k

        cos, sin = position_embeddings

        if self.use_flash_rope:
            q = apply_rotary_emb_func(q, cos, sin, False, True, cu_seqlens=None, max_seqlen=q_len)
            k = apply_rotary_emb_func(k, cos, sin, False, True, cu_seqlens=None, max_seqlen=q_len)
        else:
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_value.update(k.flatten(-2, -1), v.flatten(-2, -1), self.layer_idx, cache_kwargs)
            k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
            v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        if self.seerattn_implementation != "seer_dense":
            if self.seerattn_implementation == "seer_sparse":
                block_sparse_mask = self.attn_gate(
                    k_nope,
                    self.layer_idx,
                    k_compressed_cache,
                    q_nope,
                    block_attention_mask,
                    max_cache_len=k.shape[1],
                    position_embeddings=position_embeddings_gate_q,
                    block_position_embeddings=block_position_embeddings,
                    threshold=self.config.seerattn_threshold,
                    block_budget=self.block_budget,
                    sparsity_method=self.config.seerattn_sparsity_method,
                )
            else:
                block_sparse_mask = compute_oracle_sparse_mask(
                    q,
                    k,
                    cache_seqlens,
                    block_attention_mask,
                    self.config.seerattn_gate_block_size,
                    self.config.seerattn_sparsity_method,
                    self.config.seerattn_threshold,
                    self.block_budget,
                )

        activate_and_original_block_count = None
        if self.seerattn_output_sparsity and q_len == 1 and self.layer_idx >= self.seerattn_start_layer:
            activate_block_count = block_sparse_mask.sum()   # block_sparse_mask in shape batch, kv_heads, seq(block)
            original_block_count = block_attention_mask.sum() * self.config.num_key_value_heads # block_attention_mask in shape batch, 1, seq(block)
            activate_and_original_block_count = (activate_block_count.item(), original_block_count.item())

        if self.config.seerattn_implementation == "seer_dense" or self.layer_idx < self.seerattn_start_layer:
            attn_output = dense_flash_attention_forward(
                q,
                k,
                v,
                attention_mask=attention_mask,
                query_length=q_len,
                softmax_scale=self.scaling,
                cache_seqlens=cache_seqlens,
            )
        else:
            attn_output = sparse_flash_attention_forward(
                q,
                k,
                v,
                attention_mask=attention_mask,
                query_length=q_len,
                softmax_scale=self.scaling,
                cache_seqlens=cache_seqlens,
                block_mask=block_sparse_mask,
                block_size=self.config.seerattn_gate_block_size,
            )
        

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, activate_and_original_block_count


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: SeerAttnQwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        # self.self_attn = SeerAttnQwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.self_attn = Qwen3SeerAttention(config=config, layer_idx=layer_idx)
        self.fused_norm = config.fused_norm

        if config.fused_norm:
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        k_compressed_cache: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cache_seqlens: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        position_embeddings_gate_q: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        block_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        block_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention

        hidden_states, activate_and_original_block_count = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            k_compressed_cache=k_compressed_cache,
            cache_position=cache_position,
            cache_seqlens=cache_seqlens,
            position_embeddings=position_embeddings,
            position_embeddings_gate_q=position_embeddings_gate_q,
            block_position_embeddings=block_position_embeddings,
            block_attention_mask=block_attention_mask,
            **kwargs,
        )
        if self.fused_norm:
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,activate_and_original_block_count) 
        return outputs



class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config: SeerAttnQwen3Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            if self.config.use_flash_rope:
                emb = freqs ## to use rope func in flash attn
                cos = emb.cos().squeeze(0)
                sin = emb.sin().squeeze(0)
            else:
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos()
                sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3PreTrainedModel(PreTrainedModel):
    config_class = SeerAttnQwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SeerAttnQwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen3RMSNorm):
            module.weight.data.fill_(1.0)


class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: SeerAttnQwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        block_config = copy.deepcopy(config)
        block_config.hidden_size = config.seerattn_gate_hidden_size * config.num_attention_heads
        self.block_rotary_emb = Qwen3RotaryEmbedding(config=block_config)
        self.rotary_emb_gate_q = Qwen3RotaryEmbedding(config=block_config)

        self.gradient_checkpointing = False
        self.num_layers = config.num_hidden_layers

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        k_compressed_cache: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCache]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
            if k_compressed_cache is None:
                if past_key_values.get_seq_length() == 0:
                    self.k_compressed_cache = KCompressionCache(self.num_layers, self.config.seerattn_gate_block_size)                
                k_compressed_cache = self.k_compressed_cache


        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        cache_seqlens = torch.sum(attention_mask.to(torch.int32), dim=-1, dtype=torch.int32) 

        hidden_states = inputs_embeds


        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        if self.config.seerattn_use_rope:
            block_position_ids = position_ids[:, 0::self.config.seerattn_gate_block_size] ## downsampled position ids
            block_position_embeddings = self.block_rotary_emb(hidden_states, block_position_ids) # downsampled position embeddings
            position_embeddings_gate_q = self.rotary_emb_gate_q(hidden_states, position_ids) 
        else:
            block_position_embeddings = None
            position_embeddings_gate_q = None


        block_attention_mask = self._gen_block_attention_mask(
            attention_mask
        )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_activate_and_original_block_count = () if self.config.seerattn_output_sparsity else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_values,
                k_compressed_cache=k_compressed_cache,
                cache_position=cache_position,
                cache_seqlens=cache_seqlens,
                position_embeddings=position_embeddings,
                position_embeddings_gate_q=position_embeddings_gate_q,
                block_position_embeddings=block_position_embeddings,
                block_attention_mask=block_attention_mask,
            )

            hidden_states = layer_outputs[0]

            if self.config.seerattn_output_sparsity and layer_outputs[1] is not None:
                all_activate_and_original_block_count += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPastAndCache(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            sparsitys_info=all_activate_and_original_block_count,
            k_compressed_cache=k_compressed_cache if use_cache else None,
        )

    def _gen_block_attention_mask(
        self,
        attention_mask,
    ):
        
        mask = F.max_pool1d(
            attention_mask.to(torch.bfloat16),
            kernel_size=[self.config.seerattn_gate_block_size],
            stride=[self.config.seerattn_gate_block_size],
            ceil_mode=True,
        ).to(torch.bool)
        mask = mask.unsqueeze(1)
        return mask


class SeerDecodingQwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.num_layers = config.num_hidden_layers
        self.block_size = config.seerattn_gate_block_size

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        k_compressed_cache: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPastAndCache]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )


        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            k_compressed_cache=k_compressed_cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPastAndCache(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            sparsitys_info=outputs.sparsitys_info,
            k_compressed_cache=outputs.k_compressed_cache,
        )

    def batch_exist_generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        max_length: int = 100,
        do_sample: bool = False,
        **model_kwargs,
    ):
        """
        Modified generation loop that dynamically adjusts batch size by filtering finished sequences
        and reorganizing the KV cache.
        """
        # Initialize variables
        generation_config, model_kwargs = self._prepare_generation_config(None)
        generated = input_ids

        if isinstance(generation_config.eos_token_id, list):
            eos_token_id = generation_config.eos_token_id[0]
            eos_token_ids = torch.tensor(generation_config.eos_token_id, device=input_ids.device)
        else:
            eos_token_id = generation_config.eos_token_id
        initial_batch_size = input_ids.shape[0]

        device = input_ids.device
        finished = torch.zeros(initial_batch_size, dtype=torch.bool, device=device)
        current_kvcache = DynamicCache()
        current_kcompressed_cache = KCompressionCache(self.num_layers, self.block_size)
        
        cur_input = generated
        cur_to_orig = torch.arange(initial_batch_size, device=device)

        if do_sample:
            top_p_warper = TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=1)

        sparsitys_info_list = []
        for step in range(max_length - generated.shape[1]):
            # Forward pass: get next token logits and updated past_key_values
            torch.cuda.empty_cache()
            with torch.no_grad():
                outputs = self(
                    cur_input, 
                    attention_mask=attention_mask,
                    past_key_values=current_kvcache, 
                    k_compressed_cache=current_kcompressed_cache,
                    use_cache=True,
                    logits_to_keep=1,
            )
                

            logits = outputs.logits[:, -1, :].clone().float()
            logits = logits.to(input_ids.device)

            if outputs.sparsitys_info is not None and outputs.sparsitys_info:
                #dense prefill, step 0 is empty
                sparsitys_info_list.append(outputs.sparsitys_info)

            if do_sample:
                logits /= generation_config.temperature
                processed_logits = top_p_warper(cur_input, logits)
                probs = torch.softmax(processed_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)

            # Update the kv cache with the new keys and values.
            current_kvcache = outputs.past_key_values
            current_kcompressed_cache = outputs.k_compressed_cache

            new_tokens_full = torch.full((initial_batch_size, 1), eos_token_id,
                                        dtype=next_tokens.dtype, device=device)
            new_tokens_full[cur_to_orig] = next_tokens

            # Append the token to each sequence.
            generated = torch.cat([generated, new_tokens_full], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.size(0), 1), device=device)], dim=1)


            # Update finished flags for the active sequences.
            if isinstance(generation_config.eos_token_id, list):
                finished[cur_to_orig] |= torch.isin(next_tokens.squeeze(1), eos_token_ids)
            else:
                finished[cur_to_orig] |= (next_tokens.squeeze(1) == eos_token_id)
            # finished[cur_to_orig] |= (next_tokens.squeeze(1) == eos_token_id)

            # If all sequences are finished, break.
            if finished.all():
                break

            # Determine which sequences in the current batch (cache) are still active.
            current_finished = finished[cur_to_orig]
            active_local = ~current_finished
            if active_local.sum().item() < cur_to_orig.shape[0]:
                active_indices_local = torch.nonzero(active_local, as_tuple=False).squeeze(-1)
                # Update the kv cache using indices relative to the current cache.
                print("active_local_batches", active_indices_local, "kvlen:", attention_mask.shape[1], flush=True)
                current_kvcache.batch_select_indices(active_indices_local)
                if self.config.seerattn_implementation != "oracle_sparse":
                    current_kcompressed_cache.batch_select_indices(active_indices_local)
                if attention_mask is not None:
                    attention_mask = attention_mask[active_indices_local]

                cur_to_orig = cur_to_orig[active_indices_local]

            # Prepare the next input tokens using the updated mapping.
            cur_input = generated[cur_to_orig, -1:].clone()

        # Pad finished sequences back to original batch size if needed
        return generated, sparsitys_info_list


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, load_gate=True, *model_args, **kwargs):
        # Call the original method first
        if load_gate:
            config = SeerAttnQwen3Config.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            base_model = config.base_model
            for key in list(kwargs.keys()):
                if hasattr(config, key) and key != "torch_dtype":
                    setattr(config, key, kwargs.pop(key))
            model = super().from_pretrained(base_model, config=config, *model_args, **kwargs)

            if os.path.exists(pretrained_model_name_or_path):
                gate_weights = torch.load(os.path.join(pretrained_model_name_or_path, "attn_gate_weights.pth"))
            else:
                try: 
                    gate_weights = torch.load(
                        hf_hub_download(repo_id=pretrained_model_name_or_path, filename="attn_gate_weights.pth")
                    )
                except:
                    raise ValueError("Could not load the attention gate weights.")
                    
            model.load_state_dict(gate_weights, strict=False)
            print("Attention gate weights loaded successfully.")
        else:
            model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
    
        return model