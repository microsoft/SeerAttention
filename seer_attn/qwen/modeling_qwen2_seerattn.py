# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

# Modified by Yizhao Gao from huggingface qwen implementation
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
# from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
# from transformers.processing_utils import Unpack
from transformers.utils import (
    # LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg
from seer_attn.qwen.configuration_qwen2_seerattn import SeerAttnQwen2Config
from seer_attn.utils import BaseModelOutputWithPastAndSeer, CausalLMOutputWithPastAndSeer
from seer_attn.attn_gate import ATTNGATE_CLASSES, MultiHeadLinear
from seer_attn.modules.common import (
    repeat_kv,
    apply_rotary_pos_emb,
    get_sparse_attn_mask_from_nz_ratio,
    get_sparse_attn_mask_from_threshold
)
from einops import rearrange
import copy, math, os
from seer_attn.modules.attention_distill import attention_distill_forward
from seer_attn.modules.attention_forward import sparse_flash_attention_forward
from huggingface_hub import hf_hub_download
from seer_attn.modules.layernorm import RMSNorm
from flash_attn.layers.rotary import apply_rotary_emb_func

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "meta-qwen2/Qwen2-2-7b-hf"
_CONFIG_FOR_DOC = "Qwen2Config"


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
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


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: SeerAttnQwen2Config, device=None):
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


class Qwen2MLP(nn.Module):
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


class SeerAttnQwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: SeerAttnQwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        self.attn_gate = ATTNGATE_CLASSES[config.seerattn_gate_type](
            config.seerattn_gate_block_size, 
            self.head_dim, 
            config.seerattn_gate_hidden_size,
            num_k_head=config.num_key_value_heads, 
            num_q_head=config.num_attention_heads,
            force_double=config.seerattn_gate_force_double,
            use_flash_rope=config.use_flash_rope,
        )

        self.mask_loss_func = torch.nn.KLDivLoss()
        self.profile_file = os.environ.get("PROFILE_FILE", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        block_position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
        block_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        q_len = hidden_states.shape[1]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = rearrange(query_states, '... (h d) -> ... h d', d=self.head_dim)
        key_states = rearrange(key_states, '... (h d) -> ... h d', d=self.head_dim)
        value_states = rearrange(value_states, '... (h d) -> ... h d', d=self.head_dim)
        
            
        attn_gate_output = self.attn_gate(
            query_states, 
            key_states, 
            block_attention_mask, 
            block_position_embeddings, 
            use_softmax=not self.training and self.config.seerattn_sparsity_method == "threshold",
        )
    
        cos, sin = position_embeddings
        if self.config.use_flash_rope:
            query_states = apply_rotary_emb_func(query_states, cos, sin, False, True, cu_seqlens=None, max_seqlen=q_len)
            key_states = apply_rotary_emb_func(key_states, cos, sin, False, True, cu_seqlens=None, max_seqlen=q_len)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states.flatten(-2, -1), value_states.flatten(-2, -1), self.layer_idx, cache_kwargs)
            key_states = rearrange(key_states, '... (h d) -> ... h d', d=self.head_dim)
            value_states = rearrange(value_states, '... (h d) -> ... h d', d=self.head_dim)

        if self.training:
            # get the block (pooled) mask ground truth
            attn_output, ground_truth_mask = attention_distill_forward(
                query_states,
                key_states,
                value_states,
                softmax_scale=self.scaling,
                block_size=self.config.seerattn_gate_block_size,
                num_key_value_groups=self.num_key_value_groups,      
            )
        else: ## inference
            attn_output = sparse_flash_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
                query_length=q_len,
                softmax_scale=self.scaling,
                attn_gate_score=attn_gate_output,
                sparsity_method=self.config.seerattn_sparsity_method,
                threshold=self.config.seerattn_threshold,
                nz_ratio=self.config.seerattn_nz_ratio,
                last_block_dense=self.config.seerattn_last_block_dense,
                block_size=self.config.seerattn_gate_block_size,
                num_key_value_groups=self.num_key_value_groups,
                profile_file=self.profile_file,
            )


        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        
        if self.training:
            # remove the first quarter of the data for training stability
            ground_truth_mask = ground_truth_mask[:, :, ground_truth_mask.shape[2]//4:].to(torch.float32)
            attn_gate_output = attn_gate_output[:, :, attn_gate_output.shape[2]//4:].to(torch.float32)
            attn_gate_output = F.log_softmax(attn_gate_output, dim=-1)
            mask_loss = self.mask_loss_func(attn_gate_output, ground_truth_mask)
        else:
            mask_loss = 0.0
            attn_gate_output = None
            ground_truth_mask = None

        # In SeerAttention, output_attentions also means output attn_gate_output and ground_truth_mask
        if not kwargs.get("output_attentions", False):
            attn_gate_output = None
            ground_truth_mask = None
        return attn_output, mask_loss, None, attn_gate_output, ground_truth_mask




class SeerAttnQwen2DecoderLayer(nn.Module):
    def __init__(self, config: SeerAttnQwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = SeerAttnQwen2Attention(config=config, layer_idx=layer_idx)
        self.fused_norm = config.fused_norm
        self.mlp = Qwen2MLP(config)
        if self.fused_norm:
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        block_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        block_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, seerattn_mask_loss, self_attn_weights, mask_gate_prediction, mask_ground_truth = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
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

        outputs = (hidden_states, seerattn_mask_loss)
        if output_attentions:
            outputs += (self_attn_weights, mask_gate_prediction, mask_ground_truth)

        return outputs


class SeerAttnQwen2PreTrainedModel(PreTrainedModel):
    config_class = SeerAttnQwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SeerAttnQwen2DecoderLayer"]
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
        elif isinstance(module, MultiHeadLinear):
            module.weight.data.normal_(mean=0.0, std=std)


class SeerAttnQwen2Model(SeerAttnQwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: SeerAttnQwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [SeerAttnQwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        block_config = copy.deepcopy(config)
        block_config.hidden_size = config.seerattn_gate_hidden_size * config.num_attention_heads
        self.block_rotary_emb = Qwen2RotaryEmbedding(config=block_config)
       
        self.gradient_checkpointing = False

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
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndSeer]:
        
        if attention_mask is not None:
            if not (attention_mask == 0).any().item():
                attention_mask = None

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        block_attention_mask = self._seerattn_update_causal_mask(
            attention_mask,
            inputs_embeds,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        #Added for seerattn
        block_position_ids = position_ids[:, 0::self.config.seerattn_gate_block_size] ## downsampled position ids
        block_position_embeddings = self.block_rotary_emb(hidden_states, block_position_ids) # downsampled position embeddings



        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_mask_gate_predictions = () if output_attentions else None
        all_mask_ground_truths = () if output_attentions else None

        # added for seerattn
        total_mask_loss = 0.0

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    block_position_embeddings,
                    block_attention_mask,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    block_position_embeddings=block_position_embeddings,
                    block_attention_mask=block_attention_mask,
                )

            hidden_states = layer_outputs[0]

            mask_loss = layer_outputs[1]
            total_mask_loss += mask_loss

            if output_attentions:
                all_self_attns += (layer_outputs[2],)
                all_mask_gate_predictions += (layer_outputs[3],)
                all_mask_ground_truths += (layer_outputs[4],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPastAndSeer(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            mask_gate_predictions=all_mask_gate_predictions,
            mask_ground_truths=all_mask_ground_truths,
            mask_loss=total_mask_loss,
        )
        return output if return_dict else output.to_tuple()

    def _seerattn_update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ):  

        batch_size, seqlen = inputs_embeds.shape[:2]
        if seqlen == 1:
            return None
        
        block_size = self.config.seerattn_gate_block_size
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device
        if attention_mask is not None:
            pad_length = seqlen - torch.sum(attention_mask, dim=-1)
            pad_blocks = torch.floor(pad_length / block_size)  # Shape: [batch_size]

            min_dtype = torch.finfo(dtype).min
            number_of_blocks = math.ceil(seqlen / block_size)
            gate_mask = torch.triu(
                torch.full((number_of_blocks, number_of_blocks), min_dtype, dtype=dtype, device=device),
                diagonal=1
            )
            # Expand to shape [batch_size, 1, number_of_blocks, number_of_blocks]
            gate_mask = gate_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, number_of_blocks, number_of_blocks)

            # Create block indices along block dimension
            block_indices = torch.arange(number_of_blocks, device=device)  # Shape [number_of_blocks]

            # Create masks for query (rows) and key (columns) per batch
            # pad_blocks has shape [batch_size]. Convert to int64.
            pad_blocks_int = pad_blocks.to(torch.int64)

            # For the query side: For each batch, block index < pad_blocks[i] should be masked.
            query_mask = block_indices.unsqueeze(0) < pad_blocks_int.unsqueeze(1)  # shape: [batch_size, number_of_blocks]
            # For the key side, the same logic applies.
            key_mask = query_mask.clone()  # shape: [batch_size, number_of_blocks]

            # Expand to match gate_mask dimensions.
            # For queries: shape [batch_size, 1, number_of_blocks, 1]
            query_mask = query_mask.unsqueeze(1).unsqueeze(-1)
            # For keys: shape [batch_size, 1, 1, number_of_blocks]
            key_mask = key_mask.unsqueeze(1).unsqueeze(2)

            # Combine the masks: positions where either the query or the key block is padded.
            combined_mask = query_mask.logical_or(key_mask)

            # Set these positions in gate_mask to min_dtype.
            gate_mask = gate_mask.masked_fill(combined_mask, min_dtype)
        else:
            min_dtype = torch.finfo(dtype).min
            number_of_blocks = math.ceil(seqlen / block_size)
            gate_mask = torch.triu(torch.full((number_of_blocks, number_of_blocks), min_dtype, dtype=dtype, device=device), diagonal=1)
            gate_mask = gate_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, number_of_blocks, number_of_blocks)

        return gate_mask

class SeerAttnQwen2ForCausalLM(SeerAttnQwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = SeerAttnQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
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

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 1,
        **kwargs,
        # **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPastAndSeer]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).
                hacky change: default to 1

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if not self.training and labels is not None: ## current self-distillation training does not require loss computation, re-enable if needed
            # loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
            valid_seq_len = input_ids.shape[-1] - 1
            valid_seq_len_slide_win = torch.sum(labels[:, 1:] >= 0).item()
            loss = 0.0
            for start_idx in range(0, valid_seq_len, 16384):
                end_idx = min(start_idx + 16384, valid_seq_len)
                shift_logits = self.lm_head(hidden_states[..., start_idx:end_idx, :]).float()
                shift_labels = labels[..., start_idx + 1:end_idx + 1].contiguous()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                shift_labels = shift_labels.to(shift_logits.device)
                loss += loss_fct(shift_logits, shift_labels)
            loss /= valid_seq_len_slide_win  


        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastAndSeer(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mask_gate_predictions=outputs.mask_gate_predictions,
            mask_ground_truths=outputs.mask_ground_truths,
            mask_loss=outputs.mask_loss,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0]:]
            elif (
                input_ids.shape[1] != cache_position.shape[0]
            ):  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

                # # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                # position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, load_gate=True, *model_args, **kwargs):
        # Call the original method first
        if load_gate:
            config = SeerAttnQwen2Config.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
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