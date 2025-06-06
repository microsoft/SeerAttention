from typing import Callable, List, Optional, Tuple, Union, Any

import torch
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    add_start_docstrings,
    logging,
)
from seer_attn.qwen3.configuration_qwen3_seerattn import SeerAttnQwen3Config
from seer_attn.attn_gate import ATTNGATE_CLASSES, MultiHeadLinear, HeadPoolingLinear
from flash_attn.layers.rotary import apply_rotary_emb_func
# from seer_attn.kernels.attn_pooling_qkv_varlen import attn_pooling_qkv_varlen as attn_pooling
from seer_attn.kernels.attn_pooling_qkv_varlen_1d import attn_pooling_qkv_varlen as attn_pooling
import copy, math, os
from seer_attn.utils import BaseModelOutputWithPastAndGateloss, CausalLMOutputWithPastAndGateloss
from flash_attn.bert_padding import unpad_input, pad_input
from seer_attn.modules.common import apply_rotary_pos_emb, get_indice, RMSNorm


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Qwen3Config"


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


class SeerAttnQwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: SeerAttnQwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        print("headim", self.head_dim)
        self.num_heads = config.num_attention_heads
        print("num_heads", self.num_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_key_value_heads = config.num_key_value_heads
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
        self.attn_gate = ATTNGATE_CLASSES[config.seerattn_k_seq_pooling_type](
            config.seerattn_gate_block_size, 
            self.head_dim, 
            config.seerattn_gate_hidden_size,
            num_k_head=config.num_key_value_heads, 
            num_q_head=config.num_attention_heads,
            q_head_pooling_type=config.seerattn_q_head_pooling_type,
            use_qk_norm=config.seerattn_use_qk_norm,
        )

        self.loss_fct = torch.nn.KLDivLoss()
        self.headpooling_type = config.seerattn_q_head_pooling_type
        self.threshold = config.seerattn_training_threshold
        self.loss_slice_ratio = config.seerattn_loss_slice_ratio
        self.block_slice_mode = config.seerattn_block_slice_mode

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  
        position_embeddings_gate_q: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  
        block_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, ## block position embeddings
        block_attention_mask: Optional[torch.Tensor] = None, ## block attention mask
        unpadded_lengths: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        input_shape = hidden_states.shape[:-1]
        block_size = self.config.seerattn_gate_block_size

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        cu_seqlens, max_seqlen, _  = unpadded_lengths

        q = q.view(*q.shape[:-1], self.num_heads, self.head_dim)
        k = k.view(*k.shape[:-1], self.num_key_value_heads, self.head_dim)
        v = v.view(*v.shape[:-1], self.num_key_value_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)

        q_unrope, k_unrope = q, k

        predict_mask = self.attn_gate(
            q_unrope,
            k_unrope,   
            unpadded_lengths,
            block_attention_mask,
            position_embeddings_gate_q,
            block_position_embeddings,
            self.block_slice_mode,
        )


        cos, sin = position_embeddings

        q, k = apply_rotary_pos_emb(q.unsqueeze(0), k.unsqueeze(0), cos, sin, unsqueeze_dim=2) # unsqueeze_dim=2 because of shape
        q, k = q.squeeze(0), k.squeeze(0)

        attn_output, mask1d_ground_truth = attn_pooling(
            q,
            k,
            v,
            cu_seqlens,
            max_seqlen,
            1.0 / math.sqrt(self.head_dim),
            self.config.seerattn_gate_block_size,      
        ) 

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        gate_loss = 0.0

        bsz = cu_seqlens.shape[0] - 1
        mask1d_ground_truth = F.max_pool3d(mask1d_ground_truth, kernel_size=[self.num_key_value_groups, 1, 1], stride=[self.num_key_value_groups, 1, 1])
        mask1d_ground_truth.masked_fill_(~block_attention_mask, 0.0)
        sum = torch.sum(mask1d_ground_truth, dim=-1, keepdim=True)
        mask1d_ground_truth.div_(sum + 1e-9)
        for i in range(bsz):
            seq_len = cu_seqlens[i + 1] - cu_seqlens[i]
            start_index = int(seq_len * self.loss_slice_ratio)
            if self.block_slice_mode:
                num_block_slice = int(math.floor(seq_len / block_size))
                predict_mask_i = F.log_softmax(predict_mask[i, :, 0:num_block_slice], dim=-1)
                idx = torch.arange(num_block_slice, device=mask1d_ground_truth.device)
                idx = idx * block_size + (block_size - 1)
                mask1d_ground_truth_i = mask1d_ground_truth[i, :, idx]
                gate_loss += self.loss_fct(predict_mask_i, mask1d_ground_truth_i) 
            else:
                predict_mask_i = F.log_softmax(predict_mask[i, :, start_index:seq_len], dim=-1).to(torch.float32)
                mask1d_ground_truth_i = mask1d_ground_truth[i, :, start_index:seq_len].to(torch.float32)
                gate_loss += self.loss_fct(predict_mask_i, mask1d_ground_truth_i) 
        gate_loss = gate_loss / bsz

        
        # for i in range(bsz):
        #     seq_len = cu_seqlens[i + 1] - cu_seqlens[i]
        #     start_index = int(seq_len * self.loss_slice_ratio)
        #     mask1d_ground_truth_i = mask1d_ground_truth[i:i+1, :, start_index:seq_len]
        #     mask1d_ground_truth_i = F.max_pool3d(mask1d_ground_truth_i, kernel_size=[self.num_key_value_groups, 1, 1], stride=[self.num_key_value_groups, 1, 1])
        #     block_attention_mask_i = block_attention_mask[start_index:seq_len]
        #     mask1d_ground_truth_i.masked_fill_(~block_attention_mask_i, 0.0)
        #     for kv in range(self.num_key_value_heads):
        #         predict_mask_i = F.log_softmax(predict_mask[i, kv, start_index:seq_len], dim=-1)
        #         mask1d_ground_truth_i_kv = mask1d_ground_truth_i[0, kv]
        #         gate_loss += self.loss_fct(predict_mask_i, mask1d_ground_truth_i_kv) 
        # gate_loss = gate_loss / bsz

        return attn_output, None, past_key_value, gate_loss


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


class SeerAttnQwen3DecoderLayer(nn.Module):
    def __init__(self, config: SeerAttnQwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = SeerAttnQwen3Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        position_embeddings_gate_q: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        block_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        block_attention_mask: Optional[torch.Tensor] = None,
        unpadded_lengths: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
            block_position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*): downsampled positional embeddings
            block_attention_mask (`torch.FloatTensor`, *optional*): downsampled attention mask: downsampled attention mask
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention

        hidden_states, self_attn_weights, present_key_value, gate_loss = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            position_embeddings_gate_q=position_embeddings_gate_q,
            block_position_embeddings=block_position_embeddings,
            block_attention_mask=block_attention_mask,
            unpadded_lengths=unpadded_lengths,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,) 

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        outputs += (gate_loss, )


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
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


QWEN2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen3Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen3 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)

class SeerAttnQwen3PreTrainedModel(PreTrainedModel):
    config_class = SeerAttnQwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SeerAttnQwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

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
        elif isinstance(module, MultiHeadLinear) or isinstance(module, HeadPoolingLinear):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, Qwen3RMSNorm) or isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)


class SeerAttnQwen3Model(SeerAttnQwen3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`SeerAttnQwen3DecoderLayer`]

    Args:
        config: SeerAttnQwen3Config
    """

    def __init__(self, config: SeerAttnQwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [SeerAttnQwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        block_config = copy.deepcopy(config)
        block_config.hidden_size = config.seerattn_gate_hidden_size * config.num_attention_heads
        self.rotary_emb_gate_q = Qwen3RotaryEmbedding(config=block_config)
        self.block_rotary_emb = Qwen3RotaryEmbedding(config=block_config)
        self.gradient_checkpointing = False
        self.config = config

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
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        unpadded_lengths: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndGateloss]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if (
            use_cache and not isinstance(past_key_values, Cache) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )


        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[0], device=inputs_embeds.device # change to 0 for now!!!!!
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)


        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        cu_seqlens, max_seqlen, _ = unpadded_lengths

        block_causal_mask = self._gen_block_mask(
            max_seqlen, cu_seqlens, 
        )

        if self.config.seerattn_use_rope:
            max_seqlen_round = math.ceil(max_seqlen / self.config.seerattn_gate_block_size) * self.config.seerattn_gate_block_size
            block_position_ids = torch.arange(0, max_seqlen_round, self.config.seerattn_gate_block_size, device=position_ids.device).unsqueeze(0)
            block_position_embeddings = self.block_rotary_emb(hidden_states, block_position_ids) # downsampled position embeddings

            position_ids_q = torch.arange(0, max_seqlen, device=position_ids.device).unsqueeze(0)
            position_embeddings_gate_q = self.rotary_emb_gate_q(hidden_states, position_ids_q) # downsampled position embeddings
        else:
            block_position_embeddings = None
            position_embeddings_gate_q = None


        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        total_gate_loss = 0.0

        # print(inputs_embeds.shape)
        
        for decoder_layer in self.layers:
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
                    position_embeddings_gate_q,
                    block_position_embeddings,
                    block_causal_mask,
                    unpadded_lengths,
                    use_reentrant=False,
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
                    position_embeddings_gate_q=position_embeddings_gate_q,
                    block_position_embeddings=block_position_embeddings,
                    block_attention_mask=block_causal_mask,
                    unpadded_lengths=unpadded_lengths,
                )

            hidden_states = layer_outputs[0]

            output_count = 1

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
                output_count += 1

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                output_count += 1

            gate_loss = layer_outputs[output_count] 
            total_gate_loss += gate_loss

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPastAndGateloss(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            gate_loss=total_gate_loss,
        )


    def _gen_block_mask(
        self,
        max_seqlen,
        cu_seqlens: Optional[torch.Tensor] = None,
    ):
        block_size = self.config.seerattn_gate_block_size
        device = cu_seqlens.device
        num_block = math.ceil(max_seqlen / block_size)  

        attention_mask = torch.ones((num_block, num_block), dtype=torch.bool, device=device)
        attention_mask = torch.tril(attention_mask, diagonal=-1)
        attention_mask = torch.repeat_interleave(attention_mask, block_size, dim=0)
        attention_mask = attention_mask[0:max_seqlen]
        attention_mask = attention_mask
        return attention_mask



    # def _gen_block_mask(
    #     self,
    #     max_seqlen,
    #     cu_seqlens: Optional[torch.Tensor] = None,
    # ):
    #     block_size = self.config.seerattn_gate_block_size
    #     bsz = cu_seqlens.shape[0] - 1
    #     # Compute per-batch sequence lengths and global max
    #     seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]  # shape [bsz]
    #     num_blocks = (max_seqlen + block_size - 1) // block_size  # ceil(max_seq/block_size)
    #     device = cu_seqlens.device

    #     # Compute for each token the block index it belongs to
    #     pos = torch.arange(max_seqlen, device=device)           # [0, 1, ..., max_seq - 1]
    #     token_block = pos // block_size                        # block index for each token, shape [max_seq]
    #     block_indices = torch.arange(num_blocks, device=device)  # shape [num_blocks]

    #     # For a given token at position i, allowed block indices are all j <= token_block[i]
    #     if self.config.seerattn_ignore_last_block:
    #         base_mask = token_block.unsqueeze(1) > block_indices.unsqueeze(0)  # shape [max_seq, num_blocks]
    #     else:
    #         base_mask = token_block.unsqueeze(1) >= block_indices.unsqueeze(0)  # shape [max_seq, num_blocks]

    #     # Create a valid-token mask for each batch: valid if token position < seq_len[b]
    #     valid_tokens = pos.unsqueeze(0) < seq_lens.unsqueeze(1)  # shape [bsz, max_seq]

    #     # Expand base_mask to each batch and mask out invalid positions
    #     mask = base_mask.unsqueeze(0).expand(bsz, -1, -1)  # shape [bsz, max_seq, num_blocks]
    #     mask = mask & valid_tokens.unsqueeze(-1)            # mask invalid tokens out

    #     # Add extra singleton dimension to match shape [bsz, 1, seq, num_blocks]
    #     mask = mask.unsqueeze(1)
    #     return mask




class SeerAttnQwen3ForCausalLM(SeerAttnQwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    # _tp_plan = {"lm_head": "colwise_rep"}
    # _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = SeerAttnQwen3Model(config)
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
    
    def compute_loss(self, hidden_states, labels, token_losses=False):
        logits = self.lm_head(hidden_states)
        if len(logits.shape) > 2:
            logits = logits.transpose(-1, -2)
        # For num-valid-token-scaled loss, we sum up here and later reweight in `compute_loss` in the trainer
        return F.cross_entropy(
            logits, labels,
            ignore_index=-100,
            reduction=("sum" if getattr(self, "token_scaled_loss", False) else "mean")
        )

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
        seq_lengths: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        shifted_labels: Optional[torch.LongTensor] = None,
        seq_parallel_group: Optional[Any] = None,
    ) -> Union[Tuple, CausalLMOutputWithPastAndGateloss]:

        # print("input_ids", input_ids, "seq_lengths", seq_lengths)
        input_ids = input_ids.squeeze()
        seq_lengths = seq_lengths.squeeze()

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if seq_lengths is not None:
            if inputs_embeds is not None:
                assert len(inputs_embeds.shape) == 2, "inputs_embeds should be a 2D tensor with `seq_lengths`"
                # assert inputs_embeds.size(0) == seq_lengths.sum(), "inputs_embeds and seq_lengths should have the same batch size"
            else:
                assert len(input_ids.shape) == 1, "input_ids should be a 1D tensor with `seq_lengths`"
                # assert input_ids.size(0) == seq_lengths.sum(), "input_ids and seq_lengths should have the same batch size"

            assert attention_mask is None or attention_mask.all().item(), "attention_mask should be None or all ones for `seq_lengths`"
            assert not use_cache, "use_cache is not supported with `seq_lengths`"
            # print("seq_lengths", seq_lengths)
            # print("seq_lengths.shape", seq_lengths.shape)

            cu_seqlens = F.pad(torch.cumsum(torch.atleast_1d(seq_lengths), dim=0, dtype=torch.torch.int32), (1, 0))
            max_seqlen = seq_lengths.max().item()

            unpad_indices = get_indice(max_seqlen, cu_seqlens)

            unpadded_lengths = (cu_seqlens, max_seqlen, unpad_indices)
        elif (
            ((attention_mask is not None) and (not attention_mask.all().item()))
            and not use_cache
        ):
            if inputs_embeds is not None:
                bsz = inputs_embeds.size(0)
                inputs_embeds, unpad_indices, cu_seqlens, max_seqlen = unpad_input(inputs_embeds, attention_mask)
            else:
                bsz = input_ids.size(0)
                input_ids, unpad_indices, cu_seqlens, max_seqlen = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids = input_ids.squeeze(-1)
            unpadded_lengths = (cu_seqlens, max_seqlen, unpad_indices)
        else:
            unpadded_lengths = None

        ## hacky way to do batching
        # input_ids = input_ids.unsqueeze(0)

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
            unpadded_lengths=unpadded_lengths,
        )

        hidden_states = outputs[0]
        # hidden_states = hidden_states.squeeze(0)


        if seq_lengths is None and unpadded_lengths is not None:
            hidden_states = pad_input(hidden_states, unpad_indices, bsz, max_seqlen)

        logits = None
        loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastAndGateloss(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            gate_loss=outputs.gate_loss,
        )