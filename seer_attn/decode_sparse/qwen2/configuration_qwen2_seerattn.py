# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Qwen2 model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging


logger = logging.get_logger(__name__)


class SeerAttnQwen2Config(PretrainedConfig):

    model_type = "qwen2"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Qwen2`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        fused_norm=False,
        use_flash_rope=False,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        seerattn_sparsity_method='threshold', #[threshold or token_budget]
        seerattn_token_budget=4096,
        seerattn_threshold=0.0,
        seerattn_use_rope=True,
        seerattn_use_qk_norm=False,
        seerattn_k_seq_pooling_type='Kmaxminavg',
        seerattn_q_head_pooling_type='Qproj', # [Qproj, Qavgproj, Qavg]
        seerattn_loss_slice_ratio=0.0,
        seerattn_gate_block_size=64, # gate block size for seerattn, [32, 64, 128]
        seerattn_gate_hidden_size=128, # gate hidden size for seerattn
        seerattn_implementation="seer_sparse", # [seer_sparse, seer_dense, oracle_sparse]
        seerattn_start_layer=0,  # the first layer to use seerattn, inclusive
        seerattn_output_sparsity=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window  # we check `use_sliding_window` in the modeling code
        self.max_window_layers = max_window_layers

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        self.fused_norm = fused_norm
        self.use_flash_rope = use_flash_rope
        
        
        self.seerattn_sparsity_method = seerattn_sparsity_method
        self.seerattn_token_budget = seerattn_token_budget
        self.seerattn_threshold = seerattn_threshold
        self.seerattn_k_seq_pooling_type = seerattn_k_seq_pooling_type  # Kmaxminavg
        self.seerattn_q_head_pooling_type = seerattn_q_head_pooling_type
        self.seerattn_loss_slice_ratio = seerattn_loss_slice_ratio
        self.seerattn_gate_hidden_size = seerattn_gate_hidden_size    
        self.seerattn_gate_block_size = seerattn_gate_block_size      
        
        self.seerattn_implementation = seerattn_implementation
        self.seerattn_start_layer = seerattn_start_layer
        self.seerattn_use_rope = seerattn_use_rope
        self.seerattn_use_qk_norm = seerattn_use_qk_norm
        self.seerattn_output_sparsity = seerattn_output_sparsity
        assert self.seerattn_q_head_pooling_type in ['Qproj', 'Qavgproj', 'Qavg']
        assert self.seerattn_implementation in ['seer_sparse', 'seer_dense', 'oracle_sparse']
        assert self.seerattn_sparsity_method in ['threshold', 'nz_ratio']
        assert self.seerattn_gate_hidden_size in [64, 128, 256]
        assert self.seerattn_gate_block_size in [16, 32, 64]

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )