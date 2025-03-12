import copy
import importlib.metadata
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import time
from collections import deque
import torch
from packaging import version

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import is_hqq_available, is_optimum_quanto_available, is_torchdynamo_compiling, logging
from transformers.cache_utils import Cache

class PoolingCache(Cache):
    """
    A cache that dynamically grows with pre-allocated fixed-size buffers for efficiency.
    """

    def __init__(self, num_layers: int, block_size: int, pooling_cache_size: int = 1024) -> None:
        super().__init__()
        self.k_pooling_cache: List[torch.Tensor] = [None] * num_layers  # Placeholder
        self.k_pooling_pos: List[int] = [0] * num_layers  # Current write position
        self.k_remainder_cache: List[torch.Tensor] = [None] * num_layers  # Placeholder
        self.k_remainder_pos: List[int] = [0] * num_layers  # Current write position
        self.block_size = block_size
        self.num_layers = num_layers
        self.pooling_cache_size = pooling_cache_size

    def init_pooling_buffer(self, layer_idx: int, template: torch.Tensor):
        """Initialize pooling buffer based on the first seen tensor's shape"""
        if self.k_pooling_cache[layer_idx] is None:
            batch, num_heads, _, dim = template.shape
            self.k_pooling_cache[layer_idx] = torch.zeros(
                batch, num_heads, self.pooling_cache_size, dim,
                dtype=template.dtype,
                device=template.device
            )

    def init_remainder_buffer(self, layer_idx: int, template: torch.Tensor):
        """Initialize remainder buffer based on the first seen tensor's shape"""
        if self.k_remainder_cache[layer_idx] is None:
            batch, num_heads, _, dim = template.shape
            self.k_remainder_cache[layer_idx] = torch.zeros(
                batch, num_heads, self.block_size, dim,
                dtype=template.dtype,
                device=template.device
            )

    def update_k_remainder_cache(
        self,
        k_: torch.Tensor,
        layer_idx: int,
        q_len: int,
        block_size: int,
    ) -> torch.Tensor:
        
        if q_len > 1:  # Prefill phase
            seq_len = k_.shape[-2]
            if seq_len % block_size == 0:
                # Take last block
                remainder = k_[..., -block_size:, :]
                remainder_len = block_size
            else:
                # Take remaining tokens
                remainder = k_[..., (seq_len // block_size)*block_size:, :]
                remainder_len = remainder.shape[-2]
            
            # Initialize buffer if needed
            self.init_remainder_buffer(layer_idx, k_)
            
            # Fill the buffer
            buffer = self.k_remainder_cache[layer_idx]
            buffer[..., :remainder_len, :] = remainder
            self.k_remainder_pos[layer_idx] = remainder_len

        else:  # Decode phase
            buffer = self.k_remainder_cache[layer_idx]
            pos = self.k_remainder_pos[layer_idx]
            
            if pos < self.block_size:
                # Insert into buffer
                buffer[..., pos:pos+1, :] = k_
                self.k_remainder_pos[layer_idx] += 1
            else:
                # Overwrite from position 0 (ring buffer)
                buffer[..., 0:1, :] = k_
                self.k_remainder_pos[layer_idx] = 1

        # Return valid portion of the buffer
        return self.k_remainder_cache[layer_idx][..., :self.k_remainder_pos[layer_idx], :]

    def update_k_pooling_cache(
        self,
        k: torch.Tensor,
        layer_idx: int,
        do_q_pooling: bool,
        prefill_update_flag: bool,
        decode_update_flag: bool,

    ) -> torch.Tensor:

        if do_q_pooling:  # Prefill phase
            if prefill_update_flag:
                # Initialize buffer if needed
                self.init_pooling_buffer(layer_idx, k)
                # Cache result
                buffer = self.k_pooling_cache[layer_idx]
                pos = self.k_pooling_pos[layer_idx]
                buffer[..., pos:pos+k.shape[2], :] = k
                self.k_pooling_pos[layer_idx] += k.shape[2]

            else:
                # Initialize buffer if needed
                self.init_pooling_buffer(layer_idx, k)
                # Cache only the integer part
                integer_part = k.shape[2] - 1
                buffer = self.k_pooling_cache[layer_idx]
                pos = self.k_pooling_pos[layer_idx]
                buffer[..., pos:pos+integer_part, :] = k[..., :-1, :]
                self.k_pooling_pos[layer_idx] += integer_part

        else:  # Decode phase, use k_remainder only
            if decode_update_flag:
                # Update cache
                buffer = self.k_pooling_cache[layer_idx]
                pos = self.k_pooling_pos[layer_idx]
                buffer[..., pos:pos+1, :] = k
                self.k_pooling_pos[layer_idx] += 1
                k = buffer[..., :self.k_pooling_pos[layer_idx], :]
            else:
                k = self.k_pooling_cache[layer_idx][..., :(self.k_pooling_pos[layer_idx] + 1), :]

        return k