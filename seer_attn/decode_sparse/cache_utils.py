import copy
import importlib.metadata
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import time
from collections import deque
import torch
import math

from transformers.cache_utils import Cache

class KCompressionCache(Cache):

    def __init__(self, num_layers: int, block_size: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.block_size = block_size
        # initialize caches for each layer
        self.k_compressed: Dict[int, Optional[torch.Tensor]] = {}
        self.k_remainder: Dict[int, Optional[torch.Tensor]] = {}
        for layer in range(num_layers):
            self.k_compressed[layer] = None  
            self.k_remainder[layer] = None  

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        # Return a tuple of (k_cache, k_remainder) for a given layer.
        return [(self.k_compressed[layer_idx], self.k_remainder[layer_idx])]

    def batch_select_indices(self, indices: torch.Tensor):
        for layer in range(self.num_layers):
            self.k_compressed[layer] = self.k_compressed[layer][indices, ...]
            self.k_remainder[layer] = self.k_remainder[layer][indices, ...]

    def get_k_remainder(self, layer_idx: int) -> torch.Tensor:
        return self.k_remainder[layer_idx]

    def update(
        self,
        layer_idx: int,
        k: Optional[torch.Tensor] = None,
        k_compressed: Optional[torch.Tensor] = None,
        k_remainder: Optional[torch.Tensor] = None, 
        is_decode: bool = False,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:

        if k_compressed is not None:
            if is_decode:
                self.k_compressed[layer_idx][:, -1:, :, :] = k_compressed
            else:
                self.k_compressed[layer_idx] = k_compressed
                bsz = k_compressed.shape[0]
                self.k_remainder[layer_idx] = torch.zeros(
                    [bsz, self.block_size, k_compressed.shape[2], k_remainder.shape[3]], device=k_compressed.device, dtype=k_compressed.dtype)
                if k_remainder is not None:        
                    self.k_remainder[layer_idx][:, :k_remainder.shape[1],] = k_remainder
                
                if layer_idx == 0:
                    self.remainder_len = k_remainder.shape[1] if k_remainder is not None else 0


        elif k is not None:
            self.k_remainder[layer_idx][:, self.remainder_len:self.remainder_len + 1, :, :] = k
            if layer_idx == 0:
                self.remainder_len += 1
                self.remainder_len %= self.block_size
            if self.remainder_len == 1:
                b, _, h, d = self.k_compressed[layer_idx].shape
                dtype, devcie = self.k_compressed[layer_idx].dtype, self.k_compressed[layer_idx].device
                self.k_compressed[layer_idx] = torch.cat(
                    [self.k_compressed[layer_idx], torch.zeros([b, 1, h, d], device=devcie, dtype=dtype)], dim=1)

        return self.k_compressed[layer_idx]