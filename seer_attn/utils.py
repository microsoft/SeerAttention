import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from transformers.utils import ModelOutput


@dataclass
class BaseModelOutputWithPastAndCache(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    sparsitys_info: Optional[Tuple[Tuple[int, int], ...]] = None
    k_compressed_cache: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class CausalLMOutputWithPastAndCache(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    sparsitys_info: Optional[Tuple[Tuple[int, int], ...]] = None
    k_compressed_cache: Optional[Tuple[torch.FloatTensor, ...]] = None



@dataclass
class BaseModelOutputWithPastAndSeer(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    mask_gate_predictions: Optional[Tuple[torch.FloatTensor, ...]] = None
    mask_ground_truths: Optional[Tuple[torch.FloatTensor, ...]] = None
    mask_loss: torch.FloatTensor = None


@dataclass
class CausalLMOutputWithPastAndSeer(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    mask_gate_predictions: Optional[Tuple[torch.FloatTensor, ...]] = None
    mask_ground_truths: Optional[Tuple[torch.FloatTensor, ...]] = None
    mask_loss: torch.FloatTensor = None


@dataclass
class BaseModelOutputWithPastAndMask(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    pooling_gt: Optional[Tuple[torch.FloatTensor, ...]] = None
    predict_mask: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class CausalLMOutputWithPastAndMask(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    pooling_gt: Optional[Tuple[torch.FloatTensor, ...]] = None
    predict_mask: Optional[Tuple[torch.FloatTensor, ...]] = None