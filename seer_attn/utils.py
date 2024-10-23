import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from transformers.utils import ModelOutput

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