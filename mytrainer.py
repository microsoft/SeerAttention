import functools
import inspect

from collections import defaultdict
from typing import Any, Dict, List, Optional, Union
import torch
import random
import numpy as np

from torch import nn
from transformers import Trainer



class AttnGateTrainer(Trainer):
    def __init__(
            self, 
            orig_weight_training=False, 
            gate_loss_scale=1.0, 
            fix_mask_predictor=False, 
            *args, 
            **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.gate_loss_scale = gate_loss_scale
        self.orig_weight_training = orig_weight_training
        self.fix_mask_predictor = fix_mask_predictor

    def compute_loss(self, model, inputs, **kwargs):
        step = self.state.global_step
        outputs = model(**inputs)

        # predict_mask = outputs.get("predict_mask")
        # pooling_gt = outputs.get("pooling_gt")
        original_loss = outputs.get("loss")

        mask_loss = outputs.get("mask_loss")
        
        del outputs
            
        if self.orig_weight_training:
            tok_loss = original_loss + self.gate_loss_scale * mask_loss
        else:
            tok_loss = self.gate_loss_scale * mask_loss
        
        return tok_loss
    
     