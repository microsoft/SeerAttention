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
    def __init__(self, orig_weight_training=False, gate_loss_scale=1.0, fix_mask_predictor=False, use_mse_loss=False,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        if use_mse_loss:
            self.loss_fct = torch.nn.MSELoss()
        else:
            self.loss_fct = torch.nn.SmoothL1Loss()
        self.gate_loss_scale = gate_loss_scale
        self.orig_weight_training = orig_weight_training
        self.fix_mask_predictor = fix_mask_predictor


    def compute_loss(self, model, inputs, return_outputs=False):
        step = self.state.global_step
        outputs = model(**inputs)

        predict_mask = outputs.get("predict_mask")
        pooling_gt = outputs.get("pooling_gt")
        original_loss = outputs.get("loss")

        if not return_outputs:
            del outputs

        mask_loss = 0
        if not self.fix_mask_predictor:
            for pm, gt in zip(predict_mask, pooling_gt):
                mask_loss += self.loss_fct(pm, gt)

          
        del predict_mask
        del pooling_gt
        
        
        if self.orig_weight_training:
            tok_loss = original_loss + self.gate_loss_scale * mask_loss
        else:
            tok_loss = self.gate_loss_scale * mask_loss
        
        return (tok_loss, outputs) if return_outputs else tok_loss
    
     