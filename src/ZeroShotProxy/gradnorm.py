'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

# Nota: Este podría mejorar con dataset real (Tal vez todos)
import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np

from utils.functions.model_training import eval_model
from utils.globals import CUDA

def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net

import torch.nn.functional as F
def cross_entropy(logit, target):
    # target must be one-hot format!!
    prob_logit = F.log_softmax(logit, dim=1)
    loss = -(target * prob_logit).sum(dim=1).mean()
    return loss

def compute_nas_score(model, data_loader):

    model.train()
    model.requires_grad_(True)

    model.zero_grad()
    model = model.to(CUDA)

    network_weight_gaussian_init(model)
        
    for i, (images, masks) in enumerate(data_loader.train):
            images = images.to(CUDA)
            masks = masks.to(CUDA)
            break
        
    output = model(images)
    loss = eval_model(
                    scores=output,
                    target=masks,
                    metrics=["iou"],
                    clone=False
                )[0]
    loss.backward()
    norm2_sum = 0
    with torch.no_grad():
        for p in model.parameters():
            if hasattr(p, 'grad') and p.grad is not None:
                norm2_sum += torch.norm(p.grad) ** 2

    grad_norm = float(torch.sqrt(norm2_sum))

    return grad_norm