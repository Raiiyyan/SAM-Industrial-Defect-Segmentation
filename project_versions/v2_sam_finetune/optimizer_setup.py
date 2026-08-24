import torch
from model import IndustrialSAM
from model_setup import _ADAPTER_PREFIX

def build_optimizer(model, lr=1e-4, weight_decay=0.01):
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)