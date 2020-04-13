import torch
import numpy as np

from common.config import Config
from common.env_wrapper import EnvWrapper

def flatten(x):
    return torch.cat(x)

def flatten_dicts(x):
    return {k: flatten([s[k] for s in x]) for k in x[0].keys()}


#  n-steps x actions x envs -> actions x n-steps*envs
def flatten_lists(x):
    return [torch.cat([s[a] for s in x]) for a in range(len(x[0]))]

