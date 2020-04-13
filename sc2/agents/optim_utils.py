import torch.nn as nn

def ExponentialLR(optimizer, init_lr, epoch, gamma, decay_steps):
    '''Exponential learning rate scheduler.'''
    lr = init_lr * gamma**(epoch / decay_steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def anneal(t, t0, t1, v0, v1):
    scale = max(0.0, min(1.0, 1.0 - (t - t0) / (t1 - t0)))  # scale : 1.0 --> 0.0
    return v1 + scale * (v0 - v1)  # coef : v0 --> v1

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def init_gru(module, weight_init, bias_init, gain=1):
    weight_init(module.weight_ih_l0.data, gain=gain)
    weight_init(module.weight_hh_l0.data, gain=gain)
    bias_init(module.bias_ih_l0.data)
    bias_init(module.bias_hh_l0.data)
    return module

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
    elif isinstance(m, nn.Linear):
        init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
    elif isinstance(m, nn.GRU):
        init_gru(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))