import torch
import numpy as np

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _sample_int_layer_wise(nbatch, high, low):
    assert(high.dim()==1 and low.dim()==1)
    ndim = len(high)
    out_list = []
    for d in range(ndim):
        out_list.append( np.random.randint(low[d], high[d]+1, (nbatch,1 ) ) )
    return np.concatenate(out_list, axis=1)

def _sample_layer_wise(nbatch, high, low):
    assert(high.dim()==1 and low.dim()==1)
    nsample = len(high)
    base = torch.rand( nbatch, nsample )
    return base*(high - low) + low

def _transform(input_tensor, mapping):
    if input_tensor.dim()==1:
        input_tensor = input_tensor.unsqueeze(-1)
    return torch.gather(mapping, 1, input_tensor)

def _to_multi_hot(index_tensor, max_dim, device): # number-to-onehot or numbers-to-multihot
    if type(index_tensor)==np.ndarray:
        index_tensor = torch.from_numpy(index_tensor)
    if len(index_tensor.shape)==1:
        out = (index_tensor.unsqueeze(1) == torch.arange(max_dim).reshape(1, max_dim).to(device))
    else:
        out = (index_tensor == torch.arange(max_dim).reshape(1, max_dim).to(device))
    return out

def batch_bin_encode(bin_tensor, device):
    dim = len(bin_tensor.shape)
    feat_dim = bin_tensor.shape[-1]
    bias = 0
    unit = 50
    if dim==2:
        NB = bin_tensor.shape[0]
        output = [0]*NB
        num_iter = feat_dim//unit + 1
        for i in range(num_iter):
            ed = min(feat_dim, bias + unit)
            out = batch_bin_encode_64( bin_tensor[:, bias:ed], device)
            out_list = out.tolist()
            output = [output[j]*pow(2,unit) + val for j, val in enumerate(out_list)]
            bias += unit
            if ed==feat_dim:
                break
        return output

    elif dim==1:
        output = 0
        num_iter = feat_dim//unit + 1
        for i in range(num_iter):
            ed = min(feat_dim, bias + unit)
            out = batch_bin_encode_64( bin_tensor[bias:ed], device)
            output = output*pow(2,unit)+out
            bias += unit
            if ed==feat_dim:
                break
        return output

    else:
        print('Input type error!')
        print('input_type=')
        print(type(bin_tensor))
        print('input_shape=', bin_tensor.shape)
        assert(False)

def batch_bin_encode_64(bin_tensor, device):
    # bin_tensor: Nbatch x dim
    if type(bin_tensor)==torch.Tensor:
        if bin_tensor.dim()==2:
            return torch.mv(bin_tensor.type(torch.long), torch.from_numpy(1 << np.arange(bin_tensor.shape[-1])).to(device))
        else:
            return torch.dot(bin_tensor.type(torch.long), torch.from_numpy(1 << np.arange(bin_tensor.shape[-1])).to(device)).item()
    elif type(bin_tensor)==np.ndarray:
        return bin_tensor.dot(1 << np.arange(bin_tensor.shape[-1]))
    else:
        print('Input type error!')
        print('input_type=')
        print(type(bin_tensor))
        print('input_shape=', bin_tensor.shape)
        assert(False)
