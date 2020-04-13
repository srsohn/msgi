import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.optim_utils import init
from common.sc2_utils import masked_softmax


class FullyConv(nn.Module):
    def __init__(self, config, device, num_envs):
        super(FullyConv, self).__init__()

        # model config
        sp_hdim, fl_hdim, act_hdim = config.sp_hdim, config.fl_hdim, config.act_hdim
        option_dims = config.option_dims  # num of actions

        self.device = device
        self.num_envs, self.sz = num_envs, config.sz
        self.policy_dims = config.policy_dims()
        self.screen_dims = config.screen_dims()

        pol_dims = [pd[0] if pd[1] == False else 2 for pd in config.policy_dims()]
        self.flat_dims = [pol_dims[0], 1] + pol_dims
        self.embed_dim_fn = config.embed_dim_fn

        # preprocessing
        preproc_list = []
        for i, d in enumerate(self.screen_dims):
            preproc_list.append(nn.Conv2d(d, self.embed_dim_fn(d), kernel_size=1).to(device))
        self.preproc_convs = nn.ModuleList(preproc_list)

        # spatial operations
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, sp_hdim, kernel_size=3, padding=1),
            nn.ReLU()
        ).to(device)
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4).to(device)

        # flat operations
        self.flat = nn.Sequential(
            nn.Linear(fl_hdim, fl_hdim),
            nn.ReLU(),
            nn.Linear(fl_hdim, fl_hdim),
            nn.ReLU()
        ).to(device)

        self.fc = nn.Linear(14112 + fl_hdim, act_hdim).to(device)
        self.vf = nn.Linear(act_hdim, 1).to(device)
        self.policy = nn.Linear(act_hdim, option_dims).to(device)

    def _preprocess(self, sp_input):
        # spatial embedding
        nbatch = sp_input.size(0)
        sp_types = ['onehot', 'onehot']
        sp_list = torch.chunk(sp_input, len(self.screen_dims), dim=1)

        output_list = []
        for i, d in enumerate(self.screen_dims):
            if d > 1:
                if sp_types[i] == 'onehot':
                    if nbatch > 1:
                        onehot = torch.zeros((nbatch, d, self.sz, self.sz)).to(self.device)
                        onehot = onehot.scatter_(1, sp_list[i].long(), 1)
                    else:
                        onehot = torch.zeros((d, self.sz, self.sz)).to(self.device)
                        onehot = onehot.scatter_(0, sp_list[i].squeeze(0).long(), 1)
                        onehot = onehot.unsqueeze(0)
                    out = self.preproc_convs[i](onehot)
                    output_list.append(out)
                elif sp_types[i] == 'norm':
                    output_list.append(sp_list[i] / d)
            else:
                output_list.append(sp_list[i])
        sp_emb = torch.cat(output_list, dim=1)
        return sp_emb

    def act(self, states, update=False):
        spatials, comps, eligs, masks, steps = states

        # spatials
        sp_emb = torch.log10(spatials + 1) - 1
        sp = self.conv(sp_emb)

        # flat
        flats = torch.cat((comps, eligs, masks, steps), dim=1)
        fl = self.flat(flats)

        # hidden
        sp_max = self.maxpool(sp)
        sp_flat = sp_max.view(sp_max.size(0), -1)
        fc_in = torch.cat([sp_flat, fl], dim=1)
        fc = self.fc(fc_in)

        # value
        v = self.vf(fc).squeeze(dim=1)

        # policies
        logits = self.policy(fc)

        # action mask & sampling distribution
        option_masks = eligs * masks
        pd = masked_softmax(logits=logits, mask=option_masks)
        dists = [Categorical(probs=pd)]

        if update:
            return dists, v

        # sample
        acts = dists[0].sample()
        return acts, v

    def _sample(self, probs):
        u = torch.rand(probs.shape).to(self.device)
        return (torch.log(u) / probs).max(dim=1)

    def eval_act(self, states):
        raise NotImplementedError

    def get_value(self, states):
        spatials, comps, eligs, masks, steps = states

        # spatials
        sp_emb = torch.log10(spatials + 1) - 1
        sp = self.conv(sp_emb)

        # flat
        flats = torch.cat((comps, eligs, masks, steps), dim=1)
        fl = self.flat(flats)

        # hidden
        sp_max = self.maxpool(sp)
        sp_flat = sp_max.view(sp_max.size(0), -1)
        fc_in = torch.cat([sp_flat, fl], dim=1)
        fc = self.fc(fc_in)

        # value
        v = self.vf(fc).squeeze(dim=1)
        return v


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Policy(nn.Module):
    def __init__(self, obs_shape, feat_dim, action_dim, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                self.base = CombinedBase(obs_shape, feat_dim, action_dim=action_dim, **base_kwargs)
            elif len(obs_shape) == 1:
                self.base = MLPBase(obs_shape[0], **base_kwargs)
            else:
                raise NotImplementedError

        self.act_dim = action_dim
        self.ent_denom = math.log(self.act_dim)
        self.verbose = False

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, obs, feats, rnn_hxs, active):
        raise NotImplementedError

    def act(self, spatial, flat, active=None, rnn_hxs=None, update=False, deterministic=False):
        value, actor_features, rnn_hxs_ = self.base(spatial, flat, rnn_hxs)
        masks = flat[:, 1:self.act_dim+1]
        eligs = flat[:, self.act_dim+1:2*self.act_dim+1]
        pol_masks = masks*eligs
        if not torch.all( pol_masks.sum(1).gt(0) ):
            assert( False )
        prob_dist = masked_softmax(logits=actor_features, mask=pol_masks)
        dist = Categorical(probs=prob_dist)
        if self.is_recurrent:
            rnn_hxs_next = active.float() * rnn_hxs_ + (1 - active).float() * rnn_hxs
            value = active.float() * value
        else:
            rnn_hxs_next = None

        if update:
            return dist, value, rnn_hxs_next
        else:
            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()
            return action, value, rnn_hxs_next

    def get_value(self, spatial, flat, rnn_hxs=None):
        value, _, _ = self.base(spatial, flat, rnn_hxs)
        return value


class NNBase(nn.Module):
    def __init__(self, recurrent, feat_dim, gru_ldim, flat_ldim):
        super(NNBase, self).__init__()

        self._gru_ldim = gru_ldim
        self._flat_ldim = flat_ldim
        self._feat_dim = feat_dim
        self._recurrent = recurrent

        if recurrent:
            self.gru_cnn = nn.GRU(gru_ldim, gru_ldim)
            for name, param in self.gru_cnn.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)
            self.gru_flat = nn.GRU(feat_dim, feat_dim)
            for name, param in self.gru_flat.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._gru_ldim + self._feat_dim
        return 1

    @property
    def output_size(self):
        return self._gru_ldim + self._flat_ldim

    def _forward_gru(self, x, hxs, gru_type):
        if gru_type=='cnn':
            gru = self.gru_cnn
        else:
            gru = self.gru_flat
        if x.size(0) == hxs.size(0): # step-by-step
            x, hxs = gru(x.unsqueeze(0), hxs.unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else: # whole sequence at once
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)
            # unflatten
            x = x.view(T, N, x.size(1))
            x, hxs = gru(x, hxs.unsqueeze(0))
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs

class CombinedBase(NNBase):
    def __init__(self, obs_shape, feat_dim=64, gru_ldim=256, flat_ldim=64, action_dim=16, recurrent=False):
        super(CombinedBase, self).__init__(recurrent, feat_dim, gru_ldim, flat_ldim)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
        self.gru_ldim, self.flat_ldim = gru_ldim, flat_ldim
        obs_ch, w, h = obs_shape

        #tanh_init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        # CNN
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        self.cnn_feat_dim = convw * convh * 32
        self.cnn = nn.Sequential(
            init_(nn.Conv2d(obs_ch, 16, 5, stride=2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            init_(nn.Conv2d(16, 32, 5, stride=2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            init_(nn.Conv2d(32, 32, 5, stride=2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(self.cnn_feat_dim, gru_ldim)),
            nn.ReLU()
        )

        # Flat
        self.flat_actor = nn.Sequential(
            init_(nn.Linear(feat_dim, flat_ldim)),
            nn.ReLU(),
            init_(nn.Linear(flat_ldim, flat_ldim)),
            nn.ReLU()
        )
        self.flat_critic = nn.Sequential(
            init_(nn.Linear(feat_dim, flat_ldim)),
            nn.ReLU(),
            init_(nn.Linear(flat_ldim, flat_ldim)),
            nn.ReLU()
        )
        self.actor_linear   = init_(nn.Linear(gru_ldim + flat_ldim, action_dim))
        self.critic_linear  = init_(nn.Linear(gru_ldim + flat_ldim, 1))

        self.train()

    def forward(self, obs, feats, rnn_hxs):
        rnn_hxs_ = None
        ## cnn
        cnn_feat = self.cnn(obs)
        if self.is_recurrent:
            rnn_hxs_cnn = rnn_hxs[:,:self.gru_ldim].contiguous()
            rnn_hxs_flat = rnn_hxs[:,self.gru_ldim:].contiguous()
            cnn_feat, rnn_hxs_cnn_ = self._forward_gru(cnn_feat, rnn_hxs_cnn, gru_type='cnn')

        ## flat
        x = feats
        if self.is_recurrent:
            x, rnn_hxs_flat_ = self._forward_gru(x, rnn_hxs_flat, gru_type='flat')
            rnn_hxs_ = torch.cat( (rnn_hxs_cnn_, rnn_hxs_flat_), dim=1 )

        flat_feat_critic = self.flat_critic(x)
        flat_feat_actor = self.flat_actor(x)

        hidden_critic = torch.cat( (cnn_feat, flat_feat_critic), dim=1)
        hidden_actor  = torch.cat( (cnn_feat, flat_feat_actor), dim=1)

        return self.critic_linear(hidden_critic), self.actor_linear(hidden_actor), rnn_hxs_
