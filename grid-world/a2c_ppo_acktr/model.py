import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from a2c_ppo_acktr.distributions import Categorical, Categorical_masked, DiagGaussian, Bernoulli
from a2c_ppo_acktr.utils import init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Policy(nn.Module):
    def __init__(self, obs_shape, feat_dim, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                if feat_dim == 0:
                    self.base = CNNBase(obs_shape[0], **base_kwargs)
                else:
                    self.base = CombinedBase(obs_shape[0], feat_dim, **base_kwargs)
            elif len(obs_shape) == 1:
                self.base = MLPBase(obs_shape[0], **base_kwargs)
            else:
                raise NotImplementedError
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical_masked(self.base.output_size, num_outputs)
            self.act_dim = action_space.n
            self.ent_denom = math.log(self.act_dim)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, obs, feats, rnn_hxs, active):
        raise NotImplementedError

    def act(self, active, obs, feats, rnn_hxs, deterministic=False):
        value, actor_features, rnn_hxs_ = self.base(obs, feats, rnn_hxs)
        masks = feats[:, 2*self.act_dim:3*self.act_dim]*feats[:, :self.act_dim] # feats= [mask, tp, elig, time]. We use elig as policy-mask
        if not torch.all( masks.sum(1).gt(0) ):
            assert( False )
        dist = self.dist(actor_features, masks)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        rnn_hxs_next = active.float() * rnn_hxs_ + (1-active).float()*rnn_hxs
        value = active.float() * value

        return value, action, action_log_probs, rnn_hxs_next

    """def get_value(self, obs, feats, rnn_hxs):
        value, _, _ = self.base(obs, feats, rnn_hxs)
        return value"""

    def evaluate_actions(self, obs, feats, rnn_hxs, active, action):
        value, actor_features, rnn_hxs = self.base(obs, feats, rnn_hxs)
        pol_masks = feats[:, 2*self.act_dim:3*self.act_dim]*feats[:, :self.act_dim]

        dist = self.dist(actor_features, pol_masks)

        action_log_probs = dist.log_probs(action)
        dist_entropy = (dist.entropy()*active).sum()/self.ent_denom # masking inactive time step + normalize

        return value, action_log_probs, dist_entropy, rnn_hxs


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
    def __init__(self, obs_ch, feat_dim=64, gru_ldim=256, flat_ldim=64, recurrent=False):
        super(CombinedBase, self).__init__(recurrent, feat_dim, gru_ldim, flat_ldim)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
        self.gru_ldim, self.flat_ldim = gru_ldim, flat_ldim

        #tanh_init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        # CNN
        self.cnn_feat_dim = 32 * 8 * 8
        self.cnn = nn.Sequential(
            init_(nn.Conv2d(obs_ch, 16, 1, stride=1)),
            nn.ReLU(),
            init_(nn.Conv2d(16, 32, 3, stride=1)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 3, stride=1, padding=1)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1, padding=1)),
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
        self.critic_linear = init_(nn.Linear(gru_ldim + flat_ldim, 1))

        self.train()

    def forward(self, obs, feats, rnn_hxs):
        ## cnn
        cnn_feat = self.cnn(obs)
        rnn_hxs_cnn = rnn_hxs[:,:self.gru_ldim].contiguous()
        rnn_hxs_flat = rnn_hxs[:,self.gru_ldim:].contiguous()
        if self.is_recurrent:
            cnn_feat, rnn_hxs_cnn_ = self._forward_gru(cnn_feat, rnn_hxs_cnn, gru_type='cnn')

        ## flat
        x = feats
        if self.is_recurrent:
            x, rnn_hxs_flat_ = self._forward_gru(x, rnn_hxs_flat, gru_type='flat')

        flat_feat_critic = self.flat_critic(x)
        flat_feat_actor = self.flat_actor(x)

        hidden_critic = torch.cat( (cnn_feat, flat_feat_critic), dim=1)
        hidden_actor  = torch.cat( (cnn_feat, flat_feat_actor), dim=1)

        rnn_hxs_ = torch.cat( (rnn_hxs_cnn_, rnn_hxs_flat_), dim=1 )

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs_
