import torch
from a2c_ppo_acktr.distributions import FixedCategorical

class Random():
    def __init__(self, args):
        self.algo = 'random'
        self.act_dim = args.act_dim

    def update(self, rollouts):
        return 0, 0, 0

    def act(self, actives, feats):
        masks = feats[:, 2*self.act_dim:3*self.act_dim]*feats[:, :self.act_dim] # feats= [mask, tp, elig, time]. We use elig as policy-mask
        for i in range(actives.shape[0]):
            if actives[i]==0:
                masks[i].fill_(1.0)
        dist = FixedCategorical(masks)
        action = dist.sample()

        return action