import numpy as np
import torch

from common.sc2_utils import MAP_INFO, NUM_UNITS, NUM_FULL_UNITS, FOOD_REQ
from agents.base import BaseMetaAgent


class Explore(BaseMetaAgent):
    """Random Agent with Exploration bonus."""
    def __init__(self, args, device):
        super().__init__(args)
        self.device = device
        self.verbose = args.verbose
        self.explores = torch.tensor(np.ones((args.envs, NUM_UNITS))).float().to(device)
        self.extras = torch.tensor(np.ones((args.envs, len(MAP_INFO) - NUM_UNITS))).float().to(device)
        self.extras[:,-10:-1].fill_(0) # gas100~300,food
        self.extras[:,-17:-13].fill_(0) # mineral400,300
        self.counts = [0]*args.envs

    def get_option(self, observations, dones=None, eval_flag=True):
        comps, eligs, masks = observations['meta_states']
        player = observations['raws'][0][0].player
        idle_scvs = player.idle_worker_count
        minerals = player.minerals
        gases = player.vespene
        foods = player.food_cap - player.food_used

        food_unit_remain = (self.explores * torch.FloatTensor(FOOD_REQ)).squeeze()
        indices = food_unit_remain.nonzero()
        if len(indices) > 0:
            min_food = int(food_unit_remain[food_unit_remain.nonzero()].min())
        else:
            min_food = 10
        min_food = max(min_food, 3)

        # reset
        if dones is not None:
            for i, done in enumerate(dones):
                if done:
                    self.explores[i].fill_(1.)
                    self.counts[i] = 0

        for i in range(self.explores.shape[0]):
            if foods <= min_food and self.counts[i] < 7:
                self.explores[i][1] = 1
            else:
                self.explores[i][1] = 0

        exps = torch.cat((self.explores, self.extras), dim=1)
        avail_masks = eligs * masks * exps

        avail_ops = []
        for op in avail_masks:
            unit_op = op.clone()
            unit_op[34:] = 0
            if unit_op.sum()==0:
                avail_ops.append(np.where(op == 1)[0])
            else:
                avail_ops.append(np.where(unit_op == 1)[0])

        # select option
        options = np.array([np.random.choice(op) for op in avail_ops])
        elig_ops = [np.where(op == 1)[0] for op in eligs]

        # exclude taken option
        for i, op in enumerate(options):
            if op < 33 and op != 16:  # Building + Unit (except build SCV)
                if op == 1: # supply
                    self.counts[i] += 1
                else:
                    self.explores[i][op] = 0

        #print("[ Selected Option ]: {}".format(MAP_INFO[options[0]]['map']))
        return options, None
