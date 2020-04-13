import numpy as np

from common.sc2_utils import MAP_INFO
from agents.base import BaseMetaAgent


class Random(BaseMetaAgent):
    """Random Neural Subtask Graph Inferencer."""
    def __init__(self, args, device):
        super().__init__(args)
        self.device = device

    def get_option(self, observations, dones=None, eval_flag=True):
        comps, eligs, masks = observations['meta_states']
        avail_ops = eligs * masks
        avail_ops = [np.where(op == 1)[0] for op in avail_ops]
        elig_ops = [np.where(op == 1)[0] for op in eligs]
        try:
            options = np.array([np.random.choice(op) for op in avail_ops])
        except:
            print("ValueError: avail_ops must be non-empty")
            from IPython import embed; embed()

        #print("[ Selected Option ]: {}".format(MAP_INFO[options[0]]['map']))
        return options, None
