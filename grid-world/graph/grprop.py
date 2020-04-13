import os
import torch
import numpy as np
from graph.batch_teacher import Teacher


class GRProp(object):
    def __init__(self, graphs, args):
        self.algo = 'grprop'
        self.graphs = graphs
        self.args = args
        self.teacher =  Teacher( graphs, args )

    def act(self, active, mask_ind, tp_ind, elig_ind, eval_flag=False):
        assert(eval_flag)
        state = (None, None, None, None, mask_ind, tp_ind, elig_ind)
        return self.teacher.choose_action( state, eval_flag ).unsqueeze(-1)
