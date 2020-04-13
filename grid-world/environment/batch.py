import os, time
import torch
import numpy as np
from environment.mazebase_high import Mazebase_high
from environment.playground import Playground
from environment.mining import Mining

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from graph.graph_utils import _to_multi_hot, _transform, dotdict
from environment.graph import SubtaskGraph
from environment.batch_graph import Batch_SubtaskGraph


def _get_id_from_ind_multihot(indexed_tensor, mapping, max_dim):
    nbatch = indexed_tensor.shape[0]
    if isinstance(indexed_tensor, torch.BoolTensor):
        out = torch.zeros(nbatch, max_dim).bool()
        out.scatter_(1, mapping, indexed_tensor)
    elif isinstance(indexed_tensor, torch.ByteTensor):
        out = torch.zeros(nbatch, max_dim).byte()
        out.scatter_(1, mapping, indexed_tensor)
    else:
        print('Error! the input type must be either Bool or Byte')
        print('Input:', indexed_tensor)
        assert False
    return out

class Batch_env(object):
    def __init__(self, args):
        self.verbose = args.verbose
        self.num_envs = args.num_processes
        self.num_graphs = 500
        root = os.getcwd()
        if args.env_name=='playground':
            args.game_config = Playground()
            args.render, args.game, args.fixed_map_mode = args.render, args.env_name, True
            self.envs = [Mazebase_high(args)
                for i in range(args.num_processes)]
            self.graph = Batch_SubtaskGraph(args)
            self.observation_space = Box(low=0, high=1, shape=self.envs[0].obs_shape, dtype=np.float32)
            self.feat_dim = args.game_config.feat_dim
            self.max_task = self.envs[0].n_actions
            self.state_space = Box(low=0, high=1, shape=self.envs[0].obs_shape, dtype=np.float32)
            self.action_space = Discrete(self.envs[0].n_actions)
            self.feed_time = True
            self.feed_prev_ard = True
            self.load_graph = False
        elif args.env_name=='mining':
            args.game_config = Mining()
            args.render, args.game, args.fixed_map_mode = args.render, args.env_name, True
            self.envs = [Mazebase_high(args)
                for i in range(args.num_processes)]
            if args.mode=='meta_train':
                self.graph = Batch_SubtaskGraph(args)
                self.load_graph = False
            else:
                seed = args.seed
                if seed<1:
                    seed = 1
                args.graph_config = dict(folder=os.path.join(root,'environment','data','task_graph_mining','new'), gamename='eval1_mining_'+str(seed))
                self.graph = SubtaskGraph(args)
                self.load_graph = True
                self.num_graphs = self.graph.num_graph
            self.observation_space = Box(low=0, high=1, shape=self.envs[0].obs_shape, dtype=np.float32)
            self.feat_dim = args.game_config.feat_dim
            self.max_task = self.envs[0].n_actions
            self.state_space = Box(low=0, high=1, shape=self.envs[0].obs_shape, dtype=np.float32)
            self.action_space = Discrete(self.envs[0].n_actions)
            self.feed_time = True
            self.feed_prev_ard = True

    def reset_trial(self, nb_epi, reset_graph, graph_index = -1):
        self.max_epi = nb_epi
        obs = []
        if reset_graph:
            if self.load_graph:
                self.graph.set_graph_index(graph_index)
            else:
                self.graph.reset_graphs()
        self.ntasks = self.graph.ntasks
        rmag_tensor, ind_to_id, id_to_ind = self.graph.rmag, self.graph.ind_to_id, self.graph.id_to_ind
        for i in range(self.num_envs):
            o = self.envs[i].reset_trial(rmag_tensor[i], ind_to_id[i], id_to_ind[i])
            obs.append(o)
        #
        self._reset_batch_params()
        self.obs    = torch.stack(obs)
        self.feats  = self._get_feature()
        return self.obs, self.feats

    def step(self, act_ids):
        act_ids = act_ids.cpu()
        steps = []
        self.step_done = torch.zeros(self.num_envs).byte()
        for i in range(self.num_envs):
            if self.active[i]==1:
                act_id = act_ids[i].item()
                assert(self.id_to_ind[i][act_id]>=0)
                step, done = self.envs[i].act(act_id)
                steps.append(step)
                if done:
                    self.step_done[i] = 1
            else:
                steps.append(0)
        #
        self._update_state(act_ids, self.step_done) # 75%
        return self.obs, self.feats, self.rewards, self.active.unsqueeze(-1), steps

    def get_indexed_states(self):
        return self.mask_ind, self.tp_ind, self.elig_ind

    def get_delayed_indexed_states(self):
        return self.prev_active, self.step_done, self.mask_ind_delayed, self.tp_ind_delayed, self.elig_ind_delayed

    def get_graphs(self):
        return self.graph
    ##################  internal functions  ######################
    def _reset_batch_params(self):
        # read from graph
        self.ANDmat         = self.graph.ANDmat.float()
        self.ORmat          = self.graph.ORmat.float()
        self.b_AND          = self.graph.b_AND.unsqueeze(-1)
        self.b_OR           = self.graph.b_OR.unsqueeze(-1)
        self.rmag           = self.graph.rmag
        self.ind_to_id      = self.graph.ind_to_id
        self.id_to_ind      = self.graph.id_to_ind

        # init param
        self.id_mask        = torch.zeros( self.num_envs, self.max_task )
        for i, env in enumerate(self.envs):
            self.id_mask[i].index_fill_(0, self.ind_to_id[i], 1)
        #
        self.done_count = torch.zeros(self.num_envs).long()
        self._reset_episode()
        self.active     = torch.LongTensor( self.num_envs ).fill_(1)
        self.prev_active= self.active.clone()
        self.mask_ind_delayed, self.tp_ind_delayed, self.elig_ind_delayed = torch.LongTensor( 3, self.num_envs, self.ntasks)

    def _reset_episode(self, dones= None):
        if dones is None:
            self.mask_ind   = torch.LongTensor( self.num_envs, self.ntasks ).fill_(1)
            self.mask_id    = self.id_mask.clone()
            self.tp_ind     = torch.LongTensor( self.num_envs, self.ntasks ).fill_(0)
            self.tp_ind_pm  = torch.LongTensor( self.num_envs, self.ntasks ).fill_(-1)
            self.tp_id      = torch.LongTensor( self.num_envs, self.max_task ).fill_(0)
            self._update_elig()

            for i, env in enumerate(self.envs):
                env._reset(epi_index = self.done_count[i].item())
        else:
            self.mask_ind[dones,:] = 1
            self.mask_id[dones] = self.id_mask[dones] # checked. it's not linked. copied.
            self.tp_ind[dones,:] = 0
            self.tp_ind_pm[dones,:] = -1
            self.tp_id[dones,:] = 0
            self._update_elig(dones)

            for i, env in enumerate(self.envs):
                if dones[i]:
                    env._reset(epi_index = self.done_count[i].item())

    def _update_state(self, act_ids, step_dones):
        # 1. prepare
        active      = (self.active * step_dones.lt(1).long()).unsqueeze(-1)  # if done by step, subtask shouldn't be executed!
        act_inds    = self._get_ind_from_id(act_ids) * active
        act_ids_masked  = act_ids * active - (1-active)
        act_inds_masked = act_inds * active - (1-active)
        act_id_mask     = _to_multi_hot(act_ids_masked, self.max_task)
        act_ind_mask    = _to_multi_hot(act_inds_masked, self.ntasks)

        if not torch.all(~active.bool() + torch.gather(self.elig_ind*self.mask_ind.byte(), 1, act_inds).bool() ):
            print('Error! the executed action_index is either ineligible or maksed')
            import ipdb; ipdb.set_trace()
            assert False
        if not torch.all(~active.bool() + torch.gather(self.elig_id*self.mask_id.byte(), 1, act_ids).bool() ):
            print('Error! the executed action_ID is either ineligible or maksed')
            import ipdb; ipdb.set_trace()
            assert False

        # 2. mask, tp, elig
        ### 2-1. mask, tp, elig
        self.mask_ind.masked_fill_(act_ind_mask, 0)
        self.mask_id.masked_fill_(act_id_mask, 0)
        self.tp_ind.masked_fill_(act_ind_mask, 1)
        self.tp_ind_pm.masked_fill_(act_ind_mask, 1)
        self.tp_id.masked_fill_(act_id_mask, 1)
        self._update_elig()

        ### 2-2. backup indexed states for ILP (before reseting episode)
        self._update_delayed_states()
        ### 2-3. if episode is done, reset
        task_dones = torch.mul(self.mask_ind.float(), self.elig_ind.float()).sum(1).lt(1)
        dones = torch.max(task_dones, step_dones.bool())
        self.done_count += dones.long()
        if dones.sum()>0:
            if self.verbose>0:
                for b in range(self.num_envs):
                    if dones[b].item()==1:
                        prt = '+++ Batch#'+str(b)+': '
                        if task_dones[b].item()==1:
                            prt += 'No subtask left. '
                        if step_dones[b].item()==1:
                            prt += 'Time expired. '
                        print(prt)
            self._reset_episode(dones)  # if episode is done, reset mask, tp
        self.epi_dones = dones
        # 3. others
        if torch.any(act_inds.lt(0)) or torch.any(act_inds.gt(self.ntasks-1)):
            import ipdb; ipdb.set_trace()

        # save outputs
        self.obs    = torch.stack( [env.get_state() for env in self.envs] )
        self.rewards= torch.gather(self.rmag, 1, act_inds) * active.float()
        self.rewards= self.rewards*torch.zeros_like(self.rewards).uniform_(0.8, 1.2)
        self.prev_active.copy_(self.active)
        self.active = self.done_count.lt(self.max_epi).long()
        self.feats  = self._get_feature(act_id_mask)

    def _get_feature(self, act_id_mask=None):
        step_list   = [env.get_log_step() for env in self.envs]
        batch_size = len(step_list)
        if act_id_mask is None:
            act_id_mask, self.rewards, self.epi_dones = torch.zeros_like(self.mask_id), torch.zeros(batch_size, 1), torch.zeros(batch_size)

        feat_list = [ self.mask_id.float(), self.tp_id.float(), self.elig_id.float() ]
        if self.feed_time:
            feat_list.append(torch.stack(step_list))
            feat_list.append( torch.log10(self.max_epi +1 - self.done_count.float()).unsqueeze(-1) )
        if self.feed_prev_ard:
            feat_list.append( act_id_mask.float() )
            feat_list.append( self.rewards )
            feat_list.append( self.epi_dones.float().unsqueeze(-1) )
        return torch.cat( feat_list, dim=1 )

    def _update_delayed_states(self):# Used for ILP module
        # never gets the initial state. Instead, get the final state.
        self.mask_ind_delayed.copy_(self.mask_ind)
        self.tp_ind_delayed.copy_(self.tp_ind)
        self.elig_ind_delayed.copy_(self.elig_ind)

    def _update_elig(self, dones = None):
        if dones is None:
            indicator = self.tp_ind_pm.unsqueeze(-1).float()       #Nb x ntask x 1
            ANDmat  = self.ANDmat
            ORmat  = self.ORmat
            b_AND   = self.b_AND
            b_OR   = self.b_OR
            ind_to_id   = self.ind_to_id
        else:
            indicator   = self.tp_ind_pm[dones].unsqueeze(-1).float()       #Nb x ntask x 1
            ANDmat      = self.ANDmat[dones]
            ORmat       = self.ORmat[dones]
            b_AND       = self.b_AND[dones]
            b_OR        = self.b_OR[dones]
            ind_to_id   = self.ind_to_id[dones]
        ANDout = (torch.matmul(ANDmat, indicator)-b_AND+1).sign().ne(-1).float() #sign(A x indic + b) (+1 or 0)
        elig_ind = (torch.matmul(ORmat, ANDout)-b_OR+0.5).sign().ne(-1).squeeze(-1)
        elig_id = _get_id_from_ind_multihot(elig_ind, ind_to_id, self.max_task)

        if dones is None:
            self.elig_ind   = elig_ind
            self.elig_id    = elig_id
        else:
            self.elig_ind[dones]    = elig_ind
            self.elig_id[dones]     = elig_id

    def render_graph(self, env_name, algo, folder_name, g_ind):
        from a2c_ppo_acktr.visualize import render_dot_graph
        render_dot_graph(graph=self.graph, env_name=env_name, algo=algo, folder_name=folder_name, g_ind=g_ind, is_batch=True)

    # utils
    def _get_id_from_ind(self, input_inds):
        return _transform(input_inds, self.ind_to_id)

    def _get_ind_from_id(self, input_ids):
        return _transform(input_ids, self.id_to_ind)

    def get_subtask_lists(self):
        return self.ind_to_id

    def get_tid_to_tind(self):
        return self.id_to_ind
