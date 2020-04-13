"""GRProp"""
from agents.base import BaseMetaAgent
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from common.sc2_utils import masked_softmax, COMP_INFO, get_key, SELECT_RANGE, UNIT_RANGE, FOOD_RANGE, BUILDING_RANGE
from torch.distributions import Categorical

class GRProp(BaseMetaAgent):
    def __init__(self, graphs, args, device, params=None):
        super().__init__(args)
        if not params is None:
            self.temp = params.temp
        else:
            self.temp = 4
        self.beta_a = 4     # higher beta_a, more gap between first precondition satisfaction and last precondition satisfaction
        self.w_a = 3        # Multiplier for NOT connection
        self.ep_or = 0.8    # The portion of subtask reward assigned to precondition
        self.temp_or = 2    # Temperature of softmax in OR operation (have no meaning in SC2 unsually since there is no OR operation)
        self.verbose = args.verbose
        self.use_grad = False
        self.device = device
        if type(graphs)==list:
            self.init_graph(graphs)
        else:
            self.init_batch_graph(graphs)

    def get_option(self, obs, last_dones, eval_flag=False):
        comps, eligs, masks = obs['meta_states']

        tp_tensor = torch.stack([c.float() for c in comps])
        mask_tensor = torch.stack([m.float() for m in masks])
        elig_tensor = torch.stack([e.float() for e in eligs])
        pol_mask = torch.mul(mask_tensor, elig_tensor).detach()

        self.ntasks = len(eligs[0])
        batch_size = len(eligs)
        tid = torch.LongTensor(batch_size).fill_(-1)
        rew = (self.rew_tensor * mask_tensor)
        if self.fast_teacher_mode == 'GR': # Greedy
            logits = rew
        else:
            # The GRProp in [Sohn et al., 2018]
            tp_batch, input_indices = self._prepare_input(tp_tensor.float(), pol_mask)
            soft_reward, or_mat = self.compute_GRProp_FD(tp_batch.unsqueeze(-1), rew)
            rew_vec = soft_reward.squeeze()
            logits = torch.zeros_like(tp_tensor)
            for i, sub_ind in enumerate(input_indices):
                delta = rew_vec[i+1]-rew_vec[0]
                logits[0,sub_ind] = delta

            # if flat, switch to greedy.
            logits += (self.num_layers==1).view(-1,1).repeat(1,self.ntasks).float()*self.rew_tensor

        logits[:, get_key(COMP_INFO, 'SCV') ] += logits[:, get_key(COMP_INFO, '#idle SCV>0') ] # These subtasks are mapped to the same option
        if self.verbose:
            print('*'*40)
            print('Masked scores:')
            for i, score in enumerate(logits[0]):
                if mask_tensor[0][i]==1 and elig_tensor[0][i]==1:
                    print(COMP_INFO[i], score.item())

        # 3. masking
        prob_dist = masked_softmax(logits*self.temp, pol_mask)
        dist = Categorical(probs=prob_dist)
        option = dist.sample()
        if self.verbose:
            print('chosen option:', COMP_INFO[option[0].item()], dist.probs[0][option[0]])
        return option, None, None

    def _prepare_input(self,tp_tensor_org, pol_masks):
        tp_tensor = tp_tensor_org.clone()
        # zero-out the options that is executable multiple times
        tp_tensor[0, UNIT_RANGE] = 0.

        tp_list = [tp_tensor]
        input_indices = []
        for mask in pol_masks: # iter over batch (1)
            indices = mask.nonzero()
            for ind in indices:
                ii = ind.item()
                tp_elem = tp_tensor.clone()
                if ii in SELECT_RANGE: # select. one-hot
                    tp_elem[0, SELECT_RANGE] = 0
                    tp_elem[0, ii] = 1
                elif ii in BUILDING_RANGE: # buildings
                    tp_elem[0, ii] = 1
                    if ii==get_key(COMP_INFO, 'supplydepot'): # if supply depot
                        tp_elem[0, FOOD_RANGE] = 1 # turn-on food
                elif ii in UNIT_RANGE: # units
                    tp_elem[0,ii] += 1
                else: # mineral, gas
                    tp_elem[0,ii] = 1
                tp_list.append(tp_elem)
                input_indices.append(ii)

        tp_batch = torch.cat( tp_list, dim=0)

        return tp_batch, input_indices

    def init_graph(self, graphs):
        ### initialize self.Wa_tensor, Wo_tensor, rew_tensor
        # prepare
        self.flat_graph = False
        batch_size = len(graphs)
        self.num_layers = torch.tensor([len(g.tind_by_layer) for g in graphs]).to(self.device)
        self.max_num_layer = max([len(g.tind_by_layer) for g in graphs]) - 1
        max_NA = max([g.ANDmat.shape[0] for g in graphs])  #max total-#-A
        max_NP = max([len(g.rmag) for g in graphs])   #max total-#-P
        self.rew_tensor = torch.zeros(batch_size, max_NP).to(self.device)
        self.tind_to_tid= torch.zeros(batch_size, max_NP).long().to(self.device)
        for bind, graph in enumerate(graphs):
            self.rew_tensor[bind].copy_(graph.rmag)
            if type(graph.ind_to_id)==dict:
                for k,v in graph.ind_to_id.items():
                    self.tind_to_tid[bind][k] = v
            else:
                self.tind_to_tid[bind].copy_(graph.ind_to_id)

        if self.max_num_layer==0:
            print('Warning!! flat graph!!!')
            self.fast_teacher_mode = 'GR'
        else:
            self.fast_teacher_mode = 'GRProp'
            self.Wa_tensor  = torch.zeros(self.max_num_layer, batch_size, max_NA, max_NP).to(self.device)
            self.Wa_neg     = torch.zeros(batch_size, max_NA, max_NP).to(self.device)
            self.Wo_tensor  = torch.zeros(self.max_num_layer, batch_size, max_NP, max_NA).to(self.device)
            self.Pmask      = torch.zeros(self.max_num_layer+1, batch_size, max_NP, 1).to(self.device)
            for bind, graph in enumerate(graphs):
                tind_by_layer = graph.tind_by_layer
                num_layer = len(tind_by_layer)-1
                #
                if type(graph.rmag)==list:
                    graph.rmag = torch.tensor(graph.rmag).to(self.device)
                if num_layer>0:
                    # W_a_neg
                    ANDmat  = graph.ANDmat
                    ORmat   = graph.ORmat
                    self.Wa_neg[bind, :ANDmat.shape[0],  :ANDmat.shape[1]].copy_(ANDmat.lt(0).float()) # only the negative entries

                    abias, tbias = 0, graph.numP[0]
                    tind_tensor = torch.LongTensor(tind_by_layer[0])
                    mask = torch.zeros(max_NP).scatter_(0, tind_tensor, 1)
                    self.Pmask[0, bind].copy_(mask.unsqueeze(-1))
                    for lind in range(num_layer):
                        # prepare
                        if not torch.all((graph.W_a[lind]!=0).sum(1)>0):
                            import ipdb; ipdb.set_trace()

                        if len(graph.numA)<lind+1 or len(graph.numP)<lind+1:
                            import ipdb; ipdb.set_trace()

                        # W_a
                        na, _ = graph.W_a[lind].shape
                        wa = ANDmat[abias:abias+na, :]
                        self.Wa_tensor[lind, bind, abias:abias+na, :].copy_(wa)     # output is current layer only.

                        # W_o
                        tind = tind_by_layer[lind+1]
                        wo = ORmat[:, abias:abias+na]    # numA x numP_cumul
                        nt, _ = graph.W_o[lind].shape

                        # re-arrange to the original subtask order
                        self.Wo_tensor[lind, bind,:, abias:abias+na].copy_(wo)     # input (or) is cumulative. output is current layer only.
                        abias += na

                        tind_tensor = torch.LongTensor(tind)
                        mask = torch.zeros(max_NP).scatter_(0, tind_tensor, 1)
                        self.Pmask[lind+1, bind].copy_(mask.unsqueeze(-1))
                        tbias += nt
            self.Wa_tensor = self.Wa_tensor.gt(0).float()           # only the positive entries

    def compute_GRProp_FD(self, tp, reward):
        # The GRProp in [Sohn et al., 2018]

        # 1. forward (45% time)
        or_ = torch.max(tp, self.ep_or + (0.99-self.ep_or)*tp) * self.Pmask[0]
        A_neg = torch.matmul(self.Wa_neg, tp)     #(Nb x NA x NP) * (Nb x NP x 1) = (Nb x NA x 1)
        or_list = [or_]
        norm_denom_and = nn.Softplus(self.beta_a)(torch.ones_like(A_neg))
        for lind in range(self.max_num_layer):
            #Init
            wa = self.Wa_tensor[lind]
            wo = self.Wo_tensor[lind]
            Pmask = self.Pmask[lind+1]

            or_concat = torch.cat( or_list, dim=2 )

            #AND layer
            a_pos = ( torch.matmul(wa, or_concat).sum(-1) - wa.sum(-1).clamp(min=1) +1 ).unsqueeze(-1) # (Nb x NA x 1)
            a_hat = a_pos - self.w_a* A_neg                                           #Nb x Na x 1
            and_ = nn.Softplus(self.beta_a)(a_hat) / norm_denom_and  #Nb x Na x 1 (element-wise)

            #soft max version2
            num_next_or = wo.shape[1]
            and_rep = and_.transpose(1,2).repeat(1, num_next_or, 1)
            soft_el = ( self.masked_softmax(self.temp_or*and_rep, wo) * and_rep ).sum(dim=-1).unsqueeze(-1)

            or_= torch.max(tp.detach(), self.ep_or*soft_el + (0.99-self.ep_or)*tp) * Pmask #Nb x Np_sum x 1
            or_list.append(or_)

        # loss (soft reward)
        or_mat_layers = torch.cat(or_list, dim=2)
        or_mat = or_mat_layers.sum(dim=2)

        soft_reward = torch.matmul(or_mat, reward.transpose(0,1)) # (Nb x Nt).*(Nb x Nt) (element-wise multiply)
        return soft_reward, or_mat

    def masked_softmax(self, mat, mask, dim=2, epsilon=1e-6):
        nb, nr, nc = mat.shape
        masked_mat = mat * mask
        masked_min = torch.min(masked_mat, dim=dim, keepdim=True)[0].repeat(1, 1, nc)
        masked_nonneg_mat = (masked_mat - masked_min)*mask
        max_mat = torch.max(masked_nonneg_mat, dim=dim, keepdim=True)[0].repeat(1, 1, nc)
        exps = torch.exp(masked_nonneg_mat-max_mat)
        masked_exps = exps * mask
        masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
        prob =  masked_exps/masked_sums
        if not torch.all(prob.ge(0)):
            import ipdb; ipdb.set_trace()
        return prob
