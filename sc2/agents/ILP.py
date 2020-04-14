import os
import torch
import numpy as np
import math, itertools
from common.utils import  dotdict, batch_bin_encode, _to_multi_hot, _transform
from common.sc2_utils import MAP_INFO, COMP_INFO, LABEL_NAME, SC2_GT, get_key, GROUND_UNITS

class ILP(object):
    def __init__(self, args, dirname=''):
        # these are shared among batch
        self.tr_epi = args.tr_epi
        self.nenvs = args.envs
        self.action_dim = len(MAP_INFO)
        self.comp_dim = self.action_dim
        self.buffer_size = 200000
        self.warfare = args.warfare
        self.epi_count = 0
        self.device = 'cpu'
        self.dirname = dirname

        # rollout.
        self.step = 0

    def reset(self, envs):
        self.step       = 0
        self.comps      = torch.zeros(self.buffer_size, self.nenvs, self.action_dim, dtype=torch.uint8).to(self.device)
        self.eligs      = torch.zeros(self.buffer_size, self.nenvs, self.action_dim, dtype=torch.uint8).to(self.device)
        self.reward_sum = torch.zeros(self.nenvs, self.action_dim).to(self.device)
        self.reward_count = torch.zeros(self.nenvs, self.action_dim, dtype=torch.long).to(self.device)
        self.prev_done  = torch.zeros(self.nenvs).to(self.device)
        self.opt_count  = torch.ones(self.nenvs, self.action_dim, dtype=torch.long).to(self.device)
        self.comp_count = torch.zeros(self.nenvs, self.action_dim, dtype=torch.long).to(self.device)

    def insert(self, obs, option=None, reward=None, done=None):
        # In the first step of trial, opt,rew,done = None
        comp, elig, mask = obs['meta_states']
        comp = comp.to(self.device)
        elig = elig.to(self.device)
        mask = mask.to(self.device)

        self.comps[self.step].copy_(comp)
        self.eligs[self.step].copy_(elig)

        # reward
        if reward is not None:
            option = option.to(self.device)
            reward = reward.to(self.device)
            mask = _to_multi_hot(option, self.action_dim, self.device).long()
            self.reward_sum += reward.unsqueeze(-1)*(mask.float())
            self.reward_count += mask
            self.opt_count += mask
            self.prev_done = torch.from_numpy(done.astype(np.uint8))
            self.epi_count += int(done[0])
        self.step += 1

    def save(self, filename='temp/ILP_ep-0.pt'):
        print('Saving ILP model @ ' + filename)
        data = (
            self.comps, self.eligs, self.reward_sum, self.reward_count, self.step, self.opt_count
        )
        torch.save(data, filename)

    def load(self, filename='temp/ILP_ep-0.pt'):
        print('Loading ILP model @ ' + filename)
        data = torch.load(filename)
        self.comps, self.eligs, self.reward_sum, self.reward_count, self.step, self.opt_count = data

    def infer_graph(self, ep, PR, eval_mode=False):
        batch_size = self.nenvs
        num_step = self.step
        comps           = self.comps[:num_step,:,:]     # T x batch x 13
        elig_tensors    = self.eligs[:num_step,:,:]     # T x batch x 13
        rew_counts      = self.reward_count             # batch x 13
        reward_sums     = self.reward_sum               # batch x 13
        tlists          = [list(range(self.comp_dim))] * batch_size
        graphs = []
        for i in range(batch_size):
            comp   = comps[:,i,:]     # T x Ntask
            elig_tensor = elig_tensors[:,i,:]   # T x Ntask
            rew_count  = rew_counts[i]          # Ntask
            rew_sum  = reward_sums[i]        # Ntask
            tlist = tlists[i]
            graph = self._init_graph(tlist) #0. initialize graph

            #1. update subtask reward
            graph.rmag = self._infer_reward(i, rew_count, rew_sum, eval_mode) #mean-reward-tracking

            # 2. find the correct layer for each node
            is_not_flat, subtask_layer, tind_by_layer, tind_list = self._locate_node_layer( comp, elig_tensor ) #20%

            if is_not_flat:
                # 3. precondition of upper layers
                cond_kmap_set = self._infer_precondition(comp, elig_tensor, subtask_layer, tind_by_layer) #75%

                # 4. precision/recall
                if PR:
                    prec, rec = self._eval_graph(ep, cond_kmap_set)
                else:
                    prec, rec = 0., 0.

                #5. fill-out params
                graph.W_a, graph.W_o, graph.ANDmat, graph.ORmat = self._fillout_mat(cond_kmap_set, tind_by_layer, tind_list)
            else:
                graph.W_a, graph.W_o, graph.ANDmat, graph.ORmat = [], [], torch.zeros(0), torch.zeros(0)
                prec, rec = 0., 0.
                pass

            #5. fill-out other params.
            graph.tind_by_layer = tind_by_layer
            graph.tind_list = tind_list # doesn't include unknown subtasks (i.e., never executed)
            graph.numP, graph.numA = [], []
            for i in range( len(tind_by_layer) ):
                graph.numP.append( len(tind_by_layer[i]) )
            for i in range( len(graph.W_a) ):
                graph.numA.append( graph.W_a[i].shape[0] )
            graphs.append(graph)

        return graphs, prec, rec

    def _init_graph(self, t_list):
        # Initialize graph params after the new task is sampled
        graph = dotdict()
        graph.numP, graph.numA = [self.action_dim], []
        self.tid_list = t_list
        graph.ind_to_id = torch.zeros(self.action_dim).long()
        graph.id_to_ind = torch.zeros(self.action_dim).long()
        for tind in range(self.action_dim):
            tid = t_list[tind]
            graph.ind_to_id[tind] = tid
            graph.id_to_ind[tid] = tind
        return graph

    def _locate_node_layer(self, attr, elig_tensor ):
        # For each node (i.e., subtask), locate its layer.
        # Rule: Put the subtask in the lowest possible layer where it's precondition can be defined
        tind_list = []
        subtask_layer = torch.IntTensor(self.action_dim).fill_(-1)
        #1. update subtask_layer / tind_by_layer
        #1-0. masking
        p_count = elig_tensor.sum(0)
        num_update = attr.size(0)
        attr_dim = attr.size(1)
        first_layer_mask = (p_count==num_update)
        first_layer_mask[ get_key(LABEL_NAME, "food 1") ]=1 # manually put FOOD1 as a first layer
        ignore_mask = (p_count==0)
        infer_flag = (first_layer_mask.sum()+ignore_mask.sum()<self.action_dim)

        cand_attr_list = self._mulhot_to_list( first_layer_mask )

        #1-1. first layer & unknown (i.e. never executed).
        subtask_layer[first_layer_mask] = 0
        subtask_layer[ignore_mask] = -2
        #
        cur_layer, remaining_tind_list = [], []
        for tind in range(self.action_dim):
            if first_layer_mask[tind]==1: # first layer subtasks.
                cur_layer.append(tind)
            elif ignore_mask[tind]==1: # if never executed, assign first layer, but not in cand_list.
                cur_layer.append(tind)
            else:
                remaining_tind_list.append(tind)
        tind_by_layer = [ cur_layer ]
        tind_list += cur_layer # add first layer

        # second layer and above
        for layer_ind in range(1,self.action_dim):
            cur_layer = []
            for tind in range(self.action_dim):
                if subtask_layer[tind] == -1: # among remaining tind
                    if not self._check_selection(tind, cand_attr_list):
                        continue
                    inputs = attr.index_select(1, torch.LongTensor(cand_attr_list).to(self.device)) #nstep x #cand_ind
                    targets = elig_tensor.index_select(1, torch.LongTensor([tind]).to(self.device)).view(-1) #nstep
                    is_valid = self._check_validity(inputs, targets, cand_attr_list)
                    if is_valid: # add to cur layer
                        subtask_layer[tind] = layer_ind
                        cur_layer.append(tind)

            if len(cur_layer) > 0:
                tind_by_layer.append(cur_layer)
                tind_list += cur_layer
                attr_cur_layer = cur_layer
                cand_attr_list = cand_attr_list + attr_cur_layer
            else: # no subtask left.
                if (subtask_layer==-1).float().sum()>0:
                    print('ERROR! There exists subtask(s) whose precondition cannot be explained')
                    import ipdb; ipdb.set_trace()
                break

        # result:
        return infer_flag, subtask_layer, tind_by_layer, tind_list

    def _check_validity(self, inputs, targets, cand_attr_list):
        # check if there exists any i1 and i2 such that inputs[i1]==inputs[i2] and targets[i1]!=targets[i2]
        # if there exists, it means the node is not valid, and it should be in the higher layer in the graph.
        tb = {}
        nstep = inputs.shape[0] #new
        code_batch = batch_bin_encode(inputs, self.device)
        for i in range(nstep):
            code = code_batch[i]
            target = targets[i].item()
            if code in tb:
                if tb[code]!=target:
                    return False
            else:
                tb[code] = target

        return True

    def _infer_reward(self, i, reward_count, reward_sum, eval_mode=False):
        # There is no subtask reward given from environment in SC2
        if self.warfare:
            if eval_mode:
                rmag = torch.zeros(self.action_dim)
                rmag[ get_key(LABEL_NAME, 'SCV') ] = 0.05
                for key in GROUND_UNITS:
                    rmag[ key ] = 1.0
                if self.step > 3:
                    rmag[ get_key(LABEL_NAME, 'no-op') ] = 0.02
            else:
                if self.step==0:
                    rmag = torch.ones(self.action_dim).to(self.device)
                else:
                    build_unit_count = self.opt_count[i].clone()
                    UCB_bonus = 1/(build_unit_count+1).float()
                    rmag = torch.ones(self.action_dim).to(self.device)
                    rmag += UCB_bonus
        else:
            rmag = torch.ones(self.action_dim).to(self.device)
            for tind  in range(self.action_dim):
                count = reward_count[tind].item()
                if count > 0:
                    # MLE of mean of Gaussian = sample mean
                    rmag[tind] = reward_sum[tind] / count
        return rmag

    def _infer_precondition(self, attr, elig_tensor, subtask_layer, tind_by_layer):
        ever_elig = elig_tensor.sum(0)
        Kmap_set, cand_attr_list = [None]*self.action_dim, []
        max_layer = subtask_layer.max().item()

        for layer_ind in range(1,max_layer+1):
            if (subtask_layer==(layer_ind-1)).sum()==0:
                print('Error! Previous layer in subtask graph is empty!')
                import ipdb; ipdb.set_trace()
            attr_cur_layer = self._mulhot_to_list( subtask_layer==(layer_ind-1) )
            cand_attr_list = cand_attr_list + attr_cur_layer
            nFeature = len(cand_attr_list)
            for ind in range(self.action_dim):
                if subtask_layer[ind] == layer_ind and ever_elig[ind]>0:
                    inputs = attr.index_select(1, torch.LongTensor(cand_attr_list) ) #nstep x #cand_ind
                    targets = elig_tensor.index_select(1, torch.LongTensor([ind]) ).view(-1) #nstep
                    mask = torch.ones(nFeature, dtype=torch.int).to(self.device)

                    tree = self.cart_train(mask, inputs, targets)
                    Kmap_tensor = self.decode_cart(tree, cand_attr_list)
                    Kmap_tensor = self.simplify_Kmap(Kmap_tensor, inputs, targets, cand_attr_list)
                    Kmap_set[ind] = Kmap_tensor

        return Kmap_set

    def decode_cart(self, tree, cand_attr_list):
        indices = np.where(tree.value.squeeze()[:,0]==0)[0]

        # For each pos-leaf node, fillout K-map
        Kmap = []
        for ind in indices:
            node_ind = ind
            lexp = dict()
            while node_ind!=0:
                if node_ind in tree.children_left:
                    par_ind = np.where(tree.children_left==node_ind)[0].item()
                    sign = 0
                elif node_ind in tree.children_right:
                    par_ind = np.where(tree.children_right==node_ind)[0].item()
                    sign = 1
                else:
                    print('Error!')
                    print('node_ind=',node_ind)
                    import ipdb; ipdb.set_trace()
                cand_attr_ind = tree.feature[par_ind].item()
                attr_index = cand_attr_list[cand_attr_ind]

                lexp[attr_index] = sign
                node_ind = par_ind
            Kmap.append(lexp)

        ntask = len(COMP_INFO)
        Kmap_tensor = torch.zeros(len(Kmap), ntask).char()
        for i, lexp in enumerate(Kmap):
            for sub_ind, sign in lexp.items():
                Kmap_tensor[i, sub_ind] = sign
        return Kmap_tensor

    def cart_train(self, mask, inputs, targets):
        from sklearn import tree
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(inputs.numpy(), targets.numpy())
        return clf.tree_

    def _fillout_mat(self, Kmap_set, tind_by_layer, tind_list):
        W_a, W_o, cand_tind = [], [], []
        num_prev_or = 0
        numA_all = 0
        #1. fillout W_a/W_o
        for layer_ind in range(1, len(tind_by_layer)):

            num_prev_or = num_prev_or + len(tind_by_layer[layer_ind-1])
            num_cur_or = len(tind_by_layer[layer_ind])
            W_a_layer_padded = []
            cand_tind += tind_by_layer[layer_ind-1]
            OR_table = [None]*self.action_dim
            numA = 0
            # fill out 'W_a_layer' and 'OR_table'
            for ind in tind_by_layer[layer_ind]:
                Kmap = Kmap_set[ind]
                if len(Kmap)>0: #len(Kmap)==0 if no positive sample exists
                    OR_table[ind] = []
                    for j in range(Kmap.shape[0]):
                        ANDnode = Kmap[j,:].float()
                        #see if duplicated
                        duplicated_flag = False
                        for row in range(numA):
                            if torch.all( torch.eq(W_a_layer_padded[row],ANDnode) ):
                                duplicated_flag = True
                                and_node_index = row
                                break

                        if duplicated_flag==False:
                            W_a_layer_padded.append(ANDnode)
                            OR_table[ind].append(numA) #add the last one
                            numA = numA+1
                        else:
                            OR_table[ind].append(and_node_index) #add the AND node
            if numA>0:
                numA_all = numA_all + numA
                W_a_tensor = torch.stack(W_a_layer_padded)
                W_a.append(W_a_tensor)
            # fill out 'W_o_layer' from 'OR_table'
            W_o_layer = torch.zeros(self.action_dim, numA)
            for ind in tind_by_layer[layer_ind]:
                OR_table_row = OR_table[ind]
                for j in range(len(OR_table_row)):
                    and_node_index = OR_table_row[j]
                    W_o_layer[ind][ and_node_index ] = 1
            W_o.append(W_o_layer)

        #2. fillout ANDmat/ORmat
        if len(W_a)==0 or numA_all==0 or self.action_dim==0:
            print('Error in the inferred subtask graph')
            import ipdb; ipdb.set_trace()

        ANDmat  = torch.cat(W_a, dim=0)
        ORmat   = torch.cat(W_o, dim=1)
        return W_a, W_o, ANDmat, ORmat

    def simplify_Kmap(self, Kmap_tensor, inputs, targets, cand_attr_list):
        """
        # This function performs the following two reductions
        # A + AB  -> A
        # A + A'B -> A + B
        ###
        # Kmap_bin: binarized Kmap. (i.e., +1 -> +1, 0 -> 0, -1 -> +1)
        """
        numAND = Kmap_tensor.shape[0]
        mask = torch.Tensor(numAND).fill_(1)
        max_iter = 20
        for jj in range(max_iter):
            done = True
            remove_list = []
            Kmap_bin = Kmap_tensor.ne(0).char()
            for i in range(numAND):
                if mask[i]==1:
                    kbin_i = Kmap_bin[i]
                    for j in range(i+1, numAND):
                        if mask[j]==1:
                            kbin_j = Kmap_bin[j]
                            kbin_mul = kbin_i * kbin_j
                            if torch.all(kbin_mul.eq(kbin_i)): #i subsumes j. Either 1) remove j or 2) reduce j.
                                done = False
                                Kmap_common_j = Kmap_tensor[j] * kbin_i # common parts in j.
                                difference_tensor = torch.ne(Kmap_common_j, Kmap_tensor[i]) # (A,~B)!=(A,B) -> 'B'
                                num_diff_bits = torch.sum(difference_tensor)
                                if num_diff_bits==0: # completely subsumes--> remove j.
                                    mask[j]=0
                                else: #turn off the different bits
                                    dim_ind = torch.nonzero(difference_tensor)[0]
                                    Kmap_tensor[j][dim_ind] = 0

                            elif torch.all(kbin_mul.eq(kbin_j)): #j subsumes i. Either 1) remove i or 2) reduce i.
                                done = False
                                Kmap_common_i = Kmap_tensor[i] * kbin_j
                                difference_tensor = torch.ne(Kmap_common_i, Kmap_tensor[j])
                                num_diff_bits = torch.sum(difference_tensor)
                                if num_diff_bits==0: # completely subsumes--> remove i.
                                    mask[i]=0
                                else: #turn off the different bit.
                                    dim_ind = torch.nonzero(difference_tensor)[0]
                                    Kmap_tensor[i][dim_ind] = 0

            if done: break
        if mask.sum()< numAND:
            Kmap_tensor = Kmap_tensor.index_select(0,mask.nonzero().view(-1))

        # 2. check for redundant ones
        numA, dim = Kmap_tensor.shape
        if numA>1:
            for ind in range(numA):
                kvec = Kmap_tensor[ind].index_select(0, torch.LongTensor(cand_attr_list) )
                if self._check_coverage(kvec, inputs, targets):
                    if Kmap_tensor.sum()==0:
                        assert False, 'Error. No precondition'
                    Kmap_tensor = Kmap_tensor[ind:ind+1,:]
                    break
        return Kmap_tensor

    def _check_coverage(self, kvec, inputs, targets):
        data = inputs.int()*2-1
        evalmat = data * kvec.int().unsqueeze(0)
        pred = (evalmat==-1).sum(1)==0 # if total # wrong==0: True(condition satisfied), else: False (condition is False)

        return torch.all(pred==targets)

    ### Util

    ### Debugging tool ###
    def _check_selection(self, tind, cand_tind):
        if tind in range(17,21) or tind==13: # units from barracks or tech
            return (36 in cand_tind)
        elif tind in range(21,27) or tind==14: # units from factory
            return (37 in cand_tind)
        elif tind in range(21,33) or tind==15: # units from starport
            return (38 in cand_tind)
        else:
            return True
    ### Debugging tool ###

    def _mulhot_to_list(self, in_tensor, mapping=None):
        out_list = []
        indices = in_tensor.nonzero()
        if indices.dim()>1:
            indices = indices.squeeze()
        if mapping is None:
            if indices.dim()==0:
                new_idx = indices.item()
                if type(new_idx)==int:
                    out_list.append(new_idx)
                elif type(new_idx)==list or type(new_idx)==tuple:
                    out_list += new_idx
            else:
                for idx in indices:
                    new_idx = idx.item()
                    if type(new_idx)==int:
                        out_list.append(new_idx)
                    elif type(new_idx)==list or type(new_idx)==tuple:
                        out_list += new_idx
        else:
            if indices.dim()==0:
                new_idx = mapping[indices.item()]
                if type(new_idx)==int:
                    out_list.append(new_idx)
                elif type(new_idx)==list or type(new_idx)==tuple:
                    out_list += new_idx
            else:
                for idx in indices:
                    new_idx = mapping[idx]
                    if type(new_idx)==int:
                        out_list.append(new_idx)
                    elif type(new_idx)==list or type(new_idx)==tuple:
                        out_list += new_idx

        return out_list

    def _list_to_list(self, in_list, mapping):
        out_list = []
        for idx in in_list:
            new_idx = mapping[idx]
            if type(new_idx)==int:
                out_list.append(new_idx)
            elif type(new_idx)==tuple:
                out_list += new_idx

        return out_list

    
    def _eval_graph(self, ep, cond_kmap_set):
        # TP =  gt ^ infer
        # FP = ~gt ^ infer
        # FN =  gt ^ ~infer
        # TN = ~gt ^ ~infer
        precision, recall = np.zeros( (2,self.action_dim) )
        for ind in range(self.action_dim):
            k_infer = cond_kmap_set[ind] # (N, dim)
            k_gt = torch.tensor(SC2_GT[ind], dtype=torch.int) # (dim)
            if k_infer is None:
                precision[ind] = 1/pow(2, self._num_nonzero(k_gt) )
                recall[ind] = 1.
                continue
            k_infer = k_infer.type(torch.int)

            k_infer_, k_gt_ = self._compact(k_infer, k_gt)
            if k_infer_ is None or k_gt_ is None: # exactly same
                precision[ind] = 1.
                recall[ind] = 1.
            elif k_gt_.dim()==0:
                if k_gt_.item()==0:
                    precision[ind] = 1.
                    recall[ind] = 0.5
                else:
                    precision[ind] = 0.5
                    recall[ind] = 1.
            else:
                num_gt      = self._count_or(k_gt_)
                num_infer   = self._count_or(k_infer_) # checked!
                # 1. TP
                TP = self._get_A_B( k_infer_, k_gt_ )
                FN = num_gt - TP
                FP = num_infer - TP
                precision[ind]  = TP / (TP+FP)
                recall[ind]     = TP / (TP+FN)

        filename = os.path.join(self.dirname, 'graph_PR.txt')
        with open(filename, 'a') as f:
            string = '{}\t{:.02f}\t{:.02f}\t{:.02f}\t{:.02f}\n'.format(ep, precision.mean(), recall.mean(), 0., 0.)
            f.writelines( string )
        return precision, recall

    def _compact(self, kmap1, kmap2):
        diff_count = (kmap1!=kmap2).sum(0) # num diff elems in each dimension
        if diff_count.sum()==0:
            return None, None
        else:
            indices = diff_count.nonzero().squeeze()
            return kmap1[:,indices], kmap2[indices]
        #data = np.concatenate( (kmap1, kmap2.expand_dim), dim= )

    def _get_A_B(self, kmap1, kmap2):
        numA, feat_dim = kmap1.shape
        kmap_list = []
        #1. merge AND
        for aind in range(numA):
            k1 = kmap1[aind]
            k_mul = k1*kmap2 # vec * vec (elem-wise-mul)
            if (k_mul==-1).sum() > 0: # if there exists T^F--> empty (None)
                continue
            k_sum = k1+kmap2
            #    |  1 |  0 | -1
            #----+----------------
            #  1 |  1    1   None
            #----|
            #  0 |  1    0   -1
            #----|
            # -1 | None -1   -1
            k_sum = k_sum.clamp(min=-1, max=1)
            kmap_list.append(k_sum[None,:])
        if len(kmap_list)>0:
            kmap_mat = torch.cat(kmap_list, dim=0)
            return self._count_or( kmap_mat )
        else:
            return

    def _count_or(self, kmap_mat):
        # count the number of binary combinations that satisfy input 'kmap_mat'
        # where each row kmap_mat[ind, :] is a condition, and we take "or" of them.
        if kmap_mat.dim()==0:
            return 1
        elif kmap_mat.dim()==1 or kmap_mat.shape[0]==1: # simply count number of 0's
            return pow(2, self._num_zero(kmap_mat) )
        NA, dim = kmap_mat.shape

        # 0. prune out all-DC bits
        target_indices = []
        common_indices = []
        for ind in range(dim):
            if not torch.all(kmap_mat[:, ind]==kmap_mat[0, ind]):
                target_indices.append(ind)
            else: # if all the same, prune out.
                common_indices.append(ind)
        common_kmap = kmap_mat[0,common_indices] # (dim)
        num_common = pow(2, self._num_zero(common_kmap) )

        compact_kmap = kmap_mat.index_select(1, torch.LongTensor(target_indices)).type(torch.int8)
        numO, feat_dim = compact_kmap.shape # NO x d

        if feat_dim > 25:
            print('[Warning] there are too many non-DC features!! It will take long time')
            import ipdb; ipdb.set_trace()
        # 1. gen samples
        nelem = pow(2, feat_dim)
        bin_mat = torch.tensor(list(map(list, itertools.product([-1, 1], repeat=feat_dim))), dtype=torch.int8) # +1 / -1
        # N x d
        false_map = (compact_kmap.unsqueeze(0) * bin_mat.unsqueeze(1) == -1).sum(2) # NxNOxd --> NxNO
        true_map = (false_map==0) # NxNO
        truth_val_vec = (true_map.sum(1)>0) # if any one of NO is True, then count as True (i.e, OR operation)
        return truth_val_vec.sum().item() * num_common

    def _num_nonzero(self, kmap):
        return (kmap!=0).sum().item()

    def _num_zero(self, kmap):
        return (kmap==0).sum().item()