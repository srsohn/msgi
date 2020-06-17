import torch
import math
from graph.graph_utils import  dotdict, batch_bin_encode, _to_multi_hot, _transform

class ILP(object):
    def __init__(self, args):
        # these are shared among batch
        self.nenvs = args.num_processes
        self.action_dim = args.act_dim
        self.tr_epi = args.tr_epi
        self.bonus_mode = args.bonus_mode

        # rollout.
        self.step = 0

    def reset(self, envs):
        self.ntasks = envs.ntasks
        self.step = 0
        self.tp_inds    = torch.zeros( (self.ntasks+1)*self.tr_epi, self.nenvs, self.ntasks, dtype=torch.bool)
        self.elig_inds  = torch.zeros( (self.ntasks+1)*self.tr_epi, self.nenvs, self.ntasks, dtype=torch.bool)
        self.reward_sum = torch.zeros(self.nenvs, self.ntasks)
        self.reward_count=torch.zeros(self.nenvs, self.ntasks, dtype=torch.long)
        if self.bonus_mode>0:
            self.hash_table = [set() for i in range(self.nenvs)]
            self.base_reward = min(10.0 / self.tr_epi / self.ntasks, 1.0)
            if self.bonus_mode==2:
                self.pn_count = torch.zeros( self.nenvs, self.ntasks, 2 )

        # reset from envs
        self.tid_to_tind = envs.get_tid_to_tind()
        self.tlists = envs.get_subtask_lists()

    def insert(self, prev_active, step_done, tp_ind, elig_ind, action_id=None, reward=None):
        self.tp_inds[self.step].copy_(tp_ind.bool())
        self.elig_inds[self.step].copy_(elig_ind.bool())

        # reward
        if not (reward is None):
            action_id = action_id.cpu()
            active = (prev_active * (~step_done.bool()).long()).unsqueeze(-1)
            act_inds = self._get_ind_from_id(action_id) * active
            mask = _to_multi_hot(act_inds, self.ntasks).long() * active
            self.reward_sum += torch.zeros_like(self.reward_sum).scatter_(1, act_inds, reward)*(mask.float())
            self.reward_count += mask
            assert( torch.all( (mask.sum(1)>0)==(reward.squeeze().ne(0) ) ) )
        self.step += 1

    def compute_bonus(self, prev_active, step_done, tp_ind, elig_ind):
        batch_size = len(tp_ind)
        rewards = torch.zeros(batch_size)
        if self.bonus_mode == 0:
            pass
        elif self.bonus_mode == 1: # novel tp_ind
            tp_code = batch_bin_encode(tp_ind )
            for i in range(batch_size):
                if prev_active[i]==1 and step_done[i]==0:
                    code = tp_code[i].item()
                    if not code in self.hash_table[i]:
                        rewards[i] = self.base_reward
                        self.hash_table[i].add(code)
        elif self.bonus_mode == 2: # novel tp_ind & UCB weight ( branch: pos/neg )
            tp_code = batch_bin_encode(tp_ind )
            for i in range(batch_size):
                if prev_active[i]==1 and step_done[i]==0:
                    elig = elig_ind[i]
                    pn_count = self.pn_count[i] #referencing
                    num_task = elig.shape[0]
                    code = tp_code[i].item()
                    if code not in self.hash_table[i]:
                        self.hash_table[i].add(code)

                        # 1. add shifted-ones
                        shifted_ones = torch.zeros_like(pn_count)
                        shifted_ones.scatter_(1, elig.unsqueeze(1).long(), 1)
                        pn_count += shifted_ones

                        # 2. compute weight
                        N_all = pn_count.sum(1)
                        n_current = pn_count.gather(dim=1, index=elig.long().unsqueeze(1)).squeeze()
                        UCB_weight = (25/n_current).sqrt().sum() / num_task   # version 2 (removed log(N_all) since it will be the same for all subtasks)
                        rewards[i] = self.base_reward * UCB_weight
        else:
            assert False
        return rewards.unsqueeze(1)

    def infer_graph(self):
        batch_size = self.nenvs
        num_step = self.step
        tp_tensors      = self.tp_inds[:num_step,:,:]              # T x batch x 13
        elig_tensors    = self.elig_inds[:num_step,:,:]            # T x batch x 13
        rew_counts      = self.reward_count             # batch x 13
        rew_tensors     = self.reward_sum               # batch x 13
        tlists          = self.tlists

        graphs = []
        for i in range(batch_size):
            tp_tensor   = tp_tensors[:,i,:]     # T x Ntask
            elig_tensor = elig_tensors[:,i,:]   # T x Ntask
            rew_count  = rew_counts[i]          # Ntask
            rew_tensor  = rew_tensors[i]        # Ntask
            tlist = tlists[i]
            graph = self._init_graph(tlist) #0. initialize graph

            #1. update subtask reward
            graph.rmag = self._infer_reward(rew_count, rew_tensor) #mean-reward-tracking

            #2. find the correct layer for each node
            is_not_flat, subtask_layer, tind_by_layer, tind_list = self._locate_node_layer( tp_tensor, elig_tensor ) #20%
            if is_not_flat:
                #3. precondition of upper layers
                Kmap_set = self._infer_precondition(tp_tensor, elig_tensor, subtask_layer, tind_by_layer) #75%

                #4. fill-out params
                graph.W_a, graph.W_o, graph.ANDmat, graph.ORmat = self._fillout_mat(Kmap_set, tind_by_layer, tind_list)
            else:
                graph.W_a, graph.W_o, graph.ANDmat, graph.ORmat = [], [], torch.zeros(0), torch.zeros(0)

            #5. fill-out other params.
            graph.tind_by_layer = tind_by_layer
            graph.tind_list = tind_list # doesn't include unknown subtasks (i.e., never executed)
            graph.numP, graph.numA = [], []
            for i in range( len(tind_by_layer) ):
                graph.numP.append( len(tind_by_layer[i]) )
            for i in range( len(graph.W_a) ):
                graph.numA.append( graph.W_a[i].shape[0] )
            graphs.append(graph)

        return graphs

    def _init_graph(self, t_list):
        graph = dotdict()
        graph.numP, graph.numA = [self.ntasks], []
        self.tid_list = t_list
        graph.ind_to_id = torch.zeros(self.ntasks).long()
        graph.id_to_ind = torch.zeros(self.action_dim).long()
        for tind in range(self.ntasks):
            tid = t_list[tind]
            graph.ind_to_id[tind] = tid
            graph.id_to_ind[tid] = tind
        return graph

    def _locate_node_layer(self, tp_tensor, elig_tensor ):
        tind_list = []
        subtask_layer = torch.IntTensor(self.ntasks).fill_(-1)
        #1. update subtask_layer / tind_by_layer
        #1-0. masking
        p_count = elig_tensor.sum(0)
        num_update = tp_tensor.size(0)
        first_layer_mask = (p_count==num_update)
        ignore_mask = (p_count==0)
        infer_flag = (first_layer_mask.sum()+ignore_mask.sum()<self.ntasks)

        #1-1. first layer & unknown (i.e. never executed).
        subtask_layer[first_layer_mask] = 0
        subtask_layer[ignore_mask] = -2
        #
        cand_ind_list, cur_layer = [], []
        remaining_tind_list = []
        for tind in range(self.ntasks):
            if first_layer_mask[tind]==1: # first layer subtasks.
                cand_ind_list.append(tind)
                cur_layer.append(tind)
            elif ignore_mask[tind]==1: # if never executed, assign first layer, but not in cand_list.
                cur_layer.append(tind)
            else:
                remaining_tind_list.append(tind)
        tind_by_layer = [ cur_layer ]
        tind_list += cur_layer # add first layer
        for layer_ind in range(1,self.ntasks):
            cur_layer = []
            for tind in range(self.ntasks):
                if subtask_layer[tind] == -1: # among remaining tind
                    inputs = tp_tensor.index_select(1, torch.LongTensor(cand_ind_list) ) #nstep x #cand_ind
                    targets = elig_tensor.index_select(1, torch.LongTensor([tind]) ).view(-1) #nstep
                    is_valid = self._check_validity(inputs, targets)
                    if is_valid: # add to cur layer
                        subtask_layer[tind] = layer_ind
                        cur_layer.append(tind)

            if len(cur_layer)>0:
                tind_by_layer.append(cur_layer)
                tind_list += cur_layer
                cand_ind_list = cand_ind_list + cur_layer
            else: # no subtask left.
                assert( (subtask_layer==-1).float().sum()==0 )
                break
        return infer_flag, subtask_layer, tind_by_layer, tind_list

    def _check_validity(self, inputs, targets):
        # check if there exists any i1 and i2 such that inputs[i1]==inputs[i2] and targets[i1]!=targets[i2]
        # if there exists, it means the node is not valid, and it should be in the higher layer in the graph.
        tb = {}
        nstep = inputs.shape[0] #new
        code_batch = batch_bin_encode(inputs)
        for i in range(nstep):
            code = code_batch[i].item()
            target = targets[i].item()
            if code in tb:
                if tb[code]!=target:
                    return False
            else:
                tb[code] = target

        return True

    def _infer_reward(self, reward_count, reward_tensor): #. mean-reward
        # reward_count: Ntasks
        # reward_tensor: Ntasks
        rmean = (reward_tensor.sum()/reward_count.sum() ).item()
        rmag = torch.ones(self.ntasks).fill_(rmean)
        for tind  in range(self.ntasks):
            count = reward_count[tind].item()
            if count>0:
                rmag[tind] = reward_tensor[tind] / count
        return rmag

    def _infer_precondition(self, tp_tensor, elig_tensor, subtask_layer, tind_by_layer):
        ever_elig = elig_tensor.sum(0)
        #'subtask_layer' is already filled out in 'update()'
        Kmap_set, cand_ind_list = [ None ]*self.ntasks, []
        max_layer = subtask_layer.max().item()
        for layer_ind in range(1,max_layer+1):
            cand_ind_list = cand_ind_list + tind_by_layer[layer_ind-1] # previous layers
            nFeature = len(cand_ind_list)
            for ind in range(self.ntasks):
                if subtask_layer[ind] == layer_ind and ever_elig[ind]>0:
                    inputs = tp_tensor.index_select(1, torch.LongTensor(cand_ind_list) ) #nstep x #cand_ind
                    targets = elig_tensor.index_select(1, torch.LongTensor([ind]) ).view(-1) #nstep

                    mask = torch.ones(nFeature, dtype=torch.int)
                    root = self.cart_train(mask, inputs, targets) #1.8
                    Kmap_tensor = self.decode_cart(root, nFeature) #0.08
                    self.simplify_Kmap(Kmap_tensor) #0.12
                    Kmap_set[ind] = Kmap_tensor
                    #

        return Kmap_set

    def decode_cart(self, root, nFeature):
        Kmap = []
        stack = []
        instance = dotdict()
        instance.lexp = torch.zeros(nFeature, dtype=torch.int8)
        instance.node = root
        stack.append(instance)
        while len(stack)>0:
            node = stack[0].node
            lexp = stack[0].lexp
            stack.pop(0)
            featId = node.best_ind
            if node.gini>0 : #leaf node && positive sample
                assert(featId>=0)
                if node.left.best_ind>=0: #negation
                    instance = dotdict()
                    instance.lexp = lexp.clone()
                    instance.lexp[featId] = -1 # negative
                    instance.node = node.left
                    stack.append(instance)

                if node.right.best_ind>=0: #positive
                    instance = dotdict()
                    instance.lexp = lexp.clone()
                    instance.lexp[featId] = 1 # positive
                    instance.node = node.right
                    stack.append(instance)

            elif node.sign==0:
                lexp[featId] = -1
                Kmap.append(lexp[None,:])
            else:
                lexp[featId] = 1
                Kmap.append(lexp[None,:])


        Kmap_tensor = torch.cat(Kmap, dim=0)

        return Kmap_tensor

    def cart_train(self, mask, inputs, targets):
        nstep, ncand = inputs.shape
        root = dotdict()
        minval = 2.5 #range: 0~2.0
        assert(mask.sum()>0)
        for i in range(ncand):
            if mask[i]>0:
                left, right = self.compute_gini(inputs[:,i], targets)
                gini = left.gini + right.gini
                if minval > gini:
                    minval = gini
                    best_ind = i
                    best_left = left
                    best_right = right

        root.best_ind = best_ind
        root.gini = minval
        mask[best_ind] = 0
        if minval>0:
            if best_left.gini>0:
                best_lind = torch.nonzero(torch.eq(inputs[:,best_ind], 0) ).squeeze()
                left_input = inputs[best_lind]
                left_targets = targets[best_lind]
                root.left = self.cart_train(mask, left_input, left_targets)
            else:
                root.left = dotdict()
                root.left.gini = 0
                root.left.sign = best_left.p1
                root.left.best_ind = -1

            if best_right.gini>0:
                best_rind = torch.nonzero(inputs[:,best_ind] ).squeeze()
                right_input = inputs[best_rind]
                right_targets = targets[best_rind]
                root.right = self.cart_train(mask, right_input, right_targets)
            else:
                root.right = dotdict()
                root.right.gini = 0
                root.right.sign = best_right.p1
                root.right.best_ind = -1
        else:
            root.sign = best_right.p1 #if right is all True,: sign=1

        mask[best_ind] = 1
        return root

    def compute_gini(self, input_feat, targets):
        neg_input = ~input_feat.bool()
        neg_target = ~targets.bool()
        nn = (neg_input*neg_target).sum().item()   # count[0]
        np = (neg_input*targets).sum().item()      # count[1]
        pn = (input_feat*neg_target).sum().item()  # count[2]
        pp = (input_feat*targets).sum().item()     # count[3]
        assert(nn+np+pn+pp == input_feat.shape[0])

        left, right = dotdict(), dotdict()
        if nn+np>0:
            p0_left = nn / (nn+np)
            p1_left = np / (nn+np)
            left.gini = 1-pow( p0_left, 2)-pow( p1_left, 2)
            left.p0 = p0_left
            left.p1 = p1_left
        else:
            left.gini = 1
            left.p0 = 1
            left.p1 = 1

        if pn+pp>0:
            p0_right = pn / (pn+pp)
            p1_right = pp / (pn+pp)
            right.gini = 1-pow( p0_right, 2)-pow( p1_right, 2)
            right.p0 = p0_right
            right.p1 = p1_right
        else:
            right.gini = 1
            right.p0 = 1
            right.p1 = 1

        return left, right

    def _fillout_mat(self, Kmap_set, tind_by_layer, tind_list):
        W_a, W_o, cand_tind = [], [], []
        num_prev_or = 0
        numA_all = 0
        #1. fillout W_a/W_o
        for layer_ind in range(1, len(tind_by_layer)):
            num_prev_or = num_prev_or + len(tind_by_layer[layer_ind-1])
            num_cur_or = len(tind_by_layer[layer_ind])
            W_a_layer, W_a_layer_padded = [], []
            cand_tind += tind_by_layer[layer_ind-1]
            OR_table = [None]*self.ntasks
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
                            if torch.all( torch.eq(W_a_layer[row],ANDnode) ):
                                duplicated_flag = True
                                and_node_index = row
                                break

                        if duplicated_flag==False:
                            W_a_layer.append(ANDnode)
                            cand_tind_tensor = torch.LongTensor( cand_tind )
                            assert(cand_tind_tensor.shape[0]==ANDnode.shape[0])
                            padded_ANDnode = torch.zeros(self.ntasks).scatter_(0, cand_tind_tensor, ANDnode)
                            W_a_layer_padded.append(padded_ANDnode)
                            OR_table[ind].append(numA) #add the last one
                            numA = numA+1
                        else:
                            OR_table[ind].append(and_node_index) #add the AND node
            if numA>0:
                numA_all = numA_all + numA
                W_a_tensor = torch.stack(W_a_layer_padded)
                W_a.append(W_a_tensor)
            # fill out 'W_o_layer' from 'OR_table'
            W_o_layer = torch.zeros(self.ntasks, numA)
            for ind in tind_by_layer[layer_ind]:
                OR_table_row = OR_table[ind]
                for j in range(len(OR_table_row)):
                    and_node_index = OR_table_row[j]
                    W_o_layer[ind][ and_node_index ] = 1

            W_o.append(W_o_layer)

        #2. fillout ANDmat/ORmat
        if len(W_a)==0:
            print('Error! Subtask graph is flat')
            import ipdb; ipdb.set_trace()
        ANDmat  = torch.cat(W_a, dim=0)
        ORmat   = torch.cat(W_o, dim=1)

        return W_a, W_o, ANDmat, ORmat

    def simplify_Kmap(self, Kmap_tensor):
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

        return Kmap_tensor

    ### Util
    def _get_ind_from_id(self, input_ids):
        return _transform(input_ids, self.tid_to_tind)
