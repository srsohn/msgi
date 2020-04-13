import os
import torch
import numpy as np
from graph.graph_utils import _sample_layer_wise, _sample_int_layer_wise, batch_bin_encode

__PATH__ = os.path.abspath(os.path.dirname(__file__))

class Batch_SubtaskGraph(object):
    def __init__(self, args):
        #1. init graph param
        self.game_config = args.game_config
        self.max_task = self.game_config.nb_subtask_type
        self._set_graph_config(args) # random graph generation
        self.nbatch = args.num_processes
        self.env_name = args.env_name
        self.seed = args.seed

    def reset_graphs(self):
        torch.manual_seed(self.seed)
        if self.env_name=='mining':
            max_layer = 10
            self.layer_range = [6, max_layer]
            self.ntask_range = torch.LongTensor([ [3,3], [4,4], [2,2], [3,3], [1,2], [2,3], [1,2], [1,2], [1,3], [1,2]]).t()

            self.nlayer = np.random.randint(self.layer_range[0], self.layer_range[1]+1, 1 ).item()
            self.nt_layer = []
            for lind in range(self.nlayer):
                nt = np.random.randint(self.ntask_range[0][lind], self.ntask_range[1][lind]+1, 1 ).item()
                self.nt_layer.append(nt)
            self.ntasks     = sum(self.nt_layer)
            self.ndt_layer = [0]*max_layer

            # 2. fillout others
            tbias, self.tlist, self.dtlist = 0, [], []
            self.r_high, self.r_low = torch.zeros(2, self.ntasks)

            for lind in range(self.nlayer):
                nt, ndt = self.nt_layer[lind], self.ndt_layer[lind]
                self.tlist += list( range(tbias, tbias+nt) )
                self.dtlist += list( range(tbias+nt, tbias+nt+ndt) )

                high, low = self.rew_range[:,lind]
                self.r_high[tbias:tbias+nt+ndt].fill_(high)
                self.r_low[tbias:tbias+nt+ndt].fill_(low)

                tbias += (nt + ndt)

        Nprob = torch.tensor([0.25, 0.4, 0.1])
        nbatch = self.nbatch
        self.rmag           = torch.zeros( nbatch, self.ntasks )
        self.id_mask        = torch.zeros( nbatch, self.max_task )
        self.ind_to_id      = torch.LongTensor( nbatch, self.ntasks ).zero_()
        self.id_to_ind      = torch.LongTensor( nbatch, self.max_task ).fill_(-1)

        # 1. ind_to_id
        self.ind_to_id = torch.LongTensor( nbatch, self.ntasks ).zero_()
        for bind in range(nbatch):
            id_tensor = torch.randperm(self.max_task)[:self.ntasks]
            self.id_mask[bind].index_fill_(0, id_tensor, 1)
            self.ind_to_id[bind].copy_( id_tensor )

        base = torch.arange(self.ntasks).repeat(nbatch, 1)
        self.id_to_ind.scatter_(1, self.ind_to_id, base)

        # 2. rmag
        self.rmag = _sample_layer_wise(nbatch, self.r_high, self.r_low )

        # 3. na_layer, ndnp_layer
        no_layer    = self.nt_layer
        ndt_layer   = self.ndt_layer

        na_layer_   = _sample_int_layer_wise(nbatch, self.na_range[1], self.na_range[0] ) # nbatch x (nlayer-1)

        self.max_NA = na_layer_.sum(1).max()
        self.max_NP = sum(no_layer+ndt_layer)
        # 4.
        self.ANDmat         = torch.zeros( nbatch, self.max_NA, self.max_NP ).long()
        self.ORmat          = torch.zeros( nbatch, self.max_NP, self.max_NA )
        self.b_AND          = torch.zeros( nbatch, self.max_NA, 1 )
        self.b_OR           = torch.zeros( nbatch, self.max_NP, 1 )
        self.numP, self.numA= [], []
        self.tind_by_layer  = []
        for bind in range(nbatch):
            # prepare
            atable, otable = set(), set()
            na_layer = na_layer_[bind]
            ANDmat  = self.ANDmat[bind]
            ORmat   = self.ORmat[bind]
            obias, abias = (no_layer[0]+ndt_layer[0]), 0
            nump, numa, tlayer  = [obias], [], [ [ *range(obias) ] ]
            ocand = [*range(no_layer[0])]
            for lind in range(self.nlayer-1):
                # 4-0. prepare
                na, no, ndt = na_layer[lind].item(), no_layer[lind+1], ndt_layer[lind+1]
                nt_prev = no_layer[lind] + ndt_layer[lind]
                nanc_low, nanc_high = self.nanc_range[0, lind], self.nanc_range[1, lind]+1
                Nweights = Nprob[nanc_low:nanc_high]
                nump.append(no+ndt)
                numa.append(na)
                tlayer.append( [*range(obias, obias+no+ndt)] )

                # 4-1. AND node (non-distractors)
                i = 0
                while i<na:
                    aind = abias + i
                    # sample #pos-child & #neg-child (nac/nanc)
                    nac = np.random.randint(self.nac_range[0, lind], self.nac_range[1, lind]+1, 1 ).item()
                    nanc = 0

                    if nanc_high>nanc_low:
                        nanc = torch.multinomial(Nweights, 1, replacement=True).item() + nanc_low
                    #
                    and_row = torch.zeros(self.ntasks).long()
                    # sample one direct_child (non-distractors)
                    oind_ = obias - nt_prev + torch.randperm( no_layer[lind] )[0].item() # only non-distrators
                    and_row[oind_].fill_(1)

                    # sample nac-1 pos-children and nanc neg-children (non-distractors)
                    ocand_copy = [ o for o in ocand ]
                    ocand_copy.remove(oind_)

                    ocand_copy_tensor = torch.LongTensor( ocand_copy )
                    oinds_ = torch.randperm( len(ocand_copy) )
                    ac_oind = ocand_copy_tensor[oinds_[:nac-1]]
                    neg_ac_oind = ocand_copy_tensor[oinds_[nac-1:nac+nanc-1]]
                    and_row.index_fill_(0, ac_oind, 1)
                    and_row.index_fill_(0, neg_ac_oind, -1)
                    code = batch_bin_encode(and_row)
                    if not code in atable: # if not duplicated
                        atable.add(code)
                        i = i+1
                        ANDmat[aind] = ANDmat[aind] + and_row

                # 4-2. OR node
                i = 0
                count=0
                while i<no+ndt:
                    oind = obias + i
                    noc = np.random.randint(self.noc_range[0, lind], self.noc_range[1, lind]+1, 1 ) .item()
                    ainds = abias + torch.randperm(na)[:noc]
                    or_row = torch.zeros(self.max_NA).index_fill_(0, ainds, 1)
                    code = batch_bin_encode(or_row)
                    if not code in otable: # if not duplicated
                        otable.add(code)
                        i = i+1
                        ORmat[oind] += or_row
                    count += 1
                ocand += [ *range(obias, obias+no) ]
                obias += no + ndt
                abias += na

            self.numP.append(nump)
            self.numA.append(numa)
            self.tind_by_layer.append(tlayer)

            # should be done after determining NA
            # 4-3. distractor
            total_a = abias
            dt_inds, count = torch.LongTensor(self.dtlist), 0
            abias = 0
            for lind in range(self.nlayer-1):
                ndt = ndt_layer[lind]
                ndnp_ = np.random.randint(self.ndnp_range[0, lind], self.ndnp_range[1, lind]+1, ndt )
                for i in range(ndt):
                    oind = dt_inds[count]
                    assert(torch.all(ANDmat[:, oind]<1) )
                    ndnp = ndnp_[i].item()
                    #
                    par_ainds = abias + torch.randperm(total_a-abias)[:ndnp]
                    column = torch.zeros(self.max_NA).scatter_(0, par_ainds, -1)
                    ANDmat[:, oind].copy_(column)
                    count+=1
                abias += numa[lind]

        #5. b_AND & b_OR
        self.b_AND  = self.ANDmat.ne(0).sum(2).type(torch.float)
        self.b_OR = torch.Tensor( nbatch, self.ntasks).fill_(1)
        for i in range(nbatch):
            self.b_OR[i, :self.numP[i][0]].fill_(0) # first layer

    def get_elig(self, task_indicator_pm):
        ANDmat = self.ANDmat
        b_AND = self.b_AND
        ORmat = self.ORmat # nb_or x
        b_OR = self.b_OR

        indicator = task_indicator_pm.type(torch.float)

        ANDout = torch.addmv(-b_AND, ANDmat, indicator).sign().ne(-1).type(torch.float) #sign(A x indic + b) (+1 or 0)
        elig_hard = torch.addmv(-b_OR, ORmat, ANDout).sign().ne(-1)
        return elig_hard

    def _set_graph_config(self, args):
        if args.env_name=='playground':
            level = args.level
            max_layer = 6
            self.level = level
            self.noc_range = torch.LongTensor([[1,2]]).repeat(max_layer,1).t()
            self.nac_range = torch.LongTensor([[1,3]]).repeat(max_layer,1).t()
            self.nanc_range= torch.LongTensor([[0,2],[0,1],[0,1],[0,0],[0,0],[0,0]]).t()
            self.ndnp_range= torch.LongTensor([[2,3],[2,3],[2,3],[0,0],[0,0],[0,0]]).t()
            if level==1:
                self.ntasks = 13
                self.rew_range  = torch.tensor([ [.1,.2], [.3,.4], [.7,.9], [1.8, 2.0] ]).t()
                self.ndt_layer  = [2,1,0,0]
                self.nt_layer   = [4,3,2,1]

                self.na_range   = torch.LongTensor([ [3,5], [3,4], [2,2] ]).t()
                self.nanc_range       = torch.LongTensor([ [0,2],[0,1],[0,0]]).t()
            elif level==2:
                self.ntasks = 15
                self.rew_range  = torch.tensor([ [.1,.2], [.3,.4], [.7,.9], [1.8, 2.0] ]).t()
                self.ndt_layer  = [2,2,0,0]
                self.nt_layer   = [5,3,2,1]

                self.na_range   = torch.LongTensor([ [3,5], [3,4], [2,2] ]).t()
                self.nanc_range       = torch.LongTensor([ [0,2],[0,1],[0,0]]).t()
            elif level==3:
                self.ntasks = 16
                self.rew_range  = torch.tensor([ [.1,.2], [.3,.4], [.6,.7], [1.0,1.2], [2.0,2.2] ]).t()
                self.ndt_layer  = [1,1,1,0,0]
                self.nt_layer   = [4,3,3,2,1]

                self.na_range   = torch.LongTensor([ [3,5], [3,4], [3,4], [2,2] ]).t()
            elif level==4:
                self.ntasks = 16
                self.rew_range  = torch.tensor([ [.1,.2], [.3,.4], [.6,.7], [1.0,1.2], [1.4,1.6],[2.4,2.6] ]).t()
                self.ndt_layer  = [0,0,0,0,0,0]
                self.nt_layer   = [4,3,3,3,2,1]

                self.na_range   = torch.LongTensor([ [3,5], [3,4], [3,4], [3,4], [2,2] ]).t()
                self.nanc_range       = torch.LongTensor([ [0,2],[0,2],[0,1],[0,1],[0,0]]).t()
            else:
                assert(False)

            self.nlayer = len(self.nt_layer)
            assert(self.ntasks == sum(self.ndt_layer+self.nt_layer))
            assert(self.nlayer==len(self.ndt_layer) and self.nlayer==self.na_range.shape[1]+1 and\
                    self.nlayer==self.rew_range.shape[1])
            # 2. fillout others
            tbias, self.tlist, self.dtlist = 0, [], []
            self.r_high, self.r_low = torch.zeros(2, self.ntasks)

            for lind in range(self.nlayer):
                nt, ndt = self.nt_layer[lind], self.ndt_layer[lind]
                self.tlist += list( range(tbias, tbias+nt) )
                self.dtlist += list( range(tbias+nt, tbias+nt+ndt) )

                high, low = self.rew_range[:,lind]
                self.r_high[tbias:tbias+nt+ndt].fill_(high)
                self.r_low[tbias:tbias+nt+ndt].fill_(low)

                tbias += (nt + ndt)
        else: # Mining
            max_layer = 10
            self.layer_range = [6, max_layer]
            self.ntask_range = torch.LongTensor([ [3,3], [4,4], [2,2], [3,3], [1,2], [2,3], [1,2], [1,2], [1,3], [1,2]   ]).t()
            self.na_range    = torch.LongTensor([        [4,4], [2,2], [3,3], [2,2], [2,3], [2,3], [2,3], [2,3], [2,3]   ]).t()
            self.noc_range = torch.LongTensor([[1,2]]).repeat(max_layer,1).t()
            self.nac_range = torch.LongTensor([[1,3]]).repeat(max_layer,1).t()
            self.nanc_range= torch.LongTensor([[0,0]]).repeat(max_layer,1).t()
            self.ndnp_range= torch.LongTensor([[0,0]]).repeat(max_layer,1).t()
            self.rew_range = torch.tensor([ [.1,.1], [.1,.2], [.3,.5], [.3,.5], [.8,1.0], [1.2,1.5], [1.8,2.5], [3.0,4.5], [4.5,5.5], [5.0,7.0]   ]).t()

    # rendering
    def draw_graph(self, env_name, g_ind, epi_ind = None):

        ######## change to single graph setting
        numP, numA = self.numP[0], self.numA[0]
        ANDmat, ORmat = self.ANDmat[0].numpy(), self.ORmat[0].numpy()

        ind_to_id = self.ind_to_id[0].tolist()
        rewards = self.rmag[0].tolist()
        num_level = len(numA)
        ########

        from graphviz import Digraph
        root = os.path.join(__PATH__, 'config', env_name)
        if epi_ind is None:
            filename='./render/temp/subtask_graph_GT_{}'.format(g_ind)
        else:
            filename='./render/temp/subtask_graph_index{}_epi{}'.format(g_ind, epi_ind)
        g = Digraph(comment='subtask graph', format='png', filename=filename)
        g.attr(nodesep="0.1", ranksep="0.2")
        g.node_attr.update(fontsize="14", fontname='Arial')
        # 1. add Or nodes in the first layer
        for ind in range(numP[0]):
            sub_id = ind_to_id[ind]
            label = '\n{:+1.2f}'.format(rewards[ind])
            g.node('OR'+str(ind), label, shape='rect', height="0.1",
                    width="0.1", image=root+'/subtask{:02d}.png'.format(sub_id))

        abias, obias = 0, numP[0]
        for lind in range(num_level):
            Na, No = numA[lind], numP[lind+1]
            Amat = ANDmat[abias:abias+Na]
            Omat = ORmat[obias:obias+No]
            # Add AND nodes
            for i in range(Na):
                Aind = i + abias
                g.node('AND'+str(Aind), "", shape='ellipse',
                       style='filled', width="0.3", height="0.2", margin="0")

            # Edge OR->AND
            left, right = Amat.nonzero()
            for i in range(len(left)):
                Aind = abias + left[i]
                Oind = right[i]
                if Amat[left[i]][right[i]] < 0:
                    g.edge('OR'+str(Oind), 'AND'+str(Aind),
                           style="dashed", arrowsize="0.7")
                else:
                    g.edge('OR'+str(Oind), 'AND'+str(Aind), arrowsize="0.7")

            # Add OR nodes
            for i in range(No):
                ind = i + obias
                sub_id = ind_to_id[ind]
                label = '\n{:+1.2f}'.format(rewards[ind])
                g.node('OR'+str(ind), label, shape='rect', height="0",
                        width="0", image=root+'/subtask{:02d}.png'.format(sub_id))

            # Edge AND->OR
            left, right = Omat.nonzero()
            for i in range(len(left)):
                Oind = obias + left[i]
                Aind = right[i]
                g.edge('AND'+str(Aind), 'OR'+str(Oind),
                       arrowsize="0.7", arrowhead="odiamond")
            abias += Na
            obias += No
        g.render()

    def __str__(self): #print
        return "n_subtask="+self.nb_subtask

def test():
    pass

if __name__ == '__main__':
    test()

