import os
import torch
import numpy as np

__PATH__ = os.path.abspath(os.path.dirname(__file__))

class SubtaskGraph(object):
    def __init__(self, args, nb_subtask=0):
        #1. init/load graph
        folder = args.graph_config['folder']
        gamename = args.graph_config['gamename']
        self._load_graph(folder, gamename) # subtask_list / edges (ANDmat&ORmat) / subtask reward
        self.nbatch = args.num_processes
        self.game_config = args.game_config
        self.max_task = self.game_config.nb_subtask_type

    def _load_graph(self, folder, gamename):
        fname = os.path.join(folder, gamename+'.pkl' )
        print('loading graph @ ' +fname)
        self.graph_list = torch.load( fname )
        self.num_graph = len(self.graph_list)

    def set_graph_index(self, graph_index):
        nbatch = self.nbatch
        graph = self.graph_list[graph_index]
        # 1. ANDmat, ORmat, rmag, num's
        self.max_NA, self.max_NP = graph['ANDmat'].shape
        self.ntasks = self.max_NP
        self.ANDmat = graph['ANDmat'].long().repeat(nbatch, 1, 1)
        self.ORmat  = graph['ORmat'].long().repeat(nbatch, 1, 1)
        self.rmag = torch.tensor(graph['rmag']).repeat(nbatch, 1)

        # 2. id_mask, id_to_ind, ind_to_id
        self.id_mask    = torch.zeros( nbatch, self.max_task )
        self.id_to_ind  = torch.LongTensor( nbatch, self.max_task ).fill_(-1)

        id_tensor = graph['trind'] - 1
        self.ind_to_id  = id_tensor.repeat(nbatch, 1)
        self.id_mask.index_fill_(1, id_tensor, 1)
        base = torch.arange(self.ntasks).repeat(nbatch, 1)
        self.id_to_ind.scatter_(1, self.ind_to_id, base)

        # 3. W_a, W_o
        self.W_a, self.W_o = graph['W_a'], graph['W_o']
        self.num_level = len(self.W_a)
        self.nlayer = self.num_level+1
        tbias = self.W_a[0].shape[1]
        nump, numa, tlayer=[ tbias ], [], [ [*range(tbias)] ]
        self.num_or, self.num_and = nump[0], 0
        for lv in range(self.num_level):
            nt = self.W_o[lv].shape[0]
            nump.append( nt )
            numa.append(self.W_a[lv].shape[0])
            self.num_or = self.num_or + nump[lv + 1]
            self.num_and = self.num_and + numa[lv]
            tlayer.append([*range(tbias, tbias+nt)])
            tbias += nt
        self.numP = [ [elem for elem in nump] for i in range(nbatch) ]
        self.numA = [ [elem for elem in numa] for i in range(nbatch) ]
        self.tind_by_layer = [ [elem for elem in tlayer] for i in range(nbatch) ]

        #4. b_AND & b_OR
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

    def draw_graph(self, env_name, g_ind, epi_ind = None):
        rewards = self.rmag
        from graphviz import Digraph
        root = os.path.join(__PATH__, 'config', env_name)
        if epi_ind is None:
            filename='./render/temp/subtask_graph_GT'
        else:
            filename='./render/temp/subtask_graph_index{}_epi{}'.format(g_ind, epi_ind)
        g = Digraph(comment='subtask graph', format='png', filename=filename)
        g.attr(nodesep="0.1", ranksep="0.2")
        g.node_attr.update(fontsize="14", fontname='Arial')
        # 1. add Or nodes in the first layer
        for ind in range(self.numP[0]):
            sub_id = self.ind_to_id[ind]
            label = '\n{:+1.2f}'.format(rewards[ind])
            g.node('OR'+str(ind), label, shape='rect', height="0.1",
                    width="0.1", image=root+'/subtask{:02d}.png'.format(sub_id))

        abias, obias = 0, self.numP[0]
        for lind in range(self.num_level):
            Na, No = self.numA[lind], self.numP[lind+1]
            Amat = self.ANDmat[abias:abias+Na]
            Omat = self.ORmat[obias:obias+No]
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
                sub_id = self.ind_to_id[ind]
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

def test():
    pass

if __name__ == '__main__':
    test()

