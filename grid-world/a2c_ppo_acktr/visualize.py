# Copied from https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/visualize_atari.py
# and https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/load.py
# Thanks to the author and OpenAI team!
"""
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from scipy.signal import medfilt
matplotlib.rcParams.update({'font.size': 8})

"""
import os
from graphviz import Digraph

def render_dot_graph(graph, env_name, algo, folder_name, g_ind, is_batch=False, epi_ind=None, niter=None):
    ######## change to single graph setting
    if is_batch:
        numP, numA = graph.numP[0], graph.numA[0]
        ANDmat, ORmat = graph.ANDmat[0].numpy(), graph.ORmat[0].numpy()
        
        ind_to_id = graph.ind_to_id[0].tolist()
        rewards = graph.rmag[0].tolist()
        tind_by_layer = []
        bias = 0
        for num in numP:
            layer = list(range(bias, bias+num))
            tind_by_layer.append(layer)
            bias += num
        tind_list = list(range(len(ind_to_id)))
    else:
        numP, numA = graph.numP, graph.numA
        ANDmat, ORmat = graph.ANDmat.numpy(), graph.ORmat.numpy()
        
        ind_to_id = graph.ind_to_id.tolist()
        rewards = graph.rmag.tolist()
        tind_by_layer = graph.tind_by_layer
        tind_list = graph.tind_list
        tind_reverse_map = {v: i for i, v in enumerate(tind_list)}

    num_level = len(numA)
    ########
    __PATH__ = os.path.abspath(os.path.dirname(__file__))
    root = os.path.join(__PATH__, '../environment/config', env_name)
    if epi_ind is None:
        #filename='./render/{}_{}/{}/subtask_graph_GT_index{}'.format(env_name, algo, folder_name, g_ind)
        filename='./render/{}/GT/graph_GT_index{}'.format(env_name, g_ind)
    else:
        #filename='./render/{}_{}/{}/subtask_graph_index{}_epi{}'.format(env_name, algo, folder_name, g_ind, epi_ind)
        if niter is None:
            filename='./render/{}/{}/graph_last_index{}'.format(env_name, algo, g_ind)
        else:
            filename='./render/{}/{}/graph_last_index{}_{}'.format(env_name, algo, g_ind, niter)
    g = Digraph(comment='subtask graph', format='png', filename=filename)
    g.attr(nodesep="0.1", ranksep="0.2")
    g.node_attr.update(fontsize="14", fontname='Arial')

    # In mining, we don't shuffle index. So, index is roughly same with id
    # tind_by_layer is correct. 
    # 1. add Or nodes in the first layer
    # ORmat's first dim is 
    for ind in tind_by_layer[0]:
        sub_id = ind_to_id[ind]
        label = '\n{:+1.2f}'.format(rewards[ind])
        with g.subgraph() as c:
            c.attr(rank='same')
            c.node('OR'+str(ind), label, shape='rect', height="0.1",
                width="0.1", image=root+'/subtask{:02d}.png'.format(sub_id))

    abias, obias = 0, numP[0]
    for lind in range(num_level):
        Na, No = numA[lind], numP[lind+1]
        Amat = ANDmat[abias:abias+Na]
        Omat = ORmat[obias:obias+No]
        with g.subgraph() as c:
            # Add AND nodes
            for i in range(Na):
                Aind = i + abias
                if Amat[Aind].nonzero().sum()>0:
                    c.node('AND'+str(Aind), "", shape='ellipse',
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
        with g.subgraph() as c:
            for i in range(No):
                Oind = i + obias
                sub_ind = tind_list[Oind]
                sub_id = ind_to_id[sub_ind]
                label = '\n{:+1.2f}'.format(rewards[sub_ind])
                c.node('OR'+str(sub_ind), label, shape='rect', height="0",
                        width="0", image=root+'/subtask{:02d}.png'.format(sub_id))

        # Edge AND->OR
        left, right = Omat.nonzero()
        for i in range(len(left)):
            Oind = obias + left[i] # id = ind_to_id[Oind]
            Aind = right[i]
            g.edge('AND'+str(Aind), 'OR'+str(Oind),
                    arrowsize="0.7", arrowhead="odiamond")
        abias += Na
        obias += No
    g.render()


"""


def smooth_reward_curve(x, y):
    # Halfwidth of our smoothing convolution
    halfwidth = min(31, int(np.ceil(len(x) / 30)))
    k = halfwidth
    xsmoo = x[k:-k]
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='valid') / \
        np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='valid')
    downsample = max(int(np.floor(len(xsmoo) / 1e3)), 1)
    return xsmoo[::downsample], ysmoo[::downsample]


def fix_point(x, y, interval):
    np.insert(x, 0, 0)
    np.insert(y, 0, 0)

    fx, fy = [], []
    pointer = 0

    ninterval = int(max(x) / interval + 1)

    for i in range(ninterval):
        tmpx = interval * i

        while pointer + 1 < len(x) and tmpx > x[pointer + 1]:
            pointer += 1

        if pointer + 1 < len(x):
            alpha = (y[pointer + 1] - y[pointer]) / \
                (x[pointer + 1] - x[pointer])
            tmpy = y[pointer] + alpha * (tmpx - x[pointer])
            fx.append(tmpx)
            fy.append(tmpy)

    return fx, fy


def load_data(indir, smooth, bin_size):
    datas = []
    infiles = glob.glob(os.path.join(indir, '*.monitor.csv'))

    for inf in infiles:
        with open(inf, 'r') as f:
            f.readline()
            f.readline()
            for line in f:
                tmp = line.split(',')
                t_time = float(tmp[2])
                tmp = [t_time, int(tmp[1]), float(tmp[0])]
                datas.append(tmp)

    datas = sorted(datas, key=lambda d_entry: d_entry[0])
    result = []
    timesteps = 0
    for i in range(len(datas)):
        result.append([timesteps, datas[i][-1]])
        timesteps += datas[i][1]

    if len(result) < bin_size:
        return [None, None]

    x, y = np.array(result)[:, 0], np.array(result)[:, 1]

    if smooth == 1:
        x, y = smooth_reward_curve(x, y)

    if smooth == 2:
        y = medfilt(y, kernel_size=9)

    x, y = fix_point(x, y, bin_size)
    return [x, y]


color_defaults = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]


def visdom_plot(viz, win, folder, game, name, num_steps, bin_size=100, smooth=1):
    tx, ty = load_data(folder, smooth, bin_size)
    if tx is None or ty is None:
        return win

    fig = plt.figure()
    plt.plot(tx, ty, label="{}".format(name))

    tick_fractions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    ticks = tick_fractions * num_steps
    tick_names = ["{:.0e}".format(tick) for tick in ticks]
    plt.xticks(ticks, tick_names)
    plt.xlim(0, num_steps * 1.01)

    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')

    plt.title(game)
    plt.legend(loc=4)
    plt.show()
    plt.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close(fig)

    # Show it in visdom
    image = np.transpose(image, (2, 0, 1))
    return viz.image(image, win=win)


if __name__ == "__main__":
    from visdom import Visdom
    viz = Visdom()
    visdom_plot(viz, None, '/tmp/gym/', 'BreakOut', 'a2c', bin_size=100, smooth=1)
"""