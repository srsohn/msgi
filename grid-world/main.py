import os, time, math, copy
from collections import deque
os.environ["OMP_NUM_THREADS"] = "24"
import numpy as np

from graph.ILP import ILP
from graph.grprop import GRProp
from graph.graph_utils import dotdict
from a2c_ppo_acktr import algo
from a2c_ppo_acktr.arguments import get_args

from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage_feat
from a2c_ppo_acktr.utils import _anneal_param, weights_init, \
                                _update_print_log, _save_log, prepare_logging, \
                                _save_eval_log, _print_eval_log
from environment.batch import Batch_env

args = get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) # this need to be done before importing pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F

if args.seed > 0:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
args.save_interval = max(args.num_updates//5, 1)
if args.algo in ['random', 'grprop']:
    args.cuda = False
    args.save_flag = False
else:
    args.cuda = (not args.no_cuda and torch.cuda.is_available())
    if args.mode=='meta_train':
        args.save_flag = True
        print('saving model every ', args.save_interval)
    else:
        args.save_flag = False
print('torch.cuda.is_available()=',torch.cuda.is_available())
print('args.cuda=',args.cuda)
device = torch.device("cuda:0" if args.cuda else "cpu")

def main():
    assert args.algo in ['a2c', 'random', 'grprop']
    args.summary_interval = 2

    print('device=',device)
    print('num-processes=',args.num_processes)
    logs = prepare_logging(args)

    envs = Batch_env(args) # env_name, num_processes, gamma.
    args.act_dim = envs.action_space.n

    model_args = {'recurrent': args.recurrent_policy}
    if args.gru_ldim > 0:
        model_args['gru_ldim']=args.gru_ldim
    if args.flat_ldim > 0:
        model_args['flat_ldim']=args.flat_ldim

    if args.method in ['SGI', 'rlrl'] and args.algo in ['a2c'] and not args.load_dir=='':
        save_path = os.path.join('trained_model', args.load_dir)
        load_path = os.path.join(save_path, args.env_name + '_' + str(args.load_epi) + ".pt")
        actor_critic = torch.load( load_path )
        print('Loaded model @ '+load_path)
    else:
        actor_critic = Policy(envs.observation_space.shape, envs.feat_dim, envs.action_space, base_kwargs=model_args)
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.v_coef,
                               args.rho_v_st, lr=args.lr, lr_decay=args.lr_decay,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'random':
        agent = algo.Random(args)
    elif args.algo == 'grprop':
        agent = None
    #
    assert args.env_name=='playground' or args.env_name=='mining'
    rollouts = RolloutStorage_feat( args.ntasks*args.tr_epi+1, args.num_processes,
                    envs.observation_space.shape, envs.feat_dim, envs.action_space, args.ntasks,
                    actor_critic.recurrent_hidden_state_size)
    rollouts.to(device)
    #
    ilp = None
    if args.infer:
        ilp = ILP(args)
    if args.mode == 'meta_train': # Train sgi-meta / rl2
        assert( args.algo in ['a2c'] )
        assert( args.train )
        args.meta = True
        meta_train( args, envs, agent, rollouts, ilp, logs )
    elif args.mode == 'meta_eval': # sgi-random / hrl-baseline / eval of meta-agent / eval of rl2
        assert( not (args.infer and args.train))
        if args.method =='rlrl':
            args.tr_epi += 1 # to use S_tr as S_tst
        logs.str, logs.stst, logs.act_tst, logs.ret = np.zeros( (4, args.tr_epi) )
        meta_eval(  args, envs, agent, rollouts, ilp, logs )
    elif args.mode == 'eval':
        assert( args.train==False and args.infer==False and args.algo in ['random', 'grprop'])
        evaluate( args, envs, agent, logs )
    else:
        assert(False)

def meta_train(args, envs, agent, rollouts, ilp, logs):  # train sgi-meta / train rlrl
    start = time.time()
    num_updates = args.num_updates
    for it in range(num_updates):
        # 0. update optimizer params
        agent.entropy_coef = _anneal_param(args.rho_v_st, args.rho_v_ed, it, num_updates, args.rho_t_st, args.rho_t_ed)

        # 1. Adaptation phase. Ignore env reward (16%)
        Score_tr, act_tr, rollouts, ilp, _ = rollout_trial(args, envs, agent, args.tr_epi, \
                                                    train=args.train, rollouts=rollouts, \
                                                    infer=args.infer, ilp=ilp, reset_task=True)
        # 2. Infer the graph (52%)
        if args.infer:
            graphs = ilp.infer_graph()
            test_agent = GRProp(graphs, args)
        else:
            test_agent = agent

        if args.method == 'rlrl':
            Score_test, act_tst = torch.zeros(args.num_processes), 0
        else:
            # 3. Testing phase ( get reward ) (32%)
            Score_test, act_tst, _, _, _ = rollout_trial(args, envs, test_agent, args.test_epi)
            if args.bonus_mode == 0:
                rollouts.pad_last_step_reward(Score_test)

        # 4. Update model (<1%)
        rollouts.compute_returns(args.use_gae, args.gamma, args.tau)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        # 5. logging (0%)
        logs = _update_print_log(args, logs, it+1, num_updates, start, Score_tr, Score_test, act_tr, act_tst, rollouts)
        _save_log(args, logs, agent, value_loss, action_loss, dist_entropy) # save summary, log, model

        rollouts.after_update()

def meta_eval(args, envs, agent, rollouts, ilp, logs):
    start = time.time()
    # sgi-random/rnn-baseline/ evaluation of meta-agent/rl2
    # - eval rl2: train - x infer - x
    # - sgi-random & eval meta-agent: train - x, infer - o
    # - hrl-baseline: train - o, infer - x
    rnn_h = None
    value_loss, action_loss, dist_entropy = np.zeros( (3, args.tr_epi) )
    if args.algo=='a2c':
        rnn_h = torch.zeros(args.num_processes, agent.actor_critic.recurrent_hidden_state_size, device=device) # for hrl, eval-meta, eval-rl2
    prec, rec = np.zeros( [2, args.tr_epi] )
    for g_ind in range(envs.num_graphs):
        # 0. init trial
        if args.train:# hrl-baseline
            assert(args.method=='baseline')
            agent.init_optimizer()  # init optimizer (lr)
            agent.actor_critic.apply(weights_init) # initialize model
        if not rnn_h is None: # eval sgi-meta/eval-rl2
            rnn_h.zero_()
        for epi_ind in range(args.tr_epi):
            # 1. training trial (episode)
            if args.method=='baseline': # hrl only. reset hidden state every episode
                rnn_h.zero_()
                agent.entropy_coef = _anneal_param(args.rho_v_st, args.rho_v_ed, epi_ind, args.tr_epi, args.rho_t_st, args.rho_t_ed)
            args.global_epi = epi_ind
            Score_tr, act_tr, rollouts, ilp, rnn_h = rollout_trial(args, envs, agent, 1, rnn_h=rnn_h,\
                                                        train=args.train, rollouts=rollouts, \
                                                        infer = args.infer, ilp=ilp, reset_task = (epi_ind==0), gind=g_ind)
            if args.train:
                # 2-1. update model
                rollouts.compute_returns(args.use_gae, args.gamma, args.tau)
                vloss, aloss, dloss = agent.update(rollouts)
                value_loss[epi_ind]     += vloss
                action_loss[epi_ind]    += aloss
                dist_entropy[epi_ind]   += dloss * agent.entropy_coef
                rollouts.after_update()

            # inference
            if args.infer:
                assert(torch.all(ilp.reward_count.max(1)[0]<=epi_ind+1) )
                graphs = ilp.infer_graph()
                test_agent = GRProp(graphs, args)
            else:
                test_agent = agent

            if args.draw_graph:
                from a2c_ppo_acktr.visualize import render_dot_graph
                folder_name = 'graph{}'.format(g_ind)
                #if epi_ind==0:
                    #envs.render_graph(env_name=args.env_name, algo=args.algo, folder_name=folder_name, g_ind=g_ind) # --> 'batch.py' --> 'batch_graph.py'
                if epi_ind == args.tr_epi-1:
                    render_dot_graph(graph=graphs[0], env_name=args.env_name, algo=args.algo, folder_name=folder_name, \
                                    g_ind=g_ind, is_batch=False, epi_ind=epi_ind, niter=niter)
            if args.eval_graph:
                from a2c_ppo_acktr.visualize import eval_graph
                p, r = eval_graph(envs.graph, graphs[0])
                prec[epi_ind] += p
                rec[epi_ind] += r

            # 3. Eval tst score
            if not args.method == 'rlrl': # No need to eval. Just use next-step S_tr as S_tst
                Score_tst, act_tst, _, _, _ = rollout_trial(args, envs, test_agent, args.test_epi)

                # 4. record
                logs.stst[epi_ind]  += Score_tst.cpu().mean().item()
                logs.act_tst[epi_ind] += act_tst
            logs.str[epi_ind]   += Score_tr.cpu().mean().item()
        if args.method == 'rlrl':
            logs.stst[:-1] = logs.str[1:]
        _print_eval_log(args, logs, g_ind+1, envs.num_graphs, start)
    if args.eval_graph:
        csv_filename = args.env_name +'_PR_graph.csv'
        with open(csv_filename, 'w') as f:
            writer = csv.writer(f)
            for i in range(args.tr_epi):
                writer.writerow([i, prec[i], rec[i]])

    _save_eval_log(args, logs, envs.num_graphs, value_loss, action_loss, dist_entropy)

def evaluate( args, envs, agent, logs ):
    start = time.time()
    # Random agent
    logs.stst, logs.act_tst = 0, 0
    for g_ind in range(envs.num_graphs):
        # 1. rollout
        if args.algo == 'grprop':
            _, _ = envs.reset_trial(nb_epi=args.test_epi, reset_graph=True, graph_index=g_ind)
            graphs = envs.get_graphs()
            agent = GRProp(graphs, args)
            Score_tst, act_tst, _, _, _ = rollout_trial(args, envs, agent, args.test_epi) # leave the graph same.
        else:  # Random
            Score_tst, act_tst, _, _, _ = rollout_trial(args, envs, agent, args.test_epi, reset_task=True, gind=g_ind)

        # 2. record
        logs.stst       += Score_tst.cpu().mean().item()
        logs.act_tst    += act_tst
        _print_eval_log(args, logs, g_ind+1, envs.num_graphs, start )

    _save_eval_log(args, logs, envs.num_graphs)

def rollout_trial(args, envs, agent, nb_epi, rnn_h=None, train=False, rollouts=None, infer=False, ilp=None, reset_task=False, gind=-1):
    assert(not train or not rollouts is None)
    assert(not infer or not ilp is None)
    act_sum, score, active, step_done = 0, 0, torch.LongTensor([[1.0]]*args.num_processes), torch.LongTensor([[0.0]]*args.num_processes)
    obs, feats = envs.reset_trial(nb_epi=nb_epi, reset_graph=reset_task, graph_index=gind)
    if infer: #sgi-random / eval of meta-agent
        if reset_task:
            ilp.reset(envs)
        _, tp_ind, elig_ind = envs.get_indexed_states() # only for inference
        ilp.insert(active, step_done, tp_ind, elig_ind )
    if train:
        rollouts.init_state(obs, feats)
    elif agent.algo == 'a2c':
        if rnn_h is None:
            rnn_h = torch.zeros(args.num_processes, agent.actor_critic.recurrent_hidden_state_size, device=device) # only for hrl
    for step in range(args.ntasks * nb_epi): # unroll an episode
        ## Sample actions
        if args.verbose > 0:
            print('======  step = ', step)
        if agent.algo == 'a2c':
            if train:
                with torch.no_grad():
                    value, action, action_log_prob, rnn_h = agent.actor_critic.act(
                            rollouts.active[step], rollouts.obs[step], rollouts.feats[step],
                            rollouts.recurrent_hidden_states[step])
            else:
                with torch.no_grad():
                    _, action, _, rnn_h = agent.actor_critic.act(
                            active.to(device), obs.float().to(device), feats.to(device), rnn_h)
        elif agent.algo == 'random':
            action = agent.act(active, feats)
        elif agent.algo == 'grprop':
            mask_ind, tp_ind, elig_ind = envs.get_indexed_states()
            action = agent.act(active, mask_ind, tp_ind, elig_ind, eval_flag=True)
        else:
            assert(False)

        # 2. env step
        obs, feats, reward, active, time_cost = envs.step(action) # 58% of step1 (train trial)
        if 'global_epi' in args: # quick&dirty: for meta_eval, modify #epi manually
            feats[:, envs.max_task*3+1] = math.log10(args.tr_epi +1 - args.global_epi)
        score += reward
        act_sum += active.cpu().sum().item()

        # 3. record
        if infer:
            prev_active, step_done, mask_ind, tp_ind, elig_ind = envs.get_delayed_indexed_states() # only for inference
            ilp.insert(prev_active, step_done, tp_ind, elig_ind, action, reward )
            reward = ilp.compute_bonus(prev_active, step_done, tp_ind, elig_ind)
        if train:
            rollouts.insert(obs, feats, rnn_h, action, action_log_prob, value, reward, active)

        if active.sum(0).item()==0:
            break

    score/=nb_epi
    act_sum/=nb_epi*args.num_processes
    return score, act_sum, rollouts, ilp, rnn_h

if __name__ == "__main__":
    main()
