import os, time, csv, copy
from graph.graph_utils import dotdict
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import numpy as np

#from a2c_ppo_acktr.envs import VecNormalize
def _update_print_log(args, logs, it, num_updates, start, Score_tr, Score_tst, succ_tr, succ_tst, rollouts):
    #1. 
    s_tr    = Score_tr.cpu().mean().item()
    if type(Score_tst)==int:
        s_tst = Score_tst
    else:
        s_tst   = Score_tst.cpu().mean().item()
    Val     = rollouts.value_preds[0].cpu().mean().item()
    Return  = rollouts.rewards.cpu().sum(0).mean().item()
    total_inter = succ_tr*args.num_processes*args.tr_epi
    
    #2. update windows
    _update_window(logs.window.str, s_tr)
    _update_window(logs.window.stst, s_tst)
    _update_window(logs.window.val, Val)
    _update_window(logs.window.ret, Return)

    #3. update log
    logs.iter   = it
    logs.tot_iter = num_updates
    logs.inter_count = (logs.inter_count or 0) + total_inter
    logs.s_tr, logs.s_tst       = s_tr, s_tst
    logs.value, logs.ret        = Val, Return
    logs.succ_tr, logs.succ_tst = succ_tr, succ_tst
    
    #4. print
    if it % args.summary_interval==0 and it>1:
        elapsed_time = time.time() - start
        rem_time = (num_updates - it ) / it *elapsed_time
        fps = int(logs.inter_count / elapsed_time)
        fmt = "[{:-3d}/{}] T={:3d}K| Str={:1.2f}| Stst={:1.2f}| R={:1.02f}| V(s)={:1.2f}| Suc={:.1f}| Fps={}| Elp={:3.01f}| Rem={:3.01f}"
        print(fmt.format(it, num_updates, round(logs.inter_count/1000),
                        logs.window.str.mean(), logs.window.stst.mean(), 
                        logs.window.ret.mean(), logs.window.val.mean(), logs.succ_tst,
                        fps, elapsed_time/60.0, rem_time/60.0 ) )
    return logs

def _print_eval_log(args, logs, it, tot_it, start ):
    if it % args.summary_interval==0:
        elapsed_time = time.time() - start
        rem_time = (tot_it - it ) / it *elapsed_time
        if args.mode=='meta_eval':
            fmt = "[{:-3d}/{}]: Elp={:3.01f}| Rem={:3.01f} | Stst="
            print(fmt.format(it, tot_it, elapsed_time/60.0, rem_time/60.0 ), logs.stst/it )
        else:
            fmt = "[{:-3d}/{}]: Elp={:3.01f}| Rem={:3.01f} | Stst={:1.2f}"
            print(fmt.format(it, tot_it, elapsed_time/60.0, rem_time/60.0, logs.stst/it ) )

def _save_log(args, logs, agent, value_loss, action_loss, dist_entropy):
    n_iter = logs.iter
    num_updates = logs.tot_iter
    #1. save tensorboard summary
    if n_iter % args.summary_interval==0 and n_iter>1:
        if args.writer is None:
            args.writer = SummaryWriter(log_dir=args.run_path)
        _write_summary(logs, args.writer, agent, value_loss, action_loss, dist_entropy)

    #2. write csv
    if n_iter==1:
        with open(logs.csv_filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([n_iter, logs.s_tr, logs.s_tst, logs.ret])
    else:
        with open(logs.csv_filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([n_iter, logs.s_tr, logs.s_tst, logs.ret])

    #2. save for every interval-th episode or for the last epoch
    if args.save_flag and (n_iter % args.save_interval == 0 or n_iter == num_updates) and n_iter>1:
        save_path = os.path.join('trained_model', args.folder_name)
        try:
            os.makedirs(save_path)
        except OSError:
            pass
        # A really ugly way to save a model to CPU
        save_model = agent.actor_critic
        if args.cuda:
            save_model = copy.deepcopy(agent.actor_critic).cpu()
        torch.save(save_model, os.path.join(save_path, args.env_name + '_' + str(logs.iter) + ".pt"))

def _save_eval_log(args, logs, tot_it, value_loss=None, action_loss=None, dist_entropy=None):
    #1. 
    writer = SummaryWriter(log_dir=args.run_path)
    if args.mode=='meta_eval':
        s_tr    = logs.str/tot_it
        s_tst   = logs.stst/tot_it
        ret     = logs.ret/tot_it
        data = np.stack([np.arange(1,args.tr_epi+1), s_tr, s_tst, ret])
        for n_iter in range(0, args.num_updates, args.summary_interval):
            writer.add_scalar('performance/score_train', s_tr[-1], n_iter)
            writer.add_scalar('performance/score_test', s_tst[-1], n_iter)

        for n_iter in range(0, args.tr_epi):
            v, a, d = value_loss[n_iter].item()/tot_it, action_loss[n_iter].item()/tot_it, dist_entropy[n_iter].item()/tot_it
            writer.add_scalar('loss/total', v+a-d, n_iter)
            writer.add_scalar('loss/policy', a, n_iter)
            writer.add_scalar('loss/value', v, n_iter)
            writer.add_scalar('loss/entropy', -d, n_iter)
    else: # random / grprop
        s_tst   = logs.stst/tot_it
        data = np.array([[0, 0, s_tst, 0]]).T

        for n_iter in range(0, args.num_updates, args.summary_interval):
            writer.add_scalar('performance/score_test', s_tst, n_iter)
    
    #2. write csv
    np.savetxt(logs.csv_filename, data.T, delimiter=",")
    
def _write_summary(logs, writer, agent, value_loss, action_loss, dist_entropy):
    n_iter = logs.iter
    writer.add_scalar('loss/total', value_loss+action_loss-dist_entropy*agent.entropy_coef, n_iter)
    writer.add_scalar('loss/policy', action_loss, n_iter)
    writer.add_scalar('loss/value', value_loss, n_iter)
    writer.add_scalar('loss/entropy', -dist_entropy*agent.entropy_coef, n_iter)
    if agent.algo=='a2c':
        writer.add_scalar('optim/lr', agent.optimizer.param_groups[0]['lr'], n_iter)
        writer.add_scalar('optim/rho', agent.entropy_coef, n_iter)
        writer.add_scalar('optim/entropy', dist_entropy, n_iter)
    else:
        writer.add_scalar('optim/lr', 0, n_iter)
        writer.add_scalar('optim/rho', 0, n_iter)
        writer.add_scalar('optim/entropy', 0, n_iter)
    writer.add_scalar('performance/score_train', logs.s_tr, n_iter)
    writer.add_scalar('performance/score_test', logs.s_tst, n_iter)
    writer.add_scalar('performance/return', logs.ret, n_iter)    
    writer.add_scalar('performance/init_value', logs.value, n_iter)
    writer.add_scalar('performance/Success_test', logs.succ_tst, n_iter)

def prepare_logging(args):
    # 1. naming
    args.writer = None
    args.folder_name = args.method + '_'
    if args.env_name=='mining':
        args.folder_name = 'Mine_' + args.folder_name
    if args.method=='SGI' and args.algo in ['a2c', 'ppo']: # SGI-meta / SGI-meta-eval
        if args.bonus_mode==0:
            if not args.load_dir=='': # finetuning
                args.suffix = '_ext_'
            else:
                args.suffix = '_ext_'
        elif args.bonus_mode==1:
            args.suffix = '_uniform_'
        elif args.bonus_mode==2:
            args.suffix = '_UCB_'
        else:
            assert(False)
    else:
        args.suffix = '_'
    
    import socket
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    if args.env_name=='playground':
        args.suffix+= 'lv' +str(args.level) +'_'
    args.suffix+= 'epi'+str(args.tr_epi)+'_'
    if args.mode =='meta_eval':
        args.folder_name += 'eval_'
    args.folder_name += args.algo + args.suffix + str(args.exp_id) + str(args.seed)
    #1 tensorboard dir
    args.run_path = os.path.join('runs', args.folder_name + '_' + current_time + '_' + socket.gethostname())
    logs = dotdict()

    #2 print summary
    window = dotdict()
    window.val, window.ret, window.str, window.stst = np.zeros((4, args.summary_interval))
    logs.window = window

    #3 log
    save_path = os.path.join('logs', args.folder_name)
    try:
        os.makedirs(save_path)
    except OSError:
        pass
    if args.mode=='meta_train':
        logs.csv_filename = os.path.join('logs', args.folder_name, 'log.csv')
    else:
        logs.csv_filename = os.path.join('logs', args.folder_name, 'eval_log.csv')
    return logs

def _anneal_param(vst, ved, t, tot, tst, ted):
    progress = t / tot
    clamped_progress = min(max( (progress - tst) / (ted-tst), 0.0), 1.0)
    return vst + (ved - vst) * clamped_progress

def _update_window(window, new_value):
    if len(window.shape)==1:
        window[:-1] = window[1:]
        window[-1] = new_value
    else:
        window[:,:-1] = window[:,1:]
        window[:,-1] = new_value
    return window
    
# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None

"""
def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None"""

# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def init_gru(module, weight_init, bias_init, gain=1):
    weight_init(module.weight_ih_l0.data, gain=gain)
    weight_init(module.weight_hh_l0.data, gain=gain)
    bias_init(module.bias_ih_l0.data)
    bias_init(module.bias_hh_l0.data)
    return module

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
    elif isinstance(m, nn.Linear):
        init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
    elif isinstance(m, nn.GRU):
        init_gru(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))