import os
import time
import argparse
import torch
import shutil
import numpy as np
from absl import flags

from common import Config, EnvWrapper
from common.env import make_envs
from common.file_writer import SC2FileWriter
from common.sc2_utils import MAP_INFO
from runner import Runner

from pysc2.env import sc2_env


if __name__ == '__main__':
    torch.set_printoptions(precision=10)
    flags.FLAGS(['main.py'])
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--verbose", type=bool, nargs='?', const=True, default=False)
    parser.add_argument("--meta", type=str, default='random', choices=['random', 'explore', 'hrl','MSGI'])
    parser.add_argument("--tr_epi", type=int, default=20)
    parser.add_argument("--warfare", type=bool, nargs='?', const=True, default=False)
    parser.add_argument("--prep_time", type=int, default=20, help="War preparation time in (game time) minutes")
    parser.add_argument("--sp_hdim", type=int, default=32)
    parser.add_argument("--fl_hdim", type=int, default=128)
    parser.add_argument("--act_hdim", type=int, default=256)
    parser.add_argument("--init_ep", type=int, default=0)
    parser.add_argument("--num_iter", type=int, default=1)
    parser.add_argument("--max_noop", type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--decay_rate', type=float, default=0.99)
    parser.add_argument('--decay_steps', type=int, default=20)
    parser.add_argument('--vf_coef', type=float, default=0.25)
    parser.add_argument('--sc_weight', type=float, default=84*84)

    parser.add_argument('--e1t0', type=int, default=50) # entropy regularization for action_id
    parser.add_argument('--e1t1', type=int, default=100)
    parser.add_argument('--e1v0', type=float, default=1.)
    parser.add_argument('--e1v1', type=float, default=0.)

    parser.add_argument('--term_coef', type=float, default=1.0)
    parser.add_argument('--discount', type=float, default=0.98)
    parser.add_argument('--step_penalty', type=float, default=-0.05)
    parser.add_argument('--clip_grads', type=float, default=None)
    parser.add_argument("--run_id", type=int, default=-1)
    parser.add_argument("--log_id", type=int, default=-1)
    parser.add_argument("--spec", type=str, default='')
    parser.add_argument("--debug", type=bool, nargs='?', const=True, default=False)
    parser.add_argument("--map", type=str, default='Warfare_Zerglings_20')
    parser.add_argument("--eval", type=bool, nargs='?', const=True, default=False)
    parser.add_argument("--nhwc", type=bool, nargs='?', const=True, default=False)
    parser.add_argument("--restore", type=int, default=-1)
    parser.add_argument('--save_replay', type=bool, nargs='?', const=True, default=False)
    parser.add_argument('--save_ilp', type=bool, nargs='?', const=True, default=False)
    parser.add_argument('--load_ilp', type=bool, nargs='?', const=True, default=False)

    # sc2
    parser.add_argument("--log_device", action='store_true', default=False)
    parser.add_argument('--feature_screen_size', help='Resolution for screen feature layers.', type=int, default=84)
    parser.add_argument('--feature_minimap_size', help='Resolution for minimap feature layers.', type=int, default=64)
    parser.add_argument('--rgb_screen_size', help='Resolution for rendered screen.', type=bool, default=None)
    parser.add_argument('--rgb_minimap_size', help='Resolution for rendered minimap.', type=bool, default=None)
    parser.add_argument('--action_space', help='Which action space to use. Needed if you take both feature.', type=str, choices=sc2_env.ActionSpace._member_names_, default=None)
    parser.add_argument('--agent_race', help="Agent 1's race.", type=str, choices=sc2_env.Race._member_names_, default='terran')
    parser.add_argument('--use_feature_units', help='Whether to include feature units.', type=bool, default=True)
    parser.add_argument('--use_raw_units', help='Whether to include feature units.', type=bool, default=True)
    parser.add_argument('--use_unit_counts', help='Whether to include feature units.', type=bool, default=True)
    parser.add_argument('--step_mul', help='Game steps per agent step.', type=int, default=8)
    parser.add_argument('--disable_fog', help='Whether to disable Fog of War.', type=bool, default=False)
    parser.add_argument('--save_replay_episodes', help='Save replay after this many episodes.', type=int, default=0)
    parser.add_argument('--visualize', help='Whether to visualize with pygame.', type=bool, default=False)
    parser.add_argument('--replay_dir', help='Where to save the replay buffer.', type=str, default='PySC2Replays')
    parser.add_argument('--seed', help='RNG seed.', type=int, default=None)
    parser.add_argument('--num_timesteps', type=int, default=2e3)
    args = parser.parse_args()
    assert args.envs == 1, "The current version only supports single environment."

    # check map type
    if args.map.startswith('Warfare'):
        args.warfare = True

    opt_dim = len(MAP_INFO)
    args.obs_shape = [2, 84, 84]
    args.attr_dim = opt_dim
    args.feat_dim = 2 * opt_dim + args.attr_dim + 1
    args.action_dim = opt_dim

    # max_step from float --> int
    if len(args.map) > 5 and (args.map.startswith('Build')):
        args.max_step = int(args.num_timesteps / args.step_mul)
    else:
        args.max_step = int(args.num_timesteps / args.step_mul)
    args.game_steps_per_episode = args.max_step * args.step_mul

    # pytorch device setup
    processor = 'cpu' if args.eval or args.gpu == -1 else 'cuda:%s'%args.gpu
    device = torch.device(processor)
    data_format = "NHWC" if args.nhwc else "NCHW"

    config = Config(args, device, data_format)
    os.makedirs('weights/' + config.full_id(), exist_ok=True)
    with open('weights/' + config.full_id() + '/arguments.txt', 'w') as f:
        for key,value in vars(args).items():
            string = str(key) + ' : ' + str(value) + '\n'
            f.write( string )

    # make/clean directory for save mode
    dirname = os.path.join('results', args.map, args.meta, 'run_%d'%(args.run_id))
    if args.save_ilp or (args.meta in ['random', 'hrl'] and not args.eval):
        if os.path.exists(dirname):
            print("The savepath already exists. Deleting {}".format(dirname))
            shutil.rmtree(dirname)
        os.makedirs(dirname, exist_ok=True)

    # set up agent & ilp
    ilp = None
    if args.meta == 'MSGI':
        from agents.ILP import ILP
        ilp = ILP(args, dirname)
        agent = None
        args.infer = True
        args.train = False
    elif args.meta == 'random':
        from agents.random import Random
        agent = Random(args, device)
        args.infer = False
        args.train = False
    elif args.meta == 'explore':
        from agents.exploration import Explore
        agent = Explore(args, device)
        args.infer = False
        args.train = False
    elif args.meta == 'hrl':
        from agents.hrl import HRL
        from agents.model import FullyConv
        actor_critic = FullyConv(config, device, args.envs)
        agent = HRL(actor_critic, config, args)
        args.infer = False
        args.train = True
    else:
        raise ValueError

    # initialize training setups
    envs = EnvWrapper(make_envs(args), config, args, device)
    sc2_filewriter = SC2FileWriter(args, dirname)
    runner = Runner(envs, agent, ilp, sc2_filewriter, config, args, device, dirname)
    time.sleep(5)  # wait for SC2 to boot up

    # run eval or train
    if args.eval and not args.load_ilp:  # hrl, random
        runner.eval(num_iter=args.num_iter)
    else:
        if args.meta=='MSGI' or args.meta=='hrl': # MSGI, HRL
            if args.load_ilp:
                runner.meta_eval_load(dirname, init_ep=args.init_ep)
            else:
                runner.meta_eval_save(num_iter=args.num_iter, tr_epi=args.tr_epi)

    # save game replay
    if args.save_replay:
        prefix = "{}-{}_ID-{}".format(args.meta, args.map, args.run_id)
        print("Saving game replays to : {}".format(args.replay_dir))
        envs.save_replay(args.replay_dir, prefix)

    envs.close()
