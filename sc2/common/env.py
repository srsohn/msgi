from pysc2.env import sc2_env
from pysc2.lib.actions import FUNCTIONS
from multiprocessing import Process, Pipe

# no_op function
_NO_OP = FUNCTIONS.no_op()

def make_envs(args):
    return EnvPool([make_env(args) for i in range(args.envs)])

# based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
def make_env(args):
    def _thunk():
        players = list()
        players.append(sc2_env.Agent(sc2_env.Race[args.agent_race]))
        AIF = sc2_env.parse_agent_interface_format(
            feature_screen=args.feature_screen_size,
            feature_minimap=args.feature_minimap_size,
            rgb_screen=args.rgb_screen_size,
            rgb_minimap=args.rgb_minimap_size,
            action_space=args.action_space,
            use_feature_units=args.use_feature_units,
            use_raw_units=args.use_raw_units,
            use_unit_counts=args.use_unit_counts)
        env = sc2_env.SC2Env(
            map_name=args.map,
            players=players,
            agent_interface_format=AIF,
            step_mul=args.step_mul,
            game_steps_per_episode=args.game_steps_per_episode,
            disable_fog=args.disable_fog,
            visualize=args.visualize,
            save_replay_episodes=args.save_replay_episodes,
            replay_dir=args.replay_dir,
            random_seed=args.seed
        )
        return env
    return _thunk


# based on https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
# SC2Env::step expects actions list and returns obs list so we send [data] and process obs[0]
def worker(remote, env_fn_wrapper):
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'spec':
            remote.send((env.observation_spec(), env.action_spec()))
        elif cmd == 'step':
            obs = env.step([data])
            remote.send(obs[0])
        elif cmd == 'reset':
            obs = env.reset()
            remote.send(obs[0])
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'save_replay':
            env.save_replay(data[0], data[1])
        else:
            raise NotImplementedError


class CloudpickleWrapper(object):
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class EnvPool(object):
    def __init__(self, env_fns):
        nenvs = len(env_fns)
        self.nenvs = nenvs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn)))
            for (work_remote, env_fn) in zip(self.work_remotes, env_fns)]
        for p in self.ps:
            p.start()

    def spec(self):
        for remote in self.remotes:
            remote.send(('spec', None))
        results = [remote.recv() for remote in self.remotes]
        return results[0]

    def step(self, obs, actives, actions):
        for remote, active, action in zip(self.remotes, actives, actions):
            if active:
                remote.send(('step', action))
        results = [remote.recv() if active else ob \
                   for remote, active, ob in zip(self.remotes, actives, obs)]

        # skip first step
        firsts = [res.first() for res in results]

        if any(firsts):
            for remote, first in zip(self.remotes, firsts):
                if active and first:
                    remote.send(('step', _NO_OP))
            return [remote.recv() if active and first else res \
                    for remote, first, res in zip(self.remotes, firsts, results)]
        return results

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]

        # skip first step
        for remote in self.remotes:
            remote.send(('step', _NO_OP))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def save_replay(self, replay_dir, prefix):
        self.remotes[0].send(('save_replay', [replay_dir, prefix]))

    @property
    def num_envs(self):
        return len(self.remotes)
