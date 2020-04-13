import abc
import sys
import numpy as np
import torch

from common.sc2_utils import SELECTIONS, NUM_FULL_UNITS, MAP_INFO, SUBTASK_REWARDS


class BaseMetaAgent(metaclass=abc.ABCMeta):
    """Implementation of Abstract Base agent."""
    def __init__(self, args):
        self.args = args
        self.verbose = args.verbose
        self.gamma = args.discount
        self.meta = args.meta

        # subtask executer
        from agents.subtask_executer import SubtaskExecuter
        self.subtask_executer = SubtaskExecuter(args.envs, args)

    @abc.abstractmethod
    def get_option(self, states):
        """Given the observation, choose option (ind)."""
        pass

    def execute(self, observations, options, envs):
        """Invoke the subtask executer to interact with the environment."""
        max_step = 240
        ever_dones = np.zeros(envs.num_envs)
        rewards, discount = torch.zeros(envs.num_envs).to(self.device), 1.
        actives = np.ones(envs.num_envs)
        elapsed_step = np.zeros(envs.num_envs)
        self.subtask_executer.reset()

        for t in range(max_step):
            obs, _, _ = observations['raws']
            actions, terms = self.subtask_executer.act(observations, actives, options)

            actives *= (1 - terms)
            if (actives == 0).all():
                break

            # env step
            observations, rew, dones = envs.step(obs, actives, actions)
            rewards += rew * discount
            discount *= self.gamma
            ever_dones += dones

            actives *= (1 - dones)
            if (actives == 0).all():
                break

            elapsed_step += actives
        return observations, rewards, ever_dones, elapsed_step
