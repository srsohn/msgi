"""Hierarchical RL agent"""
from math import log
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from agents.base import BaseMetaAgent
from common.sc2_utils import anneal, ExponentialLR


class HRL(BaseMetaAgent):
    def __init__(self, actor_critic, config, args):
        super(HRL, self).__init__(args)
        self.discount, self.device = args.discount, config.device
        self.vf_coef, self.term_coef = args.vf_coef, args.term_coef
        self.e1t0, self.e1t1, self.e1v0, self.e1v1 = args.e1t0, args.e1t1, args.e1v0, args.e1v1
        self.ent_coef = args.e1v0

        # fully conv policy
        self.actor_critic = actor_critic

        # setup optimizer
        self.init_lr, self.clip_grads = args.lr, args.clip_grads
        self.decay_rate, self.decay_steps = args.decay_rate, args.decay_steps
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=args.lr)

        # save weights
        self.savepath = 'weights/%s'%self.config.full_id()

    def get_option(self, obs):
        spatials = obs['spatials']
        comps, eligs, masks = obs['meta_states']
        steps = obs['steps']
        states = (spatials, comps, eligs, masks, steps)

        # select option
        options, values = self.actor_critic.act(states)
        option_masks = eligs * masks
        return options, values, option_masks

    def get_value(self, obs):
        spatials = obs['spatials']
        comps, eligs, masks = obs['meta_states']
        steps = obs['steps']
        states = (spatials, comps, eligs, masks, steps)

        # select option
        values = self.actor_critic.get_value(states)
        return values

    def train(self, step, states, options, rewards, dones, last_value):
        options = options.unsqueeze(0)

        # compute returns
        returns = self._compute_returns(rewards, dones, last_value)

        # learning rate decay
        lr = ExponentialLR(self.optimizer, init_lr=self.init_lr, epoch=step,
                           gamma=self.decay_rate, decay_steps=self.decay_steps)

        # anneal entropy coefficents
        self.ent_coef = anneal(t=step, t0=self.e1t0, t1=self.e1t1, v0=self.e1v0, v1=self.e1v1)

        # calculate loss
        policies, values = self.actor_critic.act(states, update=True)
        log_probs = sum([p.log_prob(a) for a, p in zip(options, policies)])
        entropy = sum([p.entropy() for p in policies])

        adv = returns - values
        policy_loss = -(adv.detach() * log_probs).mean()
        entropy_loss = -self.ent_coef * entropy.mean()
        value_loss = self.vf_coef * adv.pow(2).mean()

        self.optimizer.zero_grad()
        (policy_loss + value_loss + entropy_loss).backward()
        pol_grads = [p.grad.norm(p=2) for p in self.actor_critic.policy.parameters()]

        # clip gradient
        if self.clip_grads:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.clip_grads)

        self.optimizer.step()
        return policy_loss.item(), value_loss.item(), entropy_loss.item()

    def save(self, step, save_period=1):
        if step > 0 and step % save_period == 0:
            state_dict = {
                'epoch' : step + 1,
                'state_dict' : self.actor_critic.state_dict(),
                'optimizer' : self.optimizer.state_dict()
            }
            filepath = self.savepath + '/hrl-%d'%step
            print("Saving the network @ ", filepath)
            torch.save(state_dict, filepath)

    def _compute_returns(self, rewards, dones, last_value):
        returns = torch.zeros((dones.shape[0]+1, dones.shape[1])).to(self.device)
        returns[-1] = last_value
        for t in reversed(range(dones.shape[0])):
            returns[t] = rewards[t] + self.discount * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        return returns.view(-1)
