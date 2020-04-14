import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])

class RolloutStorage_feat(object):
    def __init__(self, num_steps, num_processes, obs_shape, feat_dim, action_space, ntasks, recurrent_hidden_state_size):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.feats = torch.zeros(num_steps + 1, num_processes, feat_dim)
        self.recurrent_hidden_states = torch.zeros(num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.active = torch.zeros(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.feats = self.feats.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.active = self.active.to(device)

    def init_state(self, obs, feats):
        assert(self.step==0)
        self.obs[self.step].copy_(obs)
        self.feats[self.step].copy_(feats)
        self.active[self.step].fill_(1)

    def insert(self, obs, feats, recurrent_hidden_states, actions, action_log_probs, value_preds, rewards, active):
        self.obs[self.step + 1].copy_(obs)
        self.feats[self.step + 1].copy_(feats)
        if not (recurrent_hidden_states is None):
            self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        if not (action_log_probs is None):
            self.action_log_probs[self.step].copy_(action_log_probs)
        if not (value_preds is None):
            self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.active[self.step + 1].copy_(active)
        assert(not torch.any( (self.active[self.step].cpu()==0)*(rewards>0) )  )

        self.step = (self.step + 1) % self.num_steps

    def after_update(self): # after trial
        """self.obs[0].copy_(self.obs[-1])
        self.feats[0].copy_(self.feats[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.active[0].copy_(self.active[-1])"""
        self.active.zero_()
        self.step = 0
        self.recurrent_hidden_states.zero_()
        self.value_preds.zero_()
        self.rewards.zero_()
        self.returns.zero_()

    def pad_last_step_reward(self, last_rewards):
        for i in range(self.active.size(1)):
            last_step = (self.active[:,i,0]==0).nonzero()[0].item()-1 # last active step.
            self.rewards[last_step, i, 0] += last_rewards[i].item()

    def compute_returns(self, use_gae, gamma, tau): #Monte-Carlo
        if use_gae:
            self.value_preds[-1] = torch.zeros_like(self.rewards[0])
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.active[step + 1] - self.value_preds[step]
                gae = delta + gamma * tau * self.active[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
                # later, we compute: advantage = (self.returns[step]-self.value_preds[step]) = gae!
        else:
            self.returns[-1] = torch.zeros_like(self.rewards[0])
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                    gamma * self.active[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(num_processes, num_steps, num_processes * num_steps, num_mini_batch))
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            feats_batch = self.feats[:-1].view(-1, *self.feats.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(-1,
                self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            active_batch = self.active[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, feats_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, active_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            feats_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            active_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                feats_batch.append(self.feats[:-1, ind])
                recurrent_hidden_states_batch.append(self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                active_batch.append(self.active[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            feats_batch = torch.stack(feats_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            active_batch = torch.stack(active_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            feats_batch = _flatten_helper(T, N, feats_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            active_batch = _flatten_helper(T, N, active_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, feats_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, active_batch, old_action_log_probs_batch, adv_targ
