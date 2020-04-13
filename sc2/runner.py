import os
import time
import copy
import torch
import numpy as np
from tqdm import tqdm

from common import flatten_lists
from common.utils import dotdict
from common.sc2_utils import _update_window, warfare, save_results

from agents.grprop import GRProp


class Runner:
    def __init__(self, envs, agent, ilp, sc2_filewriter, config, args, device, dirname=None):
        self.args = args
        self.envs, self.agent, self.ilp = envs, agent, ilp
        self.gamma, self.n_steps, self.device = args.discount, args.steps, device
        self.infer, self.train = args.infer, args.train
        self.is_warfare = args.warfare
        self.prep_time = 2 * 60 * args.prep_time  # 2 env steps == 1 game sec
        self.save_ilp = args.save_ilp

        # file writer
        self.sc2_filewriter = sc2_filewriter
        self.dirname = dirname

        # restore
        self.init_update = 0
        if args.restore != -1:
            filepath = 'weights/%s/hrl-%s'%(config.full_id(), args.restore)
            print("Loading the network @ ", filepath)
            ckpt = torch.load(filepath, map_location=device)
            self.agent.actor_critic.load_state_dict(ckpt['state_dict'])
            self.agent.optimizer.load_state_dict(ckpt['optimizer'])
            self.init_update = ckpt['epoch']

        # init
        self.num_envs = self.envs.num_envs
        self.state = self.ep_rews = None
        self.logs = {'updates': self.init_update}
        self.step, self.ep_len, self.score_record, self.return_record = np.zeros((4,self.envs.num_envs))
        self.scores, self.ep_returns, self.init_values = np.zeros((3, self.envs.num_envs))

    def meta_eval_load(self, dirname, init_ep=0, eval_eps=20, test_eps=4):  # HRL & MSGI
        '''Load ILP models saved for each episode of meta evaluation
        (i.e. ILP_ep-*.pt) and calcuate their winning rate.
        '''
        for ep in range(init_ep, eval_eps):
            # load the saved ILP model
            filename = os.path.join(dirname, 'ILP_ep-%d.pt'%ep)
            self.ilp.load(filename)

            # precision & recall
            graphs, prec, rec = self.ilp.infer_graph(ep=ep, PR=True, eval_mode=True)
            print('prec, rec=', prec.mean(), rec.mean())

            # temperature annealing (BATTLECRUISER)
            params = dotdict()
            if self.args.map.startswith('Build'):
                params.temp = 1 + 49.0 * ( ep ) / ( eval_eps ) # linear scheduling (new)
            else:
                params.temp = 2 + 38.0 * ( ep ) / ( eval_eps ) # linear scheduling (old)

            test_agent = GRProp(graphs, self.args, self.device, params)
            self.rollout_trial(nb_epi=test_eps, eval_flag=True, agent=test_agent)

            # store the eval results
            mean_frame = self.cum_frames / self.nb_eval_epi
            mean_score = self.test_score / self.nb_eval_epi
            self.sc2_filewriter.store(ep=ep, mean=mean_score, data=self.ep_scores, ep_len=mean_frame)
            self.sc2_filewriter.save()

            # log
            print('[ Ep: {} | Score: {} ]'.format(ep, mean_score))

    def meta_eval_save(self, num_iter=10, tr_epi=20, test_epi=4):
        '''
          - msgi-meta: train - x, infer - o
          - hrl-baseline: train - o, infer - x
        '''
        logs = dotdict()
        logs.str, logs.stst = np.zeros((2, tr_epi))
        for i in range(num_iter):
            # reset
            self.envs.reset_task()

            if self.infer:
                self.ilp.reset(self.envs)
                graphs, prec, rec = self.ilp.infer_graph(ep=None, PR=False)
                self.agent = GRProp(graphs, self.args, self.device)

            for epi_ind in tqdm(range(tr_epi), ascii=True, desc="[Iter: {}] Episode".format(i)):
                self.cur_ep = epi_ind

                # rollout training trial (MSGI & HRL)
                self.rollout_trial(nb_epi=1)
                logs.str[epi_ind] += self.score_record

                # MSGI - infer graph
                if self.infer:
                    if self.save_ilp:
                        filename = self.dirname + '/ILP_ep-{}.pt'.format(epi_ind)
                        self.ilp.save(filename)

                    graphs, prec, rec = self.ilp.infer_graph(ep=tr_epi, PR=False)
                    self.agent = GRProp(graphs, self.args, self.device)
                    test_agent = self.agent
                else:
                    test_agent = None

                # save episode results
                save_results(ep=epi_ind, dirname=self.dirname, score=self.score_record,
                             total_counts=self.total_counts)

                # eval HRL
                if self.args.meta == 'hrl':  # HRL
                    self.rollout_trial(nb_epi=4, eval_flag=True, agent = test_agent)

                    # 4. record
                    mean_frame = self.cum_frames / self.nb_eval_epi
                    mean_score = self.test_score / self.nb_eval_epi
                    logs.stst[epi_ind]  += mean_score
                    print('[ Ep: {} | Score: {} ]'.format(epi_ind, mean_score))
                    self.sc2_filewriter.store(ep=epi_ind, mean=mean_score, data=self.ep_scores, ep_len=mean_frame)
                    self.sc2_filewriter.save()

    def rollout_trial(self, nb_epi, eval_flag=False, agent=None):
        '''
          1. HRL - train trial
          - N-step AC. However, stop after 'nb_epi' episodes.
          - result: trained policy network
          2. MSGI - train trial
          - Just run 'nb_epi' episodes, and collect data into ilp.
          - result: ilp
          3. HRL & MSGI - eval trial
          - Just run 'nb_epi' episodes, measure performance.
        '''
        assert(not self.infer or not self.ilp is None)

        if agent is None:
            agent = self.agent

        self.eval_flag = eval_flag
        if eval_flag:
            self.test_score, self.cum_frames, self.nb_eval_epi = 0., 0, 0
            self.ep_scores = np.zeros(nb_epi)

        # reset
        self.reset_trial()

        while True:
            rollout = self.collect_rollout(nb_epi, eval_flag, agent)
            if (self.active == False).all():
                break

            if not eval_flag:
                if self.train: # HRL
                    ploss, vloss, eloss = agent.train(self.logs['updates'], *rollout)
                    if self.cur_ep == 19 and self.frames > 2000:
                        self.agent.save(step=self.frames)
                    self.logs['updates'] += 1

        if eval_flag:
            assert(self.nb_eval_epi == nb_epi)

    def collect_rollout(self, nb_epi, eval_flag, agent):
        states, options, option_masks = [None]*self.n_steps, [None]*self.n_steps, [None]*self.n_steps
        rewards, values = torch.zeros((2, self.n_steps, self.envs.num_envs)).to(self.device)
        dones, prev_dones = np.zeros((2, self.n_steps, self.envs.num_envs))

        for step in range(self.n_steps):
            with torch.no_grad():
                if eval_flag:
                    if isinstance(agent, GRProp):
                        option, value, option_mask = agent.get_option(self.obs, self.last_dones, eval_flag)
                    else:
                        option, value, option_mask = agent.get_option(self.obs, self.last_dones)
                else:
                    option, value, option_mask = agent.get_option(self.obs, self.last_dones)

            options[step] = copy.deepcopy(option)
            option_masks[step] = copy.deepcopy(option_mask)

            if self.is_warfare and self.frames >= self.prep_time:
                # time is up and prepare for the battle
                self.obs, reward, done, frames, self.total_counts = warfare(self.envs, self.obs)
            else:
                self.obs, reward, done, frames = agent.execute(self.obs, option, self.envs)

                if not done:  # for non-warfare maps
                    self.total_counts = copy.deepcopy(self.envs.total_counts[0])

            # MSGI
            if self.infer and not eval_flag:
                self.ilp.insert(self.obs, option, reward, done)

            # HRL
            if self.train and not eval_flag:
                spatials = self.obs['spatials']
                comps, eligs, masks = self.obs['meta_states']
                steps = self.obs['steps']
                states[step] = [spatials, comps, eligs, masks, steps]
                rewards[step], dones[step], values[step] = reward, done, value

            # compute records
            self._compute_records(reward, done, value, frames)

            # updates & log
            self.epi_count += done
            self.active = (self.epi_count < nb_epi)
            self.last_dones = done
            self.frames += frames

            if self.active.sum() == 0:
                break

        # terminate when all episodes are finished
        if self.active.sum() == 0: # ignore current samples
            return None

        if self.train and not eval_flag:
            with torch.no_grad():
                last_value = agent.get_value(self.obs).detach()

            # convert to torch tensor
            prev_dones = torch.from_numpy(prev_dones).float().to(self.device)
            dones = torch.from_numpy(dones).float().to(self.device)
            return (flatten_lists(states), torch.cat(options), rewards, dones, self.init_values, last_value)
        else:
            return None

    def eval(self, num_iter, step=None):
        ep_len_sum, score_sum, return_sum, epi_count, cur_score, cur_return = np.zeros((6, self.envs.num_envs))
        gammas = np.ones((self.envs.num_envs))
        steps = np.zeros((self.envs.num_envs))
        meta_steps = np.zeros((self.envs.num_envs))
        print('Eval for {} iterations!'.format(num_iter))

        ep_lens = []
        score_buffer = []
        observations, _, _ = self.envs.reset()
        dones = np.zeros((self.envs.num_envs, 1))

        while True:
            with torch.no_grad():
                options, *_ = self.agent.get_option(observations, dones)

            if self.is_warfare and steps >= self.prep_time:
                # time is up and prepare for the battle
                observations, rewards, dones, frames, _ = warfare(self.envs, observations)
            else:
                observations, rewards, dones, frames = self.agent.execute(observations, options, self.envs)

            dones = np.asarray(dones, dtype=np.float32)
            rewards = np.asarray(rewards, dtype=np.float32)

            # computes cur values
            steps += frames
            cur_score += rewards
            cur_return += rewards if self.is_warfare else gammas * rewards
            gammas = (1 - dones) * np.power(self.gamma, frames) * gammas + dones

            # break if episode is done
            mask = (epi_count < num_iter)
            if mask.sum() == 0:
                break

            if (dones * mask).sum() > 0:
                # save the episode score (episode / win_or_lose / 0 / 0)
                ep = copy.deepcopy(epi_count[0])
                score = float(copy.deepcopy(cur_score) > 0)
                score_buffer.append([ep, score, 0., 0.])

                # store episode length
                ep_lens.append([ep, mask*dones*steps, 0., 0.])

                # save the stats for the episode
                if not self.args.eval:
                    save_results(ep=int(epi_count[0]), dirname=self.dirname, score=cur_score,
                                 total_counts=self.total_counts)
                print('[ Ep: {} | Score: {} ]'.format(ep, score))
            else:
                self.total_counts = copy.deepcopy(self.envs.total_counts[0])

            ep_len_sum += mask*dones*steps
            score_sum += mask*dones*cur_score
            return_sum += mask*dones*cur_return

            # reset if episode is done
            epi_count += dones*mask
            steps *= (1 - dones)
            cur_score *= (1 - dones)
            cur_return *= (1 - dones)

        div = num_iter * self.envs.num_envs
        print('\n========= Final Result =========')
        print('Avg Length =', ep_len_sum.sum()/div)
        print('Avg Score  =', score_sum.sum()/div)
        print('Avg Return =', return_sum.sum()/div)
        print('================================')

        # save the mean scores and episode lengths
        ep_lens = np.asarray(ep_lens)
        scores = np.asarray(score_buffer)
        mean_ep_lens = ep_lens[:, 1].mean()
        mean_score = scores[:, 1].mean()
        ep_lens[:, 1].fill(mean_ep_lens)
        scores[:, 1].fill(mean_score)

        if not self.args.eval:
            self.sc2_filewriter.save(mean=scores, ep_len=ep_lens)

    def _compute_records(self, reward, done, value, frames):
        if value is not None:
            value = value.cpu().numpy()
        reward = reward.cpu().numpy()

        # episode length taken
        self.step += 1
        for i in range(done.shape[0]):
            if done[i] > 0.5:
                self.ep_len_window = _update_window(self.ep_len_window, self.step[i])
        self.ep_len = (1 - done)*self.ep_len + done*self.step
        self.step = 0*done + (1 - done)*self.step

        # init value est
        if value is not None:
            self.init_values = (1 - self.last_dones)*self.init_values + self.last_dones*value

        self.scores = (1 - self.last_dones)*(reward + self.scores) + self.last_dones*reward
        self.ep_returns = (1 - self.last_dones)*(self.gammas*reward + self.ep_returns) + self.last_dones*self.gammas*reward
        self.gammas = (1 - done)*self.gamma*self.gammas + done
        self.score_record = done * self.scores + (1 - done) * self.score_record
        self.return_record = done * self.ep_returns + (1 - done) * self.return_record
        self.dones_window = _update_window(self.dones_window, self.last_dones )
        self.frames = (1 - done)*self.frames

        done = done > 0.5
        num_dones = done.sum()
        num_succ = (done * (reward > 0)).sum()

        for i in range(num_dones):
            if i < num_succ:
                self.success_window = _update_window(self.success_window, 1) # success
            else:
                self.success_window = _update_window(self.success_window, 0) # fail

        if self.eval_flag:
            self.cum_frames += frames

        if self.eval_flag and done:
            self.ep_scores[self.nb_eval_epi] = float(self.scores > 0)
            self.test_score += float(self.scores > 0)
            self.nb_eval_epi += 1

    def reset_trial(self):
        self.obs, _, _ = self.envs.reset()

        if self.infer:
            self.ilp.insert( self.obs)

        self.epi_count = np.zeros(self.envs.num_envs)
        self.frames = np.zeros(self.envs.num_envs)
        self.last_dones = np.ones(self.envs.num_envs)
        self.scores.fill(0)
        self.dones_window = np.zeros( (self.envs.num_envs, 100) )
        self.success_window, self.ep_len_window = np.zeros( (2, self.envs.num_envs*10) )
        self.gammas = self.gamma*np.ones(self.envs.num_envs)
        self.logs.update({'eps': 0, 'rew_best': 0, 'start_time': time.time(),
                     'ep_rew': np.zeros(self.envs.num_envs),
                     'dones': np.zeros(self.envs.num_envs)})
