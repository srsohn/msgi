import copy
import numpy as np
import torch

from pysc2.lib.actions import FunctionCall, FUNCTIONS
from common.config import is_spatial
from common.sc2_utils import STRUCTURES, PRODUCTIONS, FLYING, SELECTIONS, SELECTION_MAP, \
    ELIG_MAP, ADDON_ELIG_MAP, ADDON_FUNCS, NUM_UNITS, UNIT_MAP, NUM_FULL_UNITS, \
    FLYING_MAP, FLYING_ID, FULL_UNITS, FULL_UNIT_MAP, IS_TRAINING, TERRAN_FUNCTIONS, ARG_TYPES, \
    UNIT_REWARDS

class EnvWrapper:
    def __init__(self, envs, config, args, device):
        self.envs, self.config, self.device = envs, config, device
        self.epi_count = 0
        self.step_mul = args.step_mul
        self.max_step = args.max_step
        self.unit_masks = np.ones((envs.num_envs, NUM_UNITS))
        self.total_counts = np.zeros((envs.num_envs, NUM_FULL_UNITS))
        self.minerals_elig = np.ones((self.num_envs, 7))
        self.gases_elig = np.zeros((self.num_envs, 8))
        self.prev_dones = [None]*envs.num_envs
        self.prev_counts = None
        self.resources = np.zeros((envs.num_envs, 3))
        self.idle_worker_count = np.zeros((envs.num_envs, 1))
        self.no_ops = np.ones((envs.num_envs, 1))
        self.reset_task()

        # SGE completion initialize
        mineral = [50, 75, 100, 125, 150, 300, 400]
        gas = [25, 50, 75, 100, 125, 150, 200, 300]
        food = [1, 2, 3, 6]

        self.sge_minerals = np.array([mineral]*envs.num_envs)
        self.sge_gases = np.array([gas]*envs.num_envs)
        self.sge_foods = np.array([food]*envs.num_envs)
        self.sge_idle_workers = np.array([[0.]]*envs.num_envs)

    def _set_init_meta(self, results):
        observations, _, _ = results
        _, eligs, _ = observations['meta_states']
        self._init_total_counts = copy.deepcopy(self.total_counts[0])
        self._init_select_counts = copy.deepcopy(self.select_counts[0])
        self._init_unit_masks = copy.deepcopy(self.unit_masks[0])
        self._init_unit_eligs = copy.deepcopy(self.unit_eligs[0])
        self._init_resources = copy.deepcopy(self.resources[0])
        self._init_gases = copy.deepcopy(self.gases_elig[0])

    def _reset_meta(self):
        for i, done in enumerate(self.prev_dones):
            if done:
                self.total_counts[i] = self._init_total_counts
                self.select_counts[i] = self._init_select_counts
                self.prev_counts[i] = self._init_total_counts
                self.unit_masks[i] = self._init_unit_masks
                self.unit_eligs[i] = self._init_unit_eligs
                self.resources[i] = self._init_resources
                self.gases_elig[i] = self._init_gases

    def reset_task(self):
        scaler = 1
        self.reward_vec = np.tile(np.array(UNIT_REWARDS), (self.envs.num_envs, 1)) * scaler

    def reset(self):
        timesteps = self.envs.reset()
        res = self.wrap_results(timesteps)
        self._set_init_meta(res)
        self.epi_count += 1
        return res

    def step(self, obs, actives, acts):
        timesteps = self.envs.step(obs, actives, acts)
        res = self.wrap_results(timesteps)
        if any(res[2]): # if done
            self.last_obs = res
            observations, rew, done = res
            next_obs, _, _ = observations['raws']
            timesteps = self.envs.step(next_obs, actives, FUNCTIONS.no_op())
            observations, _, _ = self.wrap_results(timesteps)
            res = (observations, rew, done)
        return res

    def wrap_actions(self, actions):
        pol_mask = [torch.zeros((self.envs.num_envs)).float().to(self.device) \
                    for _ in range(1 + len(ARG_TYPES))]
        pol_mask[0].fill_(1.0)
        acts, args = actions[0], actions[1:]
        wrapped_actions = []
        for i, act in enumerate(acts):
            fn = TERRAN_FUNCTIONS[act]
            act_args = []
            for arg_type in fn.args:
                act_arg = [args[self.config.arg_idx[arg_type.name]][i]]
                pol_mask[self.config.arg_idx[arg_type.name] + 1][i] = 1.
                if arg_type.name == 'queued':
                    act_arg = [False]
                if is_spatial(arg_type.name):  # spatial args, convert to coords
                    act_arg = [act_arg[0] % self.config.sz, act_arg[0] // self.config.sz]
                act_args.append(act_arg)
            wrapped_actions.append(FunctionCall(fn.id, act_args))
        return wrapped_actions, pol_mask

    def wrap_results(self, timesteps):
        self.prev_counts = copy.deepcopy(self.total_counts)
        obs = [t.observation for t in timesteps]
        counts = [ob.unit_counts for ob in obs]
        players = [ob.player for ob in obs]
        raw_units = [ob.raw_units for ob in obs]
        last_acts = [ob.last_actions for ob in obs]
        dones = np.asarray([t.last() for t in timesteps])
        ep_steps = [t.episode_step - self.step_mul for t in timesteps]
        steps = [np.log10(self.max_step - (step / self.step_mul) + 1.0)-1 for step in ep_steps]
        firsts = np.asarray([step == 0 for step in ep_steps])
        self.unit_masks.fill(1)
        self.selects = np.zeros((self.num_envs, len(SELECTIONS)))
        self.select_counts = np.zeros((self.num_envs, len(SELECTIONS)))
        self.selects_mask = np.ones((self.num_envs, len(SELECTIONS)))
        idle_masks = np.ones((self.num_envs, 1))

        # preprocess
        steps = np.expand_dims(np.asarray(steps),axis=1)
        spatials, (act_masks, avail_acts) = self.config.preprocess(obs)

        # resources
        mineral_masks = np.ones((self.num_envs, 7))
        gas_masks = np.ones((self.num_envs, 8))
        food_masks = np.zeros((self.num_envs, 4))

        for i, player in enumerate(players):
            # ATTR 4 : count the number of idle worker
            self.idle_worker_count[i][0] = player.idle_worker_count
            self.resources[i][0] = player.minerals
            self.resources[i][1] = player.vespene
            self.resources[i][2] = player.food_cap - player.food_used

            # MASK 6 : Compare the remaining resources left for minerals, gases, and foods
            mineral_masks[i] = (self.sge_minerals[i] > player.minerals).astype('float32')
            gas_masks[i] = (self.sge_gases[i] > player.vespene).astype('float32')

        # mask and termination
        for i, unit_count in enumerate(counts):
            for id, count in unit_count:
                # ATTR 1 : add counts for flying units (ie. Barracks, Factory, Starport)
                if id in FLYING:
                    self.total_counts[i][FLYING_MAP[id]] = count

                # ATTR 2 : add (over-) counts for the entire unis on the map
                if id in FULL_UNITS:
                    self.total_counts[i][FULL_UNIT_MAP[id]] = count

                # ATTR 2 : add counts for Rich refinery
                if id == 1949:  # Rich refinery
                    self.total_counts[i][FULL_UNIT_MAP[20]] = count

                # MASK 1 : (BUILD BUILDING) if the building exists, mask out
                #          (except SupplyDepot)
                if id in STRUCTURES:
                    self.unit_masks[i][UNIT_MAP[id]] = 0
                    if id == 19 and count < 8:
                        self.unit_masks[i][UNIT_MAP[id]] = 1
                    if id == 19 and count >= 8:
                        food_masks[i].fill(0)

                # MASK 2 : (TRAIN UNITS) if SCV count exceeds 17, mask out
                if id == 45 and count > 17:
                    self.unit_masks[i][UNIT_MAP[id]] = 0
                    idle_masks[i] = 0

        current_units = [None]*self.num_envs
        training = [set()]*self.num_envs  # ids of production buildings that are training units
        for i, raw_unit in enumerate(raw_units):
            for unit in raw_unit:
                id = unit.unit_type
                if id in SELECTIONS:
                    if unit.build_progress == 100:
                        # MASK 4 : (SEL_BUILD) If the bulding is training unit,
                        #          then mask out from selection
                        if id in PRODUCTIONS and unit.order_length > 0:
                            self.selects_mask[i][SELECTION_MAP[id]] = 0
                            training[i].add(id)

                    if unit.is_selected:
                        # MASK 4 : (SEL_BUILD & SEL_UNIT) if the unit is selected,
                        #          then mask out from selection
                        self.selects_mask[i][SELECTION_MAP[id]] = 0

                        # ATTR 5 : set the selected unit to 1
                        self.selects[i][SELECTION_MAP[id]] = 1
                        current_units[i] = id

                if id in STRUCTURES and unit.build_progress < 100:
                    # ATTR 3 : for each incomplete unit, decrease the count by 1
                    self.total_counts[i][FULL_UNIT_MAP[id]] -= 1

                    # FOOD 1
                    if id == 19:
                        self.unit_masks[i][UNIT_MAP[id]] = 0
                        food_masks[i].fill(0)
                if id in FLYING and unit.is_selected:
                    # MASK 4 : (SEL_BUILD & SEL_UNIT) if the unit is selected,
                    #          then mask out from selection
                    self.selects_mask[i][SELECTION_MAP[FLYING_ID[id]]] = 0

                    # ATTR 5 : set the selected unit to 1
                    self.selects[i][SELECTION_MAP[FLYING_ID[id]]] = 1
                    current_units[i] = id

        for i in range(self.num_envs):
            if self.total_counts[i][4] > 0:
                self.gases_elig[i].fill(1)

        # eligibility for units
        idles = np.zeros((self.num_envs, 1))
        foods = np.zeros((self.num_envs, 4))
        # Note: foods must have same eligibility as supplyDepot to connect foods with Sel_scv.

        self.unit_eligs = np.zeros((self.num_envs, NUM_UNITS))
        for i, avail_act in enumerate(avail_acts):
            for id in avail_act:
                # ELIG 1 : get available actions from env (ie. build_xxx & train_yyy & sel_scv)
                if id.item() in ELIG_MAP:
                    self.unit_eligs[i][ELIG_MAP[id.item()]] = 1

                # ELIG 2 : (IDLE SCV) If Train_SCV_quick is available, set IdleSCV
                #          as eligible
                if id.item() == 490:
                    idles[i] = 1

                # ELIG 2 : (Select SCV) If select_idle_worker available, set SelectSCV
                #          as eligible
                if id.item() == 6:
                    self.select_counts[i][SELECTION_MAP[45]] = 1

                # ELIG 4 : (FOOD) If supplydepot is eligible, enable food option
                if id.item() == 91:
                    foods[i].fill(1)

                # ELIG 2 : if the addon action is available, double-map to make
                #          it seem 3 different options
                if id.item() in ADDON_FUNCS:
                    self.unit_eligs[i][ADDON_ELIG_MAP[current_units[i]]] = 1

        # MASK 2 : (TRAIN UNIT) If a building is training, mask all related
        #          options for training units
        for i, train in enumerate(training):
            for _id in train:
                acts = [ELIG_MAP[f] for f in IS_TRAINING[_id]]
                self.unit_masks[i][acts] = 0
                if _id == 18:  # CommandCenter
                    idle_masks[i] = 0

        # if done reset meta states & prev_counts
        if any(self.prev_dones):
            self._reset_meta()
        self.prev_dones = dones

        # meta-states
        mineral = np.expand_dims(self.resources[:,0], axis=1)
        gas = np.expand_dims(self.resources[:,1], axis=1)
        food = np.expand_dims(self.resources[:,2], axis=1)
        comp_counts = (self.total_counts > 0).astype('float32')
        comp_minerals = (self.sge_minerals <= mineral).astype('float32')
        comp_gases = (self.sge_gases <= gas).astype('float32')
        comp_foods = (self.sge_foods <= food).astype('float32')
        comp_idle_worker = (self.sge_idle_workers < self.idle_worker_count).astype('float32')
        comp_noop = np.zeros_like(self.no_ops)
        comp = np.concatenate((comp_counts, comp_idle_worker, self.selects, comp_minerals, comp_gases, comp_foods, comp_noop), axis=1)
        self.minerals_elig.fill(1)

        if True:
            self.gases_elig[:,1:] = comp_gases[:,:-1]
            self.minerals_elig[:,1:] = comp_minerals[:,:-1]

        # make sure the comp and elig relation for techlabs are consistent
        if comp_minerals[i][0] and comp_gases[i][0] and self.selects[i][SELECTION_MAP[21]]:  # BARRACKS
            self.unit_eligs[i][ADDON_ELIG_MAP[21]] = 1
        if comp_minerals[i][0] and comp_gases[i][0] and self.selects[i][SELECTION_MAP[27]]:  # FACTORY
            self.unit_eligs[i][ADDON_ELIG_MAP[27]] = 1
        if comp_minerals[i][0] and comp_gases[i][0] and self.selects[i][SELECTION_MAP[28]]:  # STARPORT
            self.unit_eligs[i][ADDON_ELIG_MAP[28]] = 1

        for i, comp_count in enumerate(comp_counts):
            count_idxs = comp_counts.nonzero()[1]
            for idx in count_idxs:
                if FULL_UNITS[idx] != 45 and FULL_UNITS[idx] in SELECTIONS:
                    self.select_counts[i][SELECTION_MAP[FULL_UNITS[idx]]] = 1

        eligs = np.concatenate((self.unit_eligs, idles, self.select_counts, self.minerals_elig, self.gases_elig, foods, self.no_ops), axis=1)
        masks = np.concatenate((self.unit_masks, idle_masks*(self.idle_worker_count<1), self.selects_mask, mineral_masks, gas_masks, food_masks, self.no_ops), axis=1)
        rewards = np.asarray([t.reward for t in timesteps])

        return self._timestep(spatials, comp, eligs, masks, steps, rewards, dones, obs, firsts, ep_steps)

    def _timestep(self, spatials, comp_or_attrs, eligs, masks, steps, rewards, dones, obs, firsts, ep_steps):
        # convert to tensor
        comp_or_attrs = torch.from_numpy(comp_or_attrs).float().to(self.device)
        eligs = torch.from_numpy(eligs).float().to(self.device)
        masks = torch.from_numpy(masks).float().to(self.device)
        steps = torch.from_numpy(steps).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        #ep_steps = torch.from_numpy(ep_steps).float().to(self.device)

        # get state
        observations = {
            'spatials' : spatials,
            'meta_states' : (comp_or_attrs, eligs, masks),
            'steps' : steps,
            'ep_steps' : ep_steps,
            'game_steps' : ep_steps[0]/8/2,
            'raws' : (obs, firsts, dones)
        }
        return observations, rewards, dones

    def save_replay(self, replay_dir, prefix):
        self.envs.save_replay(replay_dir, prefix)

    def spec(self):
        return self.envs.spec()

    def close(self):
        return self.envs.close()

    @property
    def num_envs(self):
        return self.envs.num_envs
