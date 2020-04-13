"""Implementation of subtask executer (or, Options)."""
import sys
import numpy as np

from pysc2.lib.actions import FUNCTIONS
from common.sc2_utils import MAP_INFO


class SubtaskExecuter:
    def __init__(self, num_envs, args):
        self.num_envs = num_envs
        self.prev_foods = [None]*num_envs
        self.supply_counts = [None]*num_envs
        self.vespene_coord = [8, 8]  # vespene geyser coord : (y, x)
        self.command_coord = [26, 33]  # command center coord : (y, x)
        self.coords = [[]]*num_envs
        self._selected = False
        self.deselect_flag = np.zeros(num_envs)
        self.select_trials = 0
        self.build_trials = 0
        self.max_select_trials = 8
        self.max_build_trials = 220
        self._max_no_op = args.max_noop

    def _get_coord(self, ob, id):
        coord = None
        exists = False
        for unit in ob.feature_units:
            if unit[0] == id:
                coord = [unit.x, unit.y]
                exists = True
                break

        # if feature unit does not work
        if coord is None and not exists:
            screen = ob.feature_screen.unit_type
            try:
                coords = np.where(screen == id)  # (Y, X) coords

                # if coord for unit is not found, return it
                if len(coords) <= 1:
                    return [None, None], exists
                if len(coords[0]) == 0 or len(coords[1]) == 0:
                    return [None, None], exists

                coords = np.stack((coords[0], coords[1]), axis=1)
                y, x = coords[np.random.randint(len(coords))]
                coord = [x, y]
                exists = True
            except ValueError as err:
                print(" ERROR ({}) : value error".format(__name__))
                from IPython import embed; embed()
        return coord, exists

    def _preprocess(self, ob):
        # slice the observation to only playable space
        ob = ob[0:70, :]
        ob[ob == 45] = 0  # filter out SCV
        return ob

    def reset(self):
        self._selected = False
        self.no_op_count = 0
        self.max_no_op = self._max_no_op
        self.deselect_flag.fill(0)
        self.select_trials = 0
        self.build_trials = 0

    def act(self, observations, actives, options):
        actions = []
        obs, firsts, dones = observations['raws']
        terms = np.zeros_like(actives)
        for i, (op, active, ob, first, done) in enumerate(zip(options, actives, obs, firsts, dones)):
            # check active
            if not active:
                actions.append(None)
                continue

            # reset env info
            if first:
                self.supply_counts[i] = 0  # SupplyDepot counts
                self.prev_foods[i] = ob.player.food_used
                self.coords[i] = []

            # extract subtask info
            map_info = MAP_INFO[op]
            func = map_info['func']
            addon = map_info['add_on']
            is_quick = map_info['cmd_quick']
            if is_quick:
                quick_func = map_info['quick_func']
            _type, target = map_info['type-target']
            avail_acts = np.array(ob.available_actions)

            # compute termination
            if _type == 'build' or addon:
                if self.deselect_flag[i]:
                    self.deselect_flag[i] = False
                    terms[i] = 1
                elif addon:
                    for id, count in ob.unit_counts:
                        if id == target[1]:
                            self.deselect_flag[i] = True
                else:
                    for id, count in ob.unit_counts:
                        if id == target:
                            self.deselect_flag[i] = True

                            # allow multiple SupplyDepots
                            if id == 19:
                                if count == self.supply_counts[i]:
                                    self.deselect_flag[i] = False
                                elif count > self.supply_counts[i]:
                                    self.deselect_flag[i] = True
                                    self.supply_counts[i] = count
                                else:
                                    print('Error. SupplyDepot count should not be less than count.')
                                    sys.exit(1)

                # build trial
                if self.build_trials == self.max_build_trials:
                    self.deselect_flag[i] = True
                    self.build_trials = 0
                else:
                    self.build_trials += 1

                # reset coords
                if terms[i]:
                    self.coords[i] = []

            elif _type in ['unit', 'idle']:
                if self.deselect_flag[i]:
                    self.deselect_flag[i] = False
                    terms[i] = 1
                else:
                    if ob.player.food_used > self.prev_foods[i]:
                        for act in ob.last_actions:
                            if act == func.id:
                                self.deselect_flag[i] = True
                        self.prev_foods[i] = ob.player.food_used
            elif _type == 'select':
                for unit in ob.feature_units:
                    if unit.unit_type == target and unit.is_selected:
                        if unit.unit_type == 45 and unit.order_length == 0:
                            self._selected = True
                        else:
                            self._selected = True

                # check if the unit is selected or the deselection flag is on
                if self._selected:
                    # if selected, max noop
                    if self.no_op_count == self.max_no_op:
                        self.no_op_count = 0
                        terms[i] = 1
                    else:
                        self.no_op_count += 1
                else:
                    if self.select_trials == self.max_select_trials:
                        self._selected = True
                        self.select_trials = 0
                    else:
                        self.select_trials += 1
            elif _type == 'gather':
                if self.deselect_flag[i]:
                    self.deselect_flag[i] = False
                    terms[i] = 1
                elif func.id in ob.last_actions:
                    self.deselect_flag[i] = True
            elif _type in ['mineral', 'gas']:
                if _type == 'mineral' and ob.player.minerals >= target:
                    terms[i] = 1
                if _type == 'gas' and ob.player.vespene >= target:
                    terms[i] = 1
            elif _type == 'food':
                if self.deselect_flag[i]:
                    self.deselect_flag[i] = False
                    terms[i] = 1
                else:
                    for id, count in ob.unit_counts:
                        if id == target and count > self.supply_counts[i]:
                            self.deselect_flag[i] = True
                            self.supply_counts[i] = count

                if terms[i]:
                    self.coords[i] = []
            elif _type == 'no_op':
                if self.no_op_count == self.max_no_op:
                    self.no_op_count = 0
                    terms[i] = 1
                else:
                    self.no_op_count += 1
            else:
                raise NotImplementedError

            # check termination
            if terms[i]:
                actions.append(None)
                continue

            # execute the subtask
            if self.deselect_flag[i]:
                actions.append(FUNCTIONS.select_point("select", self.vespene_coord))
            elif is_quick and quick_func.id in avail_acts:  # Quick actions
                if quick_func == FUNCTIONS.select_idle_worker:
                    if self._selected:
                        actions.append(FUNCTIONS.no_op())
                    else:
                        actions.append(quick_func("select"))
                else:
                    actions.append(quick_func("now"))
            elif _type == 'build' and func.id in avail_acts: # Build
                if self._check_vacant(i, ob, func, map_info):
                    actions.append(func("now", self.coords[i]))
                else:
                    actions.append(self.build_structure(i, ob, func, map_info))
            elif addon and func.id in avail_acts: # Addon
                if self._check_vacant(i, ob, func, map_info):
                    actions.append(func("now", self.coords[i]))
                else:
                    actions.append(self.build_structure(i, ob, func, map_info))
            elif _type == 'select' and func.id in avail_acts:  # Selection
                if self._selected:  # when unit is selected & is terminating
                    actions.append(FUNCTIONS.no_op())
                else:
                    coords, coord_exists = self._get_coord(ob, id=target)
                    if coord_exists and (coords[0] < 0 or coords[1] < 0):
                        print("[ ERROR ] : Got Negative coordinate.")
                        from IPython import embed; embed()
                        raise ValueError

                    # if the unit is out-of-screen, then ignore
                    if coord_exists and (coords[0] >= 84 or coords[1] >= 84):
                        self._selected = True
                        actions.append(FUNCTIONS.no_op())
                        continue

                    # if value error, ignore
                    if coord_exists and (coords[0] is None or coords[1] is None):
                        self._selected = True
                        actions.append(FUNCTIONS.no_op())
                        continue

                    # if coord exists return selection action, and if not, take noop
                    if coord_exists:
                        x, y = coords[0], coords[1]
                        actions.append(func(0, [x, y]))
                    else:
                        actions.append(FUNCTIONS.no_op())
            elif _type in ['mineral', 'gas']:
                actions.append(func())
            elif _type == 'food' and func.id in avail_acts:  # Food supply
                if self._check_vacant(i, ob, func, map_info):
                    actions.append(func("now", self.coords[i]))
                else:
                    actions.append(self.build_structure(i, ob, func, map_info))
            elif _type == 'no_op':
                actions.append(func())
            else:
                if (_type in ['build', 'food'] or addon) and self._check_available(func, avail_acts):
                    actions.append(FUNCTIONS.no_op())
                    continue

                # if selectSCV suddenly becomes unavailable, then just take noop
                if _type == 'select' and target == 45:
                    self._selected = True
                    actions.append(FUNCTIONS.no_op())
                    continue

                print('[ ERROR ] : Selected option seems not valid. Check if the condition is correct.')
                import pudb; pudb.set_trace()  # XXX DEBUG

                raise NotImplementedError
        return actions, terms

    def _check_available(self, func, avail_acts):
        """Workaround when the mineral is not enough but the subtask
        executer keep tries to take actions.
        """
        if func.id not in avail_acts:
            return True
        return False

    def _check_vacant(self, i, ob, func, map_info):
        """Check if the previously selected region is still valid."""
        if len(self.coords[i]) == 0:
            return False

        # skip Refinery
        if func.id == 79:
            return True

        # validate
        x, y = self.coords[i]
        _region = map_info['region']
        _y, _x = map_info['center']
        addon = map_info['add_on']
        screen = np.array(ob.feature_screen.unit_type)
        screen = self._preprocess(screen)

        if addon:
            x += 3
            cum_sum = np.sum(screen[y-_y:y+_y+1, x-_x:x+_x+1] * _region)
        else:
            cum_sum = np.sum(screen[y-_y:y+_y, x-_x:x+_x] * _region)

        if cum_sum == 0:
            return True
        return False

    def build_structure(self, i, ob, func, map_info):
        # check Refinery
        if func.id == 79:
            self.coords[i] = self.vespene_coord
            return func("now", self.coords[i])

        _region = map_info['region']
        _y, _x = map_info['center']
        addon = map_info['add_on']

        # get closest coordinate from Command Center
        screen = np.array(ob.feature_screen.unit_type)
        screen = self._preprocess(screen)
        Y, X = np.where(screen == 0)  # empty location
        zeros = np.stack((Y, X), axis=1)
        dists = np.linalg.norm((zeros - self.command_coord), ord=1, axis=1)  # L1 norm
        coords = zeros[dists.argsort()]  # sort & adjust coord

        H, W = screen.shape
        for (y, x) in coords:
            if (x-_x) >= 0 and (x+_x+1) <= W and (y-_y) >= 0 and (y+_y+1) <= H:
                try:
                    if addon:
                        cum_sum = np.sum(screen[y-_y:y+_y+1, x-_x:x+_x+1] * _region)
                    else:
                        cum_sum = np.sum(screen[y-_y:y+_y, x-_x:x+_x] * _region)
                except ValueError:
                    print('[ VALUE ERROR ] : Calculation error occurred while computing cumulative sum.')
                    from IPython import embed; embed()
                if cum_sum == 0:
                    self.coords[i] = [x, y]  # (x,y) coordinate
                    if addon:
                        self.coords[i] = [x - 3, y]  # addon : (x - 3, y)
                    break
        assert len(self.coords[i]) != 0, "Valid coordinate not found"
        return func("now", self.coords[i])



