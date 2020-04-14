import time
import os
from collections import deque
import numpy as np
import torch
#import imageio

class Mazebase_high(object):
    def __init__(self, args):
        self.gamma = args.gamma

        #visualization
        self._rendering = args.render
        self.render_dir = 'render'
        self.fixed_map_mode = args.fixed_map_mode

        #load game config (outcome)
        self.config = args.game_config
        if args.game=="playground":
            self.rendering_scale = 32
        elif args.game=='mining':
            self.rendering_scale = 96
        else:
            assert(not self._rendering)

        self.subtask_list = self.config.subtask_list
        self.n_actions = self.config.nb_subtask_type

        #init
        self.game_length_base = args.max_step
        self.omask = torch.zeros(len(self.config.object_list)) # needed for mining domain
        ###init state = (obs, feats)
        self.w, self.h, self.ch = args.game_config.width, args.game_config.height, args.game_config.nb_obj_type
        self.obs_shape = [self.ch, self.w, self.h, ]
        self.map_tensor = torch.zeros( self.obs_shape, dtype=torch.uint8 )

    def act(self, tid):
        if self.finished:
            return 0

        step, done = self._move_to_closest_obj(tid)
        self.step = self.step + step
        self.render()
        return step, done

    def reset_trial(self, rmag=None, subtask_id_list=None, id_to_ind=None):
        self.game_length = self.game_length_base #round(self.game_length_base * np.random.uniform(0.8,1.2,1).item())

        # reset graph and map when starting new training trial
        if not rmag is None and not subtask_id_list is None:
            self.rmag = rmag
            self.subtask_id_list = subtask_id_list
            self.id_to_ind = id_to_ind
            reset_map_flag = True
            self.ntasks = len(rmag)
        else:
            reset_map_flag = False

        return self._reset(reset_map=reset_map_flag) # reset episode (& map)

    def _reset(self, reset_map=False, obj_only=False, epi_index = 0): # reset episode
        self.epi_index = epi_index
        # reset state (except observation)
        self.step = 0
        self.finished = False

        # reset map
        if reset_map: #if not a fixed map or resetting task -> reset map.
            self.empty_list = []
            self._reset_walls()
            if not obj_only:
                self._reset_targets() #init objects + agent (+walls)
        else: #recover
            self._recover_targets()
        self.render()
        return self.get_state()

    def get_state(self):
        return self.map_tensor

    def get_log_step(self):
        return torch.log10(torch.Tensor( [self.game_length - self.step+1] ) )

    ######  map
    def _reset_walls(self):
        # boundary
        self.walls = [(0,y) for y in range(self.h)] #left wall
        self.walls = self.walls + [(self.w-1,y) for y in range(self.h)] #right wall
        self.walls = self.walls + [(x,0) for x in range(self.w)] #bottom wall
        self.walls = self.walls + [(x,self.h-1) for x in range(self.w)] #top wall

        for x in range(self.w):
            for y in range(self.h):
                if (x,y) not in self.walls:
                    self.empty_list.append( (x,y) )

        #random block (for now, no block)
        """
        nb_block = np.random.randint(self.config.nb_block[0], self.config.nb_block[1])

        for i in range(nb_block):
            pool = np.random.permute(empty_list)
            success=False
            for (x, y) in pool:
                if not self._check_block():
                    success=True
                    break
            if success==False:
                assert(False)"""

    def _recover_targets(self):
        np.copyto(self.item_map, self.item_map_BU)
        self.map_tensor.copy_(self.map_tensor_BU)
        self.agent_x = self.agent_init_pos_x
        self.agent_y = self.agent_init_pos_y

        self.omask.copy_(self.omask_BU)
        self.object_list = [item for item in self.object_list_BU]

    def _reset_targets(self):
        #reset
        self.object_list = []
        self.map_tensor.zero_()
        self.omask.zero_()
        self.item_map = np.zeros( (self.w,self.h),dtype=np.int16)

        #create objects
        self.item_map.fill(-1)
        pool = np.random.permutation(self.empty_list)
        for ind in range(self.ntasks):
            self._place_object(ind, (pool[ind][0],pool[ind][1]) )

        #create agent
        (self.agent_x, self.agent_y) = pool[self.ntasks]

        #Backup for recover
        self.agent_init_pos_x = self.agent_x
        self.agent_init_pos_y = self.agent_y
        self.item_map_BU = self.item_map.copy()
        self.map_tensor_BU = self.map_tensor.clone()
        self.omask_BU = self.omask.clone()
        self.object_list_BU = [item for item in self.object_list]

    def _place_object(self, task_ind, pos):
        subid = self.subtask_id_list[task_ind]
        (_, oid) = self.subtask_list[subid]['param']
        if ('unique' not in self.config.object_list[oid]) or (not self.config.object_list[oid]['unique']) or (self.omask[oid].item()==0):
            self.omask[oid]=1
            obj = dict(oid = oid, pos = pos)
            self.object_list.append( obj )
            self.item_map[pos[0]][pos[1]] = oid
            self.map_tensor[oid][pos[0]][pos[1]] = 1

    def _check_block(self, empty_list):
        nb_empty = len(empty_list)
        mask = np.copy(self.occupied_map)
        #
        queue = deque([empty_list[0]])
        count = 0
        while len(queue)>0:
            [x, y] = queue.popleft()
            mask[x][y]=1
            count+=1
            candidate = [ (x+1, y), (x-1, y), (x, y+1), (x, y-1) ]
            for item in candidate:
                if mask[item[0]][item[1]]:
                    queue.append(item)
        return count == nb_empty

    def _get_cur_item(self):
        return self.item_map[self.agent_x][self.agent_y]

    def _process_obj(self, action, obj):
        pos, oid = obj['pos'], obj['oid']
        obj_config = self.config.object_list[oid]
        if not 'unique' in obj_config or not obj_config['unique']: # if not unique(e.g., jewel shop)
            # remove obj
            self.item_map[pos[0]][pos[1]] = -1
            self.object_list.remove( obj )
            self.map_tensor[oid][pos[0]][pos[1]] = 0

        if action == 'transform' and self.config.env_name == 'playground':
            # if transform, add ice
            obj = dict(oid = 8, pos = pos)
            self.object_list.append( obj )
            self.item_map[pos[0]][pos[1]] = 8
            self.map_tensor[8][pos[0]][pos[1]] = 1
        else:
            pass

    def _move_to_closest_obj(self, id):
        (action, oid) = self.config.subtask_param_list[id]
        mindist=pow(self.w,2)+pow(self.h,2)
        flag = False

        for oind in range(len(self.object_list)):
            obj=self.object_list[oind]
            if obj['oid']==oid:
                (tx,ty) = obj['pos']
                dist = abs(self.agent_x-tx) + abs(self.agent_y-ty)
                if mindist>dist:
                    mindist = dist
                    target_x = tx
                    target_y = ty
                    target_obj = obj
                    flag = True
        if not flag:
            print('Error! no target object!')
            import ipdb; ipdb.set_trace()
            assert(False)
        step = mindist + 1
        done = False
        if self.step + step <=self.game_length: #
            self.agent_x = target_x # move agent
            self.agent_y = target_y
            self._process_obj(action, target_obj) # process obj & move agent
        else:
            step = self.game_length - self.step
        if self.step + step == self.game_length:
            done = True
        return step, done

    def render(self):
        scale = self.rendering_scale
        if not self._rendering:
            return
        self.screen = np.ones( (scale*self.w, scale*self.h, 3) ).astype(np.uint8)*255
        #items
        for i in range(len(self.object_list)):
            obj = self.object_list[i]
            oid = obj['oid']
            obj_img = self.config.object_image_list[oid]
            self._draw_cell(obj['pos'], obj_img[:,:,:3])

        #walls
        for wall_pos in self.walls:
            self._draw_cell(wall_pos, self.config.wall_image[:,:,:3])
        #agent
        self._draw_cell( (self.agent_x, self.agent_y) , self.config.agent_image[:,:,:3])

        # grid
        indices = np.concatenate( (np.arange(0, scale*self.w, scale), np.arange(scale-1, scale*self.w, scale) ) )
        self.screen[indices, :] = 0
        indices = np.concatenate( (np.arange(0, scale*self.h, scale), np.arange(scale-1, scale*self.h, scale) ) )
        self.screen[:, indices] = 0
        if self._rendering:
            self.save_image()

    def _draw_cell(self, pos, obj_img = None):
        scale = self.rendering_scale
        if obj_img is None:
            pass
        else:
            np.copyto(self.screen[ pos[0]*scale : pos[0]*scale+scale, pos[1]*scale : pos[1]*scale+scale, : ], obj_img  )


    def get_action_name(self, action):
        if action>=0 and action<len(self.action_meanings):
            return self.action_meanings[action]
        else:
            print('Invalid action name!')
            return "INVALID"

    def save(self, root, filename):
        filepath = os.path.join(root,'data',filename+'.npy')
        data = np.array([
            self.rng,
            self.wall_width,
            self.opening_width,
            self.basic_unit,
            self.w, self.h,
            self.wall_rows,
            self.rendering_scale,
            self.ghosts,
            self.walls,
            self.goal_pos_x, self.goal_pos_y,
            self.walls_dict,
            self.fixed_map_mode,
            self.agent_x, self.agent_y,
            self.agent_init_pos_x, self.agent_init_pos_y,
        ])
        np.save(filepath, data)

    def load(self, root, filename):
        filepath = os.path.join(root,'data',filename+'.npy')
        data = np.load(filepath)
        self.rng,\
        self.wall_width,\
        self.opening_width,\
        self.basic_unit,\
        self.w, self.h,\
        self.wall_rows,\
        self.rendering_scale,\
        self.ghosts,\
        self.walls,\
        self.goal_pos_x, self.goal_pos_y,\
        self.walls_dict,\
        self.fixed_map_mode,\
        self.agent_x, self.agent_y,\
        self.agent_init_pos_x, self.agent_init_pos_y = data

    """def save_image(self):
        if self._rendering and self.render_dir is not None:
            imageio.imwrite(self.render_dir + '/render_epi'+str(self.epi_index)+'step_' + str(self.step) + '.jpg', self.screen)
        else:
            raise ValueError('env._rendering is False and/or environment has not been reset.')"""
