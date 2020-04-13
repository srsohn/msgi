import torch
import numpy as np

from pysc2.lib import features
from common.actions import ARG_TYPES
from common.sc2_utils import TERRAN_FUNCTIONS, ACTION_SET, ACTION_REVERSE_MAP, \
    SCREEN_MAP, NUM_FULL_UNITS

CAT = features.FeatureType.CATEGORICAL

# name => dims
NON_SPATIAL_FEATURES = dict(
    player=(11,),
    game_loop=(1,),
    score_cumulative=(13,),
    available_actions=(len(TERRAN_FUNCTIONS),),
    single_select=(1, 7),
    cargo_slots_available=(1,),
    control_groups=(10, 2)
)


class Config:
    def __init__(self, args, device, data_format="NCHW",
                 embed_dim_fn=lambda x: max(1, int(round(np.log2(x))))):
        self.run_id, self.spec = args.run_id, args.spec
        self.device = device
        self.sz, self.map = args.feature_screen_size, args.map
        self.df = data_format
        self.embed_dim_fn = embed_dim_fn

        # model configs
        self.sp_hdim = args.sp_hdim
        self.fl_hdim = 259
        self.act_hdim = args.act_hdim
        self.option_dims = 86
        self.feats = {
            'screen' : ['unit_type', 'selected'],
            'non_spatial' : ['available_actions']
        }

        self.act_args = ARG_TYPES._fields
        self.arg_idx = {arg: i for i, arg in enumerate(self.act_args)}
        self.ns_idx = {f: i for i, f in enumerate(self.feats['non_spatial'])}

    def map_id(self):
        return self.map

    def full_id(self):
        if self.run_id == -1:
            return self.map_id()
        model_id = str(self.run_id) + '-' + self.spec if self.spec else str(self.run_id)
        return self.map_id() + "/" + model_id

    def policy_dims(self):
        return [(len(TERRAN_FUNCTIONS), 0)] + [(getattr(ARG_TYPES, arg).sizes[0], is_spatial(arg)) for arg in self.act_args]

    def screen_dims(self):
        return self._dims('screen')

    def minimap_dims(self):
        return self._dims('minimap')

    def non_spatial_dims(self):
        return [NON_SPATIAL_FEATURES[f] for f in self.feats['non_spatial']]

    def preprocess(self, obs):
        feat_screen = self._preprocess(obs, _type='feature_screen')
        avail_acts = self._preprocess(obs, _type='available_actions')
        return (feat_screen, avail_acts)

    def _dims(self, _type):
        return [NUM_FULL_UNITS if f.name == 'unit_type' else f.scale**(f.type == CAT) \
                for f in self._feats(_type)]

    def _feats(self, _type):
        if _type == 'feature_screen':
            _type = 'screen'
        feats = getattr(features, _type.upper() + '_FEATURES')
        return [getattr(feats, f_name) for f_name in self.feats[_type]]

    def _preprocess(self, obs, _type):
        # non-spatial
        if _type in self.feats['non_spatial']:
            non_spatials = [self._preprocess_non_spatial(ob, _type) for ob in obs]
            masks = torch.tensor([m for m, _ in non_spatials]).float().to(self.device)
            avail_acts = torch.tensor([a for _, a in non_spatials]).to(self.device)
            return masks, avail_acts

        # spatial
        spatials = np.asarray([[ob[_type][f.index] for f in self._feats(_type)] for ob in obs])

        # map terran units
        _unknown = False
        for spatial in spatials:
            unit_type = spatial[0]  # obs.unit_type
            uY, uX = np.nonzero(unit_type)
            for y, x in zip(uY, uX):
                unit_id = unit_type[y][x]
                if unit_id in SCREEN_MAP:
                    spatial[0][y][x] = SCREEN_MAP[unit_id]
                else:
                    _unknown = True
        return torch.from_numpy(spatials).float().to(self.device)

    def _preprocess_non_spatial(self, ob, _type):
        if _type == 'available_actions':
            masks = np.zeros(len(TERRAN_FUNCTIONS))
            commons = ACTION_SET.intersection(set(ob['available_actions']))
            ids = [ACTION_REVERSE_MAP[i] for i in commons]
            masks[ids] = 1
            return [masks, list(commons)]
        return ob[_type]

def is_spatial(arg):
    return arg in ['screen']
