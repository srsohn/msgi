import copy
import numpy as np
import torch

from common.actions import ARG_TYPES, _TERRAN_FUNCTIONS
from common.units import GroundUnits, Structures, Selections, Productions, AddOnUnits, \
    _COMMANDCENTER, _BARRACKS, _FACTORY, _STARPORT, _FLYING, UNITS, FULL_UNITS, SCREEN_UNITS
from pysc2.lib.actions import FUNCTIONS


# build region
_SMALL_REGION = np.array(
    [
        [ 0,  0,  0, -1, -1, -1, -1, -1, -1,  0,  0,  0],
        [ 0,  0, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0],
        [ 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0],
        [ 0,  0, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0],
        [ 0,  0,  0, -1, -1, -1, -1, -1, -1,  0,  0,  0],
     ]
)

_MID_REGION = np.array(
    [
        [ 0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0],
        [ 0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0],
        [ 0,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0],
        [ 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0],
        [ 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0],
        [ 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0],
        [ 0,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0],
        [ 0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0],
        [ 0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0]
     ]
)

_MID_REGION_ADDON = np.array(
    [
        [ 0,  0,  0,  0,  0, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0],
        [ 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0, -1, -1, -1, -1,  0,  0],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 0,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0, -1, -1, -1, -1, -1, -1,  0],
        [ 0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0, -1, -1, -1, -1,  0,  0],
        [ 0,  0,  0,  0,  0, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
     ]
)

# center coordinate
_MID_SHAPE = _MID_REGION.shape
_MID_REGION_CENTER = (int(_MID_SHAPE[0]/2), int(_MID_SHAPE[1]/2))
_SMALL_SHAPE = _SMALL_REGION.shape
_SMALL_REGION_CENTER = (int(_SMALL_SHAPE[0]/2), int(_SMALL_SHAPE[1]/2))
_MID_ADDON_SHAPE = _MID_REGION_ADDON.shape
_MID_REGION_ADDON_CENTER = (int(_MID_ADDON_SHAPE[0]/2), int(_MID_ADDON_SHAPE[1]/2))

MAP_INFO = [
    {#0
        "map" : "BuildCommandCenter",
        "type-target" : ("build", 18),
        "region" : _MID_REGION,
        "center" : _MID_REGION_CENTER,
        "func" : FUNCTIONS.Build_CommandCenter_screen,
        "cmd_quick" : False,
        "add_on" : False
    },
    {#1
        "map" : "BuildSupplyDepot",
        "type-target" : ("build", 19),
        "region" : _SMALL_REGION,
        "center" : _SMALL_REGION_CENTER,
        "func" : FUNCTIONS.Build_SupplyDepot_screen,
        "cmd_quick" : False,
        "add_on" : False
    },
    {#2
        "map" : "BuildBarracks",
        "type-target" : ("build", 21),
        "region" : _MID_REGION,
        "center" : _MID_REGION_CENTER,
        "func" : FUNCTIONS.Build_Barracks_screen,
        "cmd_quick" : False,
        "add_on" : False
    },
    {#3
        "map" : "BuildEngineeringBay",
        "type-target" : ("build", 22),
        "region" : _MID_REGION,
        "center" : _MID_REGION_CENTER,
        "func" : FUNCTIONS.Build_EngineeringBay_screen,
        "cmd_quick" : False,
        "add_on" : False
    },
    {#4
        "map" : "BuildRefinery",
        "type-target" : ("build", 20),
        #"type-target" : ("build", 1949),  # Refinery for Rich Vespene
        "region" : _MID_REGION,
        "center" : _MID_REGION_CENTER,
        "func" : FUNCTIONS.Build_Refinery_screen,
        "cmd_quick" : False,
        "add_on" : False
    },
    {#5
        "map" : "BuildFactory",
        "type-target" : ("build", 27),
        "region" : _MID_REGION,
        "center" : _MID_REGION_CENTER,
        "func" : FUNCTIONS.Build_Factory_screen,
        "cmd_quick" : False,
        "add_on" : False
    },
    {#6
        "map" : "BuildMissileTurret",
        "type-target" : ("build", 23),
        "region" : _SMALL_REGION,
        "center" : _SMALL_REGION_CENTER,
        "func" : FUNCTIONS.Build_MissileTurret_screen,
        "cmd_quick" : False,
        "add_on" : False
    },
    {#7
        "map" : "BuildSensorTower",
        "type-target" : ("build", 25),
        "region" : _SMALL_REGION,
        "center" : _SMALL_REGION_CENTER,
        "func" : FUNCTIONS.Build_SensorTower_screen,
        "cmd_quick" : False,
        "add_on" : False
    },
    {#8
        "map" : "BuildBunker",
        "type-target" : ("build", 24),
        "region" : _MID_REGION,
        "center" : _MID_REGION_CENTER,
        "func" : FUNCTIONS.Build_Bunker_screen,
        "cmd_quick" : False,
        "add_on" : False
    },
    {#9
        "map" : "BuildGhostAcademy",
        "type-target" : ("build", 26),
        "region" : _MID_REGION,
        "center" : _MID_REGION_CENTER,
        "func" : FUNCTIONS.Build_GhostAcademy_screen,
        "cmd_quick" : False,
        "add_on" : False
    },
    {#10
        "map" : "BuildArmory",
        "type-target" : ("build", 29),
        "region" : _MID_REGION,
        "center" : _MID_REGION_CENTER,
        "func" : FUNCTIONS.Build_Armory_screen,
        "cmd_quick" : False,
        "add_on" : False
    },
    {#11
        "map" : "BuildStarport",
        "type-target" : ("build", 28),
        "region" : _MID_REGION,
        "center" : _MID_REGION_CENTER,
        "func" : FUNCTIONS.Build_Starport_screen,
        "cmd_quick" : False,
        "add_on" : False
    },
    {#12
        "map" : "BuildFusionCore",
        "type-target" : ("build", 30),
        "region" : _MID_REGION,
        "center" : _MID_REGION_CENTER,
        "func" : FUNCTIONS.Build_FusionCore_screen,
        "cmd_quick" : False,
        "add_on" : False
    },
    {#13
        "map" : "BuildTechLabBarracks",
        "type-target" : ("addon", (21, 37, 46)),  # target: (Barracks, BarracksTechLab, BarracksFlying)
        "region" : _MID_REGION_ADDON,
        "center" : _MID_REGION_ADDON_CENTER,
        "func" : FUNCTIONS.Build_TechLab_screen,
        "cmd_quick" : True,
        "quick_func" : FUNCTIONS.Build_TechLab_quick,
        "add_on" : True
    },
    {#14
        "map" : "BuildTechLabFactory",
        "type-target" : ("addon", (27, 39, 43)),  # target: (Factory, FactoryTechLab, FactoryFlying)
        "region" : _MID_REGION_ADDON,
        "center" : _MID_REGION_ADDON_CENTER,
        "func" : FUNCTIONS.Build_TechLab_screen,
        "cmd_quick" : True,
        "quick_func" : FUNCTIONS.Build_TechLab_quick,
        "add_on" : True
    },
    {#15
        "map" : "BuildTechLabStarport",
        "type-target" : ("addon", (28, 41, 44)),  # target: (Starport, StarportTechLab, StarportFlying)
        "region" : _MID_REGION_ADDON,
        "center" : _MID_REGION_ADDON_CENTER,
        "func" : FUNCTIONS.Build_TechLab_screen,
        "cmd_quick" : True,
        "quick_func" : FUNCTIONS.Build_TechLab_quick,
        "add_on" : True
    },
    {#16
        "map" : "BuildSCV",
        "type-target" : ("unit", 45),
        "func" : FUNCTIONS.Train_SCV_quick,
        "cmd_quick" : True,
        "quick_func" : FUNCTIONS.Train_SCV_quick,
        "add_on" : False
    },
    {#17
        "map" : "BuildMarine",
        "type-target" : ("unit", 48),
        "func" : FUNCTIONS.Train_Marine_quick,
        "cmd_quick" : True,
        "quick_func" : FUNCTIONS.Train_Marine_quick,
        "add_on" : False
    },
    {#18
        "map" : "BuildReaper",
        "type-target" : ("unit", 49),
        "func" : FUNCTIONS.Train_Reaper_quick,
        "cmd_quick" : True,
        "quick_func" : FUNCTIONS.Train_Reaper_quick,
        "add_on" : False
    },
    {#19
        "map" : "BuildMarauder",
        "type-target" : ("unit", 51),
        "func" : FUNCTIONS.Train_Marauder_quick,
        "cmd_quick" : True,
        "quick_func" : FUNCTIONS.Train_Marauder_quick,
        "add_on" : False
    },
    {#20
        "map" : "BuildGhost",
        "type-target" : ("unit", 50),
        "func" : FUNCTIONS.Train_Ghost_quick,
        "cmd_quick" : True,
        "quick_func" : FUNCTIONS.Train_Ghost_quick,
        "add_on" : False
    },
    {#21
        "map" : "BuildWidowMine",
        "type-target" : ("unit", 498),
        "func" : FUNCTIONS.Train_WidowMine_quick,
        "cmd_quick" : True,
        "quick_func" : FUNCTIONS.Train_WidowMine_quick,
        "add_on" : False
    },
    {#22
        "map" : "BuildHellion",
        "type-target" : ("unit", 53),
        "func" : FUNCTIONS.Train_Hellion_quick,
        "cmd_quick" : True,
        "quick_func" : FUNCTIONS.Train_Hellion_quick,
        "add_on" : False
    },
    {#23
        "map" : "BuildHellbat",
        "type-target" : ("unit", 484),
        "func" : FUNCTIONS.Train_Hellbat_quick,
        "cmd_quick" : True,
        "quick_func" : FUNCTIONS.Train_Hellbat_quick,
        "add_on" : False
    },
    {#24
        "map" : "BuildCyclone",
        "type-target" : ("unit", 692),
        "func" : FUNCTIONS.Train_Cyclone_quick,
        "cmd_quick" : True,
        "quick_func" : FUNCTIONS.Train_Cyclone_quick,
        "add_on" : False
    },
    {#25
        "map" : "BuildSiegeTank",
        "type-target" : ("unit", 33),
        "func" : FUNCTIONS.Train_SiegeTank_quick,
        "cmd_quick" : True,
        "quick_func" : FUNCTIONS.Train_SiegeTank_quick,
        "add_on" : False
    },
    {#26
        "map" : "BuildThor",
        "type-target" : ("unit", 52),
        "func" : FUNCTIONS.Train_Thor_quick,
        "cmd_quick" : True,
        "quick_func" : FUNCTIONS.Train_Thor_quick,
        "add_on" : False
    },
    {#27
        "map" : "BuildBanshee",
        "type-target" : ("unit", 55),
        "func" : FUNCTIONS.Train_Banshee_quick,
        "cmd_quick" : True,
        "quick_func" : FUNCTIONS.Train_Banshee_quick,
        "add_on" : False
    },
    {#28
        "map" : "BuildLiberator",
        "type-target" : ("unit", 689),
        "func" : FUNCTIONS.Train_Liberator_quick,
        "cmd_quick" : True,
        "quick_func" : FUNCTIONS.Train_Liberator_quick,
        "add_on" : False
    },
    {#29
        "map" : "BuildMedivac",
        "type-target" : ("unit", 54),
        "func" : FUNCTIONS.Train_Medivac_quick,
        "cmd_quick" : True,
        "quick_func" : FUNCTIONS.Train_Medivac_quick,
        "add_on" : False
    },
    {#30
        "map" : "BuildViking",
        "type-target" : ("unit", 35),
        "func" : FUNCTIONS.Train_VikingFighter_quick,
        "cmd_quick" : True,
        "quick_func" : FUNCTIONS.Train_VikingFighter_quick,
        "add_on" : False
    },
    {#31
        "map" : "BuildRaven",
        "type-target" : ("unit", 56),
        "func" : FUNCTIONS.Train_Raven_quick,
        "cmd_quick" : True,
        "quick_func" : FUNCTIONS.Train_Raven_quick,
        "add_on" : False
    },
    {#32
        "map" : "BuildBattleCruiser",
        "type-target" : ("unit", 57),
        "func" : FUNCTIONS.Train_Battlecruiser_quick,
        "cmd_quick" : True,
        "quick_func" : FUNCTIONS.Train_Battlecruiser_quick,
        "add_on" : False
    },
    {# 33
        "map" : "IdleSCV",
        "type-target" : ("idle", 45),
        "func" : FUNCTIONS.Train_SCV_quick,
        "cmd_quick" : True,
        "quick_func" : FUNCTIONS.Train_SCV_quick,
        "add_on" : False
    },
    {#34
        "map" : "SelectCommandCenter",
        "type-target" : ("select", 18),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#35
        "map" : "SelectSupplyDepot",
        "type-target" : ("select", 19),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#36
        "map" : "SelectBarracks",
        "type-target" : ("select", 21),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#37
        "map" : "SelectEngineeringBay",
        "type-target" : ("select", 22),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    #{
    #    "map" : "SelectRefinery",
    #    "type-target" : ("select", 20),
    #    "func" : FUNCTIONS.select_point,
    #    "cmd_quick" : False,
    #    "quick_func" : False,
    #    "add_on" : False
    #},
    {#38
        "map" : "SelectFactory",
        "type-target" : ("select", 27),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#39
        "map" : "SelectMissileTurret",
        "type-target" : ("select", 23),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#40
        "map" : "SelectSensorTower",
        "type-target" : ("select", 25),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#41
        "map" : "SelectBunker",
        "type-target" : ("select", 24),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#42
        "map" : "SelectGhostAcademy",
        "type-target" : ("select", 26),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#43
        "map" : "SelectArmory",
        "type-target" : ("select", 29),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#44
        "map" : "SelectStarport",
        "type-target" : ("select", 28),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#45
        "map" : "SelectFusionCore",
        "type-target" : ("select", 30),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#46
        "map" : "SelectBarracksTechLab",
        "type-target" : ("select", 37),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#47
        "map" : "SelectFactoryTechLab",
        "type-target" : ("select", 39),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#48
        "map" : "SelectStarportTechLab",
        "type-target" : ("select", 41),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#49
        "map" : "SelectSCV",
        "type-target" : ("select", 45),
        "func" : FUNCTIONS.select_idle_worker,
        "cmd_quick" : True,
        "quick_func" : FUNCTIONS.select_idle_worker,
        "add_on" : False
    },
    {#50
        "map" : "SelectMarine",
        "type-target" : ("select", 48),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#51
        "map" : "SelectReaper",
        "type-target" : ("select", 49),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#52
        "map" : "SelectMarauder",
        "type-target" : ("select", 51),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#53
        "map" : "SelectGhost",
        "type-target" : ("select", 50),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#54
        "map" : "SelectWidowMine",
        "type-target" : ("select", 498),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#55
        "map" : "SelectHellion",
        "type-target" : ("select", 53),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#56
        "map" : "SelectHellbat",
        "type-target" : ("select", 484),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#57
        "map" : "SelectCyclone",
        "type-target" : ("select", 692),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#58
        "map" : "SelectSiegeTank",
        "type-target" : ("select", 33),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#59
        "map" : "SelectThor",
        "type-target" : ("select", 52),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#60
        "map" : "SelectBanshee",
        "type-target" : ("select", 55),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#61
        "map" : "SelectLiberator",
        "type-target" : ("select", 689),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#62
        "map" : "SelectMedivac",
        "type-target" : ("select", 54),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#63
        "map" : "SelectViking",
        "type-target" : ("select", 35),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#64
        "map" : "SelectRaven",
        "type-target" : ("select", 56),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#65
        "map" : "SelectBattlecruiser",
        "type-target" : ("select", 57),
        "func" : FUNCTIONS.select_point,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#66
        "map" : "Mineral50",
        "type-target" : ("mineral", 50),
        "func" : FUNCTIONS.no_op,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#67
        "map" : "Mineral75",
        "type-target" : ("mineral", 75),
        "func" : FUNCTIONS.no_op,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#68
        "map" : "Mineral100",
        "type-target" : ("mineral", 100),
        "func" : FUNCTIONS.no_op,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#69
        "map" : "Mineral125",
        "type-target" : ("mineral", 125),
        "func" : FUNCTIONS.no_op,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#70
        "map" : "Mineral150",
        "type-target" : ("mineral", 150),
        "func" : FUNCTIONS.no_op,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#71 (-15)
        "map" : "Mineral300",
        "type-target" : ("mineral", 300),
        "func" : FUNCTIONS.no_op,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#72
        "map" : "Mineral400",
        "type-target" : ("mineral", 400),
        "func" : FUNCTIONS.no_op,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#73
        "map" : "Gas25",
        "type-target" : ("gas", 25),
        "func" : FUNCTIONS.no_op,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#74
        "map" : "Gas50",
        "type-target" : ("gas", 50),
        "func" : FUNCTIONS.no_op,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#75
        "map" : "Gas75",
        "type-target" : ("gas", 75),
        "func" : FUNCTIONS.no_op,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#76 (-10)
        "map" : "Gas100",
        "type-target" : ("gas", 100),
        "func" : FUNCTIONS.no_op,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#77
        "map" : "Gas125",
        "type-target" : ("gas", 125),
        "func" : FUNCTIONS.no_op,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#78
        "map" : "Gas150",
        "type-target" : ("gas", 150),
        "func" : FUNCTIONS.no_op,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#79
        "map" : "Gas200",
        "type-target" : ("gas", 200),
        "func" : FUNCTIONS.no_op,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#80
        "map" : "Gas300",
        "type-target" : ("gas", 300),
        "func" : FUNCTIONS.no_op,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    },
    {#81
        "map" : "Food1",
        "type-target" : ("food", 19),
        "region" : _SMALL_REGION,
        "center" : _SMALL_REGION_CENTER,
        "func" : FUNCTIONS.Build_SupplyDepot_screen,
        "cmd_quick" : False,
        "add_on" : False
    },
    {#82
        "map" : "Food2",
        "type-target" : ("food", 19),
        "region" : _SMALL_REGION,
        "center" : _SMALL_REGION_CENTER,
        "func" : FUNCTIONS.Build_SupplyDepot_screen,
        "cmd_quick" : False,
        "add_on" : False
    },
    {#83
        "map" : "Food3",
        "type-target" : ("food", 19),
        "region" : _SMALL_REGION,
        "center" : _SMALL_REGION_CENTER,
        "func" : FUNCTIONS.Build_SupplyDepot_screen,
        "cmd_quick" : False,
        "add_on" : False
    },
    {#84
        "map" : "Food6",
        "type-target" : ("food", 19),
        "region" : _SMALL_REGION,
        "center" : _SMALL_REGION_CENTER,
        "func" : FUNCTIONS.Build_SupplyDepot_screen,
        "cmd_quick" : False,
        "add_on" : False
    },
    {#85
        "map" : "NO_OP",
        "type-target" : ("no_op", -1),
        "func" : FUNCTIONS.no_op,
        "cmd_quick" : False,
        "quick_func" : False,
        "add_on" : False
    }
]
""",
    40 : {
        "map" : "GatherMineral",
        "type-target" : ("gather", 341),
        "func" : FUNCTIONS.Train_Battlecruiser_quick,
        "cmd_quick" : True,
        "quick_func" : FUNCTIONS.Train_Battlecruiser_quick,
        "add_on" : False
    },
    41 : {
        "map" : "GatherGas",
        "type-target" : ("gather", 342),
        "func" : FUNCTIONS.Train_Battlecruiser_quick,
        "cmd_quick" : True,
        "quick_func" : FUNCTIONS.Train_Battlecruiser_quick,
        "add_on" : False
    },
    42 : {
        "map" : "No-op",
        "type-target" : ("noop", -1),
        "func" : FUNCTIONS.Train_Battlecruiser_quick,
        "cmd_quick" : True,
        "quick_func" : FUNCTIONS.Train_Battlecruiser_quick,
        "add_on" : False
    }
}"""

### SR
MAP_NAMES = [info['map'] for idx, info in enumerate(MAP_INFO)]
UNIT_REWARDS = [[0 ]*16 + \
                [.2] + \
                [.2, .3, 1.0, 3.6 ] + \
                [.3, .3, 0.6, 0.6, 1.8, 4.9 ] + \
                [1., 1., 0.7, 4.8, 1.0, 1. ]]
UNIT_REWARDS[0][1] = 0.01
SUBTASK_REWARDS = UNIT_REWARDS[0] + [0]*25 + [0.1] # 0.1 for no-op
FOOD_REQ = [[ 0 ]*16 + \
                [0] + \
                [1, 1, 2, 2] + \
                [2, 2, 2, 3, 3, 6] + \
                [3, 3, 2, 2, 2, 6] ]

def create_idx_mapper(actions, reverse=False):
    if reverse:
        idx_mapper = {action.id: idx for idx, action in enumerate(actions)}
    else:
        idx_mapper = {idx: action for idx, action in enumerate(actions)}
    return idx_mapper

def create_terran_action_set(actions):
    return set([action.id for action in actions])

# total number of action ids in pysc2
NUM_FUNCTIONS = len(FUNCTIONS)

# terran unit specific action specs
TERRAN_FUNCTIONS = _TERRAN_FUNCTIONS
ACTION_SET = create_terran_action_set(_TERRAN_FUNCTIONS)
ACTION_REVERSE_MAP = create_idx_mapper(_TERRAN_FUNCTIONS, reverse=True)

# terran unit hash table
NUM_UNITS = len(UNITS)  # w/o CommandCenter
NUM_FULL_UNITS = len(FULL_UNITS)  # w/ CommandCenter + Flying
STRUCTURES = set(Structures)
GROUND_UNITS = set(GroundUnits)
PRODUCTIONS = set(Productions)
SELECTIONS = set(Selections)
SELECTION_MAP = {int(unit): idx for idx, unit in enumerate(Selections)}
UNIT_MAP = {int(unit): idx for idx, unit in enumerate(UNITS)}
FULL_UNIT_MAP = {int(unit): idx for idx, unit in enumerate(FULL_UNITS)}
SCREEN_MAP = {int(unit): idx+1 for idx, unit in enumerate(SCREEN_UNITS)}

ADDON_FUNCS = {
        FUNCTIONS.Build_TechLab_screen.id,
        FUNCTIONS.Build_TechLab_quick.id,
}

ADDON_ELIG_MAP = {
    21 : 13,  # Barracks
    46 : 13,  # BarracksFlying

    27 : 14,  # Factory
    43 : 14,  # FactoryFlying

    28 : 15,  # Starport
    44 : 15   # StarportFlying
}

BUILD_IDX = [idx for idx, info in enumerate(MAP_INFO) if info['type-target'][0] == 'build']
BUILD_MAP = {info['func'].id : idx for idx, info in enumerate(MAP_INFO) \
            if info['type-target'][0] == 'build'}
_UNIT_ELIG_MAP = {info['func'].id : idx for idx, info in enumerate(MAP_INFO) \
            if info['type-target'][0] == 'unit'}
ELIG_MAP = {**BUILD_MAP, **_UNIT_ELIG_MAP}

FLYING = _FLYING
FLYING_ID = {
    46 : 21,  # BarracksFlying to Barracks
    43 : 27,  # FactoryFlying to Factory
    44 : 28   # StarportFlying to Starport
}
FLYING_MAP = {
    46 : 2,
    43 : 5,
    44 : 11
}

COMMANDCENTER = [info['func'].id for idx, info in enumerate(MAP_INFO) \
                 if info['type-target'][0] in ['unit'] and info['type-target'][1] in _COMMANDCENTER]
BARRACKS = [info['func'].id for idx, info in enumerate(MAP_INFO) \
                 if info['type-target'][0] in ['unit'] and info['type-target'][1] in _BARRACKS]
FACTORY = [info['func'].id for idx, info in enumerate(MAP_INFO) \
                 if info['type-target'][0] in ['unit'] and info['type-target'][1] in _FACTORY]
STARPORT = [info['func'].id for idx, info in enumerate(MAP_INFO) \
                 if info['type-target'][0] in ['unit'] and info['type-target'][1] in _STARPORT]
IS_TRAINING = {
    18 : COMMANDCENTER,
    21 : BARRACKS,
    27 : FACTORY,
    28 : STARPORT
}

ADDON_MASK_MAP = {
    21 : BARRACKS,  # Barracks
    46 : BARRACKS,  # BarracksFlying

    27 : FACTORY,  # Factory
    43 : FACTORY,  # FactoryFlying

    28 : STARPORT,  # Starport
    44 : STARPORT   # StarportFlying
}


def PRINT_GT_PRECOND():
    dim = SC2_GT.shape[0]
    for ind in range(dim):
            print('=========')
            print('[ {}. {} ]'.format(ind, COMP_INFO[ind]))
            cond = SC2_GT[ind]
            indices = cond.nonzero()[0]
            if len(indices)>0:
                for cind in indices:
                    print("\t{}".format(COMP_INFO[cind]))
            else:
                print('NONE')

# Subtask Pre-condition Matrix
_SC2_GT = [
#    0                            15   16                              32   33   34          40                  50                            65   66          72   73            80   81      85
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 1    0 0 0 0 0 0 0 0    0 0 0 0 0', # 0
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 1 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 1 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 1 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 1 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 1 0 0    0 0 0 1 0 0 0 0    0 0 0 0 0',
    '0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 1 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 1 0 0 0    0 0 0 1 0 0 0 0    0 0 0 0 0',  # XXX Structures
    '0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 1 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 1 0 0    0 1 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 1 0 0    0 0 0 1 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 1 0 0    0 0 0 1 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 1 0 0    0 0 0 0 0 1 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    1 0 0 0 0 0 0    1 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    1 0 0 0 0 0 0    1 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    1 0 0 0 0 0 0    1 0 0 0 0 0 0 0    0 0 0 0 0', # 15

    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    1 0 0 0 0 0 0    0 0 0 0 0 0 0 0    1 0 0 0 0', # 16
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    1 0 0 0 0 0 0    0 0 0 0 0 0 0 0    1 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    1 0 0 0 0 0 0    0 1 0 0 0 0 0 0    1 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 1 0 0 0 0    1 0 0 0 0 0 0 0    0 1 0 0 0',
    '0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 1 0 0    0 0 0 0 1 0 0 0    0 1 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 1 0 0 0 0 0    1 0 0 0 0 0 0 0    0 1 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 1 0 0 0 0    0 0 0 0 0 0 0 0    0 1 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 1 0 0 0 0    0 0 0 0 0 0 0 0    0 1 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 1 0 0    0 0 0 1 0 0 0 0    0 0 1 0 0',  # XXX Units
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 1 0 0    0 0 0 0 1 0 0 0    0 0 1 0 0',
    '0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 1 0    0 0 0 0 0 0 1 0    0 0 0 1 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 1 0 0    0 0 0 1 0 0 0 0    0 0 1 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 1 0 0    0 0 0 0 0 1 0 0    0 0 1 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 1 0 0 0 0    0 0 0 1 0 0 0 0    0 1 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 1 0 0    0 0 1 0 0 0 0 0    0 1 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 1 0 0 0 0    0 0 0 0 0 0 1 0    0 1 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 1    0 0 0 0 0 0 0 1    0 0 0 1 0', # 32

    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    1 0 0 0 0 0 0    0 0 0 0 0 0 0 0    1 0 0 0 0', # 33  XXX idleSCV

    '1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0', # 34
    '0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0', # 48
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    1    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',  # XXX Selects
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0', # 65

    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0', # 66
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    1 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 1 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 1 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',  # XXX Minerals
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 1 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 1 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 1 0    0 0 0 0 0 0 0 0    0 0 0 0 0', # 72

    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0', # 73
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    1 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 1 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 1 0 0 0 0 0    0 0 0 0 0',  # XXX Gas
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 1 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 1 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 1 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 1 0    0 0 0 0 0', # 80

    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0', # 81
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',  # XXX Food + Noop
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0 0    0 0 0 0 0', # 85
]
SC2_GT = np.array([np.fromstring(s, sep=' ') for s in _SC2_GT])

def get_key(dict_in, target_val):
    for key, val in dict_in.items():    # for name, age in dictionary.iteritems():  (for Python 2.x)
        if val == target_val:
            return key
    return -1

# Completion info map
COMP_INFO = {
    0 : 'commandcenter',
    1 : 'supplydepot',
    2 : 'barracks',
    3 : 'engineeringbay',
    4 : 'refinery',
    5 : 'factory',
    6 : 'missile turret',
    7 : 'sensor tower',
    8 : 'bunker',
    9 : 'ghost academy',
    10 : 'armory',
    11 : 'starport',
    12 : 'fusionCore',
    13 : 'barracksTechLab',
    14 : 'factoryTechLab',
    15 : 'starportTechLab',

    16 : 'SCV',
    17 : 'marine',
    18 : 'reaper',
    19 : 'marauder',
    20 : 'ghost',
    21 : 'widowMine',
    22 : 'hellion',
    23 : 'hellbat',
    24 : 'cyclone',
    25 : 'siegeTank',
    26 : 'thor',
    27 : 'banshee',
    28 : 'liberator',
    29 : 'medivac',
    30 : 'vikingFighter',
    31 : 'raven',
    32 : 'battlecruiser',

    33 : '#idle SCV>0',

    34 : "select=CommandCenter",
    35 : "select=SupplyDepot",
    36 : "select=Barracks",
    37 : "select=EngineeringBay",
    38 : "select=Factory",
    39 : "select=MissileTurret",
    40 : "select=SensorTower",
    41 : "select=Bunker",
    42 : "select=GhostAcademy",
    43 : "select=Armory",
    44 : "select=Starport",
    45 : "select=FusionCore",
    46 : "select=BarracksTechLab",
    47 : "select=FactoryTechLab",
    48 : "select=StarportTechLab",
    49 : "select=SCV",
    50 : "select=Marine",
    51 : "select=Reaper",
    52 : "select=Marauder",
    53 : "select=Ghost",
    54 : "select=WidowMine",
    55 : "select=Hellion",
    56 : "select=Hellbat",
    57 : "select=Cyclone",
    58 : "select=SiegeTank",
    59 : "select=Thor",
    60 : "select=Banshee",
    61 : "select=Liberator",
    62 : "select=Medivac",
    63 : "select=Viking",
    64 : "select=Raven",
    65 : "select=Battlecruiser",

    66 : "mineral>=50",
    67 : "mineral>=75",
    68 : "mineral>=100",
    69 : "mineral>=125",
    70 : "mineral>=150",
    71 : "mineral>=300",
    72 : "mineral>=400",

    73 : "gas>=25",
    74 : "gas>=50",
    75 : "gas>=75",
    76 : "gas>=100",
    77 : "gas>=125",
    78 : "gas>=150",
    79 : "gas>=200",
    80 : "gas>=300",

    81 : "food>=1",
    82 : "food>=2",
    83 : "food>=3",
    84 : "food>=6",
    85 : "no-op"
}

LABEL_NAME= {
    0 : 'ComCenter',
    1 : 'SupDepot',
    2 : 'Barracks',
    3 : 'EnginBay',
    4 : 'Refinery',
    5 : 'Factory',
    6 : 'MissTurret',
    7 : 'SensTower',
    8 : 'Bunker',
    9 : 'GhostAcad',
    10 : 'Armory',
    11 : 'Starport',
    12 : 'FusionCore',
    13 : 'Barracks\nTechLab',
    14 : 'Factory\nTechLab',
    15 : 'Starport\nTechLab',
    16 : 'SCV',
    17 : 'Marine',
    18 : 'Reaper',
    19 : 'Marauder',
    20 : 'Ghost',
    21 : 'WidowMine',
    22 : 'Hellion',
    23 : 'Hellbat',
    24 : 'Cyclone',
    25 : 'SiegeTank',
    26 : 'Thor',
    27 : 'Banshee',
    28 : 'Liberator',
    29 : 'Medivac',
    30 : 'Viking',
    31 : 'Raven',
    32 : 'Battlecruiser',

    33 : '#idle SCV>0',

    34 : "select\nComCenter",       #
    35 : "select\nSupplyDepot",
    36 : "select\nBarracks",        #
    37 : "select\nEngineeringBay",
    38 : "select\nFactory",         #
    39 : "select\nMissileTurret",
    40 : "select\nSensorTower",
    41 : "select\nBunker",
    42 : "select\nGhostAcademy",
    43 : "select\nArmory",
    44 : "select\nStarport",        #
    45 : "select\nFusionCore",
    46 : "select\nBarracksTechLab",
    47 : "select\nFactoryTechLab",
    48 : "select\nStarportTechLab",
    49 : "select\nSCV",             #
    50 : "select\nMarine",
    51 : "select\nReaper",
    52 : "select\nMarauder",
    53 : "select\nGhost",
    54 : "select\nWidowMine",
    55 : "select\nHellion",
    56 : "select\nHellbat",
    57 : "select\nCyclone",
    58 : "select\nSiegeTank",
    59 : "select\nThor",
    60 : "select\nBanshee",
    61 : "select\nLiberator",
    62 : "select\nMedivac",
    63 : "select\nViking",
    64 : "select\nRaven",
    65 : "select\nBattlecruiser",

    66 : "mineral\n50",
    67 : "mineral\n75",
    68 : "mineral\n100",
    69 : "mineral\n125",
    70 : "mineral\n150",
    71 : "mineral\n300",
    72 : "mineral\n400",
    73 : "gas\n25",
    74 : "gas\n50",
    75 : "gas\n75",
    76 : "gas\n100",
    77 : "gas\n125",
    78 : "gas\n150",
    79 : "gas\n200",
    80 : "gas\n300",
    81 : "food 1",
    82 : "food 2",
    83 : "food 3",
    84 : "food 6",
    85 : "no-op"
}

GROUND_UNITS = [17, 18, 19, 20, 22, 23, 24, 25, 26]

OPT_TO_COMP = []
for i in range(32):
    OPT_TO_COMP.append(i+1) # skip first command center. All thebuilding&units
OPT_TO_COMP.append( get_key(COMP_INFO, 'Select=SCV') ) # 5 selections
OPT_TO_COMP.append( get_key(COMP_INFO, 'Select=CommandCenter') )
OPT_TO_COMP.append( get_key(COMP_INFO, 'Select=Barracks') )
OPT_TO_COMP.append( get_key(COMP_INFO, 'Select=Factory') )
OPT_TO_COMP.append( get_key(COMP_INFO, 'Select=Starport') )
OPT_TO_COMP.append( list(range(46,54)) )
OPT_TO_COMP.append( [33, 34] + list(range(39,58)) )

SELECT_RANGE    = range( get_key(COMP_INFO, 'select=CommandCenter'), get_key(COMP_INFO, 'mineral>=50') )
UNIT_RANGE      = range( get_key(COMP_INFO, 'SCV'),                  get_key(COMP_INFO, '#idle SCV>0') )
BUILDING_RANGE  = range( 0,                                          get_key(COMP_INFO, 'SCV') )
FOOD_RANGE      = range( get_key(COMP_INFO, 'food>=1'),              get_key(COMP_INFO, 'no-op') )
BUILDING_RANGE_WITHOUT_TWO =range(2, get_key(COMP_INFO, 'SCV'))

_NOOP = TERRAN_FUNCTIONS[0]()


def ExponentialLR(optimizer, init_lr, epoch, gamma, decay_steps):
    '''Exponential learning rate scheduler.'''
    lr = init_lr * gamma**(epoch / decay_steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def masked_softmax(logits, mask, epsilon=1e-6):
    masked_logits = logits * mask
    minimum = masked_logits.min(dim=1, keepdim=True)[0]
    masked_nonneg_logits = (masked_logits - minimum) * mask
    maximum = masked_nonneg_logits.max(dim=1, keepdim=True)[0]
    exps = torch.exp(masked_nonneg_logits - maximum)
    masked_exps = exps * mask
    exp_sums = masked_exps.sum(dim=1, keepdim=True) + epsilon
    probs = masked_exps / exp_sums
    return probs

def encode_prev_act(acts, dones, sz, device):
    # XXX deepcopy in order to prevent the property
    # of the original variable changing
    acts = [act.cpu().numpy() for act in acts]
    _acts = copy.deepcopy(acts)

    # coord-to-heat map
    Nbatch = _acts[1].shape[0]
    screen = _one_hot(_acts[1], sz*sz).reshape( (Nbatch, 1, sz, sz) )

    # coord-to-norm-coord
    act_list = [_acts[0]] + _norm_coord(_acts[1], sz) + _acts[2:]
    out_list = []
    for act in act_list:
        out_list.append(np.expand_dims(act, axis=1))

    # if done, set zero
    for idx, done in enumerate(dones.tolist()):
        if done:
            screen[idx].fill(0)
            for act in out_list:
                act[idx] = 0

    prev_acts = [screen] + out_list
    return [torch.from_numpy(prev_act).float().to(device) for prev_act in prev_acts]

def _one_hot(indices, depth):
    num_row = len(indices)
    out_array = np.zeros( (num_row, depth) )
    out_array[np.arange(num_row), indices] = 1
    return out_array

def _norm_coord(flat_coord, sz):
    norm_coord_x = ( (flat_coord//sz) / (sz-1) )*2 -1
    norm_coord_y = ( (flat_coord%sz) / (sz-1) )*2 -1
    return [norm_coord_x, norm_coord_y]

def _update_window(window, new_value):
    if len(window.shape)==1:
        window[:-1] = window[1:]
        window[-1] = new_value
    else:
        window[:,:-1] = window[:,1:]
        window[:,-1] = new_value
    return window

def anneal(t, t0, t1, v0, v1):
    scale = max(0.0, min(1.0, 1.0 - (t - t0) / (t1 - t0)))  # scale : 1.0 --> 0.0
    return v1 + scale * (v0 - v1)  # coef : v0 --> v1

def warfare(envs, observations):
    '''Warfare between the units trained by the agent and the enemy units.'''
    print("======== [ WARFARE BEGINS IN 2 MINS ] ===========")
    before_war = True
    actives = np.ones(envs.num_envs)
    actions = [_NOOP]*envs.num_envs

    # print current units
    total_counts = envs.total_counts[0]
    PRINT_CURRENT_UNITS(total_counts)

    while True:
        obs, _, _ = observations['raws']
        observations, rewards, dones = envs.step(obs, actives, actions)

        if before_war and 18 not in [id for id, _ in obs[0].unit_counts]:
            # store the # units for logging
            total_counts = np.copy(envs.total_counts[0])
            before_war = False

        if any(dones):
            print("====================== WAR RESULT ========================")
            if rewards[0] > 0:
                print("Victory!! The agent has won the battle. [Reward : {}]".format(rewards[0]))
            else:
                print("Defeat!! The agent has been defeated. [Reward : {}]".format(rewards[0]))
            print("==========================================================")
            frames = 0
            break
    return observations, rewards, dones.astype(np.uint8), frames, total_counts

def save_results(ep, dirname, score, total_counts):
    if isinstance(score, np.ndarray):
        score = score.item()

    filename = dirname + '/result_ep-{}.txt'.format(ep)
    with open(filename, 'a') as f:
        string = 'ep: {:02d}\nscore: {:.02f}\n\n'.format(ep, score)
        f.writelines(string)

        inds = total_counts.nonzero()
        for ind in inds[0]:
            string = '{}:{}\n'.format(FULL_UNITS[ind].name, int(total_counts[ind]))
            f.writelines(string)

def encode_attr( attrs ):
    attrs = attrs.long()
    from common.utils import batch_bin_encode
    count_bin = (attrs[:, :NUM_FULL_UNITS]>0).long()
    idle_scvs = (attrs[:, 33]>0).long()
    selects = attrs[:, 34:39]
    #addons = attrs[:, 39]
    minerals = attrs[:, 39]
    gases = attrs[:, 40]
    foods = attrs[:, 41]
    #######
    code = batch_bin_encode(count_bin)
    #
    code *= 2
    code += idle_scvs
    #
    code *=5
    code += selects.nonzero()[:,1]
    """
    #
    code *=2
    code += addons"""
    #
    code *= 20
    code += torch.clamp(minerals//25, 0,19)
    #
    code *= 20
    code += torch.clamp(gases//25, 0,19)
    #
    code *= 8
    code += torch.clamp(foods, 0,7)
    return code

def PRINT_SUBTASKS(vec):
    if len(vec)==1:
        vec = vec[0]
    inds = vec.nonzero()
    for ind in inds:
        ii = ind.item()
        print( COMP_INFO[ii] )

def PRINT_CURRENT_UNITS(total_counts):
    inds = total_counts.nonzero()
    for ind in inds[0]:
        print( FULL_UNITS[ind], int(total_counts[ind]) )
