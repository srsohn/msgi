'''Terran units.'''
import enum

class Neutral(enum.IntEnum):
    """Neutral units."""
    MineralField = 341
    VespeneGeyser = 342
    #MineralField = 146
    #RichVespeneGeyser = 344

class FullTerranUnits(enum.IntEnum):
    """Full Terran units."""
    CommandCenter = 18
    SupplyDepot = 19
    Barracks = 21
    EngineeringBay = 22
    Refinery = 20
    #Refinery = 1949  # Rich Refinery
    Factory = 27
    MissileTurret = 23
    SensorTower = 25
    Bunker = 24
    GhostAcademy = 26
    Armory = 29
    Starport = 28
    FusionCore = 30
    BarracksTechLab = 37
    #BarracksReactor = 38
    FactoryTechLab = 39
    #FactoryReactor = 40
    StarportTechLab = 41
    #StarportReactor = 42
    BarracksFlying = 46
    FactoryFlying = 43
    StarportFlying = 44
    SCV = 45
    Marine = 48
    Reaper = 49
    Marauder = 51
    Ghost = 50
    WidowMine = 498
    Hellion = 53
    Hellbat = 484
    Cyclone = 692
    SiegeTank = 33
    Thor = 52
    Banshee = 55
    Liberator = 689
    Medivac = 54
    VikingFighter = 35
    Raven = 56
    Battlecruiser = 57


"""Used in the Attribute : total count"""
class TerranUnits(enum.IntEnum):
    """Full Terran units."""
    CommandCenter = 18
    SupplyDepot = 19
    Barracks = 21
    EngineeringBay = 22
    Refinery = 20  # and Rich refinery
    Factory = 27
    MissileTurret = 23
    SensorTower = 25
    Bunker = 24
    GhostAcademy = 26
    Armory = 29
    Starport = 28  # 11
    FusionCore = 30
    BarracksTechLab = 37
    #BarracksReactor = 38
    FactoryTechLab = 39
    #FactoryReactor = 40
    StarportTechLab = 41
    #StarportReactor = 42
    #BarracksFlying = 46
    #FactoryFlying = 43
    #StarportFlying = 44
    SCV = 45  # 19
    Marine = 48
    Reaper = 49
    Marauder = 51
    Ghost = 50
    WidowMine = 498
    Hellion = 53
    Hellbat = 484
    Cyclone = 692
    SiegeTank = 33
    Thor = 52
    Banshee = 55
    Liberator = 689
    Medivac = 54
    VikingFighter = 35
    Raven = 56
    Battlecruiser = 57


class Structures(enum.IntEnum):
    """Terran target building units."""
    CommandCenter = 18
    SupplyDepot = 19
    Barracks = 21
    EngineeringBay = 22
    Refinery = 20
    Factory = 27
    MissileTurret = 23
    SensorTower = 25
    Bunker = 24
    GhostAcademy = 26
    Armory = 29
    Starport = 28
    FusionCore = 30
    BarracksTechLab = 37
    #BarracksReactor = 38
    FactoryTechLab = 39
    #FactoryReactor = 40
    StarportTechLab = 41
    #StarportReactor = 42

class GroundUnits(enum.IntEnum):
    """Terran target ground units."""
    SCV = 45
    Marine = 48
    Reaper = 49
    Marauder = 51
    Ghost = 50
    WidowMine = 498
    Hellion = 53
    Hellbat = 484
    Cyclone = 692
    SiegeTank = 33
    Thor = 52
    Banshee = 55
    Liberator = 689
    Medivac = 54
    VikingFighter = 35
    Raven = 56
    Battlecruiser = 57


class Selections(enum.IntEnum):
    """Terran target building units."""
    CommandCenter = 18  # --- 0
    SupplyDepot = 19
    Barracks = 21
    EngineeringBay = 22
    #Refinery = 20
    Factory = 27
    MissileTurret = 23  # --- 5
    SensorTower = 25
    Bunker = 24
    GhostAcademy = 26
    Armory = 29
    Starport = 28  # -------- 10
    FusionCore = 30
    BarracksTechLab = 37
    FactoryTechLab = 39
    StarportTechLab = 41
    SCV = 45  # ------------- 15
    Marine = 48
    Reaper = 49
    Marauder = 51
    Ghost = 50
    WidowMine = 498  # ------ 20
    Hellion = 53
    Hellbat = 484
    Cyclone = 692
    SiegeTank = 33
    Thor = 52  # ------------ 25
    Banshee = 55
    Liberator = 689
    Medivac = 54
    VikingFighter = 35
    Raven = 56
    Battlecruiser = 57  # --- 31


class Productions(enum.IntEnum):
    """Terran production building units."""
    CommandCenter = 18
    Barracks = 21
    Factory = 27
    Starport = 28

class AddOnUnits(enum.IntEnum):
    """Terran production building units."""
    Barracks = 21
    Factory = 27
    Starport = 28

class CommandCenterUnits(enum.IntEnum):
    """Terran target ground units."""
    SCV = 45

class BarracksUnits(enum.IntEnum):
    """Terran target ground units."""
    Marine = 48
    Reaper = 49
    Marauder = 51
    Ghost = 50

class FactoryUnits(enum.IntEnum):
    """Terran target ground units."""
    WidowMine = 498
    Hellion = 53
    Hellbat = 484
    Cyclone = 692
    SiegeTank = 33
    Thor = 52

class StarportUnits(enum.IntEnum):
    """Terran target ground units."""
    Banshee = 55
    Liberator = 689
    Medivac = 54
    VikingFighter = 35
    Raven = 56
    Battlecruiser = 57

class FlyingUnits(enum.IntEnum):
  BarracksFlying = 46
  FactoryFlying = 43
  StarportFlying = 44


# list of available Terran units
_COMMANDCENTER = [id for id in CommandCenterUnits]
_BARRACKS = [id for id in BarracksUnits]
_FACTORY = [id for id in FactoryUnits]
_STARPORT = [id for id in StarportUnits]
_FLYING = [id for id in FlyingUnits]
FULL_UNITS = [id for id in TerranUnits]
UNITS = [id for _type in [Structures, GroundUnits] for id in _type]
SCREEN_UNITS = [id for _type in [Neutral, FullTerranUnits] for id in _type]
