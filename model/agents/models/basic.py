from model.agents.reward import RewardExploration, RewardEnergy
from ..agent import Agent
from ..observer import MapEnergyObserver, UnitPositionObserver, UnitEnergyObserver

class BasicAgent(Agent):
  def __init__(self):
    super().__init__()
    self.add_observer(MapEnergyObserver())
    self.add_observer(UnitPositionObserver(0))
    self.add_observer(UnitPositionObserver(1))
    self.add_observer(UnitPositionObserver(2))
    self.add_observer(UnitPositionObserver(3))
    self.add_observer(UnitPositionObserver(4))
    self.add_observer(UnitPositionObserver(5))
    self.add_observer(UnitPositionObserver(6))
    self.add_observer(UnitPositionObserver(7))
    self.add_observer(UnitPositionObserver(8))
    self.add_observer(UnitPositionObserver(9))
    self.add_observer(UnitPositionObserver(10))
    self.add_observer(UnitPositionObserver(11))
    self.add_observer(UnitPositionObserver(12))
    self.add_observer(UnitPositionObserver(13))
    self.add_observer(UnitPositionObserver(14))
    self.add_observer(UnitPositionObserver(15))
    self.add_observer(UnitEnergyObserver())

    self.set_reward(RewardExploration() + 0.1 * RewardEnergy())
