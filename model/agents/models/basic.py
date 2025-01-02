import torch
from model.agents.reward.models.exploration import RewardExploration
from ..agent import Agent
from ..observer import MapEnergyObserver, PlayerObserver

class BasicAgent(Agent):
  def __init__(self):
    super().__init__()
    self.add_observer(MapEnergyObserver())
    self.add_observer(PlayerObserver(0))
    self.add_observer(PlayerObserver(1))
    self.add_observer(PlayerObserver(2))
    self.add_observer(PlayerObserver(3))
    self.add_observer(PlayerObserver(4))
    self.add_observer(PlayerObserver(5))
    self.add_observer(PlayerObserver(6))
    self.add_observer(PlayerObserver(7))
    self.add_observer(PlayerObserver(8))
    self.add_observer(PlayerObserver(9))
    self.add_observer(PlayerObserver(10))
    self.add_observer(PlayerObserver(11))
    self.add_observer(PlayerObserver(12))
    self.add_observer(PlayerObserver(13))
    self.add_observer(PlayerObserver(14))
    self.add_observer(PlayerObserver(15))

    self.set_reward(RewardExploration())
