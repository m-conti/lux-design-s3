from game_interface import Obs, unit_max_energy
from ..base import RewardBase

class RewardEnergy(RewardBase):
  def get(self, obs: Obs, next_obs: Obs):
    return obs.allies.energy / unit_max_energy