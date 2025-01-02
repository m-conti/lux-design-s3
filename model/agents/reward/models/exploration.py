from operator import mul
import numpy as np
from game_interface import Obs, n_agents, map_size
from ..base import RewardBase

class RewardExploration(RewardBase):
  def get(self, obs: Obs, next_obs: Obs):
    return np.full(n_agents, next_obs.sensor_mask.sum().astype(np.float32) / mul(*map_size))