import numpy as np
from game_interface import Obs, n_agents
from ..base import RewardBase

class RewardExploration(RewardBase):
  def get(self, obs: Obs, next_obs: Obs):
    return np.full(n_agents, next_obs.sensor_mask.astype(np.float32).sum())