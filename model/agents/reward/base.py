from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

from game_interface import Obs

RewardData = NDArray[np.float32]

class RewardBase(ABC):
  @abstractmethod
  def get(self, obs: Obs, next_obs: Obs) -> RewardData:
    """Get the reward for the current observation."""

  def __add__(self, other: "RewardBase") -> "RewardBase":
    return RewardAdd(self, other)

  def __mul__(self, scalar: "float | RewardBase") -> "RewardBase":
    if isinstance(scalar, RewardBase):
      return RewardMul(self, scalar)
    return RewardScaler(self, scalar)

  def __rmul__(self, scalar: float) -> "RewardBase":
    return RewardScaler(self, scalar)


class RewardAdd(RewardBase):
  def __init__(self, reward1: RewardBase, reward2: RewardBase):
    self.reward1 = reward1
    self.reward2 = reward2

  def get(self, *args, **kwargs) -> RewardData:
    return self.reward1.get(*args, **kwargs) + self.reward2.get(*args, **kwargs)


class RewardMul(RewardBase):
  def __init__(self, reward1: RewardBase, reward2: RewardBase):
    self.reward1 = reward1
    self.reward2 = reward2

  def get(self, *args, **kwargs) -> RewardData:
    return self.reward1.get(*args, **kwargs) * self.reward2.get(*args, **kwargs)


class RewardScaler(RewardBase):
  def __init__(self, reward: RewardBase, scalar: float):
    self.reward = reward
    self.scalar = scalar

  def get(self, *args, **kwargs) -> RewardData:
    return self.reward.get(*args, **kwargs) * self.scalar