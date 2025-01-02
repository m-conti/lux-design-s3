
from abc import abstractmethod
import torch
from game_interface import Obs, PlayerAction
from model.agents.observer import ObserverBase
from model.agents.reward import RewardBase


class Agent:
  observers: list[ObserverBase]
  reward: RewardBase

  def __init__(self):
    self.observers = []
    pass

  def add_observer(self, observer: ObserverBase) -> "Agent":
    self.observers.append(observer)
    return self

  def set_reward(self, reward: RewardBase) -> "Agent":
    self.reward = reward
    return self

  @abstractmethod
  def _sample_actions(self, obs_tensor: torch.Tensor) -> PlayerAction:
    """Sample actions from the agent's policy."""



  def get_conv_tensor(self, obs: Obs):
    tensors = [tensor for observer in self.observers if (tensor := observer.extract_conv(obs)) is not None]
    if len(tensors) == 0:
      return torch.tensor([])
    return torch.stack(tensors)
  
  def get_fc_tensor(self, obs: Obs):
    tensors = [tensor for observer in self.observers if (tensor := observer.extract_fc(obs)) is not None]
    if len(tensors) == 0:
      return torch.tensor([])
    return torch.cat(tensors)
  
  def get_tensors(self, obs: Obs):
    return self.get_conv_tensor(obs), self.get_fc_tensor(obs)

  def update(self, obs: Obs):
    for observer in self.observers:
      observer.update(obs)

  def reset(self):
    for observer in self.observers:
      observer.reset()

  def get_reward(self, obs: Obs, next_obs: Obs):
    return self.reward.get(obs, next_obs)