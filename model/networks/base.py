from abc import abstractmethod
from typing import cast
import numpy as np
import torch
import torch.nn as nn

from game_interface import PlayerAction, PlayerActionBatch


class BaseNN(nn.Module):
  @abstractmethod
  def __init__(self, conv_size: int, fc_size: int):
      super().__init__()
      self.conv_size = conv_size
      self.fc_size = fc_size
  
  @classmethod
  def from_tensors(cls, conv_source: torch.Tensor, fc_source: torch.Tensor) -> 'BaseNN':
    return cls(conv_source.size(0), fc_source.size(0))

  @classmethod
  def from_model(cls, model: 'BaseNN') -> 'BaseNN':
    new = cls(model.conv_size, model.fc_size)
    new.load_state_dict(model.state_dict())
    return new

  @abstractmethod
  def _sample_actions(self, conv_tensor: torch.Tensor, fc_tensor: torch.Tensor) -> PlayerActionBatch:
    pass

  @torch.no_grad()
  def sample_actions(self, conv_tensor: torch.Tensor, fc_tensor: torch.Tensor) -> PlayerActionBatch:
    return self._sample_actions(conv_tensor, fc_tensor)
  
  def sample_one_action(self, conv_tensor: torch.Tensor, fc_tensor: torch.Tensor) -> PlayerAction:
    return cast(PlayerAction, self.sample_actions(conv_tensor.unsqueeze(0), fc_tensor.unsqueeze(0)).squeeze(0))