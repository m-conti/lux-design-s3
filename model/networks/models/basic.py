import torch
from ..base import BaseNN
import torch.nn as nn
import torch.nn.functional as F

from game_interface import n_actions, PlayerActionBatch


class CNN(BaseNN):
  def __init__(self, conv_size: int, fc_size: int):
      super().__init__(conv_size, fc_size)

      size_out_1 = conv_size * 4
      self.conv1 = nn.Conv2d(conv_size, size_out_1, 3)

      size_out_2 = size_out_1 * 2
      self.conv2 = nn.Conv2d(size_out_1, size_out_2, 3)

      size_out_3 = size_out_2 * 2
      self.conv3 = nn.Conv2d(size_out_2, size_out_3, 3)

      self.conv_out_size = size_out_3 * 4 * 4
      self.fc1 = nn.Linear((self.conv_out_size) + fc_size, 128)
      self.fc2 = nn.Linear(128, 16 * n_actions)

  def forward(self, x: torch.Tensor, fc_tensor: torch.Tensor) -> torch.Tensor:
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
      x = F.max_pool2d(x, 2)
      x = F.relu(self.conv3(x))
      x = F.max_pool2d(x, 2)
      x = x.view(-1, self.conv_out_size)
      x = torch.cat((x, fc_tensor), dim=1)
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
      x = x.view(-1, 16, 5)

      return x

  def _sample_actions(self, conv_tensor: torch.Tensor, fc_tensor: torch.Tensor) -> PlayerActionBatch:
    batch_size = conv_tensor.size(0)
    out = self.forward(conv_tensor, fc_tensor)
    actions = torch.zeros((batch_size, 16, 3), dtype=torch.int32)
    actions[:, :, 0] = out.argmax(2).int()

    return actions.numpy()