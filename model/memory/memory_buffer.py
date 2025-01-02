import collections
import random
import torch

from game.luxai_s3.env import PlayerAction
from model.agents.reward import RewardData

class MemoryBuffer:
  def __init__(self, limit: int):
    self.buffer: collections.deque[
      tuple[
        torch.Tensor,   # obs_conv
        torch.Tensor,   # obs_fc
        PlayerAction,   # action
        RewardData,     # reward
        torch.Tensor,   # next_obs_conv
        torch.Tensor,   # next_obs_fc
        torch.Tensor,   # alive
        torch.Tensor,   # next_alive
      ]
    ] = collections.deque(maxlen=limit)

  def put(
    self,
    obs_conv: torch.Tensor,
    obs_fc: torch.Tensor,
    action: PlayerAction,
    reward: RewardData,
    next_obs_conv: torch.Tensor,
    next_obs_fc: torch.Tensor,
    alive: torch.Tensor,
    next_alive: torch.Tensor,
  ):
    self.buffer.append((obs_conv, obs_fc, action, reward, next_obs_conv, next_obs_fc, alive, next_alive))

  def sample(self, batch_size: int):
    batch = random.sample(self.buffer, batch_size)
    obs_conv_tensor = torch.empty((batch_size, *batch[0][0].shape), dtype=torch.float)
    obs_fc_tensor = torch.empty((batch_size, *batch[0][1].shape), dtype=torch.float)
    action_tensor = torch.empty((batch_size, *batch[0][2].shape), dtype=torch.float)
    reward_tensor = torch.empty((batch_size, *batch[0][3].shape), dtype=torch.float)
    next_obs_conv_tensor = torch.empty((batch_size, *batch[0][4].shape), dtype=torch.float)
    next_obs_fc_tensor = torch.empty((batch_size, *batch[0][5].shape), dtype=torch.float)
    alive_tensor = torch.empty((batch_size, *batch[0][6].shape), dtype=torch.float)
    next_alive_tensor = torch.empty((batch_size, *batch[0][7].shape), dtype=torch.float)

    for i, (obs_conv, obs_fc, action, reward, next_obs_conv, next_obs_fc, alive, next_alive) in enumerate(batch):
      obs_conv_tensor[i] = obs_conv
      obs_fc_tensor[i] = obs_fc
      action_tensor[i] = torch.tensor(action)
      reward_tensor[i] = torch.tensor(reward)
      next_obs_conv_tensor[i] = next_obs_conv
      next_obs_fc_tensor[i] = next_obs_fc
      alive_tensor[i] = alive
      next_alive_tensor[i] = next_alive

    return (
      obs_conv_tensor,
      obs_fc_tensor,
      action_tensor,
      reward_tensor,
      next_obs_conv_tensor,
      next_obs_fc_tensor,
      alive_tensor,
      next_alive_tensor,
    )

  def __len__(self):
    return len(self.buffer)
