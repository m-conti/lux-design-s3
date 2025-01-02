
import torch
from ..base import ObserverBase
from game_interface import Obs, map_size


class UnitPositionObserver(ObserverBase):
  def __init__(self, player_id: int):
    self.player = player_id
    super().__init__()

  def extract_conv(self, obs: Obs) -> torch.Tensor:
    map_position = torch.zeros(map_size, dtype=torch.float)
    if not obs.allies.alive[self.player]:
       return map_position
    map_position[obs.allies.position[self.player]] = 1
    return map_position
