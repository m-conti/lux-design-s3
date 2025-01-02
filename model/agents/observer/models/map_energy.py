
import torch
from ..base import ObserverBase
from game_interface import Obs, map_max_energy


class MapEnergyObserver(ObserverBase):
  def extract_conv(self, obs: Obs) -> torch.Tensor:
    return torch.tensor(obs.map_features.energy / map_max_energy, dtype=torch.float)
