
import torch
from ..base import ObserverBase
from game_interface import Obs


class MapEnergyObserver(ObserverBase):
  def extract_conv(self, obs: Obs) -> torch.Tensor:
    return torch.tensor(obs.map_features.energy, dtype=torch.float)
