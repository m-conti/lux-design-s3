
import torch
from ..base import ObserverBase
from game_interface import Obs, unit_max_energy


class UnitEnergyObserver(ObserverBase):
  def extract_fc(self, obs: Obs) -> torch.Tensor | None:
    return torch.tensor(obs.allies.energy / unit_max_energy, dtype=torch.float)