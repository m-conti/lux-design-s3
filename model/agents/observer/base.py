
from abc import ABC
from game_interface import Obs

import torch

class ObserverBase(ABC):
  def __init__(self):
    pass

  def extract_conv(self, obs: Obs) -> torch.Tensor | None:
    """Extract the features from the observation."""
    return None

  def extract_fc(self, obs: Obs) -> torch.Tensor | None:
    """Extract the features from the observation."""
    return None

  def update(self, obs: Obs):
    """Update the observer with the new observation."""
    pass

  def reset(self):
    """Reset the observer to its initial state."""
    pass

  def __add__(self, other: "ObserverBase") -> "ObserverBase":
    return ObserverAdd(self, other)

  def __mul__(self, scalar: "float | ObserverBase") -> "ObserverBase":
    if isinstance(scalar, ObserverBase):
      return ObserverMul(self, scalar)
    return ObserverScaler(self, scalar)

  def __rmul__(self, scalar: float) -> "ObserverBase":
    return ObserverScaler(self, scalar)


class ObserverAdd(ObserverBase):
  def __init__(self, observer1: ObserverBase, observer2: ObserverBase):
    super().__init__()
    self.observer1 = observer1
    self.observer2 = observer2

  def extract_conv(self, obs: Obs) -> torch.Tensor | None:
    result1 = self.observer1.extract_conv(obs)
    result2 = self.observer2.extract_conv(obs)
    if result1 is None:
      return result2
    if result2 is None:
      return result1
    return result1 + result2

  
  def extract_fc(self, obs: Obs) -> torch.Tensor | None:
    result1 = self.observer1.extract_fc(obs)
    result2 = self.observer2.extract_fc(obs)
    if result1 is None:
      return result2
    if result2 is None:
      return result1
    return result1 + result2

  
  def update(self, obs: Obs) -> None:
    self.observer1.update(obs)
    self.observer2.update(obs)


class ObserverMul(ObserverBase):
  def __init__(self, observer1: ObserverBase, observer2: ObserverBase):
    self.observer1 = observer1
    self.observer2 = observer2
    super().__init__()

  def extract_conv(self, obs: Obs) -> torch.Tensor | None:
    result1 = self.observer1.extract_conv(obs)
    result2 = self.observer2.extract_conv(obs)
    if result1 is None:
      return result2
    if result2 is None:
      return result1
    return result1 * result2
  
  def extract_fc(self, obs: Obs) -> torch.Tensor | None:
    result1 = self.observer1.extract_fc(obs)
    result2 = self.observer2.extract_fc(obs)
    if result1 is None:
      return result2
    if result2 is None:
      return result1
    return result1 * result2

  
  def update(self, obs: Obs) -> None:
    self.observer1.update(obs)
    self.observer2.update(obs)


class ObserverScaler(ObserverBase):
  def __init__(self, observer: ObserverBase, scalar: float):
    self.observer = observer
    self.scalar = scalar
    super().__init__()

  def extract_conv(self, obs: Obs) -> torch.Tensor | None:
    result = self.observer.extract_conv(obs)
    if result is None:
      return result
    return result * self.scalar

  def extract_fc(self, obs: Obs) -> torch.Tensor | None:
    result = self.observer.extract_fc(obs)
    if result is None:
      return result
    return result * self.scalar

  def update(self, obs: Obs) -> None:
    self.observer.update(obs)