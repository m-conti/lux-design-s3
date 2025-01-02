from dataclasses import dataclass
from game.luxai_runner.episode import json_to_html
from game.luxai_s3.state import serialize_env_actions, serialize_env_states
from game.luxai_s3.wrappers import LuxAIS3GymEnv, PlayerName
from game_interface import GameParams, GodObs
from typing import Any, Literal
import gymnasium as gym
import os
import flax
import flax.serialization
from pathlib import Path
import numpy as np

@dataclass
class Episode:
  metadata: dict[str, Any]
  states: list[dict[str, Any]]
  actions: list[dict[str, Any]]
  params: dict[str, Any]

  def __init__(self):
    self.metadata = {}
    self.states = []
    self.actions = []
    self.params = {}


class RecordInterface(gym.Wrapper):
  def __init__(self, save_dir: str = "records", save_on_close: bool = True, save_on_reset: bool = True):
    self.env = LuxAIS3GymEnv(numpy_output=True)
    super().__init__(self.env) # type: ignore
    self.episode = Episode()
    self.episode_id = 0
    self.save_dir = save_dir
    self.save_on_close = save_on_close
    self.save_on_reset = save_on_reset
    self.episode_steps = 0
    Path(save_dir).mkdir(parents=True, exist_ok=True)

  def __del__(self):
    self.close()

  def reset(  # type: ignore
      self, *, seed: int | None = None, options: dict[str, Any] | None = None
  ) -> tuple[GodObs, GameParams]:
    if self.save_on_reset and self.episode_steps > 0:
      self._save_episode_and_reset()
    obs, config = self.env.reset(seed=seed, options=options)

    self.episode.metadata["seed"] = seed
    self.episode.params = flax.serialization.to_state_dict(config["full_params"])
    self.episode.states.append(config["state"])
    return GodObs.from_dict(obs), GameParams.from_dict(config["params"])

  def step(  # type: ignore
    self, action: Any
  ) -> tuple[
    GodObs,
    dict[PlayerName, np.ndarray[tuple[Literal[1]], np.dtype[np.float32]]],
    dict[PlayerName, np.ndarray[tuple[Literal[1]], np.dtype[np.bool_]]],
    dict[PlayerName, np.ndarray[tuple[Literal[1]], np.dtype[np.bool_]]],
    dict[str, Any],
  ]:
    obs, reward, terminated, truncated, info = self.env.step(action)
    self.episode_steps += 1
    self.episode.states.append(info["final_state"])
    self.episode.actions.append(action)
    return GodObs.from_dict(obs), reward, terminated, truncated, info

  def serialize_episode_data(self, episode: Episode | None = None):
    if episode is None:
      episode = self.episode
    ret = dict()
    ret["observations"] = serialize_env_states(episode.states)  # type: ignore
    if len(episode.actions) > 0:
      ret["actions"] = serialize_env_actions(episode.actions)
    ret["metadata"] = episode.metadata
    ret["params"] = episode.params
    return ret

  def save_episode(self, save_path: str):
    episode = self.serialize_episode_data()
    with open(save_path, "w") as f:
      f.write(json_to_html(episode))
    self.episode = Episode()

  def _save_episode_and_reset(self):
    """saves to generated path based on self.save_dir and episoe id and updates relevant counters"""
    self.save_episode(
        os.path.join(self.save_dir, f"episode_{self.episode_id}.html")
    )
    self.episode_id += 1
    self.episode_steps = 0

  def close(self):
    if self.save_on_close and self.episode_steps > 0:
      self._save_episode_and_reset()

  def init(self):
    _, config =self.reset()
    return self.step({
      "player_0": np.zeros((16, 3), dtype=np.int32),
      "player_1": np.zeros((16, 3), dtype=np.int32) 
    }), config