from game.luxai_s3.wrappers import LuxAIS3GymEnv
from game.luxai_s3.env import N_Players, N_Agents, PlayerName, PlayerAction, PlayerActionBatch, n_agents, n_players
from dataclasses import dataclass
from typing import Any, Literal
from numpy.typing import NDArray
import numpy as np

n_actions = 5
map_size = (24, 24)

@dataclass(frozen=True)
class Units:
  position: np.ndarray[tuple[Literal[1]], np.dtype[np.int8]]
  energy: np.ndarray[tuple[Literal[1]], np.dtype[np.int16]]
  alive: np.ndarray[tuple[Literal[1]], np.dtype[np.bool_]]


@dataclass(frozen=True)
class MapFeatures:
  energy: NDArray
  tile_type: NDArray


@dataclass(frozen=True)
class Obs:
  allies: Units
  enemies: Units
  sensor_mask: NDArray[np.bool]
  map_features: MapFeatures
  relic_nodes: NDArray
  relic_nodes_mask: NDArray
  team_points: NDArray
  team_wins: NDArray
  steps: int
  match_steps: int

  @staticmethod
  def from_dict(obs: dict[str, Any], team_id: int):
    return Obs(
      allies=Units(obs["units"]["position"][team_id], obs["units"]["energy"][team_id], obs["units_mask"][team_id]),
      enemies=Units(obs["units"]["position"][1 - team_id], obs["units"]["energy"][1 - team_id], obs["units_mask"][1 - team_id]),
      sensor_mask=obs["sensor_mask"],
      map_features=MapFeatures(obs["map_features"]["energy"], obs["map_features"]["tile_type"]),
      relic_nodes=obs["relic_nodes"],
      relic_nodes_mask=obs["relic_nodes_mask"],
      team_points=obs["team_points"],
      team_wins=obs["team_wins"],
      steps=obs["steps"],
      match_steps=obs["match_steps"],
    )

  def get_available_units(self):
    return np.where(self.allies.alive)[0]

  def get_available_relics(self):
    return np.where(self.relic_nodes_mask)[0]


@dataclass(frozen=True)
class GodObs:
  player_0: Obs
  player_1: Obs

  @staticmethod
  def from_dict(obs: dict[PlayerName, Any]):
    return GodObs(
      Obs.from_dict(obs["player_0"], 0),
      Obs.from_dict(obs["player_1"], 1),
    )

@dataclass(frozen=True)
class GameParams:
  max_units: int
  match_count_per_episode: int
  max_steps_in_match: int
  map_height: int
  map_width: int
  num_teams: int
  unit_move_cost: int
  unit_sap_cost: int
  unit_sap_range: int
  unit_sensor_range: int

  @staticmethod
  def from_dict(env_params: dict[str, Any]):
    return GameParams(
      max_units=env_params["max_units"],
      match_count_per_episode=env_params["match_count_per_episode"],
      max_steps_in_match=env_params["max_steps_in_match"],
      map_height=env_params["map_height"],
      map_width=env_params["map_width"],
      num_teams=env_params["num_teams"],
      unit_move_cost=env_params["unit_move_cost"],
      unit_sap_cost=env_params["unit_sap_cost"],
      unit_sap_range=env_params["unit_sap_range"],
      unit_sensor_range=env_params["unit_sensor_range"],
    )


class GameInterface(LuxAIS3GymEnv):
  def __init__(self):
    super().__init__(numpy_output=True)

  def reset(  # type: ignore
    self, *, seed: int | None = None, options: dict[str, Any] | None = None
  ) -> tuple[GodObs, GameParams]:
    obs, config = super().reset(seed=seed, options=options)
    return GodObs.from_dict(obs), GameParams.from_dict(config['params'])

  def step(  # type: ignore
    self, actions
  ) -> tuple[
    GodObs,
    dict[PlayerName, np.ndarray[tuple[Literal[1]], np.dtype[np.float32]]],
    dict[PlayerName, np.ndarray[tuple[Literal[1]], np.dtype[np.bool_]]],
    dict[PlayerName, np.ndarray[tuple[Literal[1]], np.dtype[np.bool_]]],
    dict[str, Any],
  ]:
    obs, reward, terminated, truncated, info = super().step(actions)
    return GodObs.from_dict(obs), reward, terminated, truncated, info

  def init(self):
    _, config =self.reset()
    return self.step({
      "player_0": np.zeros((16, 3), dtype=np.int32),
      "player_1": np.zeros((16, 3), dtype=np.int32) 
    }), config
