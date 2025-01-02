import time
from typing import Callable, Type
from torch import nn, optim
import torch
from config import SAMPLING_DEVICE, TRAINING_DEVICE
from game_interface import GameInterface
from record_interface import RecordInterface
from model.agents import Agent
from model.logger import WandbLogger
from model.networks import BaseNN
from model.memory import MemoryBuffer


class Loop:
  game: GameInterface | RecordInterface
  model: nn.Module
  training_model: nn.Module

  player_0: Agent
  player_1: Agent

  memory: MemoryBuffer
  optimizer: optim.Optimizer

  logger: WandbLogger | None

  def __init__(
    self,
    AgentInit: Type[Agent],
    ModelInit: Type[BaseNN],
    optimizer: Callable[..., optim.Optimizer],
    memory: MemoryBuffer,
    lr: float,
    monitoring: bool = False,
   ):
    self.game = RecordInterface() if monitoring else GameInterface()
    self.player_0 = AgentInit()
    self.player_1 = AgentInit()

    (obs, _) = self.game.reset()

    self.model = ModelInit.from_tensors(self.player_0.get_conv_tensor(obs.player_0), self.player_0.get_fc_tensor(obs.player_0)).to(SAMPLING_DEVICE)
    self.training_model = ModelInit.from_model(self.model).to(TRAINING_DEVICE)

    self.memory = memory
    self.optimizer = optimizer(self.model.parameters(), lr=lr)

  def launch(self):
    game_finished = False
    turn = 0

    # Reset the agents to their initial state
    self.player_0.reset()
    self.player_1.reset()

    ((obs, _, _, _, _), config) = self.game.init()

    # Get the tensors for the first state
    self.player_0.update(obs.player_0)
    self.player_1.update(obs.player_1)
    (conv_tensor_0, fc_tensor_0) = self.player_0.get_tensors(obs.player_0)
    (conv_tensor_1, fc_tensor_1) = self.player_1.get_tensors(obs.player_1)

    while not game_finished:
      turn += 1

      # Get the actions of the player for the current state
      action_0 = self.model.sample_one_action(conv_tensor_0, fc_tensor_0)
      action_1 = self.model.sample_one_action(conv_tensor_1, fc_tensor_1)

      # Step the game
      (next_obs, _, _, done, _) = self.game.step({ "player_0": action_0, "player_1": action_1 })

      # Update the agents observers
      self.player_0.update(next_obs.player_0)
      self.player_1.update(next_obs.player_1)

      # Get the next tensors
      (next_conv_tensor_0, next_fc_tensor_0) = self.player_0.get_tensors(next_obs.player_0)
      (next_conv_tensor_1, next_fc_tensor_1) = self.player_1.get_tensors(next_obs.player_1)

      # Compute the reward
      reward_0 = self.player_0.get_reward(obs.player_0, next_obs.player_0)
      reward_1 = self.player_1.get_reward(obs.player_1, next_obs.player_1)

      # Save to the batch
      self.memory.put(
        conv_tensor_0,
        fc_tensor_0,
        action_0,
        reward_0,
        next_conv_tensor_0,
        next_fc_tensor_0,
        torch.from_numpy(obs.player_0.allies.alive),
        torch.from_numpy(next_obs.player_0.allies.alive),
      )
      self.memory.put(
        conv_tensor_1,
        fc_tensor_1,
        action_1,
        reward_1,
        next_conv_tensor_1,
        next_fc_tensor_1,
        torch.from_numpy(obs.player_1.allies.alive),
        torch.from_numpy(next_obs.player_1.allies.alive),
      )

      # Set the next state as the current state
      conv_tensor_0 = next_conv_tensor_0
      fc_tensor_0 = next_fc_tensor_0
      conv_tensor_1 = next_conv_tensor_1
      fc_tensor_1 = next_fc_tensor_1
      obs = next_obs

      # end the game if it is done
      if done["player_0"].item() or done["player_1"].item():
        game_finished = True
    
    return (turn, )
  
  def session(self, n: int):
    start_time = time.time()
    fps = []
    for _ in range(n):
      (turns, ) = self.launch()
      fps.append(turns / (time.time() - start_time))

    return self
  
  def train(self, n: int, batch_size: int):
    pass

  def log(self):
    if self.logger is None:
      self.logger = WandbLogger({"algo": "idqn", "project": "minimal-marl"}, "test")
    # self.logger.log()
    