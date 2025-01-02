import time
from typing import Callable, Type
from torch import nn, optim
import torch
from torch.nn import functional as F
from config import SAMPLING_DEVICE, TRAINING_DEVICE
from game_interface import GameInterface
from record_interface import RecordInterface
from model.agents import Agent
from model.logger import WandbLogger
from model.networks import BaseNN
from model.memory import MemoryBuffer


class Loop:
  game: GameInterface
  game_with_record: RecordInterface
  model: nn.Module
  training_model: nn.Module

  player_0: Agent
  player_1: Agent

  memory: MemoryBuffer
  optimizer: optim.Optimizer

  logger: WandbLogger

  def __init__(
    self,
    AgentInit: Type[Agent],
    ModelInit: Type[BaseNN],
    optimizer: Callable[..., optim.Optimizer],
    memory: MemoryBuffer,
    log_name: str,
    lr: float,
   ):
    self.logger = WandbLogger({"algo": "idqn", "project": "minimal-marl"}, log_name)

    self.game = GameInterface()
    self.game_with_record = RecordInterface()
    self.player_0 = AgentInit()
    self.player_1 = AgentInit()

    (obs, _) = self.game.reset()

    self.model = ModelInit.from_tensors(self.player_0.get_conv_tensor(obs.player_0), self.player_0.get_fc_tensor(obs.player_0)).to(SAMPLING_DEVICE)
    self.training_model = ModelInit.from_model(self.model).to(TRAINING_DEVICE)

    self.memory = memory
    self.optimizer = optimizer(self.model.parameters(), lr=lr)


  def launch(self, with_record: bool = False, epsilon: float = 0.0):
    game_finished = False
    turn = 0

    game = self.game_with_record if with_record else self.game

    # Reset the agents to their initial state
    self.player_0.reset()
    self.player_1.reset()

    ((obs, _, _, _, _), config) = game.init()

    # Get the tensors for the first state
    self.player_0.update(obs.player_0)
    self.player_1.update(obs.player_1)
    (conv_tensor_0, fc_tensor_0) = self.player_0.get_tensors(obs.player_0)
    (conv_tensor_1, fc_tensor_1) = self.player_1.get_tensors(obs.player_1)
    score: float = 0.0

    while not game_finished:
      turn += 1

      # Get the actions of the player for the current state
      action_0 = self.model.sample_one_action(conv_tensor_0, fc_tensor_0, epsilon)
      action_1 = self.model.sample_one_action(conv_tensor_1, fc_tensor_1, epsilon)

      # Step the game
      (next_obs, _, _, done, _) = game.step({ "player_0": action_0, "player_1": action_1 })

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

      # Update the score
      score += (reward_0 * obs.player_0.allies.alive).sum()

      # end the game if it is done
      if done["player_0"].item() or done["player_1"].item():
        game_finished = True
    
    return (turn, score)

  def record(self, epsilon: float = 0.0):
    self.launch(with_record=True, epsilon=epsilon)
    return self

  def session(self, n: int, epsilon: float = 0.0):
    for _ in range(n):
      start_time = time.time()
      (turns, score) = self.launch(epsilon=epsilon)
      self.logger.set_run(turns / (time.time() - start_time), score)

    return self
  
  def update_training_model(self):
    self.training_model.load_state_dict(self.model.state_dict())

  def train(self, n: int, batch_size: int):
    self.model.to(TRAINING_DEVICE)

    train_loss = 0
    GAMMA = 0.99

    for _ in range(n):
      (obs_conv, obs_fc, action, reward, next_obs_conv, next_obs_fc, alive, next_alive) = self.memory.sample(batch_size)

      result = self.model(obs_conv.to(TRAINING_DEVICE), obs_fc.to(TRAINING_DEVICE)).cpu()
      # get the Q value of the action taken
      action_prediction = result.gather(2, action[:, :, 0].unsqueeze(-1).long()).squeeze(-1)

      # get the Q' value of the best action
      result_prime = self.training_model(next_obs_conv.to(TRAINING_DEVICE), next_obs_fc.to(TRAINING_DEVICE)).cpu().max(dim=2).values

      # compute the target
      target = alive * (reward + GAMMA * next_alive * result_prime)
      loss = F.smooth_l1_loss(action_prediction * alive, target.detach())

      train_loss += loss.item()

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

    self.model.to(SAMPLING_DEVICE)

    self.logger.set_train(train_loss / n)
    return self

  def log(self):
    self.logger.log()
    