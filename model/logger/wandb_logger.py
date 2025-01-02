
from typing import Any


class WandbLogger:
  step = 0

  buffer_size: int = 0
  fps: list
  scores: list
  episode = 0
  epsilon: float = 0.0
  train_loss: float = 0.0

  def __init__(self,config: dict[str, Any], name: str):
    import wandb
    self.run = wandb.init(project=config['project'], name=name)
    self.run.config.update(config)
    self.run.config.update({'run_name': name})
    self.reset()

  def log(self):
    metrics = {
      'episode': self.episode,
      'buffer-size': self.buffer_size,
      'train-loss': self.train_loss,
      'fps': self.get_fps(),
      'score': self.get_score(),
      'epsilon': self.epsilon,
    }
    self.run.log(metrics, step=self.step)
    self.step += 1
    self.reset()

  def finish(self):
    self.run.finish()

  def reset(self):
    self.test_score = 0
    self.buffer_size = 0
    self.train_loss = 0
    self.fps = []
    self.scores = []
  
  def get_fps(self):
    return sum(self.fps) / len(self.fps) if len(self.fps) > 0 else 0
  
  def __str__(self) -> str:
    return f"FPS: {self.get_fps()} | Score: {self.get_score():.2f} | Buffer Size: {self.buffer_size} | Train Loss: {self.train_loss:.6f} | Epsilon: {self.epsilon:.2f}"
  
  def get_score(self):
    return sum(self.scores) / len(self.scores) if len(self.scores) > 0 else 0

  def set_run(self, fps: float, score: float):
    self.episode += 1
    self.fps.append(fps)
    self.scores.append(score)

  def set_train(self, train_loss: float):
    self.train_loss = train_loss

  def set_data(self, epsilon: float, buffer_size: int):
    self.epsilon = epsilon
    self.buffer_size = buffer_size

  def __del__(self):
    self.finish()