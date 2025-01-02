import torch
from tqdm import tqdm
from loop import Loop
from model.agents import BasicAgent
from model.networks import CNN
from model.memory import MemoryBuffer
from torch import optim

def main(
  lr: float,
):

  loop = Loop(
    AgentInit=BasicAgent,
    ModelInit=CNN,
    optimizer=optim.Adam,
    memory=MemoryBuffer(limit=50_000),
    lr=lr,
    log_name="with-save-model",
  )

  min_epsilon = 0.1
  max_epsilon = 0.9
  max_episodes = 8_000

  for i in tqdm(range(1, max_episodes + 1)):
    epsilon = max(min_epsilon, max_epsilon - (max_epsilon - min_epsilon) * (i / (0.4 * max_episodes)))
    loop.session(2, epsilon)
    loop.train(4, 256)

    if i % 10 == 0:
      loop.logger.set_data(epsilon=epsilon, buffer_size=len(loop.memory.buffer))
      print(f"{loop.logger}")
      loop.logger.log()

    if i % 20 == 0:
      loop.update_training_model()
    
    if i % 50 == 0:
      torch.save(loop.model, f"models_weights/{loop.model.save_name}_{i // 50}.pth")
      loop.record()


def init(
  profiling: bool = False,
  **kwargs
):

  call = "main(**kwargs)"
  # Run main function with functions performance profiling
  if profiling:
    import cProfile
    cProfile.run(call, sort="cumulative")
  # Run main function
  else:
    eval(call)



if __name__ == "__main__":
  init(
    profiling=False,
    lr=0.001,
  )
  pass