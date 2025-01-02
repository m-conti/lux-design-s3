from loop import Loop
from model.agents import BasicAgent
from model.networks import CNN
from model.memory import MemoryBuffer
from torch import optim

def main(
  monitoring: bool,
  lr: float,
):

  loop = Loop(
    AgentInit=BasicAgent,
    ModelInit=CNN,
    optimizer=optim.Adam,
    memory=MemoryBuffer(limit=1000),
    lr=lr,
    monitoring=monitoring,
  )

  loop.session(10)

  pass


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
    monitoring=True,
    lr=0.001,
  )
  pass