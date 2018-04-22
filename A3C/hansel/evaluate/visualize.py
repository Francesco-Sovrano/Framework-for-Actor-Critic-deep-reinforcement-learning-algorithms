
import argparse

parser = argparse.ArgumentParser(description='Evaluate an agent from a training checkpoint')
ARGS = parser.parse_args()
print("ARGS:", ARGS)

from agent import A3C_Agent


configs = {
   'userinterface': 'curses',
   'verbose': 3,
   'gui': True,
   'memory_size': 0,
   'test': False,
   'timer_ms': 50
}

agent = A3C_Agent(configs)
agent.run()

