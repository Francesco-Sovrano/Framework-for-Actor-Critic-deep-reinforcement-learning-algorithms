
import argparse

parser = argparse.ArgumentParser(description='Evaluate an agent from a training checkpoint')
parser.add_argument('--legacy', '-l', help='use legacy agent', action='store_true')
ARGS = parser.parse_args()
print("ARGS:", ARGS)

if not ARGS.legacy:
    from agent import A3C_Agent
else:
    from legacy_agent import A3C_Agent


configs = {
   'userinterface': 'curses',
   'verbose': 3,
   'gui': True,
   'memory_size': 0,
   'test': False,
   'timer_ms': 250
}

agent = A3C_Agent(configs)
agent.run()

