
import argparse

parser = argparse.ArgumentParser(description='Evaluate an agent from a training checkpoint')
parser.add_argument('--nruns', '-n', type=int, default=200, help='number of runs over which averaging the evaluation')
parser.add_argument('--steps', '-s', type=int, default=500, help='maximum number of steps per run')
parser.add_argument('--filepath', '-f', default='stats.json', help='evaluation output file path')
parser.add_argument('--legacy', '-l', help='use legacy agent', action='store_true')
ARGS = parser.parse_args()
print("ARGS:", ARGS)

import json

if not ARGS.legacy:
    from agent import A3C_Agent
else:
    from legacy_agent import A3C_Agent


configs = {'gui': False,
           'episodes_for_evaluation': ARGS.nruns}
agent = A3C_Agent(configs)

for run in range(ARGS.nruns):
    agent.environment.reset()
    
    if run % (ARGS.nruns // 20) == 0:
        print("run number: %s..." % run)
    
    for step in range(ARGS.steps):
        terminal = agent.act()
        
        if terminal:
            break

agent.environment.game.stop()
stats = agent.environment.get_statistics()

with open(ARGS.filepath, mode='w') as f:
    json.dump(stats, f, indent=4)

