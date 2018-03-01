from .baseagent import BaseAgent
import random


class RandomAgent(BaseAgent):
	configuration_manager_style = "single"

	def act(self):
		actions = self.rb.get_actions()
		action = random.choice(actions)
		_,__, terminal = self.rb.send_command(action)
		return terminal


if __name__ == '__main__':
	configs = {
		'userinterface': 'curses',
		'verbose': 3,
		'gui': True,
		'rogue': 'rogue',
		'memory_size': 0,
		'test': False,
		'state_generator': 'SingleLayer_StateGenerator',
		'reward_generator': 'E_D_W_RewardGenerator',
		'max_step_count': 500
	}

	agent = RandomAgent(configs)
	agent.run()
