
import numpy as np

import train
Application = train.Application

from roguelib_module.baseagent import BaseAgent


class A3C_Agent(BaseAgent):
    def __init__(self, configs):
        self.app = Application()
        self.app.load_checkpoint()
        self.model = self.app.build_network(0)
        self.model.set_weights(self.app.global_weigths)
        self.environment = self.app.environment
        super().__init__(configs)

    def _create_rogue(self, configs):
        self.environment.reset()
        return self.environment.game

    def act(self):
        prev_state = self.environment.last_state
        last_action = self.environment.last_action
        last_reward = self.environment.last_reward
        last_action_reward = self.model.concat_action_and_reward(last_action, last_reward)
        last_situation = self.environment.last_situation

        agent = self.model.get_agent(last_situation)
        pi, value = agent.run_policy_and_value(prev_state, last_action_reward)
        action = np.random.choice(range(len(pi)), p=pi)

        _new_state, _reward, win, lose = self.environment.process(action)

        return win or lose
