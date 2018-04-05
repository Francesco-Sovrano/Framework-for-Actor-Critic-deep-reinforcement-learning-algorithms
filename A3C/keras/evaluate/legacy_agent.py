
import numpy as np

from legacy.train_noconv import Application
import tensorflow as tf
from options import flags

from roguelib_module.baseagent import BaseAgent
from environment.environment import Environment


class A3C_Agent(BaseAgent):
    def __init__(self, configs):
        app = Application()
        app.sess = tf.Session()
        app.device = '/cpu:0'
        app.build_global_network(tf.placeholder(tf.float64))
        app.load_checkpoint()
        self.app = app
        self.model = app.global_network
        self.environment = Environment.create_environment(flags.env_type, -1)
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
        pi, value = agent.run_policy_and_value(self.app.sess, prev_state, last_action_reward)
        action = np.random.choice(range(len(pi)), p=pi)

        _new_state, _reward, win, lose = self.environment.process(action)

        return win or lose
