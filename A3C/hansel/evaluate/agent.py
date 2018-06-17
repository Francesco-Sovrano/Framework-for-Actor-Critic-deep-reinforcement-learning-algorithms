
import numpy as np

from main import Application
import tensorflow as tf
import options
flags = options.get()

from roguelib_module.baseagent import BaseAgent
from environment.environment import Environment


class A3C_Agent(BaseAgent):
    def __init__(self, configs):
        app = Application()
        app.sess = tf.Session()
        app.device = '/cpu:0'
        app.build_global_network(tf.placeholder(tf.float64))
        app.trainers = []
        app.load_checkpoint()
        self.app = app
        self.model = app.global_network
        self.environment = Environment.create_environment(flags.env_type, -1)

        self.episode_index = 1

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

        if flags.show_all_screenshots:
            self.record_screen(pi, value)

        action = np.random.choice(range(len(pi)), p=pi)

        new_state, _reward, win, lose = self.environment.process(action)

        if win and flags.show_all_screenshots:
            last_action = self.environment.last_action
            last_reward = self.environment.last_reward
            last_action_reward = self.model.concat_action_and_reward(last_action, last_reward)
            new_situation = self.environment.last_situation
            agent = self.model.get_agent(new_situation)
            pi, value = agent.run_policy_and_value(self.app.sess, new_state, last_action_reward)
            self.record_screen(pi, value)

        return win or lose

    def game_over(self):
        self.episode_index += 1
        super().game_over()

    def record_screen(self, pi, value):
        screen = self.environment.game.get_screen()[:]  # type: list[str]
        decorate = False
        if decorate:
            screen.append('')

            acts_names = ['left', 'down', 'up', 'right', 'descent']
            v_name = 'value'

            all_names = acts_names + [v_name]
            names_space = max(len(name) for name in all_names) + 2
            perc_space = 50

            for a_i, (name, prob) in enumerate(zip(acts_names, pi)):
                line = name + ' ' * (names_space - len(name))
                line += '['
                n_perc = int(round(prob * perc_space))
                line += '=' * n_perc + ' ' * (perc_space - n_perc)
                line += ']'
                screen.append(line)

            value_line = v_name + ' ' * (names_space - len(v_name)) + str(value)
            screen.append(value_line)

        step = str(self.environment.game.step_count)
        step = '0' * (3 - len(step)) + step
        fname = flags.log_dir + '/screenshots/ep%sst%s.txt' % (self.episode_index, step)
        with open(fname, mode='w') as file:
            print(*screen, sep='\n', file=file)


