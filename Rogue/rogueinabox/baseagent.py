from .ui.UIManager import UIManager
from .box import RogueBox
from abc import ABC, abstractmethod


class BaseAgent(ABC):

	def __init__(self, configs):
		self.configs = configs
		self.rb = self._create_rogue(configs)
		self._init_ui(configs)

	def _create_rogue(self, configs):
		"""
		:type configs: dict
		:rtype: RogueBox
		"""
		rb = RogueBox(configs["rogue"], configs["state_generator"], configs["reward_generator"], configs["max_step_count"])
		rb.reset()
		return rb

	def _init_ui(self, configs):
		if self.configs["gui"]:
			self._pending_action_timer = None
			self.ui = UIManager.init(configs["userinterface"], self.rb)
			self.ui.on_key_press(self._keypress_callback)
			self._timer_value = configs["timer_ms"]
			self._pending_action_timer = self.ui.on_timer_end(self._timer_value, self._act_callback)
		else:
			self.ui = None

	@abstractmethod
	def act(self):
		"""
		:rtype : bool
		:return: whether next state is terminal
		"""
		pass

	def run(self):
		if self.configs["gui"]:
			self.ui.start_ui()
		else:
			while(self.rb.is_running()):
				self.act()

	def _keypress_callback(self, event):
		if event.char == 'q' or event.char == 'Q':
			self.rb.quit_the_game()
			exit()
		elif event.char == 'r' or event.char == 'R':
			# we need to stop the agent from acting
			# or it will try to write to a closed pipe
			self.ui.cancel_timer(self._pending_action_timer)
			self.rb.reset()
			self._pending_action_timer = self.ui.on_timer_end(self._timer_value, self._act_callback)

	def game_over(self):
		# This must stay a separate method because of the interaction with the Judges
		# Takes care of restarting rogue and the agent
		self.rb.reset()

	def _act_callback(self):
		terminal = self.act()
		self.ui.draw_from_rogue()
		if not self.rb.game_over(self.rb.get_screen()):
			# renew the callback
			self._pending_action_timer = self.ui.on_timer_end(self._timer_value, self._act_callback)
		else:
			self.game_over()
