# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import h5py
from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense, LSTM, Concatenate, Activation, Lambda, TimeDistributed
from keras.initializers import Initializer
from keras import backend as K


from lib.layers import ReLu
from lib import a3c
from lib.keras import load_optmizer_from_hdf5_file, save_optmizer_to_hdf5_file

import options
flags = options.get()


class Initializer_muupan(Initializer):
	# weight initialization based on muupan's code
	# https://github.com/muupan/async-rl/blob/master/a3c_ale.py

	def __init__(self, input_channels):
		self.d = 1.0 / np.sqrt(input_channels)

	def __call__(self, shape, dtype=None):
		return K.random_uniform(shape, minval=-self.d, maxval=self.d, dtype=dtype)


def identity_loss(y_true, y_pred):
	"""
	Implements a keras loss functions that simply outputs the loss computed in y_pred and ignores y_true.
	N.B. a dummy y_true must still be provided to keras .fit() method
	"""
	return y_pred


class MultiAgentModel(object):
	def __init__( self, id, state_shape, agents_count, action_size, local_t_max, device ):
		self._id = id
		self._device = device
		self.action_size = action_size
		# input size
		self.agent_count = agents_count
		self._agent_state_shape = state_shape
		self._agent_list = []
		""":type : list[A3CModel]"""
		# create networks
		for i in range(self.agent_count):
			self._agent_list.append ( A3CModel( str(id)+"_"+str(i), self._agent_state_shape, action_size, local_t_max, device ) )

	def iterate_agents(self):
		return iter(self._agent_list)

	def get_agent( self, id ):
		return self._agent_list[id]
		
	def reset(self):
		for agent in self._agent_list:
			agent.reset_state()
		
	def concat_action_and_reward(self, action, reward):
		"""
		Return one hot vectored action and reward.
		"""
		action_reward = np.zeros([self.action_size+1])
		action_reward[action] = 1.0
		action_reward[-1] = float(reward)
		return action_reward

	def compile(self, optimizer=None):
		for agent in self._agent_list:
			agent.compile(optimizer=optimizer)

	def get_weights(self):
		return [agent.get_weights() for agent in self._agent_list]

	def set_weights(self, weights):
		for w, agent in zip(weights, self._agent_list):
			agent.set_weights(w)

	def save_optimizers(self, hdf5_file):
		"""
		:type hdf5_file: h5py.File
		"""
		for agent in self._agent_list:
			agent.save_optimizer(hdf5_file)

	def load_optimizers(self, hdf5_file):
		"""
		:type hdf5_file: h5py.File
		"""
		for agent in self._agent_list:
			agent.load_optimizer(hdf5_file)


class A3CModel(object):

	def __init__( self, id, state_shape, action_size, local_t_max, device ):
		self._id = id
		self._device = device
		self._action_size = action_size
		self._state_shape = state_shape

		batch_size = 1
		batch_state_shape = (batch_size, None) + state_shape
		lstm_neurons = 256

		input_state = Input(batch_shape=batch_state_shape, name='input_state')
		input_action_reward = Input(batch_shape=(batch_size, None, action_size+1), name='input_act_reward')
		input_taken_act = Input(batch_shape=(batch_size, None, action_size), name='input_taken_act')
		input_advantage = Input(batch_shape=(batch_size, None), name='input_advantage')
		input_R = Input(batch_shape=(batch_size, None), name='input_R')

		init_conv1 = Initializer_muupan(1)
		init_conv2 = Initializer_muupan(16)
		init_tower_dense = Initializer_muupan(32*state_shape[0]*state_shape[1])
		init_out_dense = Initializer_muupan(lstm_neurons)

		tower = TimeDistributed(Conv2D(16, kernel_size=(3,3), padding='same',
									   kernel_initializer=init_conv1, bias_initializer=init_conv1)
								, name='tower_conv1')(input_state)
		tower = TimeDistributed(ReLu())(tower)
		tower = TimeDistributed(Conv2D(32, kernel_size=(3,3), padding='same',
									   kernel_initializer=init_conv2, bias_initializer=init_conv2)
								, name='tower_conv2')(tower)
		tower = TimeDistributed(ReLu())(tower)
		tower = TimeDistributed(Flatten())(tower)
		tower = TimeDistributed(Dense(lstm_neurons, kernel_initializer=init_tower_dense,
									  bias_initializer=init_tower_dense))(tower)
		tower = TimeDistributed(ReLu(), name='tower_out')(tower)

		lstm_input = Concatenate()([tower, input_action_reward])
		self.lstm = LSTM(lstm_neurons, stateful=True, return_sequences=True, recurrent_activation='sigmoid')
		lstm_out = self.lstm(lstm_input)

		value_out = TimeDistributed(Dense(1, kernel_initializer=init_out_dense, bias_initializer=init_out_dense),
									name='value_out')(lstm_out)

		policy_out = TimeDistributed(Dense(action_size, kernel_initializer=init_out_dense, bias_initializer=init_out_dense),
									 name='policy_dense')(lstm_out)
		policy_out = TimeDistributed(Activation('softmax', name='softmax'), name='policy_out')(policy_out)

		# the a3c loss needs to be computed on both policy and value outputs, however keras .compile() method
		# expects a loss for each output independently, so this loss must be computed as part of the model
		# keras .compile() method will receive the identity_loss declared above
		loss_out = Lambda(lambda x: a3c.a3c_loss(input_taken_act=x[0], input_advantage=x[1], input_R=x[2],
						                         policy_out=x[3], value_out=x[4], entropy_beta=flags.entropy_beta),
						  name='loss_out')([input_taken_act, input_advantage, input_R, policy_out, value_out])

		# as noted above we will use an identiy_loss, which simply outputs the loss computed by the model, however
		# keras .fit() method still require an y_true, so this dummy y_true will be provided
		self.dummy_loss_out = np.zeros((1, local_t_max, 1))

		# the playing and training net share all the weights, they need to be separate however because keras
		# requires a fixed batch size for stateful LSTMs and for the loss calculation in training
		self.playing_net = Model(inputs=[input_state, input_action_reward],
								 outputs=[policy_out, value_out])

		self.traning_net = Model(inputs=[input_state, input_action_reward, input_taken_act, input_advantage, input_R],
							     outputs=[loss_out])

	def compile(self, optimizer):
		# as described in __init__(), the loss is computed by the model and given to keras with the identity_loss
		self.traning_net.compile(loss=identity_loss, optimizer=optimizer)

	def get_weights(self):
		return self.traning_net.get_weights()

	def set_weights(self, weights):
		return self.traning_net.set_weights(weights)

	def get_state(self):
		return [K.get_value(s) for s in self.lstm.states]

	def set_state(self, state):
		for s_val, s in zip(state, self.lstm.states):
			K.set_value(s, s_val)

	def reset_state(self):
		self.playing_net.reset_states()

	def _run_inputs(self, s_t, last_action_reward):
		return [np.array([[x]]) for x in [s_t, last_action_reward]]

	def run_policy_and_value(self, s_t, last_action_reward):
		"""
		This is used when forward propagating, so the step size is 1.
		"""
		pi_out, v_out = self.playing_net.predict(self._run_inputs(s_t, last_action_reward))
		return (pi_out[0,0], v_out[0,0])
		
	def run_value(self, s_t, last_action_reward):
		"""
		This  is used for calculating V for bootstrapping at the end of LOCAL_T_MAX time step sequence
		N.B. the LSTM state will be updated
		"""
		_, v_out = self.playing_net.predict(self._run_inputs(s_t, last_action_reward))
		return v_out[0,0]

	def fit(self, base_input, base_last_action_reward_input, base_a, base_adv, base_r):
		dummy_loss_out = self.dummy_loss_out[:, :len(base_input)]
		x = [np.expand_dims(x, axis=0) for x in [base_input, base_last_action_reward_input, base_a, base_adv, base_r]]
		self.traning_net.fit(x, dummy_loss_out, shuffle=False, batch_size=1, verbose=0, epochs=1)

	def save_optimizer(self, hdf5_file):
		save_optmizer_to_hdf5_file(self.traning_net, self._id, hdf5_file)

	def load_optimizer(self, hdf5_file):
		load_optmizer_from_hdf5_file(self.traning_net, self._id, hdf5_file,
									 custom_objects={'identity_loss': identity_loss})
