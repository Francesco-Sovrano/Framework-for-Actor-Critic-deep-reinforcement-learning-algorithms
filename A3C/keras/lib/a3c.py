
from keras import backend as K


def a3c_loss(input_taken_act=None, input_advantage=None, input_R=None, policy_out=None, value_out=None,
			 entropy_beta=None):
	# [base A3C]

	input_taken_act = input_taken_act[0]
	input_advantage = input_advantage[0]
	input_R = input_R[0]
	policy_out = policy_out[0]
	value_out = value_out[0, :, 0]

	# Avoid NaN with clipping when value in pi becomes zero
	log_pi = K.log(K.clip(policy_out, 1e-20, 1.0))

	# Policy entropy
	entropy = -K.sum(policy_out * log_pi, axis=-1)

	# Policy loss (output)
	policy_loss = -K.sum(K.sum(log_pi * input_taken_act, axis=-1) * input_advantage + entropy * entropy_beta)

	# Value loss (output)
	l2_loss_input = input_R - value_out
	l2_loss = K.sum(l2_loss_input ** 2) / 2
	# (Learning rate for Critic is half of Actor's, so multiply by 0.5)
	value_loss = 0.5 * l2_loss

	total_loss = policy_loss + value_loss
	return total_loss
