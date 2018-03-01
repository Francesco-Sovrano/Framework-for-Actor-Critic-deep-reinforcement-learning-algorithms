
# this will check that


import numpy as np
from keras.models import Model
from keras.layers import Input
import keras.backend as K

from environment.environment import Environment
from conversion import convert
import train
Application = train.Application


env = Environment.create_environment('rogue', 1)
state_shape = env.get_state_shape()
act_size = env.get_action_size()

global_weigths, tf_app, d = convert(conv=False)
sess = tf_app.sess
tf_net = tf_app.global_network.get_agent(0)

keras_app = Application()
keras_app.load_checkpoint()
keras_model = keras_app.build_network(0)
keras_model.set_weights(keras_app.global_weigths)
keras_model = keras_model.get_agent(0)

# ker = d['net_-1_0/base_lstm-1_0/basic_lstm_cell/kernel:0']
# print("ker:")
# print(ker)

tf_tower_dense = sess.graph.get_tensor_by_name('net_-1_0/base_lstm-1_0/tower_out:0')
k_tower_layer = keras_model.playing_net.get_layer('tower_out')
k_tower_dense = Model(inputs=keras_model.playing_net.inputs, outputs=[k_tower_layer(k_tower_layer.input)])

tf_policy_before_softmax = sess.graph.get_tensor_by_name('net_-1_0/base_policy-1_0/policy_before_softmax:0')
k_before_softmax_layer = keras_model.playing_net.get_layer('policy_dense')
k_before_softmax = Model(inputs=keras_model.playing_net.inputs, outputs=[k_before_softmax_layer(k_before_softmax_layer.input)])

tf_lstm = sess.graph.get_tensor_by_name('net_-1_0/base_lstm-1_0/lstm_output:0')
k_policy_input = Input(batch_shape=(1, None, 256))
k_policy_layer = keras_model.playing_net.get_layer('policy_out')
k_policy = Model(inputs=k_policy_input, outputs=k_policy_layer(k_before_softmax_layer(k_policy_input)))

tf_lstm_input = sess.graph.get_tensor_by_name('net_-1_0/base_lstm-1_0/lstm_input:0')
k_lstm_layer = keras_model.playing_net.get_layer('lstm_1')
k_lstm_input = k_lstm_layer.input
k_lstm = Model(inputs=keras_model.playing_net.inputs, outputs=k_lstm_layer(k_lstm_input))



np.random.seed(42)

st = np.ones(shape=state_shape)
ar = np.zeros(shape=act_size+1)
ar[0] = 1

policy_input = np.ones(shape=(1,1,256))
lstm_input = np.ones(shape=(1,1,262))

tf_net.reset_state()
keras_model.reset_state()

for _ in range(4):
    st = np.random.random(state_shape)
    ar = np.random.random(act_size+1)
    tf_p, tf_v = tf_net.run_policy_and_value(sess, st, ar)
    k_p, k_v = keras_model.run_policy_and_value(st, ar)
    eq1 = np.array_equal(tf_p, k_p)
    eq2 = np.array_equal(tf_v, k_v[0])
    print("tf_p:", tf_p, 'tf_v:', tf_v)
    print(" k_p:", k_p, ' k_v:', k_v[0])
    print("eq:", eq1 and eq2)
    print("eq1:", eq1, "eq2:", eq2)

    tf_tower = sess.run(tf_tower_dense,
                        feed_dict={tf_net.base_input: [st],
                                   tf_net.base_last_action_reward_input: [ar]})
    k_tower = k_tower_dense.predict([np.array([[x]]) for x in [st, ar]])
    # print("tow shapes:", tf_tower.shape, k_tower.shape)
    eq = np.array_equal(tf_tower, k_tower[0])
    print("eq tower out:", eq)
    print()

for _ in range(4):
    st = np.random.random(state_shape)
    ar = np.random.random(act_size + 1)

    tf_net.reset_state()
    keras_model.reset_state()

    tf_bef_softmax_out = sess.run(tf_policy_before_softmax,
                        feed_dict={tf_net.base_input: [st],
                                   tf_net.base_last_action_reward_input: [ar],
                                   tf_net.base_initial_lstm_state0:
                                       tf_net.base_lstm_state_out[0],
                                   tf_net.base_initial_lstm_state1:
                                       tf_net.base_lstm_state_out[1]})
    k_bef_softmax_out = k_before_softmax.predict([np.array([[x]]) for x in [st, ar]])
    # print("bef softmax shapes:", tf_bef_softmax_out.shape, k_bef_softmax_out.shape)
    # print('tf_p:', tf_bef_softmax_out[0], '\n k_p:', k_bef_softmax_out[0,0])
    eq = np.array_equal(tf_bef_softmax_out, k_bef_softmax_out[0])
    print("eq bef softmax:", eq)

    tf_net.reset_state()
    keras_model.reset_state()

    tf_policy_output = sess.run(tf_net.base_pi,
                                feed_dict={tf_lstm: policy_input[0]})
    k_policy_output = k_policy.predict(policy_input)
    # print('tf_p:', tf_policy_output, '\n k_p:', k_policy_output[0])
    eq = np.array_equal(tf_policy_output, k_policy_output[0])
    print("eq policy out:", eq)

    tf_lstm_out = sess.run(tf_lstm,
                           feed_dict={tf_net.base_input: [st],
                                      tf_net.base_last_action_reward_input: [ar],
                                      tf_net.base_initial_lstm_state0:
                                          tf_net.base_lstm_state_out[0],
                                      tf_net.base_initial_lstm_state1:
                                          tf_net.base_lstm_state_out[1]
                                      })
    k_lstm_out = k_lstm.predict([np.array([[x]]) for x in [st, ar]])[0]
    # print("lstm shapes:", tf_lstm_out.shape, k_lstm_out.shape)
    eq = np.array_equal(tf_lstm_out, k_lstm_out)
    diff = tf_lstm_out - k_lstm_out
    print("eq lstm out:", eq)
    print("diff sum", diff.sum(), "avg", diff.mean())

    print()
