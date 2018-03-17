
import argparse
import numpy as np
import os
import tensorflow as tf

import train
Keras_Application = train.Application


def convert(conv=False):
    if conv:
        print("converting model with convolutions after lstm")
        from legacy.train import Application as TF_Application
    else:
        print("converting model WITHOUT convolutions after lstm")
        from legacy.train_noconv import Application as TF_Application

    os.makedirs('log', exist_ok=True)
    os.makedirs('log/performance', exist_ok=True)

    if not os.path.isfile(train.flags.checkpoint_dir + '/checkpoint'):
        print("Please place a tensorflow checkpoint in", os.path.realpath(train.flags.checkpoint_dir))
        exit(1)

    tf_app = TF_Application()
    sess = tf_app.sess = tf.Session()
    tf_app.device = '/cpu:0'
    tf_app.build_global_network(tf.placeholder(tf.float32))
    tf_app.load_checkpoint()

    d = create_tensor_vals_dict(sess)

    keras_app = Keras_Application()
    keras_app.global_t = tf_app.global_t
    keras_app.wall_t = tf_app.wall_t

    weights = []

    for a in range(tf_app.global_network.agent_count):
        w = convert_agent_weights(a, d, conv=conv)
        weights.append(w)

    keras_app.global_weigths = weights
    keras_app.save_checkpoint()

    return weights, tf_app, d


def create_tensor_vals_dict(sess):
    vars = [v for v in tf.trainable_variables() if '-1_' in v.name]
    d = {v.name: sess.run(v) for v in vars}
    return d


def convert_agent_weights(agent_index, tensor_vals_dict, conv=False):
    a = agent_index
    d = tensor_vals_dict

    w = []

    # tower
    name = 'net_-1_%s/base_conv-1_%s/' % (a, a)
    w.append(d[name + 'weight_tower1_conv1:0'])
    w.append(d[name + 'bias_tower1_conv1:0'])
    w.append(d[name + 'weight_tower1_conv2:0'])
    w.append(d[name + 'bias_tower1_conv2:0'])
    name = 'net_-1_%s/base_lstm-1_%s/' % (a, a)
    w.append(d[name + 'W_base_fc1-1_%s:0' % a])
    w.append(d[name + 'b_base_fc1-1_%s:0' % a])

    # lstm
    name = 'net_-1_%s/base_lstm-1_%s/basic_lstm_cell/' % (a, a)
    tf_kernel = d[name + 'kernel:0']
    tf_bias = d[name + 'bias:0']
    k_kernel = np.empty_like(tf_kernel)
    k_bias = np.empty_like(tf_bias)

    lstm_neurons = int(tf_kernel.shape[-1]) // 4

    # keras stores lstm weights in a different way than tensorflow
    # in particular the weights are divided in 4 blocks
    # tensorflow's 2nd and 3rd blocks are swapped in keras
    k_kernel[:, : lstm_neurons] = tf_kernel[:, : lstm_neurons]
    k_kernel[:, lstm_neurons: 2 * lstm_neurons] = tf_kernel[:, 2 * lstm_neurons: 3 * lstm_neurons]
    k_kernel[:, 2 * lstm_neurons: 3 * lstm_neurons] = tf_kernel[:, lstm_neurons: 2 * lstm_neurons]
    k_kernel[:, 3 * lstm_neurons:] = tf_kernel[:, 3 * lstm_neurons:]

    k_bias[: lstm_neurons] = tf_bias[: lstm_neurons]
    # tensorflow adds a constant to the forget bias
    k_bias[lstm_neurons: 2 * lstm_neurons] = tf_bias[2 * lstm_neurons: 3 * lstm_neurons] + 1
    k_bias[2 * lstm_neurons: 3 * lstm_neurons] = tf_bias[lstm_neurons: 2 * lstm_neurons]
    k_bias[3 * lstm_neurons:] = tf_bias[3 * lstm_neurons:]

    w.append(k_kernel[:-lstm_neurons])  # lstm tower input weights
    w.append(k_kernel[-lstm_neurons:])  # lstm old state weights
    w.append(k_bias)

    # policy and value
    name_p = 'net_-1_%s/base_policy-1_%s/' % (a, a)
    name_v = 'net_-1_%s/base_value-1_%s/' % (a, a)
    if conv:
        for i in range(1, 3):
            w.append(d[name_p + 'weight_policy_conv%s:0' % i])
            w.append(d[name_p + 'bias_policy_conv%s:0' % i])
            w.append(d[name_v + 'weight_value_conv%s:0' % i])
            w.append(d[name_v + 'bias_value_conv%s:0' % i])
    w.append(d[name_p + 'W_base_fc_p-1_%s:0' % a])
    w.append(d[name_p + 'b_base_fc_p-1_%s:0' % a])
    w.append(d[name_v + 'W_base_fc_v-1_%s:0' % a])
    w.append(d[name_v + 'b_base_fc_v-1_%s:0' % a])

    return w


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a checkpoint from tensorflow to keras.')
    parser.add_argument('--conv', '-c', help='use this flag if there were convolutions after the lstm'
                        , action='store_true')
    ARGS = parser.parse_args()
    print("ARGS:", ARGS)
    convert(conv=ARGS.conv)
