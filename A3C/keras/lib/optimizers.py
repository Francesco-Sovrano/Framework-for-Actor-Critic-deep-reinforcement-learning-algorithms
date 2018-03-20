# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.training import training_ops
from tensorflow.python.training import slot_creator
from keras.optimizers import RMSprop, Optimizer
import keras.backend as K
import tensorflow as tf


class CustomRMSprop(Optimizer):

    def __init__(self,
                 learning_rate=0.0007,
                 decay=0.9,
                 momentum=0.0,
                 epsilon=1e-10,
                 clipnorm=40.0,
                 max_iterations=10**8,
                 **kwargs):
        super().__init__(**kwargs)

        self._name = self.__class__.__name__
        with K.name_scope(self._name):
            self.learning_rate = K.variable(learning_rate, name="learning_rate")
            self.decay = K.variable(decay, name="decay")
            self.momentum = K.variable(momentum, name="momentum")
            self.epsilon = K.variable(epsilon, name="epsilon")
            self.iterations = K.variable(0, dtype='int64', name="iterations")
        self.inital_lr = learning_rate
        self.initial_decay = decay
        self.clipnorm = clipnorm
        self.max_iterations = max_iterations

        self._slots = {}

    def _create_slots(self, var_list):
        for v in var_list:
            # 'val' is Variable's initial value tensor.
            val = tf.constant(1.0, dtype=v.dtype, shape=v.get_shape())
            self._get_or_make_slot(v, val, "rms", self._name)
            self._zeros_slot(v, "momentum", self._name)

    def _slot_dict(self, slot_name):
        named_slots = self._slots.get(slot_name, None)
        if named_slots is None:
            named_slots = {}
            self._slots[slot_name] = named_slots
        return named_slots

    def _get_or_make_slot(self, var, val, slot_name, op_name):
        named_slots = self._slot_dict(slot_name)
        if var not in named_slots:
            named_slots[var] = slot_creator.create_slot(var, val, op_name)
        return named_slots[var]

    def get_slot(self, var, name):
        named_slots = self._slots.get(name, None)
        if not named_slots:
            return None
        return named_slots.get(var, None)

    def _zeros_slot(self, var, slot_name, op_name):
        named_slots = self._slot_dict(slot_name)
        if var not in named_slots:
            named_slots[var] = slot_creator.create_zeros_slot(var, op_name)
        return named_slots[var]

    # TODO: in RMSProp native code, memcpy() (for CPU) and
    # cudaMemcpyAsync() (for GPU) are used when updating values,
    # and values might tend to be overwritten with results from other threads.
    # (Need to check the learning performance with replacing it)
    def _apply_dense(self, grad, var):
        rms = self.get_slot(var, "rms")
        mom = self.get_slot(var, "momentum")
        return training_ops.apply_rms_prop(
            var, rms, mom,
            self.learning_rate,
            self.decay,
            self.momentum,
            self.epsilon,
            grad,
            use_locking=False).op

    def get_updates(self, loss, params):
        with tf.control_dependencies(None):
            self._create_slots(params)

        grads = self.get_gradients(loss, params)

        self.updates = [K.update_add(self.iterations, 1)]

        if self.initial_decay > 0:
            new_lr = self.inital_lr
            new_lr *= (self.max_iterations - K.cast(self.iterations, K.dtype(self.decay))) / self.max_iterations
            self.updates.append(K.update(self.learning_rate, new_lr))

        for par, grad in zip(params, grads):
            self.updates.append(self._apply_dense(grad, par))

        return self.updates

    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'decay': float(K.get_value(self.decay)),
                  'momentum': float(K.get_value(self.momentum)),
                  'epsilon': float(K.get_value(self.epsilon)),
                  'clipnorm': self.clipnorm,
                  'max_iterations': self.max_iterations}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
