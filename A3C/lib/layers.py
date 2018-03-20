
from keras.layers import Activation


relu_index = 1

def ReLu(name=None):
    """
    :param name:
    :rtype: Activation
    """
    global relu_index
    if name is None:
        name = 'relu_%s' % relu_index
        relu_index += 1
    return Activation('relu', name=name)
