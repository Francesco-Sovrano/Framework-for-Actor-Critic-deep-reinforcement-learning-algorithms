# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import misc
from skimage import exposure
from math import ceil

from .conv2d import Conv2D

class MaxPool(Conv2D):

	def __init__( self, name, input, kernel, stride, padding="SAME" ):
		if type(input) is Conv2D:
			i = input.output
		else:
			i = input
		super().__init__(name, input, kernel, stride, i[2], padding)
		
	def build_variable(self, input_placeholder):
		if self.net != None:
			input_placeholder = self.net.build_variable( input_placeholder )
		ksize = [1, self.kernel[0], self.kernel[1], 1]
		strides = [1, self.stride[0], self.stride[1], 1]
		return tf.nn.max_pool ( input_placeholder, ksize, strides, padding = self.padding )