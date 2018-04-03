from __future__ import division

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from abc import ABCMeta, abstractmethod

from stack import Stack

class Controller(nn.Module):

	__metaclass__ = ABCMeta

	def __init__(self, read_size):
		super(Controller, self).__init__()
		self.read_size = read_size
	
	@abstractmethod
	def forward(self, x):
		return NotImplementedError("Must define forward pass.")

	def init_stack(self, batch_size):
		self.read = Variable(torch.zeros([batch_size, self.read_size]))
		self.stack = Stack(batch_size, self.read_size)

	def trace(self, trace_X):
		"""
		Visualize stack activations for a single training sample.
		Draws a graphic representation of these stack activations.
		@param trace_X [1, max_length, input_size] tensor
		"""
		self.eval()
		self.init_stack(1)
		max_length = trace_X.shape[1]
		data = np.zeros([2 + self.read_size, max_length]) # 2 + len(v)
		for j in xrange(1, max_length):
			self.forward(trace_X[:, j - 1, :])
			data[0,j] = self.u.data.numpy()
			data[1,j] = self.d.data.numpy()
			data[2:,j] = self.v.data.numpy()
		plt.imshow(data, cmap="hot", interpolation="nearest")
		plt.show()