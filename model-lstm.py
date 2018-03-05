import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from stack import Stack

torch.manual_seed(1)

class LSTM_Controller(nn.Module):

	def __init__(self, input_size, read_size, output_size):
		super(LSTM_Controller, self).__init__()

		self.input_size = input_size
		self.read_size = read_size
		self.output_size = output_size

		# 								Input dim , output dim
		self.lstm = nn.LSTM(input_size + read_size, 2 + read_size + output_size)

		#initialize weights
		self.lstm.weight_hh_l0.data.uniform_(-.1, .1)
		self.lstm.weight_ih_l0.data.uniform_(-.1, .1)
		self.lstm.bias_hh_l0.data.fill_(0)
		self.lstm.bias_ih_l0.data.fill_(0)

	def forward(self, x):
		#expects 3D input tensors:
		# 1. indexes words in the sequence (always 1 for now)
		# 2. indexes over the minibatch
		# 3. indexes components of a word embedding
		lstm_input = torch.cat([x, self.read], 1)[None,:,:]
		output, hidden = self.lstm(lstm_input, self.hidden)
		self.hidden = hidden	#update hidden state for next time

		read_params = F.sigmoid(output[:,:,:2 + self.read_size].squeeze())
		u, d, v = read_params[:,0].contiguous(), read_params[:,1].contiguous(), read_params[:,2:].contiguous()
		self.read = self.stack.forward(v, u, d)

		return output[:,:,2 + self.read_size:].squeeze()

	def init_stack(self, batch_size):
		self.read = Variable(torch.zeros([batch_size, self.read_size]))
		self.stack = Stack(batch_size, self.read_size)

		# create an initial hidden state of 0s for the lstm
		# needed to "clean out" the model for re-use on new inputs
		lstm_hidden_shape = (1, batch_size, 2+self.output_size + self.read_size)
		self.hidden = (autograd.Variable(torch.zeros(lstm_hidden_shape)),
						autograd.Variable(torch.zeros(lstm_hidden_shape)))
