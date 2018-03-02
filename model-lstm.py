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

	def __init__(self, batch_size, input_size, read_size, output_size):
		super(LSTM_Controller, self).__init__()

		self.input_size = input_size
		self.read_size = read_size
		self.batch_size = batch_size
		self.output_size = output_size

		# 								Input dim , output dim
		self.lstm = nn.LSTM(input_size + read_size, 2 + read_size + output_size)
		self.hidden = (autograd.Variable(torch.zeros(1, 1, 2 + read_size + output_size)),
						autograd.Variable(torch.zeros((1, 1, 2 + read_size + output_size))))
		self.lstm.weight_hh_l0.data.uniform_(-.1, .1)
		self.lstm.weight_ih_l0.data.uniform_(-.1, .1)
		self.lstm.bias_hh_l0.data.fill_(0)
		self.lstm.bias_ih_l0.data.fill_(0)

	def forward(self, x):
		self.hidden = (autograd.Variable(torch.zeros(1, 1, 2 + self.read_size + self.output_size)),
						autograd.Variable(torch.zeros((1, 1, 2 + self.read_size + self.output_size))))
		output, hidden = self.lstm(torch.cat([x, self.read], 1).view(1, 1, -1), self.hidden)
		output = output.view(-1 , 2 + self.read_size + self.output_size)
		self.hidden = hidden

		read_params = F.sigmoid(output[:,:2 + self.read_size])
		# u, d, v = read_params[:,0], read_params[:,1], read_params[:,2:]
		u, d, v = read_params[:,0].contiguous(), read_params[:,1].contiguous(), read_params[:,2:].contiguous()
		self.read = self.stack.forward(v, u, d)
		return output[:,2 + self.read_size:]
		# return F.softmax(..) # gets applied in loss function

	def init_stack(self):
		self.read = Variable(torch.zeros([self.batch_size, self.read_size]))
		self.stack = Stack(self.batch_size, self.read_size)
