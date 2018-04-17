import torch
import torch.autograd as autograd
from torch.autograd import Variable
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from model import Controller as AbstractController

torch.manual_seed(1)

class Controller(AbstractController):

	def __init__(self, num_embeddings, embedding_size, read_size, output_size, **args):
		
		super(Controller, self).__init__(read_size, **args)

		# initialize the controller parameters
		self.embed = nn.Embedding(num_embeddings, embedding_size)
		self.embed.weight.data.uniform_(-.1, .1)
		self.linear = nn.Linear(embedding_size + read_size, 2 + read_size + output_size)
		self.linear.weight.data.uniform_(-.1, .1)
		self.linear.bias.data.fill_(0)
		
	def forward(self, x):
		# print x.shape, self.read.shape
		hidden = self.embed(x)
		output = self.linear(torch.cat([hidden, self.read], 1))
		read_params = F.sigmoid(output[:,:2 + self.read_size])
		self.u, self.d, self.v = read_params[:,0].contiguous(), read_params[:,1].contiguous(), read_params[:,2:].contiguous()
		self.read_stack(v.data, u.data, d.data)

		return output[:,2 + self.read_size:] #should not apply softmax