import torch
import torch.autograd as autograd
from torch.autograd import Variable
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from stack import Stack

torch.manual_seed(1)

class FFController(nn.Module):

	def __init__(self, batch_size, num_embeddings, embedding_size, read_size, output_size):
		super(FFController, self).__init__()

		self.read_size = read_size
		self.batch_size = batch_size

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
		# u, d, v = read_params[:,0], read_params[:,1], read_params[:,2:]
		u, d, v = read_params[:,0].contiguous(), read_params[:,1].contiguous(), read_params[:,2:].contiguous()
		self.read = self.stack.forward(v.data, u.data, d.data)
		# return F.softmax(output[:,2 + self.read_size:]) # log softmax?
		return output[:,2 + self.read_size:] #should not apply softmax

	def init_stack(self):
		self.read = torch.zeros([self.batch_size, self.read_size])
		self.stack = Stack(self.batch_size, self.read_size)