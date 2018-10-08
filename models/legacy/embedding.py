import torch
import torch.autograd as autograd
from torch.autograd import Variable
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import random

from legacy.model import Model as Model

torch.manual_seed(1)

class EmbeddingModel(Model):

	def __init__(self, num_embeddings, embedding_size, read_size, output_size, **args):
		
		super(EmbeddingModel, self).__init__(read_size, **args)

		# Initialize the embedding parameters
		self.embed = nn.Embedding(num_embeddings, embedding_size)
		self.init_normal(self.embed.weight)

		# Initialize the linear parameters
		self.linear = nn.Linear(embedding_size + self.get_read_size(), 2 + self.get_read_size() + output_size)
		Model.init_normal(self.linear.weight)
		self.linear.bias.data.fill_(0)
		
	def forward(self, x):
		# print x.shape, self.read.shape
		hidden = self.embed(x)
		output = self.linear(torch.cat([hidden, self.read], 1))
		read_params = torch.sigmoid(output[:,:2 + self.get_read_size()])
		self.u, self.d, self.v = read_params[:,0].contiguous(), read_params[:,1].contiguous(), read_params[:,2:].contiguous()
		self.read_stack(v.data, u.data, d.data)

		return output[:,2 + self.get_read_size():] #should not apply softmax