import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class Stack(nn.Module):

	def __init__(self, embedding_size):
		super(Stack, self).__init__()

		# initialize the datastructure
		self.V = torch.FloatTensor(0, embedding_size)
		self.s = torch.FloatTensor(0)

		# an instance of the ReLU function
		# self.relu = nn.ReLU()
		self.relu = lambda x: max(0, x)

	def forward(self, v, u, d):
		"""
		@param v vector of dimension [1, embedding_size] to push
		@param u pop signal in (0, 1)
		@param d push signal in (0, 1)
		@return read vector of dimension [1, embedding_size]
		"""

		# update V
		self.V = torch.cat([self.V, v], 0)

		# update s
		s_len = self.s.shape[0] if self.s.shape else 0
		s = torch.FloatTensor(s_len + 1)
		for i in xrange(s_len):
			old = sum(self.s[j] for j in xrange(i + 1, s_len)) # TODO indexing right?
			s[i] = self.relu(self.s[i] - self.relu(u - old))
		s[s_len] = d
		self.s = s

		# calculate r
		r = torch.zeros([1, v.shape[1]])
		for i in xrange(s_len + 1):
			old = sum(self.s[j] for j in xrange(i, s_len + 1)) # TODO indexing right?
			# print "weight: {}, relu: {}".format(self.s[i], self.relu(old))
			r += min(self.s[i], self.relu(old)) * self.V[i,:]

		return r

	def log(self):
		"""
		Prints a representation of the stack to stdout.
		"""
		if not self.V.shape:
			print "[Empty stack]"
			return
		for i in xrange(len(self.V)):
			print "\t".join(str(x) for x in self.V[i,:]), "\t|\t", self.s[i]
