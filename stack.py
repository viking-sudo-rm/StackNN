import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

def relu(t):
	z = torch.zeros(t.shape)
	return torch.max(z, t)

class Stack(nn.Module):

	def __init__(self, batch_size, embedding_size):
		super(Stack, self).__init__()

		# initialize tensors
		self.V = torch.FloatTensor(0)
		self.s = torch.FloatTensor(0)

		self.batch_size = batch_size
		self.embedding_size = embedding_size

	def forward(self, v, u, d):
		"""
		@param v [batch_size, embedding_size] matrix to push
		@param u [batch_size,] vector of pop signals in (0, 1)
		@param d [batch_size,] vector of push signals in (0, 1)
		@return [batch_size, embedding_size] read matrix
		"""

		# update V, which is of size [t, bach_size, embedding_size]
		v = v.view(1, self.batch_size, self.embedding_size)
		self.V = torch.cat([self.V, v], 0)

		# update s, which is of size [t, batch_size]
		old_t = self.s.shape[0] if self.s.shape else 0
		s = torch.FloatTensor(old_t + 1, self.batch_size)
		for i in xrange(old_t):
			old = torch.sum(self.s[i + 1:old_t,:], 0) if i + 1 < old_t else torch.zeros(self.batch_size)
			s[i,:] = relu(self.s[i,:] - relu(u - old))
		s[old_t,:] = d
		self.s = s

		# calculate r
		r = torch.zeros([self.batch_size, self.embedding_size])
		for i in xrange(old_t + 1):
			if i + 1 == old_t + 1:
				continue
			old = torch.sum(self.s[i + 1:old_t + 1,:], 0)
			r += torch.min(self.s[i,:], relu(old)) * self.V[i,:,:]

		return r

	def log(self):
		"""
		Prints a representation of the stack to stdout.
		"""
		if not self.V.shape:
			print "[Empty stack]"
			return
		for b in xrange(self.batch_size):
			if b > 0:
				print "----------------------------"
			for i in xrange(self.V.shape[0]):
				print "\t".join(str(x) for x in self.V[i, b,:]), "\t|\t", self.s[i, b]

if __name__ == "__main__":
	stack = Stack(2, 2)
	out = stack.forward(torch.FloatTensor([[1, 2], [3, 4]]), torch.FloatTensor([1, 0]), torch.FloatTensor([0, 1]))
	stack.log()
	print out
	print
	out = stack.forward(torch.FloatTensor([[11, 22], [33, 44]]), torch.FloatTensor([1, 0]), torch.FloatTensor([1, 0]))
	stack.log()
	print out
