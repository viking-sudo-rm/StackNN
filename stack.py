import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
		("Give it to me".split(), "ENGLISH"),
		("No creo que sea una buena idea".split(), "SPANISH"),
		("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

test_data = [("Yo creo que si".split(), "SPANISH"),
	("it is lost on me".split(), "ENGLISH")]

# word_to_ix maps each word in the vocab to a unique integer, which will be its
# index into the Bag of words vector
word_to_ix = {}
for sent, _ in data + test_data:
	for word in sent:
		if word not in word_to_ix:
			word_to_ix[word] = len(word_to_ix)
print(word_to_ix)

EMBEDDING_SIZE = 50

class Stack(nn.Module):  # inheriting from nn.Module!

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


# def make_bow_vector(sentence, word_to_ix):
# 	vec = torch.zeros(len(word_to_ix))
# 	for word in sentence:
# 		vec[word_to_ix[word]] += 1
# 	return vec.view(1, -1)


# def make_target(label, label_to_ix):
# 	return torch.LongTensor([label_to_ix[label]])


stack = Stack(3)
stack.log()
print "pushed", stack.forward(torch.FloatTensor([[1, 2, 3]]), 0, 1)
stack.log()
print "popped", stack.forward(torch.FloatTensor([[4, 5, 6]]), 1, 1)
stack.log()
# print "pushed", stack.forward(torch.FloatTensor([[7, 8, 9]]), 1, 1)

# # the model knows its parameters.  The first output below is A, the second is b.
# # Whenever you assign a component to a class variable in the __init__ function
# # of a module, which was done with the line
# # self.linear = nn.Linear(...)
# # Then through some Python magic from the Pytorch devs, your module
# # (in this case, BoWClassifier) will store knowledge of the nn.Linear's parameters
# for param in model.parameters():
# 	print(param)

# # To run the model, pass in a BoW vector, but wrapped in an autograd.Variable
# sample = data[0]
# bow_vector = make_bow_vector(sample[0], word_to_ix)
# log_probs = model(autograd.Variable(bow_vector))
# print(log_probs)
