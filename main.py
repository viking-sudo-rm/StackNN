import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from stack import Stack

torch.manual_seed(1)

class FFController(nn.Module):

	def __init__(self, input_size, read_size, output_size):
		super(FFController, self).__init__()

		# self.input_size = input_size
		self.read_size = read_size
		# self.output_size = output_size

		# initialize relevant fields for the model
		self.linear = nn.Linear(input_size + read_size, 2 + read_size + output_size)
		self.stack = Stack(read_size)
		self.read = torch.zeros([1, read_size])

	def forward(self, x):
		# print x.shape, self.read.shape
		output = self.linear(torch.cat([x, self.read], 1))
		read_params = F.sigmoid(output[:1,:2 + self.read_size])
		u, d, v = read_params[0,0], read_params[0,1], read_params[:1,2:]
		self.read = self.stack.forward(v.data, u.data[0], d.data[0])
		return F.log_softmax(output[0,2 + self.read_size:])

model = FFController(3, 10, 2)
pred = model(autograd.Variable(torch.FloatTensor([[1, 2, 3]]), requires_grad=True))
print pred

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