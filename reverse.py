from __future__ import division

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.utils import shuffle

#m = __import__("model-bare")
m = __import__("model-lstm")

# Language parameters
MIN_LENGTH = 1
MEAN_LENGTH = 10
STD_LENGTH = 2
MAX_LENGTH = 12

# Hyperparameters
LEARNING_RATE = .1 # .01 and .1 seem to work well?
BATCH_SIZE = 10 # 10 is the best I've found
READ_SIZE = 1 # was using 4 before

EPOCHS = 100

#model = m.FFController(3, READ_SIZE, 3)
model = m.LSTM_Controller(3, READ_SIZE, 3)
try: model.cuda()
except AssertionError: pass

criterion = nn.CrossEntropyLoss()

def randstr():
	length = min(max(MIN_LENGTH, int(random.gauss(MEAN_LENGTH, STD_LENGTH))), MAX_LENGTH)
	return [random.randint(0, 1) for _ in xrange(length)]

reverse = lambda s: s[::-1]
onehot = lambda b: torch.FloatTensor([1. if i == b else 0. for i in xrange(3)])

def get_tensors(B):
	X_raw = [randstr() for _ in xrange(B)]

	# initialize X to one-hot encodings of NULL
	X = torch.FloatTensor(B, 2 * MAX_LENGTH, 3)
	X[:,:,:2].fill_(0)
	X[:,:,2].fill_(1)

	# initialize Y to NULL
	Y = torch.LongTensor(B, 2 * MAX_LENGTH)
	Y.fill_(2)

	for i, x in enumerate(X_raw):
		y = reverse(x)
		for j, char in enumerate(x):
			X[i,j,:] = onehot(char)
			Y[i,j + len(x)] = y[j]
	return Variable(X), Variable(Y)

train_X, train_Y = get_tensors(800)
dev_X, dev_Y = get_tensors(100)
test_X, test_Y = get_tensors(100)

def train(train_X, train_Y):
	model.train()
	total_loss = 0.

	for batch, i in enumerate(xrange(0, len(train_X.data) - BATCH_SIZE, BATCH_SIZE)):

		digits_correct = 0
		digits_total = 0
		batch_loss = 0.

		X, Y = train_X[i:i+BATCH_SIZE,:,:], train_Y[i:i+BATCH_SIZE,:]
		model.init_stack(BATCH_SIZE)
		for j in xrange(2 * MAX_LENGTH):

			a = model.forward(X[:,j,:])

			indices = Y[:,j] != 2
			valid_a = a[indices.view(-1, 1)].view(-1, 3)
			valid_Y = Y[:,j][indices]

			if len(valid_a) == 0: continue

			_, valid_y_ = torch.max(valid_a, 1)
			digits_total += len(valid_a)
			digits_correct += len(torch.nonzero((valid_y_ == valid_Y).data))
			batch_loss += criterion(valid_a, valid_Y)

		# update the weights
		optimizer.zero_grad()
		batch_loss.backward()
		optimizer.step()

		total_loss += batch_loss.data
		if batch % 10 == 0:
			mean_loss = sum(batch_loss.data)
			print "batch {}: loss={:.4f}, acc={:.2f}".format(batch, mean_loss, digits_correct / digits_total)

def evaluate(test_X, test_Y):
	model.eval()
	total_loss = 0.
	digits_correct = 0
	digits_total = 0
	model.init_stack(len(test_X.data))
	for j in xrange(2 * MAX_LENGTH):

		a = model.forward(test_X[:,j,:])

		indices = test_Y[:,j] != 2
		valid_a = a[indices.view(-1, 1)].view(-1, 3)
		valid_Y = test_Y[:,j][indices]

		if len(valid_a) == 0: continue

		_, valid_y_ = torch.max(valid_a, 1)
		digits_total += len(valid_a)
		digits_correct += len(torch.nonzero((valid_y_ == valid_Y).data))
		total_loss += criterion(valid_a, valid_Y)

	mean_loss = sum(total_loss.data)
	print "epoch {}: loss={:.4f}, acc={:.2f}".format(epoch, mean_loss, digits_correct / digits_total)

# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print "hyperparameters: lr={}, batch_size={}, read_dim={}".format(LEARNING_RATE, BATCH_SIZE, READ_SIZE)
for epoch in xrange(EPOCHS):
	print "-- starting epoch {} --".format(epoch)
	perm = torch.randperm(800)
	train_X, train_Y = train_X[perm], train_Y[perm]
	train(train_X, train_Y)
	evaluate(dev_X, dev_Y)
