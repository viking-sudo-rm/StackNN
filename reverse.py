from __future__ import division

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.utils import shuffle

m = __import__("model-bare")

# Hyperparameters
LEARNING_RATE = .01 # .01 and .1 seem to work well?
BATCH_SIZE = 1 # 10 is the best I've found
READ_SIZE = 1 # was using 4 before

CUDA = False
EPOCHS = 10

model = m.FFController(1, 3, READ_SIZE, 2)
if CUDA:
    model.cuda()

criterion = nn.CrossEntropyLoss()

def randstr():
	length = max(1, int(random.gauss(10, 2)))
	return [random.randint(0, 1) for _ in xrange(length)]

reverse = lambda s: s[::-1]
floatT = lambda b: Variable(torch.FloatTensor([[1. if i == b else 0. for i in xrange(3)]]))
longT = lambda b: Variable(torch.LongTensor([b]))

zero = lambda: Variable(torch.FloatTensor([[1., 0., 0.]]))
end = lambda: Variable(torch.FloatTensor([[0., 0., 1.]]))

train_X = [randstr() for _ in xrange(800)]
dev_X = [randstr() for _ in xrange(100)]
test_X = [randstr() for _ in xrange(100)]

train_Y = [reverse(x) for x in train_X]
dev_Y = [reverse(x) for x in dev_X]
test_Y = [reverse(x) for x in test_X]

train_X = [map(floatT, X) for X in train_X]
dev_X = [map(floatT, X) for X in dev_X]
test_X = [map(floatT, X) for X in test_X]

train_Y = [map(longT, X) for X in train_Y]
dev_Y = [map(longT, X) for X in dev_Y]
test_Y = [map(longT, X) for X in test_Y]

def train(train_X, train_Y):
	model.train()
	total_loss = 0.

	for batch, i in enumerate(xrange(0, len(train_X) - BATCH_SIZE, BATCH_SIZE)):
		
		digits_correct = 0
		digits_total = 0
		batch_loss = 0.

		for j in xrange(i, i + BATCH_SIZE):
			X, Y = train_X[j], train_Y[j]
			model.init_stack()
			for x in X:
				model.forward(x)
			for y in Y:
				a = model.forward(end())
				_, y_ = torch.max(a, 1)

				digits_total += 1
				digits_correct += len(torch.nonzero((y_ == y).data))

				batch_loss += criterion(a, y)
		
		# update the weights
		optimizer.zero_grad()
		batch_loss.backward()
		optimizer.step()
		
		total_loss += batch_loss.data
		if batch % (len(test_X) // BATCH_SIZE // 10) == 0:
			print "batch {}: loss={:.2f}, acc={:.2f}".format(batch, sum(batch_loss.data) / BATCH_SIZE, digits_correct / digits_total)

def evaluate(test_X, test_Y):
	model.eval()
	total_loss = 0.
	digits_correct = 0
	digits_total = 0
	for j in xrange(len(test_X)):
		X, Y = train_X[j], train_Y[j]
		model.init_stack()
		for x in X:
			model.forward(x)
		for y in Y:
			a = model.forward(end())
			_, y_ = torch.max(a, 1)

			digits_total += 1
			digits_correct += len(torch.nonzero((y_ == y).data))

			total_loss += criterion(a, y)

	print "epoch {}: loss={:.2f}, acc={:.2f}".format(epoch, sum(total_loss.data) / len(test_X), digits_correct / digits_total)


# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print "hyperparameters: lr={}, batch_size={}, read_dim={}".format(LEARNING_RATE, BATCH_SIZE, READ_SIZE)
for epoch in xrange(EPOCHS):
	print "-- starting epoch {} --".format(epoch)
	train_X, train_Y = shuffle(train_X, train_Y)
	train(train_X, train_Y)
	evaluate(dev_X, dev_Y)