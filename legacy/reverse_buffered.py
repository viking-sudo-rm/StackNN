from __future__ import division

import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from models import BufferedModel


# Language parameters
MIN_LENGTH = 1
MEAN_LENGTH = 10
STD_LENGTH = 2
MAX_LENGTH = 12
TIME_FN = lambda n: n # Number of iterations as a function of input size.

# Hyperparameters
LEARNING_RATE = .01  # .01 is baseline -- .1 doesn't work!
LAMBDA = .01
BATCH_SIZE = 10  # 10 is baseline
READ_SIZE = 2  # 2 is baseline

EPOCHS = 30

model = BufferedModel(3, READ_SIZE, 3)
try:
    model.cuda()
except AssertionError:
    pass

criterion = nn.CrossEntropyLoss()


def randstr():
    length = min(max(MIN_LENGTH, int(random.gauss(MEAN_LENGTH, STD_LENGTH))),
                 MAX_LENGTH)
    return [random.randint(0, 1) for _ in xrange(length)]


reverse = lambda s: s[::-1]
onehot = lambda b: torch.FloatTensor([1. if i == b else 0. for i in xrange(3)])


# def get_tensors(B):
#     X_raw = [randstr() for _ in xrange(B)]
#
#     # initialize X to one-hot encodings of NULL
#     X = torch.FloatTensor(B, MAX_LENGTH, 3)
#     X[:, :, :2].fill_(0)
#     X[:, :, 2].fill_(1)
#
#     # initialize Y to NULL
#     Y = torch.LongTensor(B, MAX_LENGTH)
#     Y.fill_(2)
#
#     for i, x in enumerate(X_raw):
#         y = reverse(x)
#         for j, char in enumerate(x):
#             X[i, j, :] = onehot(char)
#             Y[i, j] = y[j]
#     return Variable(X), Variable(Y)


def get_tensors(B):
    X_raw = [randstr() for _ in xrange(B)]

    # initialize X to one-hot encodings of NULL
    X = torch.FloatTensor(B, 2 * MAX_LENGTH, 3)
    X[:, :, :2].fill_(0)
    X[:, :, 2].fill_(1)

    # initialize Y to NULL
    Y = torch.LongTensor(B, 2 * MAX_LENGTH)
    Y.fill_(2)

    for i, x in enumerate(X_raw):
        y = reverse(x)
        for j, char in enumerate(x):
            X[i, j, :] = onehot(char)
            Y[i, j + len(x)] = y[j]
    return Variable(X), Variable(Y)

train_X, train_Y = get_tensors(800)
dev_X, dev_Y = get_tensors(100)
test_X, test_Y = get_tensors(100)
trace_X, _ = get_tensors(1)


def train(train_X, train_Y):
    model.train()
    total_loss = 0.

    for batch, i in enumerate(
            xrange(0, len(train_X.data) - BATCH_SIZE + 1, BATCH_SIZE)):

        digits_correct = 0
        digits_total = 0
        batch_loss = 0.

        X, Y = train_X[i:i + BATCH_SIZE, :, :], train_Y[i:i + BATCH_SIZE, :]

        # # Buffered model
        zero = Variable(torch.zeros(BATCH_SIZE, 3))
        num_iterations = TIME_FN(2 * MAX_LENGTH)
        model.init_model(BATCH_SIZE, X)
        for j in xrange(num_iterations):
            model.forward()
        for j in xrange(MAX_LENGTH):
            model._buffer_out.pop(1.)
            a = model._buffer_out.read(1.)

            # # Normal seq2seq
            # model.init_stack(BATCH_SIZE)
            # for j in xrange(2 * MAX_LENGTH):
            # 	a = model.forward(X[:,j,:])

            indices = Y[:, j] != 2
            valid_a = a[indices.view(-1, 1)].view(-1, 3)
            valid_Y = Y[:, j][indices]

            if len(valid_a) == 0: continue

            _, valid_y_ = torch.max(valid_a, 1)
            digits_total += len(valid_a)
            digits_correct += len(torch.nonzero((valid_y_ == valid_Y).data))
            batch_loss += criterion(valid_a, valid_Y)

        # Add regularization loss and reset the tracker.
        batch_loss += model.get_and_reset_reg_loss()

        # update the weights
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.data
        if batch % 10 == 9:
            mean_loss = sum(batch_loss.data)
            print "batches {}-{}: loss={:.4f}, acc={:.2f}".format(batch - 9,
                                                                  batch,
                                                                  mean_loss,
                                                                  digits_correct
                                                                  / digits_total)


def evaluate(test_X, test_Y):
    model.eval()
    total_loss = 0.
    digits_correct = 0
    digits_total = 0

    len_X = test_X.size(0)

    # # Buffered model
    zero = Variable(torch.zeros(len_X, 3))
    num_iterations = TIME_FN(2 * MAX_LENGTH)
    model.init_model(len_X, test_X)
    for j in xrange(num_iterations):
        model.forward()
    for j in xrange(MAX_LENGTH):
        model._buffer_out.pop(1.)
        a = model._buffer_out.read(1.)

        # # Normal seq2seq
        # model.init_stack(len(test_X.data))
        # for j in xrange(2 * MAX_LENGTH):
        # 	a = model.forward(test_X[:,j,:])

        indices = test_Y[:, j] != 2
        valid_a = a[indices.view(-1, 1)].view(-1, 3)
        valid_Y = test_Y[:, j][indices]

        if len(valid_a) == 0: continue

        _, valid_y_ = torch.max(valid_a, 1)
        digits_total += len(valid_a)
        digits_correct += len(torch.nonzero((valid_y_ == valid_Y).data))
        total_loss += criterion(valid_a, valid_Y)

    mean_loss = sum(total_loss.data)
    print "epoch {}: loss={:.4f}, acc={:.2f}".format(epoch, mean_loss,
                                                     digits_correct /
                                                     digits_total)


optimizer = optim.Adam(model.parameters(),
                       lr=LEARNING_RATE,
                       weight_decay=LAMBDA,
                       )
print "hyperparameters: lr={}, lambda={} batch_size={}, read_dim={}".format(
    LEARNING_RATE, LAMBDA, BATCH_SIZE, READ_SIZE)

for epoch in xrange(EPOCHS):
    print "-- starting epoch {} --".format(epoch)
    perm = torch.randperm(800)
    train_X, train_Y = train_X[perm], train_Y[perm]
    train(train_X, train_Y)
    evaluate(dev_X, dev_Y)
model.trace(trace_X)
