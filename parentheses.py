from __future__ import division

import random

import torch
import torch.nn as nn
import torch.optim as optim
from nltk import PCFG
from nltk.parse.generate import generate
from torch.autograd import Variable

m = __import__("model-bare")

# Language parameters
DEPTH = 5
MIN_LENGTH = 1
STD_LENGTH = 2
MAX_LENGTH = 2 ** (DEPTH - 1)
MEAN_LENGTH = MAX_LENGTH / 2

# Hyperparameters
LEARNING_RATE = .01  # .01 and .1 seem to work well?
BATCH_SIZE = 10  # 10 is the best I've found
READ_SIZE = 2  # was using 4 before

CUDA = False
EPOCHS = 30

grammar = PCFG.fromstring("""
    S -> S S [.2]
    S -> '(' S ')' [.2] | '(' ')' [.2]
    S -> '[' S ']' [.2] | '[' ']' [.2]
""")

parenthesis_strings = list(generate(grammar, depth=5))

model = m.FFController(3, READ_SIZE, 3)
if CUDA:
    model.cuda()

criterion = nn.CrossEntropyLoss()

code_for = {u'(': 0, u')': 1, u'[': 2, u']': 3, u'#': 4}


def randstr():
    string = random.choice(parenthesis_strings)
    return [code_for[s] for s in string]
    # return [random.randint(0, 1) for _ in xrange(length)]


reverse = lambda s: s[::-1]
onehot = lambda b: torch.FloatTensor([float(i == b) for i in xrange(len(code_for))])


def get_tensors(B):
    X_raw = [randstr() for _ in xrange(B)]

    # initialize X to one-hot encodings of NULL
    X = torch.FloatTensor(B, MAX_LENGTH, len(code_for))
    X[:, :, :len(code_for) - 1].fill_(0)
    X[:, :, len(code_for) - 1].fill_(1)

    # initialize Y to NULL
    Y = torch.LongTensor(B)
    Y.fill_(0)

    for i, x in enumerate(X_raw):
        rand = int(random.gauss(MEAN_LENGTH, STD_LENGTH))
        length = min(max(MIN_LENGTH - 1, rand), len(x) - 1)

        for j, char in enumerate(x[:length]):
            X[i, j, :] = onehot(char)

        Y[i] = x[length]

    return Variable(X), Variable(Y)


train_X, train_Y = get_tensors(800)

# print train_X[0, 0, :]
# print train_Y[0]

dev_X, dev_Y = get_tensors(100)
test_X, test_Y = get_tensors(100)


def train(train_X, train_Y):
    model.train()
    total_loss = 0.

    for batch, i in enumerate(xrange(0, len(train_X.data) - BATCH_SIZE, BATCH_SIZE)):

        digits_correct = 0
        digits_total = 0
        batch_loss = 0.

        X, Y = train_X[i:i + BATCH_SIZE, :], train_Y[i:i + BATCH_SIZE]
        model.init_stack(BATCH_SIZE)
        valid_X = (X[:, :, len(code_for) - 1] == 1)
        lengths_X = torch.sum(valid_X, 1)

        A = Variable(torch.FloatTensor(B, len(code_for)))

        for j in xrange(MAX_LENGTH):
            a = model.forward(X[:, j, :])
            A[:, j] = torch.transpose(a, 0, 1)



            batch_loss += criterion(valid_a, valid_Y)

        # update the weights
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.data
        if batch % 10 == 0:
            print "batch {}: loss={:.4f}, acc={:.2f}".format(batch, sum(batch_loss.data) / BATCH_SIZE,
                                                             digits_correct / digits_total)


def evaluate(test_X, test_Y):
    model.eval()
    total_loss = 0.
    digits_correct = 0
    digits_total = 0
    model.init_stack(len(test_X.data))
    for j in xrange(2 * MAX_LENGTH):

        a = model.forward(test_X[:, j, :])

        indices = test_Y[:, j] != 2
        valid_a = a[indices.view(-1, 1)].view(-1, 3)
        valid_Y = test_Y[:, j][indices]

        if len(valid_a) == 0: continue

        _, valid_y_ = torch.max(valid_a, 1)
        digits_total += len(valid_a)
        digits_correct += len(torch.nonzero((valid_y_ == valid_Y).data))
        total_loss += criterion(valid_a, valid_Y)

    # def my_print(x):
    #     print str(x)
    #
    # total_loss.register_hook(my_print)
    # total_loss.backward()
    print "Gradient: " + str(model.linear.weight.grad)
    print "epoch {}: loss={:.4f}, acc={:.2f}".format(epoch, sum(total_loss.data) / len(test_X),
                                                     digits_correct / digits_total)


finaltestx, finaltesty = get_tensors(1)

# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print "hyperparameters: lr={}, batch_size={}, read_dim={}".format(LEARNING_RATE, BATCH_SIZE, READ_SIZE)
for epoch in xrange(EPOCHS):
    print "-- starting epoch {} --".format(epoch)
    perm = torch.randperm(800)
    train_X, train_Y = train_X[perm], train_Y[perm]
    train(train_X, train_Y)
    evaluate(dev_X, dev_Y)

    model.init_stack(len(finaltestx.data))
    for j in xrange(finaltestx.data.shape[1]):
        a = model.forward(finaltestx[:, j, :])
        print finaltestx[:, j, :], a, model.stack.log()
    print model.state_dict()
