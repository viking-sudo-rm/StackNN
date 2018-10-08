"""
    Code for building language models from (P)CFGs
"""
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.utils import shuffle
from nltk.parse.generate import generate
from nltk import CFG, PCFG
import random
from itertools import izip

from models.vanilla import VanillaModel

# Language parameters
MAX_LENGTH = 25  # bound on length of string (or prefix thereof)

# Hyperparameters
LEARNING_RATE = .01  # .01 and .1 seem to work well?
BATCH_SIZE = 10  # 10 is the best I've found
READ_SIZE = 2  # length of vectors on the stack

EPOCHS = 30

#############################################################
# grammars and coding

# each name_grammar is followed by:
# name_code_for = the codes for the terminals, and the fill code (last)
# name_predict = a list of the terminals to predict in a string
# name_sample_depth = max depth of derivations to produce sample

# Dyck language on two kinds of parentheses
# terminals to predict: ), ]
parentheses_grammar = PCFG.fromstring("""
S -> S S [0.20]
S -> '(' S ')' [0.20] | '(' ')' [0.20]
S -> '[' S ']' [0.20] | '[' ']' [0.20]
""")
parentheses_code_for = {u'(': 0, u')': 1, u'[': 2, u']': 3, '#': 4}
parentheses_predict = [u')', u']']
parentheses_sample_depth = 5
# depth = 5 yields 15,130 strings

# center-marked "palindromes" (using primed symbols to predict)
# terminals to predict: a1, b1
reverse_grammar = PCFG.fromstring("""
S -> "a" S "a1" [0.48]
S -> "b" S "b1" [0.48]
S -> "c" [0.04]
""")
reverse_code_for = {u"a": 0, u"b": 1, u"a1": 2, u"b1": 3, u"c": 4, u"#": 5}
reverse_predict = [u"a1", u"b1"]
reverse_sample_depth = 12
# depth = 12 yields 2047 strings

# agreement grammar that Bob sent (2/8/18)
# modified: removed VP from S productions 
# terminals to predict: Auxsing, Auxplur
agreement_grammar = PCFG.fromstring("""
S -> NPsing "Auxsing" [0.5]
S -> NPplur "Auxplur" [0.5]
NP -> NPsing [0.5]
NP -> NPplur [0.5]
NPsing -> NPsing PP [0.1]
NPplur -> NPplur PP [0.1]
NPsing -> NPsing Relsing [0.1]
NPplur -> NPplur Relplur [0.1]
NPsing -> "Det" "Nsing" [0.8]
NPplur -> "Det" "Nplur" [0.8]
PP -> "Prep" NP [1.0]
Relsing -> "Rel" "Auxsing" VP [0.9]
Relsing -> Relobj [0.1]
Relplur -> "Rel" "Auxplur" VP [0.9]
Relplur -> Relobj [0.1]
Relobj -> "Rel" NPsing "Auxsing" "Vtrans" [0.5]
Relobj -> "Rel" NPplur "Auxplur" "Vtrans" [0.5]
VP -> "Vintrans" [0.75]
VP -> "Vtrans" NP [0.25]
""")
agreement_code_for = {u"Det": 0, u"Nsing": 1, u"Nplur": 2, u"Auxsing": 3,
                      u"Auxplur": 4, u"Rel": 5, u"Prep": 6, u"Vintrans": 7,
                      u"Vtrans": 8, u"#": 9}
agreement_predict = [u"Auxsing", u"Auxplur"]
agreement_sample_depth = 8
# depth 8 yields 1718 strings

############################################
# change these assignments to select grammar to predict for experiment

grammar = agreement_grammar
code_for = agreement_code_for
to_predict = agreement_predict
sample_depth = agreement_sample_depth
############################################

# report symbols to predict and sample_depth
print "to_predict = {}".format(to_predict)
print "sample_depth = {}".format(sample_depth)

# need onehot here to make a list of codes to predict
onehot = lambda b: torch.FloatTensor(
    [1. if i == b else 0. for i in xrange(len(code_for))])

# list of codes of symbols to predict
to_predict_codes = [onehot(code_for[s]) for s in to_predict]


# function to test if a symbol code is in list to predict
def in_predict_codes(code):
    for i in xrange(len(to_predict_codes)):
        if ((code == to_predict_codes[i]).all()):
            return True
    return False


# sample_strings = all strings from grammar of depth at most sample_depth
sample_strings = list(generate(grammar, depth=sample_depth))

# report #, min length and max length for strings in sample_strings
print("number of sample strings = {}".format(len(sample_strings)))
sample_lengths = [len(s) for s in sample_strings]
print("min length = {}, max length = {}".format(min(sample_lengths),
                                                max(sample_lengths)))

# sanity check: report one random string from sample_strings
print "random sample string = {}".format(random.choice(sample_strings))

#################################

model = VanillaModel(len(code_for), READ_SIZE, len(code_for))
try:
    model.cuda()
except AssertionError:
    pass

# Requires PyTorch 0.3.x
criterion = nn.CrossEntropyLoss(reduction='none')


def generate_sample(grammar, prod, frags):
    """
    Generate random sentence using PCFG.
    @see https://stackoverflow.com/questions/15009656/how-to-use-nltk-to
    -generate-sentences-from-an-induced-grammar
    """
    if prod in grammar._lhs_index:  # Derivation
        derivations = grammar._lhs_index[prod]
        derivation = random.choice(derivations)
        for d in derivation._rhs:
            generate_sample(grammar, d, frags)
    else:
        # terminal
        frags.append(str(prod))


reverse = lambda s: s[::-1]


# uniform random choice from  sample_strings
def sample_randstr():
    string = random.choice(sample_strings)
    return [code_for[s] for s in string]


# generate from PCFG grammar using probabilities
def randstr():
    string = []
    generate_sample(grammar, grammar.start(), string)
    #    print string
    return [code_for[s] for s in string]


# currently uses sample_randstr()
def get_tensors(B):
    X_raw = [sample_randstr() for _ in xrange(B)]

    # initialize X to one-hot encodings of NULL
    X = torch.FloatTensor(B, MAX_LENGTH, len(code_for))
    X[:, :, :len(code_for) - 1].fill_(0)
    X[:, :, len(code_for) - 1].fill_(1)

    for i, x in enumerate(X_raw):
        for j, char in enumerate(x if len(x) < MAX_LENGTH else x[:MAX_LENGTH]):
            X[i, j, :] = onehot(char)
    return Variable(X)


train_X = get_tensors(800)
dev_X = get_tensors(100)
test_X = get_tensors(100)
trace_X = get_tensors(1)


def train(train_X):
    model.train()
    total_loss = 0.
    num_correct = 0
    num_total = 0

    # avged per mini-batch

    for batch, i in enumerate(
            xrange(0, len(train_X.data) - BATCH_SIZE, BATCH_SIZE)):

        batch_loss = 0.

        X = train_X[i:i + BATCH_SIZE, :, :]
        model.init_stack(BATCH_SIZE)

        valid_X = (X[:, :, len(code_for) - 1] != 1).type(torch.FloatTensor)

        ############################################
        # this set valid_X to 0 for any symbol not in list to predict

        for k in xrange(BATCH_SIZE):
            for j in xrange(MAX_LENGTH):
                if (not (in_predict_codes(X[k, j, :].data))):
                    valid_X[k, j] = 0

                    #        print "X[0,:,:] = {}".format(X[0,:,:])
                    #        print "valid_X[0] = {}".format(valid_X[0])
                    ############################################

        for j in xrange(1, MAX_LENGTH):
            a = model.forward(X[:, j - 1, :])
            _, y = torch.max(X[:, j, :], 1)
            _, y_pred = torch.max(a, 1)

            batch_loss += torch.mean(valid_X[:, j] * criterion(a, y))
            num_correct += sum(
                (valid_X[:, j] * (y_pred == y).type(torch.FloatTensor)).data)
            num_total += sum(valid_X[:, j].data)

        # update the weights
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.data
        if batch % 10 == 0:
            print "batch {}: loss={:.4f}, acc={:.2f}".format(batch, sum(
                batch_loss.data), num_correct / num_total)


def evaluate(test_X):
    model.eval()
    model.init_stack(len(test_X.data))

    total_loss = 0.
    num_correct = 0
    num_total = 0

    valid_X = (test_X[:, :, len(code_for) - 1] != 1).type(torch.FloatTensor)

    ######################################
    for k in xrange(len(test_X.data)):
        for j in xrange(MAX_LENGTH):
            if (not in_predict_codes(test_X[k, j, :].data)):
                valid_X[k, j] = 0
                ######################################

    y_prev = None

    for j in xrange(1, MAX_LENGTH):
        a = model.forward(test_X[:, j - 1, :])
        _, y = torch.max(test_X[:, j, :], 1)
        _, y_pred = torch.max(a, 1)

        total_loss += torch.mean(valid_X[:, j] * criterion(a, y))
        num_correct += sum(
            (valid_X[:, j] * (y_pred == y).type(torch.FloatTensor)).data)
        num_total += sum(valid_X[:, j].data)

        # print str(F.softmax(a[0, :])) + ", " + str(y_prev) + ", " + str(y[0])
        y_prev = y[0]

    # print model.state_dict()
    print "epoch {}: loss={:.4f}, acc={:.2f}".format(epoch,
                                                     sum(total_loss.data),
                                                     num_correct / num_total)


# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print "hyperparameters: lr={}, batch_size={}, read_dim={}".format(
    LEARNING_RATE, BATCH_SIZE, READ_SIZE)
for epoch in xrange(EPOCHS):
    print "-- starting epoch {} --".format(epoch)
    perm = torch.randperm(800)
    train_X = train_X[perm]
    train(train_X)
    evaluate(dev_X)
