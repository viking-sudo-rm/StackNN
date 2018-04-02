from __future__ import division

import random

import torch
import torch.nn as nn
from torch.autograd import Variable

from base import Task


class ReverseTask(Task):
    """
    String Reversal
    """

    def __init__(self,
                 min_length=1,
                 mean_length=10,
                 std_length=2,
                 max_length=12,
                 learning_rate=0.1,
                 batch_size=10,
                 read_size=1,
                 cuda=False,
                 epochs=100,
                 model=None,
                 criterion=nn.CrossEntropyLoss(),
                 verbose=False):
        """

        :param min_length:
        :param mean_length:
        :param std_length:
        :param max_length:
        :param learning_rate:
        :param batch_size:
        :param read_size:
        :param cuda:
        :param epochs:
        :param model:
        :param criterion:
        """
        super(ReverseTask, self).__init__(max_x_length=max_length * 2,
                                          max_y_length=max_length * 8,
                                          learning_rate=learning_rate,
                                          batch_size=batch_size,
                                          read_size=read_size,
                                          cuda=cuda,
                                          epochs=epochs,
                                          model=model,
                                          criterion=criterion,
                                          verbose=verbose)

        self.min_length = min_length
        self.mean_length = mean_length
        self.std_length = std_length
        self.max_length = max_length

    """ Model Training """

    def _evaluate_step(self, x, y, a, j):
        """


        :type x: torch.FloatTensor
        :param x: The input data, consisting of a set of
            strings of 0s and 1s

        :type y: torch.LongTensor
        :param y: The output data, consisting of NULLs followed
            by the input strings backwards

        :type a: torch.LongTensor
        :param a: The output of the neural network during this
            training step

        :type j: int
        :param j: The number of this trial

        :rtype: tuple
        :return:
        """
        indices = (y[:, j] != 2)
        valid_a = a[indices.view(-1, 1)].view(-1, 3)
        valid_y = y[:, j][indices]
        if len(valid_a) == 0:
            return None, None, None

        _, valid_y_ = torch.max(valid_a, 1)

        total = len(valid_a)
        correct = len(torch.nonzero((valid_y_ == valid_y).data))
        loss = self.criterion(valid_a, valid_y)

        return loss, correct, total

    """ Data Generation """

    def randstr(self):
        """
        Generates a random string of 0s and 1s. The length of the
        string is between self.min_length and self.max_length. The
        average length of the string is self.mean_length. The standard
        deviation of the length of the string is self.std_length.

        :rtype: list
        :return: A sequence of 0s and 1s
        """
        length = int(random.gauss(self.mean_length, self.std_length))
        length = min(max(self.min_length, length), self.max_length)
        return [random.randint(0, 1) for _ in xrange(length)]

    def get_tensors(self, b):
        """
        Generates a dataset containing correct input and output
        values for the reversal task. An input value is a string
        of 0s and 1s in one-hot encoding, followed by NULLs. An
        output value is a sequence of NULLs of the same length
        as the input, followed by the reverse of the input string,
        as a sequence of raw characters.

        For example, the following is a valid input-output pair.
            input: [0., 1., 0.], [1., 0., 0.],
                    [0., 0., 1.], [0., 0., 1.]
            output: NULL, NULL, 0, 1

        :type b: int
        :param b: The number of examples in the dataset

        :rtype: tuple
        :return: A Variable containing the input values and a
            Variable containing the output values
        """
        x_raw = [self.randstr() for _ in xrange(b)]

        # Initialize x to one-hot encodings of NULL
        x = torch.FloatTensor(b, 2 * self.max_length, 3)
        x[:, :, :2].fill_(0)
        x[:, :, 2].fill_(1)

        # Initialize y to NULL
        y = torch.LongTensor(b, 8 * self.max_length)
        y.fill_(2)

        for i, s in enumerate(x_raw):
            t = ReverseTask.reverse(s)
            for j, char in enumerate(s):
                x[i, j, :] = ReverseTask.one_hot(char)
                y[i, j + len(s)] = t[j]

        return Variable(x), Variable(y)

    @staticmethod
    def reverse(s):
        """
        Reverses a string.

        :type s: str
        :param s: A string

        :rtype: str
        :return: s, backwards
        """
        return s[::-1]

    @staticmethod
    def one_hot(b):
        """
        Computes the following one-hot encoding:
            0 -> [1., 0., 0.]
            1 -> [0., 1., 0.]
            2 -> [0., 0., 1.]

        0 and 1 represent alphabet symbols.
        2 represents NULL.

        :type b: int
        :param b: 0, 1, or 2

        :rtype: torch.FloatTensor
        :return: The one-hot encoding of b
        """
        return torch.FloatTensor([float(i == b) for i in xrange(3)])
