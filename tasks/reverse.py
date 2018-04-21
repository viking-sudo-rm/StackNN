from __future__ import division

import random

import torch
import torch.nn as nn
from torch.autograd import Variable

from base import Task
from models.vanilla import Controller as FFStackController


class ReverseTask(Task):
    """
    String Reversal
    """

    def __init__(self,
                 min_length=1,
                 max_length=12,
                 mean_length=10,
                 std_length=2.,
                 batch_size=10,
                 criterion=nn.CrossEntropyLoss(),
                 cuda=False,
                 epochs=30,
                 learning_rate=0.01,
                 l2_weight=0.01,
                 model=None,
                 model_type=FFStackController,
                 read_size=2,
                 verbose=True):
        """
        Constructor for the ReverseTask object. The only information
        that needs to be specified by the user is information about the
        distribution of the strings appearing in the input data.

        :type min_length: int
        :param min_length: The shortest possible length of an input
            string

        :type max_length: int
        :param max_length: The longest possible length of an input
            string

        :type mean_length: int
        :param mean_length: The average length of an input string

        :type std_length: float
        :param std_length: The standard deviation of the length of an
            input string

        :type batch_size: int
        :param batch_size: The number of trials in each batch

        :type criterion: nn.modules.loss._Loss
        :param criterion: The error function used for training the model

        :type cuda: bool
        :param cuda: If True, CUDA functionality will be used

        :type epochs: int
        :param epochs: The number of training epochs that will be
            performed when executing an experiment

        :type learning_rate: float
        :param learning_rate: The learning rate used for training

        :type l2_weight: float
        :param l2_weight: The amount of l2 regularization used for
            training

        :param model: The model that will be trained and evaluated.
            This parameter is being kept for compatibility with older
            code. Please use the model_type parameter instead in order
            to automatically instantiate models.

        :type model_type: type
        :param model_type: The model that will be trained and evaluated.
            For this task, please pass the *type* of the model to the
            constructor, not an instance of the model class

        :type read_size: int
        :param read_size: The length of the vectors stored on the neural
            data structure

        :type verbose: bool
        :param verbose: If True, the progress of the experiment will be
            displayed in the console
        """
        super(ReverseTask, self).__init__(batch_size=batch_size,
                                          criterion=criterion,
                                          cuda=cuda,
                                          epochs=epochs,
                                          learning_rate=learning_rate,
                                          l2_weight=l2_weight,
                                          max_x_length=max_length * 2,
                                          max_y_length=max_length * 8,
                                          model=model,
                                          model_type=model_type,
                                          read_size=read_size,
                                          verbose=verbose)

        self.min_length = min_length
        self.mean_length = mean_length
        self.std_length = std_length
        self.max_length = max_length

    def reset_model(self, model_type):
        """
        Instantiates a neural network model of a given type that is
        compatible with this Task. This function must set self.model to
        an instance of model_type

        :type model_type: type
        :param model_type: A type from the models package. Please pass
            the desired model's *type* to this parameter, not an
            instance thereof

        :return: None
        """
        self.model = model_type(3, self.read_size, 3)

    """ Model Training """

    def _evaluate_step(self, x, y, a, j):
        """
        Computes the loss, number of guesses correct, and total number
        of guesses at the jth time step. The loss for a string is
        considered to be 0 if the neural network is still reading the
        input string.

        :type x: Variable
        :param x: The input data, represented as a 3D tensor. Each
            example consists of a string of 0s and 1s, followed by
            "null"s. All symbols are in one-hot representation

        :type y: Variable
        :param y: The output data, represented as a 2D tensor. Each
            example consists of a sequence of "null"s, followed by a
            string backwards. All symbols are represented numerically

        :type a: Variable
        :param a: The output of the neural network at the jth time step,
            represented as a 2D vector. For each i, a[i, :] is the
            output of the neural network at the jth time step, in one-
            hot representation

        :type j: int
        :param j: This function is called during the jth time step of
            the neural network's computation

        :rtype: tuple
        :return: The loss, number of correct guesses, and number of
            total guesses at the jth time step
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

    def get_data(self):
        """
        Generates training and testing datasets for this task using the
        self.get_tensors method.

        :return: None
        """
        self.train_x, self.train_y = self.get_tensors(800)
        self.test_x, self.test_y = self.get_tensors(100)
        return

    def randstr(self):
        """
        Generates a random string of 0s and 1s. The length of the string
        is between self.min_length and self.max_length. The average
        length of the string is self.mean_length. The standard deviation
        of the length of the string is self.std_length.

        :rtype: list
        :return: A sequence of "0"s and "1"s
        """
        length = int(random.gauss(self.mean_length, self.std_length))
        length = min(max(self.min_length, length), self.max_length)
        return [random.randint(0, 1) for _ in xrange(length)]

    def get_tensors(self, b):
        """
        Generates a dataset containing correct input and output values
        for the reversal task. An input value is a sequence of 0s and 1s
        in one-hot encoding, followed by "null"s. An output value is a
        sequence of "null"s of the same length as the input, followed by
        the reverse of the input string, as a sequence of raw characters.

        For example, the following is a valid input-output pair.
            input: [0., 1., 0.], [1., 0., 0.],
                    [0., 0., 1.], [0., 0., 1.]
            output: null, null, 0, 1

        :type b: int
        :param b: The number of examples in the dataset

        :rtype: tuple
        :return: A Variable containing the input values and a Variable
            containing the output values
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
        2 represents "null."

        :type b: int
        :param b: 0, 1, or 2

        :rtype: torch.FloatTensor
        :return: The one-hot encoding of b
        """
        return torch.FloatTensor([float(i == b) for i in xrange(3)])
