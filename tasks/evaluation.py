from __future__ import division

import operator
import random
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from torch.autograd import Variable

from tasks.base import FormalTask
from models import BufferedModel
from controllers.feedforward import LinearSimpleStructController
from controllers.recurrent import RNNSimpleStructController
from structs import Stack


class EvaluationTask(FormalTask):

    """
    Abstract class for experiments where the controller is incrementally
    fed a sequence and at every iteration has to evaluate a given
    function over all the sequence elements it has seen by that point.
    """


    class Params(FormalTask.Params):

        def __init__(self, **kwargs):
            self.max_length = kwargs.get("max_length", 12)
            super(EvaluationTask.Params, self).__init__(**kwargs)
            self.null = u"2"
            self.max_x_length = self.max_length
            self.max_y_length = self.max_length


    @property
    def input_size(self):
        return self.alphabet_size

    @property
    def output_size(self):
        return self.alphabet_size

    def _init_alphabet(self, null):
        return {u"0": 0, u"1": 1, u"2": 2}

    """ Model Training """

    def _evaluate_step(self, x, y, a, j):
        """
        Computes the loss, number of guesses correct, and total number
        of guesses at the jth time step.

        :type x: Variable
        :param x: The input data, represented as a 3D tensor

        :type y: Variable
        :param y: The output data, represented as a 2D tensor

        :type a: Variable
        :param a: The output of the neural network at the jth time step,
            represented as a 2D vector

        :type j: int
        :param j: This function is called during the jth time step of
            the neural network's computation

        :rtype: tuple
        :return: The loss, number of correct guesses, and number of
            total guesses at the jth time step
        """
        indices = (y[:, j] != self.alphabet[self.null])
        # Indexing conventions changed with PyTorch version.
        valid_a = a[indices.view(-1)].view(-1, self.alphabet_size)
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

    @abstractmethod
    def sample_str(self):
        """
        Sample a random string of 0s and 1s from the distribution
        relevant to the given experiment.

        :rtype: list
        :return: A sequence of "0"s and "1"s
        """
        raise NotImplementedError("Missing implementation for sample_str")

    def get_tensors(self, num_tensors):
        """
        Generates a dataset containing correct input and output values
        for the EvaluationTask. An input value is a sequence of 0s and
        1s in one-hot encoding. An output value is a sequence of
        integers such that the ith integer is the result of computing
        eval_func on the first i elements of the input sequence.

        For example, the following is a valid input-output pair.
            Input: [1, 0, 1]
            Output: [eval_func([1]), eval_func([1, 0]),
                     eval_func([1, 0, 1])]

        :type num_tensors: int
        :param num_tensors: The number of examples in the dataset

        :rtype: tuple
        :return: A Variable containing the input values and a Variable
            containing the output values
        """
        x_raw = [self.sample_str() for _ in xrange(num_tensors)]
        y_raw = [[self.eval_func(s[:j + 1]) for j in xrange(len(s))]
                 for s in x_raw]

        x_sent = [[unicode(w) for w in s] for s in x_raw]
        y_sent = [[unicode(w) for w in s] for s in y_raw]

        x_var = self.sentences_to_one_hot(self.max_length, *x_sent)
        y_var = self.sentences_to_codes(self.max_length, *y_sent)

        return x_var, y_var

    @abstractmethod
    def eval_func(self, s):
        """
        The function evaluated over successive sequences of inputs that
        the neural network has to learn. For example, if the controller
        input is
        
            [1, 0, 1, 1],
        
        it will be trained to produce
        
            [eval_func([1]),
             eval_func([1, 0]),
             eval_func([1, 0, 1]),
             eval_func([1, 0, 1, 1])].
        
        Implementation of eval_func depends on the specific task.

        :type s: list
        :param s: A sequence of 0s and 1s (of type int)

        :rtype: int
        :return: The result of computing the target function (that the
            controller has to learn) on s
        """
        raise NotImplementedError("Missing implementation for eval_func")

    """ Data Visualization """

    @property
    def generic_example(self):
        """
        The string for visualizations.

        TODO: Make this a function of the grammar.
        """
        return [u"#"]


class XORTask(EvaluationTask):
    """
    XOR evaluation task: the controller is fed strings of length n and is
    given 2n time steps to output a string whose ith bit is the xor of
    the first i bits in the input string. We define the xor of single-
    bit inputs to be equal to the input bit. For example, if the input
    is

        1 0 1 0 1 1 0,

    the controller will have to output

        xor(1)
        xor(1, 0)
        xor(1, 0, 1)
        xor(1, 0, 1, 0)
        xor(1, 0, 1, 0, 1)
        xor(1, 0, 1, 0, 1, 1)
        xor(1, 0, 1, 0, 1, 1, 0),

    which is equivalent to

        1 1 0 0 1 0 0.
    """


    class Params(EvaluationTask.Params):

        def __init__(self, **kwargs):
            self.str_length = kwargs.get("str_length", 12)
            # Set a new default value for the time function.
            time_function = kwargs.get("time_function", lambda t: 2 * t)
            super(XORTask.Params, self).__init__(max_length=self.str_length,
                                                 time_function=time_function,
                                                 **kwargs)


    """ Model Training """

    def sample_str(self):
        """
        Generates a random string of 0s and 1s. The length of the string
        is self.str_length.

        :rtype: list
        :return: A sequence of 0s and 1s
        """
        return [random.randint(0, 1) for _ in xrange(self.str_length)]

    def eval_func(self, s):
        return reduce(operator.xor, s, 0)


class DelayedXORTask(XORTask):
    def get_tensors(self, num_tensors):
        """
        Generates a dataset containing correct input and output values
        for the EvaluationTask. An input value is a sequence of 0s and
        1s in one-hot encoding. An output value is a sequence of
        integers such that the ith integer is the result of computing
        eval_func on the first i elements of the input sequence.

        For example, the following is a valid input-output pair.
            Input: [1, 0, 1]
            Output: [eval_func([1]), eval_func([1, 0]),
                     eval_func([1, 0, 1])]

        :type num_tensors: int
        :param num_tensors: The number of examples in the dataset

        :rtype: tuple
        :return: A Variable containing the input values and a Variable
            containing the output values
        """
        x_raw = [self.sample_str() for _ in xrange(num_tensors)]
        y_raw = [[self.eval_func(s[:j]) for j in xrange(len(s))]
                 for s in x_raw]

        x_sent = [[unicode(w) for w in s] for s in x_raw]
        y_sent = [[unicode(w) for w in s] for s in y_raw]

        x_var = self.sentences_to_one_hot(self.max_length, *x_sent)
        y_var = self.sentences_to_codes(self.max_length, *y_sent)

        return x_var, y_var
