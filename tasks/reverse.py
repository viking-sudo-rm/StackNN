from __future__ import division

import random

import torch
import torch.nn as nn
from torch.autograd import Variable

from base import Task
from models import VanillaController
from shmetworks.feedforward import LinearSimpleStructShmetwork
from structs import Stack


class ReverseTask(Task):
    """
    String Reversal
    """

    def __init__(self,
                 min_length=1,
                 max_length=12,
                 mean_length=10,
                 std_length=2.,
                 num_symbols=2,
                 batch_size=10,
                 clipping_norm=None,
                 criterion=nn.CrossEntropyLoss(),
                 cuda=False,
                 epochs=100,
                 early_stopping_steps=5,
                 hidden_size=10,
                 learning_rate=0.01,
                 load_path=None,
                 l2_weight=0.01,
                 model_type=VanillaController,
                 shmetwork_type=LinearSimpleStructShmetwork,
                 read_size=2,
                 save_path=None,
                 struct_type=Stack,
                 time_function=(lambda t: t),
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

        :type num_symbols: int
        :param num_symbols: The number of possible symbols appearing in
            input and output strings, not including NULL

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch

        :type clipping_norm:
        :param clipping_norm:

        :type criterion: nn.modules.loss._Loss
        :param criterion: The error function used for training the model

        :type cuda: bool
        :param cuda: If True, CUDA functionality will be used

        :type epochs: int
        :param epochs: The number of training epochs that will be
            performed when executing an experiment

        :type hidden_size: int
        :param hidden_size: The size of state vectors

        :type learning_rate: float
        :param learning_rate: The learning rate used for training

        :type load_path: str
        :param load_path: The neural shmetwork will be initialized to a
            saved shmetwork located in this path. If load_path is set to
            None, then the shmetwork will be initialized to an empty state

        :type l2_weight: float
        :param l2_weight: The amount of l2 regularization used for
            training

        :type model_type: type
        :param model_type: The type of Controller that will be trained
            and evaluated

        :type shmetwork_type: type
        :param shmetwork_type: The type of neural shmetwork that will drive
            the Controller

        :type read_size: int
        :param read_size: The length of the vectors stored on the neural
            data structure

        :type save_path: str
        :param save_path: If this param is not set to None, then the
            neural shmetwork will be saved to the path specified by this
            save_path

        :type struct_type: type
        :param struct_type: The type of neural data structure that will
            be used by the Controller

        :type time_function: function
        :param time_function: A function mapping the length of an input
            to the number of computational steps the shmetwork will
            perform on that input

        :type verbose: bool
        :param verbose: If True, the progress of the experiment will be
            displayed in the console
        """
        self.num_symbols = num_symbols
        super(ReverseTask, self).__init__(batch_size=batch_size,
                                          clipping_norm=clipping_norm,
                                          criterion=criterion,
                                          cuda=cuda,
                                          epochs=epochs,
                                          early_stopping_steps=early_stopping_steps,
                                          hidden_size=hidden_size,
                                          learning_rate=learning_rate,
                                          load_path=load_path,
                                          l2_weight=l2_weight,
                                          max_x_length=max_length * 2,
                                          max_y_length=max_length * 8,
                                          model_type=model_type,
                                          shmetwork_type=shmetwork_type,
                                          null=unicode(num_symbols),
                                          read_size=read_size,
                                          save_path=save_path,
                                          struct_type=struct_type,
                                          time_function=time_function,
                                          verbose=verbose)

        self.min_length = min_length
        self.mean_length = mean_length
        self.std_length = std_length
        self.max_length = max_length

    def reset_model(self, model_type, shmetwork_type, struct_type, **kwargs):
        self.model = model_type(self.alphabet_size,
                                self.read_size,
                                self.alphabet_size,
                                shmetwork_type=shmetwork_type,
                                struct_type=struct_type,
                                **kwargs)

    def _init_alphabet(self, null):
        return {unicode(i): i for i in xrange(self.num_symbols + 1)}

    """ Model Training """

    def _evaluate_step(self, x, y, a, j):
        """
        Computes the loss, number of guesses correct, and total number
        of guesses at the jth time step. The loss for a string is
        considered to be 0 if the neural shmetwork is still reading the
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
        :param a: The output of the neural shmetwork at the jth time step,
            represented as a 2D vector. For each i, a[i, :] is the
            output of the neural shmetwork at the jth time step, in one-
            hot representation

        :type j: int
        :param j: This function is called during the jth time step of
            the neural shmetwork's computation

        :rtype: tuple
        :return: The loss, number of correct guesses, and number of
            total guesses at the jth time step
        """
        indices = (y[:, j] != self.alphabet[self.null])
        # Indexing semantics in the line below were changed in different versions of pytorch.
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
        return

    def randstr(self):
        """
        Generates a random string over self.alphabet, not including
        NULLs. The lengths of the strings generated by this function
        have a Gaussian distribution with the following properties.
            Minimum Length: self.min_length
            Maximum Length: self.max_length
            Average Length: self.mean_length
            Standard Deviation: self.std_length

        :rtype: list
        :return: A sequence of "0"s and "1"s
        """
        length = int(random.gauss(self.mean_length, self.std_length))
        length = min(max(self.min_length, length), self.max_length)
        s = [random.randint(0, self.num_symbols - 1) for _ in xrange(length)]
        return [unicode(w) for w in s]

    def get_tensors(self, num_tensors):
        """
        Generates a dataset containing correct input and output values
        for the reversal task. An input value is a sequence of n-many
        symbols for some n. An output value is a sequence of n-many
        NULLs, followed by the input value backwards. Input and output
        values are padded to their maximum lengths with NULLs.

        For example, the following is a valid input-output pair,
        assuming that u"2" is the null symbol.
            Input: [u"1", u"0", u"2", u"2"]
            Output: [u"2", u"2", u"0", u"1"]

        :type num_tensors: int
        :param num_tensors: The number of examples in the dataset

        :rtype: tuple
        :return: A Variable containing the input values and a Variable
            containing the output values
        """
        x_raw = [self.randstr() for _ in xrange(num_tensors)]
        y_raw = [[self.null for _ in xrange(len(s))] + s[::-1] for s in x_raw]

        x_var = self.sentences_to_one_hot(2 * self.max_length, *x_raw)
        y_var = self.sentences_to_codes(8 * self.max_length, *y_raw)

        return x_var, y_var


class CopyTask(ReverseTask):
    """
    String Copying
    """

    def get_tensors(self, num_tensors):
        """
        Generates a dataset containing correct input and output values
        for the copy task. The input and output values are identical.

        :type num_tensors: int
        :param num_tensors: The number of examples in the dataset

        :rtype: tuple
        :return: A Variable containing the input values and a Variable
            containing the output values
        """
        x_raw = [self.randstr() for _ in xrange(num_tensors)]

        x_var = self.sentences_to_one_hot(2 * self.max_length, *x_raw)
        y_var = self.sentences_to_codes(2 * self.max_length, *x_raw)

        return x_var, y_var


class ReverseDeletionTask(ReverseTask):
    """
    Reverse the result of deleting the second half of the
    alphabet symbols from the input string.
    Example: 12200313011 => 1101001  over the alphabet {0,1,2,3}
    """

    def get_tensors(self, num_tensors):
        """
        Generates a dataset containing correct input and output values
        for the reverse deletion task.

        :type num_tensors: int
        :param num_tensors: The number of examples in the dataset

        :rtype: tuple
        :return: A Variable containing the input values and a Variable
            containing the output values
        """
        x_raw = [self.randstr() for _ in xrange(num_tensors)]
        y_raw = [[self.null for _ in xrange(len(s))] + self.reverse_with_delete(s) for s in x_raw]

        x_var = self.sentences_to_one_hot(2 * self.max_length, *x_raw)
        y_var = self.sentences_to_codes(8 * self.max_length, *y_raw)

        return x_var, y_var

    def reverse_with_delete(self, s):
        large_symbol = self.num_symbols//2
        t = []
        for symbol in s:
            if int(symbol) < large_symbol:
                t.append(symbol)
        return t[::-1]
