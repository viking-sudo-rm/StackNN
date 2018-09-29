from __future__ import division

import operator
import random
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from torch.autograd import Variable

from base import Task
from models import BufferedController
from shmetworks.feedforward import LinearSimpleStructShmetwork
from shmetworks.recurrent import RNNSimpleStructShmetwork
from structs import Stack


class EvaluationTask(Task):
    """
    Abstract class for experiments where the shmetwork is incrementally
    fed a sequence and at every iteration has to evaluate a given
    function over all the sequence elements it has seen by that point.
    """
    __metaclass__ = ABCMeta

    def __init__(self,
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
                 max_length=12,
                 # TODO: strings are len 12 regardless.
                 model_type=BufferedController,
                 shmetwork_type=LinearSimpleStructShmetwork,
                 read_size=2,
                 reg_weight=1.,
                 save_path=None,
                 struct_type=Stack,
                 time_function=(lambda t: t),
                 verbose=True):
        """
        Constructor for the EvaluationTask object.

        :type max_length: int
        :param max_length: The longest possible length of an input
            string

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch

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

        :type model_type: type
        :param model_type: The type of Controller that will be trained
            and evaluated

        :type shmetwork_type: type
        :param shmetwork_type: The type of neural shmetwork that will drive
            the Controller

        :type read_size: int
        :param read_size: The length of the vectors stored on the neural
            data structure

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
        super(EvaluationTask, self).__init__(batch_size=batch_size,
                                             clipping_norm=clipping_norm,
                                             criterion=criterion,
                                             cuda=cuda,
                                             epochs=epochs,
                                             early_stopping_steps=early_stopping_steps,
                                             hidden_size=hidden_size,
                                             learning_rate=learning_rate,
                                             load_path=load_path,
                                             l2_weight=l2_weight,
                                             max_x_length=max_length,
                                             max_y_length=max_length,
                                             model_type=model_type,
                                             shmetwork_type=shmetwork_type,
                                             null=u"2",
                                             read_size=read_size,
                                             reg_weight=reg_weight,
                                             save_path=save_path,
                                             struct_type=struct_type,
                                             time_function=time_function,
                                             verbose=verbose)

        self.max_length = max_length

    def reset_model(self, model_type, shmetwork_type, struct_type,
                    reg_weight=1., **kwargs):
        """
        Instantiates a neural shmetwork model of a given type that is
        compatible with this Task. This function must set self.model to
        an instance of model_type

        :type model_type: type
        :param model_type: A type from the models package

        :type shmetwork_type: type
        :param shmetwork_type: The type of the Shmetwork that will perform
            the neural shmetwork computations

        :type struct_type: type
        :param struct_type: The type of neural data structure that this
            Controller will operate

        :return: None
        """
        self.model = model_type(self.alphabet_size,
                                self.read_size,
                                self.alphabet_size,
                                shmetwork_type=shmetwork_type,
                                struct_type=struct_type,
                                reg_weight=reg_weight)

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
        :param a: The output of the neural shmetwork at the jth time step,
            represented as a 2D vector

        :type j: int
        :param j: This function is called during the jth time step of
            the neural shmetwork's computation

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
        return

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
        the neural shmetwork has to learn. For example, if the shmetwork
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
            shmetwork has to learn) on s
        """
        raise NotImplementedError("Missing implementation for eval_func")


class XORTask(EvaluationTask):
    """
    XOR evaluation task: the shmetwork is fed strings of length n and is
    given 2n time steps to output a string whose ith bit is the xor of
    the first i bits in the input string. We define the xor of single-
    bit inputs to be equal to the input bit. For example, if the input
    is

        1 0 1 0 1 1 0,

    the shmetwork will have to output

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

    def __init__(self,
                 batch_size=10,
                 clipping_norm=None,
                 criterion=nn.CrossEntropyLoss(),
                 cuda=False,
                 epochs=30,
                 early_stopping_steps=5,
                 hidden_size=10,
                 learning_rate=0.01,
                 load_path=None,
                 l2_weight=0.01,
                 model_type=BufferedController,
                 shmetwork_type=RNNSimpleStructShmetwork,
                 read_size=2,
                 reg_weight=1.,
                 save_path=None,
                 struct_type=Stack,
                 str_length=12,
                 time_function=(lambda t: 2 * t),
                 verbose=True):
        """
        Constructor for the XORTask object. The only information that
        needs to be specified by the user is how long the input strings
        are.
        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch
        :type clipping_norm: float
        :param clipping_norm:
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
        :type model_type: type
        :param model_type: The type of Controller that will be trained
            and evaluated
        :type shmetwork_type: type
        :param shmetwork_type: The type of neural shmetwork that will drive
            the Controller
        :type read_size: int
        :param read_size: The length of the vectors stored on the neural
            data structure
        :type struct_type: type
        :param struct_type: The type of neural data structure that will
            be used by the Controller
        :type str_length: int
        :param str_length: The number of bits in each input string
        :type verbose: bool
        :param verbose: If True, the progress of the experiment will be
            displayed in the console
        """
        super(XORTask, self).__init__(batch_size=batch_size,
                                      clipping_norm=clipping_norm,
                                      criterion=criterion,
                                      cuda=cuda,
                                      epochs=epochs,
                                      early_stopping_steps=early_stopping_steps,
                                      hidden_size=hidden_size,
                                      learning_rate=learning_rate,
                                      load_path=load_path,
                                      l2_weight=l2_weight,
                                      max_length=str_length,
                                      model_type=model_type,
                                      shmetwork_type=shmetwork_type,
                                      read_size=read_size,
                                      reg_weight=reg_weight,
                                      save_path=save_path,
                                      struct_type=struct_type,
                                      time_function=time_function,
                                      verbose=verbose)

        self.str_length = str_length

        # TIME_FN = n??

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
