from __future__ import division

from abc import ABCMeta, abstractmethod

import random

import torch
import torch.nn as nn
from torch.autograd import Variable

from base import Task
from models import BufferedController
from models import VanillaController 
from models.networks.feedforward import LinearSimpleStructNetwork
from structs import Stack

class EvaluationTask(Task):
    """
    Abstract class for experiments where the network is incrementally
    fed a sequence and at every iteration has to evaluate a given
    function over all the sequence elements it has seen by that point.
    """
    __metaclass__ = ABCMeta

    def __init__(self,
                 batch_size=10,
                 criterion=nn.CrossEntropyLoss(),
                 cuda=False,
                 epochs=30,
                 hidden_size=10,
                 learning_rate=0.01,
                 load_path=None,
                 l2_weight=0.01,
                 max_length=10,
                 model=None,
                 model_type=BufferedController,
                 network_type=LinearSimpleStructNetwork,
                 read_size=2,
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

        :param model: The model that will be trained and evaluated.
            This parameter is being kept for compatibility with older
            code. Please use the model_type parameter instead in order
            to automatically instantiate models

        :type model_type: type
        :param model_type: The type of Controller that will be trained
            and evaluated

        :type network_type: type
        :param network_type: The type of neural network that will drive
            the Controller

        :type read_size: int
        :param read_size: The length of the vectors stored on the neural
            data structure

        :type struct_type: type
        :param struct_type: The type of neural data structure that will
            be used by the Controller

        :type time_function: function
        :param time_function: A function mapping the length of an input
            to the number of computational steps the network will
            perform on that input

        :type verbose: bool
        :param verbose: If True, the progress of the experiment will be
            displayed in the console
        """
        super(EvaluationTask, self).__init__(batch_size=batch_size,
                                          criterion=criterion,
                                          cuda=cuda,
                                          epochs=epochs,
                                          hidden_size=hidden_size,
                                          learning_rate=learning_rate,
                                          load_path=load_path,
                                          l2_weight=l2_weight,
                                          max_x_length=max_length,
                                          max_y_length=max_length,
                                          model=model,
                                          model_type=model_type,
                                          network_type=network_type,
                                          read_size=read_size,
                                          save_path=save_path,
                                          struct_type=struct_type,
                                          time_function=time_function,
                                          verbose=verbose)

        self.max_length = max_length


    def reset_model(self, model_type, network_type, struct_type, **kwargs):
        """
        Instantiates a neural network model of a given type that is
        compatible with this Task. This function must set self.model to
        an instance of model_type

        :type model_type: type
        :param model_type: A type from the models package. Please pass
            the desired model's *type* to this parameter, not an
            instance thereof

        :type network_type: type
        :param network_type: The type of the Network that will perform
            the neural network computations

        :type struct_type: type
        :param struct_type: The type of neural data structure that this
            Controller will operate

        :return: None
        """
        self.model = model_type(3, self.read_size, 3,
                                network_type=network_type,
                                struct_type=struct_type)

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

    def sample_str(self):
        """
        Sample a random string of 0s and 1s from the distribution relevant
        to the given experiment.

        :rtype: list
        :return: A sequence of "0"s and "1"s
        """
        raise NotImplementedError("Missing implementation for sample_str")

    def get_tensors(self, b):
        """
        Generates a dataset containing correct input and output values
        for the reversal task. An input value is a sequence of 0s and 1s
        in one-hot encoding. An output value is a sequence of integers,
        the result of computing eval_func on the first i elements of the
        input sequence for 1 <= i <= {length of input sequence}.

        For example, the following is a valid input-output pair.
            input: [0., 1.], [1., 0.], [0., 1.]
            output: eval_func([1]), eval_func([1, 0]), eval_func([1, 0, 1])

        :type b: int
        :param b: The number of examples in the dataset

        :rtype: tuple
        :return: A Variable containing the input values and a Variable
            containing the output values
        """
        x_raw = [self.sample_str() for _ in xrange(b)]

        # Initialize x to store one-hot encodings of the 
        x = torch.FloatTensor(b, self.max_length, 3)

        # Initialize y to store the bits (and null placeholders after
        # the string of bits ends)
        y = torch.LongTensor(b, self.max_length)
        y.fill_(2)

        for i, s in enumerate(x_raw):
            for j, char in enumerate(s):
                x[i, j, :] = EvaluationTask.one_hot(char)
                y[i, j] = self.eval_func(s[:j])

        #print("x and y: ")
        #print("x: {}".format(x))
        #print("y: {}".format(y))
        return Variable(x), Variable(y)

    @abstractmethod
    def eval_func(self, s):
        """
        The function evaluated over successive sequences of inputs that the
        neural network has to learn.  For example, if the network input is
        
        [1, 0, 1, 1],
        
        it will be trained to produce
        
        [eval_func([1]),
         eval_func([1, 0]),
         eval_func([1, 0, 1]),
         eval_func([1, 0, 1, 1])].
        
        Implementation of eval_func depends on the specific task.

        :type s: list
        :param s: A sequence of 0's and 1's (of type int)

        :rtype: int
        :return: The result of computing the target function (that the network has
            to learn) on s
        """
        raise NotImplementedError("Missing implementation for eval_func")

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

