from __future__ import division

import random

import torch
import torch.nn as nn
from torch.autograd import Variable

from base import Task
from evaluation import EvaluationTask
from models import BufferedController
from models import VanillaController 
from models.networks.feedforward import LinearSimpleStructNetwork
from models.networks.recurrent import RNNSimpleStructNetwork
from structs import Stack

import operator

class XORTask(EvaluationTask):
    """
    XOR evaluation task: network is fed strings of length n and is given
    2n time steps to output a string whose ith bit is the xor of the first
    i bits in the input string.  (We define the xor of single bit inputs to
    be equal to the input bit.)  e.g., if the input is

        1 0 1 0 1 1 0

    the network will have to output

        xor(1)
        xor(1, 0)
        xor(1, 0, 1)
        xor(1, 0, 1, 0)
        xor(1, 0, 1, 0, 1)
        xor(1, 0, 1, 0, 1, 1)
        xor(1, 0, 1, 0, 1, 1, 0)

    which is equivalent to

        1 1 0 0 1 0 0.
    """

    def __init__(self,
                 batch_size=10,
                 clipping_norm=None,
                 criterion=nn.CrossEntropyLoss(),
                 cuda=False,
                 epochs=30,
                 hidden_size=10,
                 learning_rate=0.01,
                 load_path=None,
                 l2_weight=0.01,
                 model=None,
                 model_type=BufferedController,
                 network_type=RNNSimpleStructNetwork,
                 read_size=2,
                 save_path=None,
                 str_length=12,
                 struct_type=Stack,
                 verbose=True):
        """
        Constructor for the XORTask object. The only information
        that needs to be specified by the user is how long the input
        strings are.

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

        :type str_length: int
        :param str_length: The number of bits in each input string

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
        super(XORTask, self).__init__(batch_size=batch_size,
                                      clipping_norm=clipping_norm,
                                      criterion=criterion,
                                      cuda=cuda,
                                      epochs=epochs,
                                      hidden_size=hidden_size,
                                      learning_rate=learning_rate,
                                      load_path=load_path,
                                      l2_weight=l2_weight,
                                      max_length=str_length,
                                      model=model,
                                      model_type=model_type,
                                      network_type=network_type,
                                      read_size=read_size,
                                      save_path=save_path,
                                      struct_type=struct_type,
                                      time_function=(lambda t: 2*t),
                                      verbose=verbose)

        self.str_length = str_length

    """ Model Training """

    def sample_str(self):
        """
        Generates a random string of 0s and 1s. The length of the string
        is between self.min_length and self.max_length. The average
        length of the string is self.mean_length. The standard deviation
        of the length of the string is self.std_length.

        :rtype: list
        :return: A sequence of "0"s and "1"s
        """
        return [random.randint(0, 1) for _ in xrange(self.str_length)]

    def eval_func(self, s):
        return reduce(operator.xor, s, 0)
