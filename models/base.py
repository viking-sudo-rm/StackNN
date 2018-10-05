from __future__ import division

from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from torch.autograd import Variable

from structs.base import Struct


class Model(nn.Module):
    """
    Abstract class for creating policy controllers (models) that
    operate a neural data structure, such as a neural stack or a neural
    queue. To create a custom model, create a class inhereting from
    this one that overrides self.__init__ and self.forward.
    """
    __metaclass__ = ABCMeta

    def __init__(self, read_size, struct_type):
        """
        Constructor for the Model object.

        :type read_size: int
        :param read_size: The size of the vectors that will be placed on
            the neural data structure

        :type struct_type: type
        :param struct_type: The type of neural data structure that this
            Model will operate
        """
        super(Model, self).__init__()
        self._struct_type = struct_type
        self._struct = None

        self._controller = None

        self._read_size = read_size
        self._read = None

    def _init_struct(self, batch_size):
        """
        Initializes the neural data structure to an empty state.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch where
            this Model is used

        :return: None
        """
        if issubclass(self._struct_type, Struct):
            self._read = Variable(torch.zeros([batch_size, self._read_size]))
            self._struct = self._struct_type(batch_size, self._read_size)
            self._reg_loss = torch.zeros([batch_size, self._read_size])

    @abstractmethod
    def _init_buffer(self, batch_size, xs):
        """
        Initializes the input and output buffers. The input buffer will
        contain a specified collection of values. The output buffer will
        be empty.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch where
            this Model is used

        :type xs: Variable
        :param xs: An array of values that will be placed on the input
            buffer. The dimensions should be [batch size, t, read size],
            where t is the maximum length of a string represented in xs

        :return: None
        """
        raise NotImplementedError("Missing implementation for _init_buffer")

    def _init_controller(self, batch_size):
        """
        Initializes the controller.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch where
            this Model is used

        :return: None
        """
        self._controller.init_controller(batch_size)

    def init_model(self, batch_size, xs):
        """
        Resets the neural data structure and other Model components
        to an initial state. This function is called at the beginning of
        each mini-batch.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch where
            this Model is used

        :return: None
        """
        self._init_struct(batch_size)
        self._init_buffer(batch_size, xs)
        self._init_controller(batch_size)

    """ Neural Network Computation """

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Computes the output of the neural network given an input. The
        controller should push a value onto the neural data structure and
        pop one or more values from the neural data structure, and
        produce an output based on this information and recurrent state
        if available.

        :return: The output of the neural network
        """
        raise NotImplementedError("Missing implementation for forward")

    """ Public Accessors and Properties """

    def get_read_size(self):
        return self._read_size

    @property
    def controller_type(self):
        return type(self._controller)

    @property
    def struct_type(self):
        return self._struct_type

    """ Analytical Tools """

    def trace(self, *args, **kwargs):
        """
        Draws a graphic representation of the neural data structure
        instructions produced by the Model's Controller at each time
        step for a single input.

        :return: None
        """
        pass

    """ Compatibility """

    def init_stack(self, batch_size, **kwargs):
        self.init_model(batch_size, **kwargs)

    def get_and_reset_reg_loss(self):
        """Method overriden for buffered regularization.

        The default method just returns a zero vector.

        """
        return self._reg_loss

    def print_experiment_start(self):
        """Print model-specific hyperparameters at the start of an experiment."""
        print "Model Type: " + str(type(self).__name__)
        print "Controller Type: " + str(self.controller_type.__name__)
        print "Struct Type: " + str(self.struct_type.__name__)
