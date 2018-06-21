from __future__ import division

from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from torch.autograd import Variable

from structs.simple import SimpleStruct


class AbstractController(nn.Module):
    """
    Abstract class for creating policy networks (controllers) that
    operate a neural data structure, such as a neural stack or a neural
    queue. To create a custom controller, create a class inhereting from
    this one that overrides self.__init__ and self.forward.
    """
    __metaclass__ = ABCMeta

    def __init__(self, read_size, struct_type):
        """
        Constructor for the Controller object.

        :type read_size: int
        :param read_size: The size of the vectors that will be placed on
            the neural data structure

        :type struct_type: type
        :param struct_type: The type of neural data structure that this
            Controller will operate
        """
        super(AbstractController, self).__init__()
        self._struct_type = struct_type
        self._struct = None

        self._network = None

        self._read_size = read_size
        self._read = None

    def _init_struct(self, batch_size):
        """
        Initializes the neural data structure to an empty state.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch where
            this Controller is used

        :return: None
        """
        if issubclass(self._struct_type, SimpleStruct):
            self._read = Variable(torch.zeros([batch_size, self._read_size]))
            self._struct = self._struct_type(batch_size, self._read_size)

    @abstractmethod
    def _init_buffer(self, batch_size, xs):
        """
        Initializes the input and output buffers. The input buffer will
        contain a specified collection of values. The output buffer will
        be empty.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch where
            this Controller is used

        :type xs: Variable
        :param xs: An array of values that will be placed on the input
            buffer. The dimensions should be [batch size, t, read size],
            where t is the maximum length of a string represented in xs

        :return: None
        """
        raise NotImplementedError("Missing implementation for _init_buffer")

    def _init_network(self, batch_size):
        """
        Initializes the network.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch where
            this Controller is used

        :return: None
        """
        self._network.init_network(batch_size)

    def init_controller(self, batch_size, xs):
        """
        Resets the neural data structure and other Controller components
        to an initial state. This function is called at the beginning of
        each mini-batch.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch where
            this Controller is used

        :return: None
        """
        self._init_struct(batch_size)
        self._init_buffer(batch_size, xs)
        self._init_network(batch_size)

    """ Neural Network Computation """

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Computes the output of the neural network given an input. The
        network should push a value onto the neural data structure and
        pop one or more values from the neural data structure, and
        produce an output based on this information and recurrent state
        if available.

        :return: The output of the neural network
        """
        raise NotImplementedError("Missing implementation for forward")

    """ Public Accessors and Properties """

    def get_read_size(self):
        """
        Public accessor for self._read_size.
        """
        return self._read_size

    """ Analytical Tools """

    def trace(self, *args, **kwargs):
        """
        Draws a graphic representation of the neural data structure
        instructions produced by the Controller's Network at each time
        step for a single input.

        :return: None
        """
        pass

    """ Compatibility """

    def init_stack(self, batch_size, **kwargs):
        self.init_controller(batch_size, **kwargs)
