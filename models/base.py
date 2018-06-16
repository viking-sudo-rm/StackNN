from __future__ import division

from abc import ABCMeta, abstractmethod

import torch.nn as nn


class Controller(nn.Module):
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
        super(Controller, self).__init__()
        self._struct_type = struct_type
        self._struct = None

        self._network = None

        self._read_size = read_size
        self._read = None

    @abstractmethod
    def init_struct(self, batch_size):
        """
        Initializes the neural data structure to an empty state.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch where
            this Controller is used

        :return: None
        """
        raise NotImplementedError("Missing implementation for init_struct")

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
        return

    """ Compatibility """

    def init_stack(self, batch_size, **kwargs):
        self.init_struct(batch_size, **kwargs)
        return
