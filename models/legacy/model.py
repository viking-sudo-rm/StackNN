from __future__ import division

from abc import ABCMeta, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from structs import Stack


class Model(nn.Module):
    """
    Abstract class for creating policy controllers (models) that
    operate a neural data structure, such as a neural stack or a neural
    queue. To create a custom model, create a class inhereting from
    this one that overrides self.__init__ and self.forward.
    """
    __metaclass__ = ABCMeta

    def __init__(self, read_size, struct_type=Stack, k=None):
        """
        Constructor for the Model object.

        :type read_size: int
        :param read_size: The size of the vectors that will be placed on
            the neural data structure

        :type struct_type: type
        :param struct_type: The type of neural data structure that this
            Model will operate. Please pass the *class* for the
            data structure type to this parameter, not a specific
            instance of that class
        """
        super(Model, self).__init__()
        self.read_size = read_size
        self.struct_type = struct_type
        self.k = k

        self.read = None
        self.stack = None

    @abstractmethod
    def forward(self, x):
        """
        Computes the output of the neural network given an input. The
        controller should push a value onto the neural data structure and
        pop one or more values from the neural data structure, and
        produce an output based on this information and recurrent state
        if available.

        :param x: The input to the neural network

        :return: The output of the neural network
        """
        raise NotImplementedError("Missing implementation for forward")

    def init_stack(self, batch_size):
        """
        Initializes the neural data structure to contain a given number
        of zero vectors.

        :type batch_size: int
        :param batch_size: The total number of vectors that may be
            placed onto the neural data structure

        :return: None
        """
        self.read = Variable(torch.zeros([batch_size, self.read_size]))
        self.stack = Stack(batch_size, self.read_size, k=self.k)

    def read_stack(self, v, u, d):

        self.read = self.stack.forward(self.v, self.u, self.d)

        # If we want multiple read vectors
        if self.k is not None:
            self.read = tf.cat(tf.unbind(self.read, dim=1), dim=1)
            # TODO verify this 

    def get_read_size(self):
        """
        Return the effective read size, taking k into account.
        """
        k = 1 if self.k is None else self.k
        return self.read_size * k

    @staticmethod
    def init_normal(tensor):
        n = tensor.data.shape[0]
        tensor.data.normal_(0, 1. / np.sqrt(n))

    def trace(self, trace_X):
        """
        Visualize stack activations for a single training sample.
        Draws a graphic representation of these stack activations.
        @param trace_X [1, max_length, input_size] tensor
        """
        self.eval()
        self.init_stack(1)
        max_length = trace_X.data.shape[1]
        data = np.zeros([2 + self.read_size, max_length])  # 2 + len(v)
        for j in xrange(1, max_length):
            self.forward(trace_X[:, j - 1, :])
            data[0, j] = self.u.data.numpy()
            data[1, j] = self.d.data.numpy()
            data[2:, j] = self.v.data.numpy()
        plt.imshow(data, cmap="hot", interpolation="nearest")
        plt.show()

    """ Compatibility """

    def init_struct(self, *args, **kwargs):
        self.init_stack(*args, **kwargs)
        return
