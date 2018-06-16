from __future__ import division

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

from models.base import Controller
from networks.feedforward import LinearSimpleStructNetwork
from structs.simple import Stack


class VanillaController(Controller):
    """
    A simple Controller that uses a SimpleStruct as its data structure.
    """

    def __init__(self, input_size, read_size, output_size,
                 network_type=LinearSimpleStructNetwork, struct_type=Stack):
        """
        Constructor for the VanillaController object.

        :type input_size: int
        :param input_size: The size of the vectors that will be input to
            this Controller

        :type read_size: int
        :param read_size: The size of the vectors that will be placed on
            the neural data structure

        :type output_size: int
        :param output_size: The size of the vectors that will be output
            from this Controller

        :type struct_type: type
        :param struct_type: The type of neural data structure that this
            Controller will operate

        :type network_type: type
        :param network_type: The type of the Network that will perform
            the neural network computations
        """
        super(VanillaController, self).__init__(read_size, struct_type)
        self._read = None
        self._network = network_type(input_size, read_size, output_size)

        return

    def init_struct(self, batch_size):
        """
        Initializes the neural data structure to an empty state.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch where
            this Controller is used

        :return: None
        """
        self._read = Variable(torch.zeros([batch_size, self._read_size]))
        self._struct = self._struct_type(batch_size, self._read_size)

        return

    """ Neural Network Computation """

    def forward(self, x):
        """
        Computes the output of the neural network given an input. The
        network should push a value onto the neural data structure and
        pop one or more values from the neural data structure, and
        produce an output based on this information and recurrent state
        if available.

        :param x: The input to the neural network

        :return: The output of the neural network
        """
        if self._read is None:
            raise RuntimeError("The data structure has not been initialized.")

        output, (v, u, d) = self._network(x, self._read)
        self._read = self._struct(v, u, d)

        return output

    """ Analytical Tools """

    def trace(self, trace_x):
        """
        Draws a graphic representation of the neural data structure
        instructions produced by the Controller's Network at each time
        step for a single input.

        :type trace_x: Variable
        :param trace_x: An input string

        :return: None
        """
        self.eval()
        self.init_struct(1)

        max_length = trace_x.data.shape[1]

        self._network.start_log(max_length)
        for j in xrange(max_length):
            self.forward(trace_x[:, j, :])
        self._network.stop_log()

        plt.imshow(self._network.log_data, cmap="hot", interpolation="nearest")
        plt.show()

        return
