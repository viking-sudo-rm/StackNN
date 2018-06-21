from __future__ import division

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

from base import AbstractController
from networks.feedforward import LinearSimpleStructNetwork
from structs import Stack, Operation
from structs.buffers import InputBuffer, OutputBuffer
from structs.regularization import InterfaceRegTracker


class BufferedController(AbstractController):
    """
    A Controller that reads inputs from a differentiable input buffer
    and writes outputs to a differentiable output buffer. At each step
    of computation, the controller must read something from the input
    buffer, interact with the neural data structure, and write something
    to the output buffer.
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
        super(BufferedController, self).__init__(read_size, struct_type)
        self._input_size = input_size
        self._read = None
        self._e_in = None

        self._network = network_type(input_size, read_size, output_size,
                                     n_args=4, discourage_pop=True)
        self._buffer_in = None
        self._buffer_out = None

        # TODO:
        #   * To disable regularization, can set this to None.
        #   * The weight of the regularization should a Model and Task
        # parameter.
        self._reg_tracker = InterfaceRegTracker(1.)

        return

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
        self._e_in = Variable(torch.zeros(batch_size))

        self._buffer_in = InputBuffer(batch_size, self._input_size)
        self._buffer_out = OutputBuffer(batch_size, self._input_size)

        self._buffer_in.init_contents(xs.permute(1, 0, 2))
        self._buffer_in.set_reg_tracker(self._reg_tracker, Operation.pop)
        self._buffer_out.set_reg_tracker(self._reg_tracker, Operation.push)

    """ Neural Network Computation """

    def forward(self):
        """
        Computes the output of the neural network given an input. The
        network should push a value onto the neural data structure and
        pop one or more values from the neural data structure, and
        produce an output based on this information and recurrent state
        if available.

        :return: The output of the neural network
        """
        x = self._buffer_in(self._e_in)

        output, (v, u, d, e_in, e_out) = self._network(x, self._read)
        self._e_in = e_in
        self._read = self._struct(v, u, d)

        self._buffer_out(output, e_out)

    """ Public Accessors """

    def read_output(self):
        """
        Returns the next symbol from the output buffer.

        :rtype: Variable
        :return: The value read from the output buffer after popping
            with strength 1
        """
        self._buffer_out.pop(1.)
        return self._buffer_out.read(1.)

    """ Analytical Tools """

    def trace(self, trace_x, num_steps):
        """
        Draws a graphic representation of the neural data structure
        instructions produced by the Controller's Network at each time
        step for a single input.

        :type trace_x: Variable
        :param trace_x: An input string

        :type num_steps: int
        :param num_steps: The number of computation steps to perform on
            the input

        :return: None
        """
        self.eval()
        self.init_controller(1, trace_x)

        self._network.start_log(num_steps)
        for j in xrange(num_steps):
            self.forward()
        self._network.stop_log()

        plt.imshow(self._network.log_data, cmap="hot", interpolation="nearest")
        plt.show()

    def get_and_reset_reg_loss(self):
        if self._reg_tracker is None:
            return 0.
        return self._reg_tracker.get_and_reset()
