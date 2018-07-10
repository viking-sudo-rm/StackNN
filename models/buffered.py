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
                 network_type=LinearSimpleStructNetwork,
                 struct_type=Stack,
                 read_reg_weight=1.,
                 write_reg_weight=1.,
                 push_reg_weight=1.,
                 pop_reg_weight=1.,
                 **kwargs):
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

        :type read_reg_weight: float
        :param read_reg_weight: Regularization weight for reading from
            the input buffer

        :type write_reg_weight: float
        :param write_reg_weight: Regularization weight for writing to the
            output buffer

        :type push_reg_weight: float
        :param push_reg_weight: Regularization weight for pushing to the
            stack

        :type read_reg_weight: float
        :param read_reg_weight: Regularization weight for popping from the
            stack

        """
        super(BufferedController, self).__init__(read_size, struct_type)
        self._input_size = input_size
        self._output_size = output_size
        self._read_size = read_size

        self._read = None
        self._e_in = None

        self._network = network_type(input_size, read_size, output_size,
                                     n_args=4, discourage_pop=True, **kwargs)
        self._buffer_in = None
        self._buffer_out = None

        self._reg_tracker = InterfaceRegTracker()
        self._read_reg_weight = read_reg_weight
        self._write_reg_weight = write_reg_weight
        self._push_reg_weight = push_reg_weight
        self._pop_reg_weight = pop_reg_weight

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

    """ Reporting """

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

        x_labels = ["x_" + str(i) for i in xrange(self._input_size)]
        y_labels = ["y_" + str(i) for i in xrange(self._output_size)]
        i_labels = ["Pop", "Push", "Input", "Output"]
        v_labels = ["v_" + str(i) for i in xrange(self._read_size)]
        labels = x_labels + y_labels + i_labels + v_labels

        plt.imshow(self._network.log_data, cmap="hot", interpolation="nearest")
        plt.title("Trace")
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.show()

    def trace_step(self, trace_x, num_steps, step=True):
        """
        Steps through the neural network's computation. The network will
        read an input and produce an output. At each time step, a
        summary of the network's state and actions will be printed to
        the console.

        :type trace_x: Variable
        :param trace_x: A single input string

        :type num_steps: int
        :param num_steps: The number of computation steps to perform

        :type step: bool
        :param step: If True, the user will need to press Enter in the
            console after each computation step

        :return: None
        """
        if trace_x.data.shape[0] != 1:
            raise ValueError("You can only trace one input at a time!")

        self.eval()
        self.init_controller(1, trace_x)

        x_end = self._input_size
        y_end = x_end + self._output_size
        push = y_end + 1
        e_in = push + 1
        e_out = e_in + 1
        v_start = e_out + 1

        self._network.start_log(num_steps)
        for j in xrange(num_steps):
            print "\n-- Step {} of {} --".format(j, num_steps)

            self.forward()

            i = self._network.log_data[:x_end, j]
            o = self._network.log_data[x_end:y_end, j].round(decimals=4)
            u = self._network.log_data[y_end, j].round(decimals=4)
            d = self._network.log_data[push, j].round(decimals=4)
            e_in = self._network.log_data[e_in, j].round(decimals=4)
            e_out = self._network.log_data[e_out, j].round(decimals=4)
            v = self._network.log_data[v_start:, j].round(decimals=4)
            r = self._struct.read(1).data.numpy()[0].round(decimals=4)

            print "\nInput: " + str(i)
            print "Input Strength: " + str(e_in)
            print "Output: " + str(o)
            print "Output Strength: " + str(e_out)

            print "\nPop Strength: " + str(u)

            print "\nPush Vector: " + str(v)
            print "Push Strength: " + str(d)

            print "\nRead Vector: " + str(r)
            print "Struct Contents: "
            self._struct.print_summary(0)

            if step:
                raw_input("\nPress Enter to continue\n")
        self._network.stop_log()

    def get_and_reset_reg_loss(self):
        """If there is a regularization tracker, return the loss term from it."""

        if self._reg_tracker is None:
            return super(BufferedController, self).get_and_reset_reg_loss()

        loss = self._reg_tracker.loss
        self._reg_tracker.reset()
        return loss

    def print_experiment_start(self):
        """Overriden to print buffered-specific params."""
        super(BufferedController, self).print_experiment_start()
        print "Reg Weight: " + str(self._reg_tracker.reg_weight)
