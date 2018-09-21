"""
Feedforward networks for use in Controllers.
"""

from __future__ import division

import torch
import torch.nn as nn

from base import SimpleStructNetwork
from stacknn_utils.errors import unused_init_param


class LinearSimpleStructNetwork(SimpleStructNetwork):
    """
    A single linear layer producing instructions compatible with
    SimpleStructs (see structs.simple.SimpleStruct).
    """

    def __init__(self, input_size, read_size, output_size,
                 n_args=2, discourage_pop=True, **kwargs):
        """
        Constructor for the LinearSimpleStruct object.

        :type input_size: int
        :param input_size: The size of input vectors to this Network

        :type read_size: int
        :param read_size: The size of vectors placed on the neural data
            structure

        :type output_size: int
        :param output_size: The size of vectors output from this Network

        :type n_args: int
        :param n_args: The number of struct instructions, apart from the
            value to push onto the struct, that will be computed by the
            network. By default, this value is 2: the push strength and
            the pop strength

        :type discourage_pop: bool
        :param discourage_pop: If True, then weights will be initialized
            to discourage popping
        """
        super(LinearSimpleStructNetwork, self).__init__(input_size,
                                                        read_size,
                                                        output_size,
                                                        n_args=n_args)

        for param_name, arg_value in kwargs.iteritems():
            unused_init_param(param_name, arg_value, self)

        # Create a Linear Module object
        nn_input_size = self._input_size + self._read_size
        nn_output_size = self._n_args + self._read_size + self._output_size
        self._linear = nn.Linear(nn_input_size, nn_output_size)

        # Initialize Module weights
        LinearSimpleStructNetwork.init_normal(self._linear.weight)
        self._linear.bias.data.fill_(0)
        if discourage_pop:
            self._linear.bias.data[0] = -1.  # Discourage popping
            if n_args >= 4:
                self._linear.bias.data[2] = 1.  # Encourage reading
                self._linear.bias.data[3] = 1.  # Encourage writing

    def forward(self, x, r):
        """
        Computes an output and data structure instructions using a
        single linear layer.

        :type x: Variable
        :param x: The input to this Network

        :type r: Variable
        :param r: The previous item read from the neural data structure

        :rtype: tuple
        :return: A tuple of the form (y, (v, u, d)), interpreted as
            follows:
                - output y
                - pop a strength u from the data structure
                - push v with strength d to the data structure
        """
        nn_output = self._linear(torch.cat([x, r], 1))

        output = nn_output[:, self._n_args + self._read_size:].contiguous()

        read_params = torch.sigmoid(nn_output[:, :self._n_args + self._read_size])
        v = read_params[:, self._n_args:].contiguous()
        instructions = tuple(read_params[:, j].contiguous()
                             for j in xrange(self._n_args))

        self._log(x, torch.sigmoid(output), v, *instructions)

        return output, ((v,) + instructions)
