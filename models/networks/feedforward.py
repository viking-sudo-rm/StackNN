"""
Feedforward networks for use in Controllers.
"""

from __future__ import division

import torch
import torch.nn as nn
from torch.nn.functional import sigmoid

from base import SimpleStructNetwork


class LinearSimpleStructNetwork(SimpleStructNetwork):
    """
    A single linear layer producing instructions compatible with
    SimpleStructs (see structs.simple.SimpleStruct).
    """

    def __init__(self, input_size, read_size, output_size,
                 discourage_pop=True):
        """
        Constructor for the LinearSimpleStruct object.

        :type input_size: int
        :param input_size: The size of input vectors to this Network

        :type read_size: int
        :param read_size: The size of vectors placed on the neural data
            structure

        :type output_size: int
        :param output_size: The size of vectors output from this Network

        :type discourage_pop: bool
        :param discourage_pop: If True, then weights will be initialized
            to discourage popping
        """
        super(LinearSimpleStructNetwork, self).__init__(input_size, read_size,
                                                        output_size)

        # Create a Linear Module object
        nn_input_size = self._input_size + self._read_size
        nn_output_size = 2 + self._read_size + self._output_size
        self._linear = nn.Linear(nn_input_size, nn_output_size)

        # Initialize Module weights
        LinearSimpleStructNetwork.init_normal(self._linear.weight)
        self._linear.bias.data.fill_(0)
        if discourage_pop:
            self._linear.bias.data[0] = -1.  # Discourage popping
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

        output = nn_output[:, 2 + self._read_size:]

        read_params = sigmoid(nn_output[:, :2 + self._read_size])
        v = read_params[:, 2:].contiguous()
        u = read_params[:, 0].contiguous()
        d = read_params[:, 1].contiguous()

        self._log(v, u, d)

        return output, (v, u, d)
