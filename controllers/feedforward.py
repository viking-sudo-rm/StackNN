"""
Feedforward controllers for use in Models.
"""

from __future__ import division
from math import ceil

import torch
import torch.nn as nn

from base import SimpleStructController
from stacknn_utils.errors import unused_init_param

class DeepSimpleStructController(SimpleStructController):
    """
    A fully connected multilayer network producing instructions compatible
    with SimpleStructs (see structs.simple.SimpleStruct).
    """
    def __init__(self, input_size, read_size, output_size,
                 n_args=2, discourage_pop=True, n_hidden_layers=2,
                 non_linearity=nn.ReLU, **kwargs):
        """
        Constructor for the DeepSimpleStructController object.

        :type input_size: int
        :param input_size: The size of input vectors to this Controller

        :type read_size: int
        :param read_size: The size of vectors placed on the neural data
            structure

        :type output_size: int
        :param output_size: The size of vectors output from this Controller

        :type n_args: int
        :param n_args: The number of struct instructions, apart from the
            value to push onto the struct, that will be computed by the
            controller. By default, this value is 2: the push strength and
            the pop strength

        :type discourage_pop: bool
        :param discourage_pop: If True, then weights will be initialized
            to discourage popping

        :type n_hidden_layers: int
        :param n_hidden_layers: How many feedforward layers

        :type non_linearity: Module
        :param non_linearity: Non-linearity to apply to hidden layers
        """
        super(DeepSimpleStructController, self).__init__(input_size,
                                                        read_size,
                                                        output_size,
                                                        n_args=n_args)

        for param_name, arg_value in kwargs.iteritems():
            unused_init_param(param_name, arg_value, self)

        # Create a Multilayer NN
        nn_input_size = self._input_size + self._read_size
        nn_output_size = self._n_args + self._read_size + self._output_size
        nn_hidden_size = int(ceil((nn_input_size+nn_output_size)/2.0))
        nn_sizes_list = [nn_input_size] + [nn_hidden_size]*n_hidden_layers

        self._network = nn.Sequential()
        for i in range(n_hidden_layers):
            self._network.add_module('lin'+str(i), nn.Linear(nn_sizes_list[i], nn_sizes_list[i+1]))
            self._network.add_module('relu'+str(i), non_linearity())
        self._network.add_module('out', nn.Linear(nn_sizes_list[-1], nn_output_size))

        # Initialize Module weights
        self.discourage_pop = discourage_pop
        self._network.apply(self.init_weights)

    def init_weights(self, module):
        """
        Initializes a linear layer with values drawn from a normal
        distribution

        :type module: Module
        :param module: The module (layer) to initialize

        :return: None
        """
        if type(module) == nn.Linear:
            DeepSimpleStructController.init_normal(module.weight)
            module.bias.data.fill_(0)
            if self.discourage_pop:
                module.bias.data[0] = -1. # Discourage popping
                if self._n_args >= 4:
                    module.bias.data[2] = 1.  # Encourage reading
                    module.bias.data[3] = 1.  # Encourage writing

    def forward(self, x, r):
        """
        Computes an output and data structure instructions using a
        multi-layer nn.

        :type x: Variable
        :param x: The input to this Controller

        :type r: Variable
        :param r: The previous item read from the neural data structure

        :rtype: tuple
        :return: A tuple of the form (y, (v, u, d)), interpreted as
            follows:
                - output y
                - pop a strength u from the data structure
                - push v with strength d to the data structure
        """
        nn_output = self._network(torch.cat([x, r], 1))

        output = nn_output[:, self._n_args + self._read_size:].contiguous()

        read_params = torch.sigmoid(nn_output[:, :self._n_args + self._read_size])
        v = read_params[:, self._n_args:].contiguous()
        instructions = tuple(read_params[:, j].contiguous()
                             for j in xrange(self._n_args))

        self._log(x, torch.sigmoid(output), v, *instructions)

        return output, ((v,) + instructions)


class LinearSimpleStructController(SimpleStructController):
    """
    A single linear layer producing instructions compatible with
    SimpleStructs (see structs.simple.SimpleStruct).
    """

    def __init__(self, input_size, read_size, output_size,
                 n_args=2, discourage_pop=True, **kwargs):
        """
        Constructor for the LinearSimpleStruct object.

        :type input_size: int
        :param input_size: The size of input vectors to this Controller

        :type read_size: int
        :param read_size: The size of vectors placed on the neural data
            structure

        :type output_size: int
        :param output_size: The size of vectors output from this Controller

        :type n_args: int
        :param n_args: The number of struct instructions, apart from the
            value to push onto the struct, that will be computed by the
            controller. By default, this value is 2: the push strength and
            the pop strength

        :type discourage_pop: bool
        :param discourage_pop: If True, then weights will be initialized
            to discourage popping
        """
        super(LinearSimpleStructController, self).__init__(input_size,
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
        LinearSimpleStructController.init_normal(self._linear.weight)
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
        :param x: The input to this Controller

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
