"""
Recurrent networks for use in Controllers.
"""

from __future__ import division

import torch
import torch.nn as nn
from torch.nn.functional import sigmoid
from torch.autograd import Variable

from base import SimpleStructNetwork

# https://pytorch.org/docs/stable/nn.html#lstmcell

class LSTMSimpleStructNetwork(SimpleStructNetwork):
    """
    An LSTM producing instructions compatible with SimpleStructs (see
    structs.Simple.SimpleStruct).
    """

    def __init__(self, input_size, read_size, output_size,
                 n_args=2, discourage_pop=True):
        """
        Constructor for the LSTMSimpleStructNetwork object.

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
        super(LSTMSimpleStructNetwork, self).__init__(input_size,
                                                      read_size,
                                                      output_size,
                                                      n_args=n_args)

        # Create an LSTM Module object
        nn_input_size = self._input_size + self._read_size
        nn_output_size = self._n_args + self._read_size + self._output_size
        self._lstm = nn.LSTMCell(nn_input_size, nn_output_size)

        # Initialize Module weights
        LSTMSimpleStructNetwork.init_normal(self._lstm.weight_hh)
        LSTMSimpleStructNetwork.init_normal(self._lstm.weight_ih)
        self._lstm.bias_hh.data.fill_(0)
        self._lstm.bias_ih.data.fill_(0)

        if discourage_pop:
            pass

    def init_hidden(self, batch_size):
        # Initialize hidden state
        lstm_hidden_shape = (batch_size, self._lstm.hidden_size)
        self._hidden = Variable(torch.zeros(lstm_hidden_shape))
        self._cell_state = Variable(torch.zeros(lstm_hidden_shape))

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
        self._hidden, self._cell_state = self._lstm(
            torch.cat([x, r], 1), (self._hidden, self._cell_state))

        output = self._hidden[:, self._n_args + self._read_size:].squeeze()

        read_params = self._hidden[:, :self._n_args + self._read_size]
        read_params = sigmoid(read_params.squeeze())
        v = read_params[:, self._n_args:].contiguous()
        instructions = tuple(read_params[:, j].contiguous()
                             for j in xrange(self._n_args))

        self._log(v, *instructions)

        return output, ((v,) + instructions)
