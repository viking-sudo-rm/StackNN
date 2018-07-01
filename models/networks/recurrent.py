"""
Recurrent networks for use in Controllers.
"""
from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import sigmoid

from base import SimpleStructNetwork
from stacknn_utils.errors import unused_init_param


class RNNSimpleStructNetwork(SimpleStructNetwork):
    """
    An RNN producing instructions compatible with SimpleStructs (see
    structs.Simple.SimpleStruct).
    """

    def __init__(self, input_size, read_size, output_size,
                 discourage_pop=True, hidden_size=10, n_args=2, **kwargs):
        """
        Constructor for the RNNSimpleStructNetwork object.

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

        :type hidden_size: int
        :param hidden_size: The size of the hidden state vector

        :type n_args: int
        :param n_args: The number of struct instructions, apart from the
            value to push onto the struct, that will be computed by the
            network. By default, this value is 2: the push strength and
            the pop strength
        """
        super(RNNSimpleStructNetwork, self).__init__(input_size,
                                                     read_size,
                                                     output_size,
                                                     n_args=n_args)

        for param_name, arg_value in kwargs.iteritems():
            unused_init_param(param_name, arg_value, self)

        self._hidden = None

        # Create an RNN Module object
        nn_input_size = self._input_size + self._read_size
        nn_output_size = self._n_args + self._read_size + self._output_size
        self._rnn = nn.RNNCell(nn_input_size, hidden_size)
        self._linear = nn.Linear(hidden_size, nn_output_size)

        # Initialize Module weights
        RNNSimpleStructNetwork.init_normal(self._rnn.weight_hh)
        RNNSimpleStructNetwork.init_normal(self._rnn.weight_ih)
        self._rnn.bias_hh.data.fill_(0)
        self._rnn.bias_ih.data.fill_(0)

        RNNSimpleStructNetwork.init_normal(self._linear.weight)
        self._linear.bias.data.fill_(0)

        if discourage_pop:
            pass

    def _init_hidden(self, batch_size):
        """
        Initializes the hidden state of the LSTM cell to zeros.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch where
            this Controller is used

        :return: None
        """
        rnn_hidden_shape = (batch_size, self._rnn.hidden_size)
        self._hidden = Variable(torch.zeros(rnn_hidden_shape))

    def init_network(self, batch_size):
        self._init_hidden(batch_size)

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
        self._hidden = self._rnn(torch.cat([x, r], 1), self._hidden)
        nn_output = self._linear(self._hidden)

        output = nn_output[:, self._n_args + self._read_size:].contiguous()

        read_params = sigmoid(nn_output[:, :self._n_args + self._read_size])
        v = read_params[:, self._n_args:].contiguous()
        instructions = tuple(read_params[:, j].contiguous()
                             for j in xrange(self._n_args))

        self._log(x, sigmoid(output), v, *instructions)

        return output, ((v,) + instructions)


class LSTMSimpleStructNetwork(SimpleStructNetwork):
    """
    An LSTM producing instructions compatible with SimpleStructs (see
    structs.Simple.SimpleStruct).

    https://pytorch.org/docs/stable/nn.html#lstmcell
    """

    def __init__(self, input_size, read_size, output_size,
                 discourage_pop=True, hidden_size=10, n_args=2, **kwargs):
        """
        Constructor for the LSTMSimpleStructNetwork object.

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

        :type hidden_size: int
        :param hidden_size: The size of state vectors

        :type n_args: int
        :param n_args: The number of struct instructions, apart from the
            value to push onto the struct, that will be computed by the
            network. By default, this value is 2: the push strength and
            the pop strength
        """
        super(LSTMSimpleStructNetwork, self).__init__(input_size,
                                                      read_size,
                                                      output_size,
                                                      n_args=n_args)

        for param_name, arg_value in kwargs.iteritems():
            unused_init_param(param_name, arg_value, self)

        self._hidden = None
        self._cell_state = None

        # Create an LSTM Module object
        nn_input_size = self._input_size + self._read_size
        nn_output_size = self._n_args + self._read_size + self._output_size
        self._lstm = nn.LSTMCell(nn_input_size, hidden_size)
        self._linear = nn.Linear(hidden_size, nn_output_size)

        # Initialize Module weights
        LSTMSimpleStructNetwork.init_normal(self._lstm.weight_hh)
        LSTMSimpleStructNetwork.init_normal(self._lstm.weight_ih)
        self._lstm.bias_hh.data.fill_(0)
        self._lstm.bias_ih.data.fill_(0)

        LSTMSimpleStructNetwork.init_normal(self._linear.weight)
        self._linear.bias.data.fill_(0)

        if discourage_pop:
            pass

    def _init_hidden(self, batch_size):
        """
        Initializes the hidden state of the LSTM cell to zeros.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch where
            this Controller is used

        :return: None
        """
        lstm_hidden_shape = (batch_size, self._lstm.hidden_size)
        self._hidden = Variable(torch.zeros(lstm_hidden_shape))
        self._cell_state = Variable(torch.zeros(lstm_hidden_shape))

    def init_network(self, batch_size):
        self._init_hidden(batch_size)

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
        nn_output = self._linear(self._hidden)

        output = nn_output[:, self._n_args + self._read_size:].contiguous()

        read_params = sigmoid(nn_output[:, :self._n_args + self._read_size])
        v = read_params[:, self._n_args:].contiguous()
        instructions = tuple(read_params[:, j].contiguous()
                             for j in xrange(self._n_args))

        self._log(x, sigmoid(output), v, *instructions)

        return output, ((v,) + instructions)


class GRUSimpleStructNetwork(SimpleStructNetwork):
    """
    An GRU producing instructions compatible with SimpleStructs (see
    structs.Simple.SimpleStruct).
    """

    def __init__(self, input_size, read_size, output_size,
                 discourage_pop=True, hidden_size=10, n_args=2, **kwargs):
        """
        Constructor for the GRUSimpleStructNetwork object.

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

        :type hidden_size: int
        :param hidden_size: The size of the hidden state vector

        :type n_args: int
        :param n_args: The number of struct instructions, apart from the
            value to push onto the struct, that will be computed by the
            network. By default, this value is 2: the push strength and
            the pop strength
        """
        super(GRUSimpleStructNetwork, self).__init__(input_size,
                                                     read_size,
                                                     output_size,
                                                     n_args=n_args)

        for param_name, arg_value in kwargs.iteritems():
            unused_init_param(param_name, arg_value, self)

        self._hidden = None

        # Create an GRU Module object
        nn_input_size = self._input_size + self._read_size
        nn_output_size = self._n_args + self._read_size + self._output_size
        self._GRU = nn.GRUCell(nn_input_size, hidden_size)
        self._linear = nn.Linear(hidden_size, nn_output_size)

        # Initialize Module weights
        GRUSimpleStructNetwork.init_normal(self._GRU.weight_hh)
        GRUSimpleStructNetwork.init_normal(self._GRU.weight_ih)
        self._GRU.bias_hh.data.fill_(0)
        self._GRU.bias_ih.data.fill_(0)

        GRUSimpleStructNetwork.init_normal(self._linear.weight)
        self._linear.bias.data.fill_(0)

        if discourage_pop:
            pass

    def _init_hidden(self, batch_size):
        """
        Initializes the hidden state of the GRU cell to zeros.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch where
            this Controller is used

        :return: None
        """
        GRU_hidden_shape = (batch_size, self._GRU.hidden_size)
        self._hidden = Variable(torch.zeros(GRU_hidden_shape))

    def init_network(self, batch_size):
        self._init_hidden(batch_size)

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
        self._hidden = self._GRU(torch.cat([x, r], 1), self._hidden)
        nn_output = self._linear(self._hidden)

        output = nn_output[:, self._n_args + self._read_size:].contiguous()

        read_params = sigmoid(nn_output[:, :self._n_args + self._read_size])
        v = read_params[:, self._n_args:].contiguous()
        instructions = tuple(read_params[:, j].contiguous()
                             for j in xrange(self._n_args))

        self._log(x, sigmoid(output), v, *instructions)

        return output, ((v,) + instructions)
