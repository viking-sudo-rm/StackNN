"""
Recurrent controllers for use in Models.
"""
from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable

from base import SimpleStructController
from stacknn_utils.errors import unused_init_param


class RNNSimpleStructController(SimpleStructController):
    """
    An RNN producing instructions compatible with SimpleStructs (see
    structs.Simple.SimpleStruct).
    """

    def __init__(self, input_size, read_size, output_size,
                 discourage_pop=True, hidden_size=10, n_args=2, **kwargs):
        """
        Constructor for the RNNSimpleStructController object.

        :type input_size: int
        :param input_size: The size of input vectors to this Controller

        :type read_size: int
        :param read_size: The size of vectors placed on the neural data
            structure

        :type output_size: int
        :param output_size: The size of vectors output from this Controller

        :type discourage_pop: bool
        :param discourage_pop: If True, then weights will be initialized
            to discourage popping

        :type hidden_size: int
        :param hidden_size: The size of the hidden state vector

        :type n_args: int
        :param n_args: The number of struct instructions, apart from the
            value to push onto the struct, that will be computed by the
            controller. By default, this value is 2: the push strength and
            the pop strength
        """
        super(RNNSimpleStructController, self).__init__(input_size,
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
        RNNSimpleStructController.init_normal(self._rnn.weight_hh)
        RNNSimpleStructController.init_normal(self._rnn.weight_ih)
        self._rnn.bias_hh.data.fill_(0)
        self._rnn.bias_ih.data.fill_(0)

        RNNSimpleStructController.init_normal(self._linear.weight)
        self._linear.bias.data.fill_(0)

        if discourage_pop:
            self._linear.bias.data[0] = -1.  # Discourage popping
            if n_args >= 4:
                self._linear.bias.data[2] = 1.  # Encourage reading
                self._linear.bias.data[3] = 1.  # Encourage writing

    def _init_hidden(self, batch_size):
        """
        Initializes the hidden state of the LSTM cell to zeros.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch where
            this Model is used

        :return: None
        """
        rnn_hidden_shape = (batch_size, self._rnn.hidden_size)
        self._hidden = Variable(torch.zeros(rnn_hidden_shape))

    def init_controller(self, batch_size):
        self._init_hidden(batch_size)

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
        self._hidden = self._rnn(torch.cat([x, r], 1), self._hidden)
        nn_output = self._linear(self._hidden)

        output = nn_output[:, self._n_args + self._read_size:].contiguous()

        read_params = torch.sigmoid(nn_output[:, :self._n_args + self._read_size])
        v = read_params[:, self._n_args:].contiguous()
        instructions = tuple(read_params[:, j].contiguous()
                             for j in xrange(self._n_args))

        self._log(x, torch.sigmoid(output), v, *instructions)

        return output, ((v,) + instructions)


class LSTMSimpleStructController(SimpleStructController):
    """
    An LSTM producing instructions compatible with SimpleStructs (see
    structs.Simple.SimpleStruct).

    https://pytorch.org/docs/stable/nn.html#lstmcell
    """

    def __init__(self, input_size, read_size, output_size,
                 discourage_pop=True, hidden_size=10, n_args=2, **kwargs):
        """
        Constructor for the LSTMSimpleStructController object.

        :type input_size: int
        :param input_size: The size of input vectors to this Controller

        :type read_size: int
        :param read_size: The size of vectors placed on the neural data
            structure

        :type output_size: int
        :param output_size: The size of vectors output from this Controller

        :type discourage_pop: bool
        :param discourage_pop: If True, then weights will be initialized
            to discourage popping

        :type hidden_size: int
        :param hidden_size: The size of state vectors

        :type n_args: int
        :param n_args: The number of struct instructions, apart from the
            value to push onto the struct, that will be computed by the
            controller. By default, this value is 2: the push strength and
            the pop strength
        """
        super(LSTMSimpleStructController, self).__init__(input_size,
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
        LSTMSimpleStructController.init_normal(self._lstm.weight_hh)
        LSTMSimpleStructController.init_normal(self._lstm.weight_ih)
        self._lstm.bias_hh.data.fill_(0)
        self._lstm.bias_ih.data.fill_(0)

        LSTMSimpleStructController.init_normal(self._linear.weight)
        self._linear.bias.data.fill_(0)

        if discourage_pop:
            self._linear.bias.data[0] = -1.  # Discourage popping
            if n_args >= 4:
                self._linear.bias.data[2] = 1.  # Encourage reading
                self._linear.bias.data[3] = 1.  # Encourage writing

    def _init_hidden(self, batch_size):
        """
        Initializes the hidden state of the LSTM cell to zeros.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch where
            this Model is used

        :return: None
        """
        lstm_hidden_shape = (batch_size, self._lstm.hidden_size)
        self._hidden = Variable(torch.zeros(lstm_hidden_shape))
        self._cell_state = Variable(torch.zeros(lstm_hidden_shape))

    def init_controller(self, batch_size):
        self._init_hidden(batch_size)

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
        self._hidden, self._cell_state = self._lstm(
            torch.cat([x, r], 1), (self._hidden, self._cell_state))
        nn_output = self._linear(self._hidden)

        output = nn_output[:, self._n_args + self._read_size:].contiguous()

        read_params = torch.sigmoid(nn_output[:, :self._n_args + self._read_size])
        v = read_params[:, self._n_args:].contiguous()
        instructions = tuple(read_params[:, j].contiguous()
                             for j in xrange(self._n_args))

        self._log(x, torch.sigmoid(output), v, *instructions)

        return output, ((v,) + instructions)


class GRUSimpleStructController(SimpleStructController):
    """
    An GRU producing instructions compatible with SimpleStructs (see
    structs.Simple.SimpleStruct).
    """

    def __init__(self, input_size, read_size, output_size,
                 discourage_pop=True, hidden_size=10, n_args=2, **kwargs):
        """
        Constructor for the GRUSimpleStructController object.

        :type input_size: int
        :param input_size: The size of input vectors to this Controller

        :type read_size: int
        :param read_size: The size of vectors placed on the neural data
            structure

        :type output_size: int
        :param output_size: The size of vectors output from this Controller

        :type discourage_pop: bool
        :param discourage_pop: If True, then weights will be initialized
            to discourage popping

        :type hidden_size: int
        :param hidden_size: The size of the hidden state vector

        :type n_args: int
        :param n_args: The number of struct instructions, apart from the
            value to push onto the struct, that will be computed by the
            controller. By default, this value is 2: the push strength and
            the pop strength
        """
        super(GRUSimpleStructController, self).__init__(input_size,
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
        GRUSimpleStructController.init_normal(self._GRU.weight_hh)
        GRUSimpleStructController.init_normal(self._GRU.weight_ih)
        self._GRU.bias_hh.data.fill_(0)
        self._GRU.bias_ih.data.fill_(0)

        GRUSimpleStructController.init_normal(self._linear.weight)
        self._linear.bias.data.fill_(0)

        if discourage_pop:
            self._linear.bias.data[0] = -1.  # Discourage popping
            if n_args >= 4:
                self._linear.bias.data[2] = 1.  # Encourage reading
                self._linear.bias.data[3] = 1.  # Encourage writing

    def _init_hidden(self, batch_size):
        """
        Initializes the hidden state of the GRU cell to zeros.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch where
            this Model is used

        :return: None
        """
        GRU_hidden_shape = (batch_size, self._GRU.hidden_size)
        self._hidden = Variable(torch.zeros(GRU_hidden_shape))

    def init_controller(self, batch_size):
        self._init_hidden(batch_size)

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
        self._hidden = self._GRU(torch.cat([x, r], 1), self._hidden)
        nn_output = self._linear(self._hidden)

        output = nn_output[:, self._n_args + self._read_size:].contiguous()

        read_params = torch.sigmoid(nn_output[:, :self._n_args + self._read_size])
        v = read_params[:, self._n_args:].contiguous()
        instructions = tuple(read_params[:, j].contiguous()
                             for j in xrange(self._n_args))

        self._log(x, torch.sigmoid(output), v, *instructions)

        return output, ((v,) + instructions)
