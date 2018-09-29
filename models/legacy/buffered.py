from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model import Model as Model
from structs.legacy.queue import Queue


class Model(Model):
    N_ARGS = 4

    def __init__(self, input_size, read_size, output_size, **args):
        super(Model, self).__init__(read_size, **args)
        self.input_size = input_size

        # initialize the model parameters
        self.linear = nn.Linear(input_size + self.get_read_size(),
                                Model.N_ARGS + self.get_read_size() +
                                output_size)

        # Careful! The way we initialize weights seems to really matter
        # self.linear.weight.data.uniform_(-.1, .1) # THIS ONE WORKS
        Model.init_normal(self.linear.weight)
        self.linear.bias.data.fill_(0)
        self.linear.bias.data[0] = -1.  # Discourage popping
        self.linear.bias.data[2] = 1.  # Encourage reading
        self.linear.bias.data[3] = 1.  # Encourage writing

    def forward(self):
        x = self.buffer_in.forward(Variable(self.zero), self.e_in, 0.)

        output = self.linear(torch.cat([x, self.read], 1))
        read_params = F.sigmoid(
            output[:, :Model.N_ARGS + self.get_read_size()])

        self.u = read_params[:, 0].contiguous()
        self.d = read_params[:, 1].contiguous()
        self.e_in = read_params[:, 2].contiguous()
        self.e_out = read_params[:, 3].contiguous()
        self.v = read_params[:, Model.N_ARGS:].contiguous()

        self.read_stack(self.v, self.u, self.d)
        out = output[:, Model.N_ARGS + self.get_read_size():].contiguous()
        self.buffer_out.forward(out, 0., self.e_out)

    def init_stack_and_buffer(self, batch_size, X, pad):
        super(Model, self).init_stack(batch_size)

        # Always push zeros onto queue
        self.zero = torch.zeros(batch_size, self.input_size)

        self.buffer_in = Queue(batch_size, self.input_size)
        self.buffer_in.enqueue_all(X.permute(1, 0, 2), pad)
        self.e_in = Variable(torch.zeros(batch_size))

        self.buffer_out = Queue(batch_size, self.input_size)
        self.e_out = Variable(torch.zeros(batch_size))
