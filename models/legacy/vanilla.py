from __future__ import division

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from model import Model as Model


class Model(Model):
    def __init__(self, input_size, read_size, output_size, **args):
        super(Model, self).__init__(read_size, **args)

        # initialize the model parameters
        self.linear = nn.Linear(input_size + self.get_read_size(),
                                2 + self.get_read_size() + output_size)

        # Careful! The way we initialize weights seems to really matter
        # self.linear.weight.data.uniform_(-.1, .1) # THIS ONE WORKS
        Model.init_normal(self.linear.weight)
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        output = self.linear(torch.cat([x, self.read], 1))
        read_params = F.sigmoid(output[:, :2 + self.get_read_size()])
        self.u, self.d, self.v = read_params[:, 0].contiguous(), read_params[:,
                                                                 1].contiguous(), read_params[
                                                                                  :,
                                                                                  2:].contiguous()
        self.read_stack(self.v, self.u, self.d)
        return output[:, 2 + self.get_read_size():]
