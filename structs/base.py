from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from torch.autograd import Variable


class Struct(nn.Module):
    """
    Abstract class for implementing neural data structures, such as
    stacks, queues, and dequeues. Data structures inheriting from this
    class are intended to be used in neural networks that are trained in
    mini-batches. Thus, the actions of the structures are performed for
    each trial in a mini-batch simultaneously.
    """
    __metaclass__ = ABCMeta

    def __init__(self, batch_size, embedding_size):
        """
        Constructor for the Struct object.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch (see
            class introduction)

        :type embedding_size: int
        :param embedding_size: The size of the vectors stored in this
            Struct
        """
        super(Struct, self).__init__()
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self._zeros = Variable(torch.zeros(batch_size))

        self.contents = Variable(torch.FloatTensor(0))
        self.strengths = Variable(torch.FloatTensor(0))

    def forward(self, v, u, d):
        self.pop(u)
        self.push(v, d)
        return self.read(1)

    @abstractmethod
    def pop(self, strength):
        raise NotImplementedError("Missing implementation for pop")

    @abstractmethod
    def push(self, value, strength):
        raise NotImplementedError("Missing implementation for push")

    @abstractmethod
    def read(self, strength):
        raise NotImplementedError("Missing implementation for read")
