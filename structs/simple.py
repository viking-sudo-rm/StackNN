from abc import ABCMeta, abstractmethod

import torch
from torch.autograd import Variable
from torch.nn.functional import relu

from base import Struct


def first_to_last(num_steps):
    return xrange(num_steps)


def last_to_first(num_steps):
    return reversed(xrange(num_steps))


def top(num_steps):
    return num_steps


def bottom(num_steps):
    return 0


class SimpleStruct(Struct):
    """
    Simple structs
    """
    __metaclass__ = ABCMeta

    def __init__(self, batch_size, embedding_size):
        """
        Constructor for the SimpleStruct object.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch (see
            class introduction)

        :type embedding_size: int
        :param embedding_size: The size of the vectors stored in this
            Struct
        """
        super(Struct, self).__init__(batch_size, embedding_size)
        self._t = 0

    @abstractmethod
    def _pop_indices(self, num_steps):
        raise NotImplementedError("Missing implementation for _pop_indices")

    @abstractmethod
    def _push_index(self, num_steps):
        raise NotImplementedError("Missing implementation for _push_index")

    @abstractmethod
    def _read_indices(self, num_steps):
        raise NotImplementedError("Missing implementation for _read_indices")

    def pop(self, strength):
        u = strength
        for i in self._pop_indices(self._t):
            s = relu(self.strengths[i, :] - u)
            u = relu(u - self.strengths[i, :])
            s[i, :] = s

        return

    def push(self, value, strength):
        v = value.view(1, self.batch_size, self.embedding_size)
        if self._t == 0:
            self.contents = v
        else:
            i = self._push_index(self._t)
            if i == 0:
                self.contents = torch.cat([v, self.contents], 0)
            elif i == self._t + 1:
                self.contents = torch.cat([self.contents, v], 0)
            else:
                bottom = self.contents[0:i, :, :]
                top = self.contents[i:, :, :]
                self.contents = torch.cat([bottom, v, top], 0)

        self._t += 1
        return

    def read(self, strength):
        r = Variable(torch.zeros([self.batch_size, self.embedding_size]))
        s = torch.FloatTensor([strength]).repeat(self.batch_size)
        for i in self._read_indices(self._t):
            s_i = torch.min(self.strengths[i, :], relu(s))
            s = relu(s - s_i)
            s_i = s_i.view(self.batch_size, 1)
            r += s_i.repeat(1, self.embedding_size) * self.contents[i, :, :]

        return r


class Stack(SimpleStruct):
    def _pop_indices(self, num_steps):
        return last_to_first(num_steps)

    def _push_index(self, num_steps):
        return top(num_steps)

    def _read_indices(self, num_steps):
        return last_to_first(num_steps)


class Queue(SimpleStruct):
    def _pop_indices(self, num_steps):
        return last_to_first(num_steps)

    def _push_index(self, num_steps):
        return bottom(num_steps)

    def _read_indices(self, num_steps):
        return last_to_first(num_steps)
