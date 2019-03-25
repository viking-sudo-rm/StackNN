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

    To create a custom data structure, you must create a class
    inheriting from this one that implements the pop, push, and read
    operations. Please see the documentation for self.pop. self.push,
    and self.read for more details.
    """
    __metaclass__ = ABCMeta

    def __init__(self, batch_size, embedding_size):
        """
        Constructor for the Struct object. The data of the Struct are
        stored in two parts. self.contents is a matrix containing a list
        of vectors. Each of these vectors is assigned a number known as
        its "strength," which is stored in self.strengths. For an item
        in self.contents to be assigned a strength of 1 means that it is
        fully in the data structure, and for it to have a strength of 0
        means it is deleted (but perhaps was previously in the
        structure).

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

        return

    def forward(self, v, u, d, r=None):
        """
        Performs the following three operations:
            - Pop something from the data structure
            - Push something onto the data structure
            - Read an element of the data structure.

        :type v: torch.FloatTensor
        :param v: The value that will be pushed to the data structure

        :type u: float
        :param u: The total strength of values that will be popped from
            the data structure

        :type d: float
        :param d: The strength with which v will be pushed to the data
            structure

        :rtype: torch.FloatTensor
        :return: The value read from the data structure
        """
        self.pop(u)
        self.push(v, d)

        if r is not None:
            read_strength = r
        elif self._read_strength is not None:
            # TODO: Should deprecate this option.
            read_strength = self._read_strength
        else:
            read_strength = 1

        return self.read(read_strength)

    @abstractmethod
    def pop(self, strength):
        """
        Removes something from the data structure. This function needs
        to modify self.strengths, since deleted material remains in
        self.contents, but is assigned a strength of 0.

        :type strength: float
        :param strength: The quantity of items to pop, measured in terms
            of total strength

        :return: None
        """
        raise NotImplementedError("Missing implementation for pop")

    @abstractmethod
    def push(self, value, strength):
        """
        Adds something to the data structure. This function needs to
        modify both self.contents and self.strengths.

        :type value: torch.FloatTensor
        :param value: The value to be added to the data structure

        :type strength: float
        :param strength: The strength with which value will be added to
            the data structure

        :return: None
        """
        raise NotImplementedError("Missing implementation for push")

    @abstractmethod
    def read(self, strength):
        """
        Reads a value from the data structure. This function should not
        modify anything, but should return the value read.

        :type strength: float
        :param strength: The quantity of items to read, measured in
            terms of total strength.

        :rtype: torch.FloatTensor
        :return: The item(s) read from the data structure. These should
            be combined into a single tensor
        """
        raise NotImplementedError("Missing implementation for read")

    @property
    def read_strength(self):
        return 1.
