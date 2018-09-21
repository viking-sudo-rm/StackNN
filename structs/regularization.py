from __future__ import print_function, division

import torch
from torch.autograd import Variable

from stacknn_utils.testcase import testcase, test_module, is_close

# Useful for debugging. Make sure it is larger than test set.
_MAX_COUNT = 100000


def binary_reg_fn(strengths):
    """ Function that is low around 0 and 1. """
    term = 3.25 * strengths - 1.625
    return 1 / (1 + torch.pow(term, 12))


class InterfaceRegTracker(object):
    """
    Compute arbitrary regularization function on struct interface.
    """

    def __init__(self, reg_weight, reg_fn=binary_reg_fn):
        """
        Constructor for StructInterfaceLoss.

        :type reg_weight: float
        :param reg_weight: Linear weight for regularization loss.

        :type reg_fn: function
        :param reg_fn: Regularization function to apply over 1D tensor

        """
        self._reg_weight = reg_weight
        self._reg_fn = reg_fn
        self._loss = Variable(torch.zeros([1]))
        self._count = 0

    @property
    def reg_weight(self):
        return self._reg_weight
    
    @property
    def loss(self):
        return self._reg_weight * self._loss / self._count

    def regularize(self, strengths):
        assert self._count < _MAX_COUNT, \
            "Max regularization count exceeded. Are you calling reg_tracker.reset()?"
        losses = self._reg_fn(strengths)
        self._loss += torch.sum(losses)
        self._count += len(losses)

    def reset(self):
        self._loss = Variable(torch.zeros([1]))
        self._count = 0


@testcase(InterfaceRegTracker)
def test_simple_reg_fn():
    """ Test whether regularization is correctly calculated. """
    reg_fn = lambda strengths: 2 * strengths
    reg_tracker = InterfaceRegTracker(1., reg_fn=reg_fn)
    strengths = Variable(torch.ones([10]))
    reg_tracker.regularize(strengths)
    result = sum(reg_tracker.loss.data)
    assert result == 2., \
        "{} != {}".format(result, 2.)


@testcase(binary_reg_fn)
def test_binary_reg_fn():
    """ Tests whether some values of the function are correct. """
    inputs = Variable(torch.Tensor([0, .5, 1]))
    outputs = binary_reg_fn(inputs).data
    expected = torch.Tensor([0.0029409, 1, 0.0029409])
    assert is_close(outputs, expected).all(), \
        "{} != {}".format(outputs.tolist(), expected.tolist())


if __name__ == "__main__":
    test_module(globals())
