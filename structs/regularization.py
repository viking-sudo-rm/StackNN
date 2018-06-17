from __future__ import print_function, division

import torch
from torch.autograd import Variable
from testcase import testcase, test_module, is_close


def binary_reg_fn(strengths):
    """ Function that is low around 0 and 1. """
    term = 3.25 * strengths - 1.625
    return 1 / (1 + torch.pow(term, 12))


class InterfaceRegTracker(object):

    """
    Compute arbitrary regularization function on struct interface.
    """

    def __init__(self, weight_decay, reg_fn=binary_reg_fn):
        """
        Constructor for StructInterfaceLoss.

        :type weight_decay: float
        :param weight_decay: Weight for regularization.

        :type reg_fn: function
        :param reg_fn: Regularization function to apply over 1D tensor

        """
        self._weight_decay = weight_decay
        self._reg_fn = reg_fn
        self._loss = Variable(torch.zeros([1]))
        self._count = 0

    @property
    def loss(self):
        return self._weight_decay * self._loss / self._count

    def regularize(self, strengths):
        losses = self._reg_fn(strengths)
        self._loss += torch.sum(losses)
        self._count += len(losses)

    def reset(self):
        loss = self.loss
        self._loss = Variable(torch.zeros([1]))
        self._count = 0
        return loss


@testcase(InterfaceRegTracker)
def test_simple_reg_fn():
    """ Test whether regularization is correctly calculated. """
    reg_fn = lambda strengths: 2 * strengths
    reg_tracker= InterfaceRegTracker(1., reg_fn=reg_fn)
    strengths = Variable(torch.ones([10]))
    reg_tracker.regularize(strengths)
    result = sum(reg_tracker.loss.data)
    assert result == 2.,  \
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
