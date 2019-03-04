from __future__ import absolute_import
from __future__ import print_function

import torch
from torch.autograd import Variable

from .simple import Stack, Queue
# from ..stacknn_utils.testcase import testcase, test_module, is_close


def is_close(x, y):
    return x - y < 0.001


"""Unit test cases for the neural data structures."""

# @testcase(Stack)
def test_stack():
    """Stack example from Grefenstette paper."""
    stack = Stack(1, 1)
    out = stack(
        Variable(torch.FloatTensor([[1]])),
        Variable(torch.FloatTensor([[0]])),
        Variable(torch.FloatTensor([[.8]])),
    )
    stack.log()
    assert is_close(out.data[0,0], .8)
    out = stack(
        Variable(torch.FloatTensor([[2]])),
        Variable(torch.FloatTensor([[.1]])),
        Variable(torch.FloatTensor([[.5]])),
    )
    stack.log()
    assert is_close(out.data[0,0], 1.5)
    out = stack(
        Variable(torch.FloatTensor([[3]])),
        Variable(torch.FloatTensor([[.9]])),
        Variable(torch.FloatTensor([[.9]])),
    )
    stack.log()
    assert is_close(out.data[0,0], 2.8)

# @testcase(Queue)
def test_queue():
    """Adapts example from Grefenstette paper for queues."""
    queue = Queue(1, 1)
    out = queue(
        Variable(torch.FloatTensor([[1]])),
        Variable(torch.FloatTensor([[0]])),
        Variable(torch.FloatTensor([[.8]])),
    )
    queue.log()
    assert is_close(out.data[0,0], .8)
    out = queue(
        Variable(torch.FloatTensor([[2]])),
        Variable(torch.FloatTensor([[.1]])),
        Variable(torch.FloatTensor([[.5]])),
    )
    queue.log()
    assert is_close(out.data[0,0], 1.3)
    out = queue(
        Variable(torch.FloatTensor([[3]])),
        Variable(torch.FloatTensor([[.9]])),
        Variable(torch.FloatTensor([[.9]])),
    )
    queue.log()
    assert is_close(out.data[0,0], 2.7)


# XXX: Note: stacknn_tuils package is not compatible with Python 3.
# Therefore, I made these into dirty functions instead of labelled test cases.
# The custom stuff in stacknn_utils should probably be mostly deprecated anyway.

def run_tests():
    test_stack()
    print("Stack tests passed!")
    test_queue()
    print("Queue tests passed!")
