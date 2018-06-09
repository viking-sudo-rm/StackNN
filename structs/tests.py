from __future__ import print_function
import torch
from torch.autograd import Variable

from testcase import testcase, test_module, EPSILON
from simple import Stack, Queue

"""

Unit test cases for the neural data structures.

"""


@testcase(Stack)
def test_stack():
    """ Stack example from Grefenstette paper. """
    stack = Stack(1, 1)
    out = stack.forward(
        Variable(torch.FloatTensor([[1]])),
        Variable(torch.FloatTensor([[0]])),
        Variable(torch.FloatTensor([[.8]])),
    )
    stack.log()
    assert (out == .8).data.all()
    out = stack.forward(
        Variable(torch.FloatTensor([[2]])),
        Variable(torch.FloatTensor([[.1]])),
        Variable(torch.FloatTensor([[.5]])),
    )
    stack.log()
    assert (out == 1.5).data.all()
    out = stack.forward(
        Variable(torch.FloatTensor([[3]])),
        Variable(torch.FloatTensor([[.9]])),
        Variable(torch.FloatTensor([[.9]])),
    )
    stack.log()
    assert (out == 2.8).data.all() # Some kind of weird roundoff error here.

@testcase(Queue)
def test_queue():
    """ Adapts example from Grefenstette paper for queues. """
    queue = Queue(1, 1)
    out = queue.forward(
        Variable(torch.FloatTensor([[1]])),
        Variable(torch.FloatTensor([[0]])),
        Variable(torch.FloatTensor([[.8]])),
    )
    queue.log()
    assert (out == .8).data.all()
    out = queue.forward(
        Variable(torch.FloatTensor([[2]])),
        Variable(torch.FloatTensor([[.1]])),
        Variable(torch.FloatTensor([[.5]])),
    )
    queue.log()
    assert (out == 1.3).data.all()
    out = queue.forward(
        Variable(torch.FloatTensor([[3]])),
        Variable(torch.FloatTensor([[.9]])),
        Variable(torch.FloatTensor([[.9]])),
    )
    queue.log()
    assert (out == 2.7).data.all() # Some kind of weird roundoff error here.


if __name__ == "__main__":
    test_module(globals())