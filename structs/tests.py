from testcase import testcase, test_module
from simple import Stack, Queue

"""

Unit test cases for the neural data structures.

"""

@testcase(Stack)
def test_stack():
	""" Test if the stack works. """
	assert 1 == 4

@testcase(Queue)
def test_queue():
	""" Test if the queue works. """
	assert 2 == 2

if __name__ == "__main__":
	test_module(globals())