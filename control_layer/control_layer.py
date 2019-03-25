from __future__ import print_function

from numpy.testing import assert_approx_equal
import torch


class ControlLayer(torch.nn.Module):

    """Layer to convert a vector to stack instructions."""

    def __init__(self, input_size, stack_size, vision):
        """Construct a ControlLayer object.

        Args:
            input_size: The length of the vectors inputted to the ControlLayer.
            stack_size: The size of the vectors on the stack.
            vision: The maximum depth with which we can read and pop from the stack.
        """
        super().__init__()
        self._vector_map = torch.nn.Linear(input_size, stack_size)
        self._push_map = torch.nn.Linear(input_size, 1)
        self._pop_map = torch.nn.Linear(input_size, vision)
        self._read_map = torch.nn.Linear(input_size, vision)

    def forward(self, input_vector):
        # First, we calculate the vector that should be pushed, and with how much weight.
        push_vector = torch.tanh(self._vector_map(input_vector))
        push_strength = torch.sigmoid(self._push_map(input_vector))

        # Next, we compute a distribution for popping and return its expectation.
        pop_distribution = torch.softmax(self._pop_map(input_vector), 1)
        pop_strength = self._get_expectation(pop_distribution)

        # Finally, we compute a separate distribution for reading.
        read_distribution = torch.softmax(self._read_map(input_vector), 1)
        read_strength = self._get_expectation(read_distribution)

        return push_vector, push_strength.squeeze(), pop_strength.squeeze(), read_strength.squeeze()

    @staticmethod
    def _get_expectation(distribution):
        """Take the expected value of a pop/read distribution."""
        values = torch.arange(distribution.size(1)).unsqueeze(1)
        return torch.mm(distribution, values.float())


def test_expectation():
    distribution = torch.Tensor([[.2, .4, .4]])
    expectation = ControlLayer._get_expectation(distribution)
    expectation = expectation.squeeze().item()
    assert_approx_equal(expectation, 1.2)
    print("Expectation test passed!")   


if __name__ == "__main__":
    test_expectation()
