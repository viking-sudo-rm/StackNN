from simple import Queue


class InputBuffer(Queue):
    """
    A read-only neural queue.
    """

    def forward(self, u):
        """
        Skip the push step.

        :type u: float
        :param u: The total strength of values that will be popped from
            the data structure

        :rtype: torch.FloatTensor
        :return: The value read from the data structure
        """
        self.pop(u)
        return self.read(1.)


class OutputBuffer(Queue):
    """
    A write-only neural queue.
    """

    def forward(self, v, d):
        """
        Only perform the push step.

        :type v: torch.FloatTensor
        :param v: The value that will be pushed to the data structure

        :type d: float
        :param d: The strength with which v will be pushed to the data
            structure

        :return: None
        """
        self.push(v, d)
