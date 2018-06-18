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

        :param v:
        :param d:
        :return:
        """
        self.push(v, d)
        return
