class ControlInstructions:

    """Stores the instruction values produced by a control layer.

    TODO: Perhaps stack.forward should be able to take this object as an argument.
    """

    def __init__(self,
                 push_vectors,
                 push_strengths,
                 pop_strengths,
                 read_strengths,
                 pop_distributions=None,
                 read_distributions=None):
        self.push_vectors = push_vectors

        self.push_strengths = push_strengths
        self.pop_strengths = pop_strengths
        self.read_strengths = read_strengths

        self.pop_distributions = pop_distributions
        self.read_distributions = read_distributions

    def make_tuple(self):
        return self.push_vectors, self.pop_strengths, self.push_strengths, self.read_strengths

    def __len__(self):
        return len(self.push_vectors)
