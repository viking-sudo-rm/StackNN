from __future__ import division

from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
from torch.autograd import Variable


class Visualizer(object):
    """Abstract class for a visualizer."""

    __metaclass__ = ABCMeta

    def __init__(self, task):
        self._task = task

    @abstractmethod
    def visualize(self, input_seq):
        """This method should visualize the LSTM activations over time."""
        raise NotImplementedError("Abstract method visualize not implemented.")

class LSTMVisualizer(Visualizer):
    """ Visualize the activations in an LSTM over time. """

    def visualize(self, input_seq, max_length):
        num_steps = len(input_seq)
        input_var = self._task.sentences_to_one_hot(num_steps, input_seq)
        model = self._task.model
        
        model.eval()
        model.init_controller(1, input_var)

        model._network.start_log(num_steps)
        hidden_states = []
        # cell_states = []
        for j in xrange(num_steps):
            model.forward()
            hidden_states.append(model._network._hidden)
            # cell_states.append(model._network._cell_state)
        model._network.stop_log()

        hidden_seqs = zip(*hidden_states.data)
        for hidden_seq in hidden_seqs:
            plt.plot(hidden_seq)
        plot.show()

if __name__ == "__main__":
    pass