from __future__ import division
from __future__ import print_function

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

    def visualize_generic_example(self):
        self.visualize(self._task.generic_example)


class LSTMVisualizer(Visualizer):
    """ Visualize the activations in an LSTM over time. """

    def visualize(self, input_seq):
        num_steps = len(input_seq)
        input_var = self._task.sentences_to_one_hot(num_steps, input_seq)
        model = self._task.model
        
        model.eval()
        model.init_model(1, input_var)

        model._controller.start_log(num_steps)
        cell_states = []
        for j in xrange(num_steps):
            model.forward()
            cell_states.append(model._controller._cell_state.data)
        model._controller.stop_log()

        cell_seqs = zip(*[state.tolist() for state in cell_states])
        for cell_seq in cell_seqs:
            full_cell_seq = [[0.]]
            full_cell_seq.extend(cell_seq)
            plt.plot(full_cell_seq)
        plt.title("LSTM Cell States for " + "".join(input_seq))
        plt.ylabel("Cell State")
        plt.xlabel("Index")
        plt.show()

class StackVisualizer(Visualizer):
    """ Visualizes the values pushed and popped from the stack. """

    def visualize(self, input_seq):
        num_steps = len(input_seq)
        input_var = self._task.sentences_to_one_hot(num_steps, input_seq)
        self._task.model.trace(input_var, num_steps)
