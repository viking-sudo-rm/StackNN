from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod, abstractproperty
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

    @abstractproperty
    def generic_example(self):
        raise NotImplementedError("Abstract property generic_example not implemented.")

    def visualize_generic_example(self):
        self.visualize(self.generic_example)

class LSTMVisualizer(Visualizer):
    """ Visualize the activations in an LSTM over time. """

    def visualize(self, input_seq):
        num_steps = len(input_seq)
        input_var = self._task.sentences_to_one_hot(num_steps, input_seq)
        model = self._task.model
        
        model.eval()
        model.init_controller(1, input_var)

        model._network.start_log(num_steps)
        cell_states = []
        for j in xrange(num_steps):
            model.forward()
            cell_states.append(model._network._cell_state.data)
        model._network.stop_log()

        cell_seqs = zip(*[state.tolist() for state in cell_states])
        for cell_seq in cell_seqs:
            plt.plot(cell_seq)
        plt.title("LSTM Cell States for " + "".join(input_seq))
        plt.ylabel("Cell State")
        plt.xlabel("Index")
        plt.show()

    @property
    def generic_example(self):
        return [u'1', u'1', u'1', u'2', u'1', u'1', u'2', u'1', u'1', u'2', u'1', u'2', u'2', u'1', u'2', u'2', u'2', u'2', u'2', u'1', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'0']
    
