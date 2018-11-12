from __future__ import division

from copy import copy
import random

import torch
import torch.nn as nn
from torch.autograd import Variable

from models import VanillaModel
from controllers.feedforward import LinearSimpleStructController
from structs import Stack
from tasks.language_modeling import LanguageModelingTask


class OrderedCountingTask(LanguageModelingTask):

    """
    An OrderedCountingTask models strings with a form like

    a^nb^nc^{2n}.

    More explicitly, each string in the language should be made up of
    blocks of each symbol, and the length of each block should be
    parameterizable in some value n. The whole language can be generated
    by building the unique string corresponding to every possible n.

    Loss from all indices in the string affects training. However, the
    evaluation accuracy only looks at the end of words.
    """


    class Params(LanguageModelingTask.Params):

        def __init__(self, **kwargs):
            self.length_fns = kwargs.get("length_fns", [lambda n: n, lambda n: n])
            self.min_n = kwargs.get("min_n", 1)
            self.max_n = kwargs.get("max_n", 100)
            self.evaluate_all = kwargs.get("evaluate_all", False)
            max_length = self._get_length(self.max_n)
            to_predict = [u"#"]
            if self.evaluate_all:
                to_predict.extend(self._get_char(i) for i in xrange(max_length))
            super(OrderedCountingTask.Params, self).__init__(to_predict,
                                                             null=u"#",
                                                             mask_null=False,
                                                             max_length=max_length,
                                                             **kwargs)

        @staticmethod
        def _get_char(i):
            """
            Get the character to represent symbol i. For example, this
            function maps 0 to u"a".
            """
            return unicode(chr(97 + i))

        def _get_length(self, n):
            """Returns the length of the string parameterized by n."""
            return sum(length_fn(n) for length_fn in self.length_fns)

    """ Core Logic """

    @property
    def input_size(self):
        return self.alphabet_size

    @property
    def output_size(self):
        return self.alphabet_size

    def _init_alphabet(self, null):
        x_length = len(self.length_fns)
        alphabet = {self._get_char(i): i for i in xrange(x_length)}
        alphabet[u"#"] = x_length
        return alphabet

    """ Data Generation """

    def get_data(self):
        """
        Generate a dataset of all valid strings for n in range
            min_n <= n <= max_n.

        The training and test set are randomly assigned by splitting
        this data set.

        TODO: This method in the API should really be renamed to not
        have a "get".
        """
        all_x, all_y = self._get_tensors()
        shuffled_indices = torch.randperm(len(all_x))
        all_x = all_x[shuffled_indices]
        all_y = all_y[shuffled_indices]
        split_index = len(all_x) // 5

        self.test_x = all_x[:split_index, :, :]
        self.test_y = all_y[:split_index, :]

        self.train_x = all_x[split_index:, :, :]
        self.train_y = all_y[split_index:, :]

    def _get_tensors(self):
        """
        Generate a dataset including all strings in the language with
            self.min_n <= n <= self.max_n.
        """
        valid_n_range = lambda: xrange(self.min_n, self.max_n)
        x_strings = [self._get_x_string(n) for n in valid_n_range()]
        y_strings = [self._get_y_string(x_string) for x_string in x_strings]
        
        x_var = self.sentences_to_one_hot(self.max_length, *x_strings)
        y_var = self.sentences_to_codes(self.max_length, *y_strings)

        return x_var, y_var

    def _get_x_string(self, n):
        x_string = []
        for i, length_fn in enumerate(self.length_fns):
            length = length_fn(n)
            x_string.extend(self._get_char(i) for _ in xrange(length))
        return x_string

    def _get_y_string(self, x_string):
        return x_string[1:]

    """ Data Visualization """

    @property
    def generic_example(self):
        """
        The string for visualizations.
        """
        return self._get_x_string(5)
