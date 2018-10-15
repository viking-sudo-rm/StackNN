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

    The number of symbols and number of occurences of each symbol are
    configurable via task parameters.
    """

    def __init__(self,
                 length_fns=[lambda n: n, lambda n: n],
                 min_n=1,
                 max_n=1000,
                 batch_size=10,
                 clipping_norm=None,
                 criterion=nn.CrossEntropyLoss(),
                 cuda=False,
                 epochs=100,
                 early_stopping_steps=5,
                 hidden_size=10,
                 learning_rate=0.01,
                 load_path=None,
                 l2_weight=0.01,
                 model_type=VanillaModel,
                 controller_type=LinearSimpleStructController,
                 read_size=2,
                 save_path=None,
                 struct_type=Stack,
                 time_function=(lambda t: t),
                 verbose=True):

        self.min_n = min_n
        self.max_n = max_n
        self.length_fns = length_fns
        self.max_length = self._get_length(max_n)
        to_predict = [self._get_char(i) for i in xrange(len(length_fns))]

        super(OrderedCountingTask, self).__init__(to_predict=to_predict,
                                                  batch_size=batch_size,
                                                  clipping_norm=clipping_norm,
                                                  criterion=criterion,
                                                  cuda=cuda,
                                                  epochs=epochs,
                                                  early_stopping_steps=early_stopping_steps,
                                                  hidden_size=hidden_size,
                                                  learning_rate=learning_rate,
                                                  load_path=load_path,
                                                  l2_weight=l2_weight,
                                                  model_type=model_type,
                                                  controller_type=controller_type,
                                                  null=u"#",
                                                  read_size=read_size,
                                                  save_path=save_path,
                                                  struct_type=struct_type,
                                                  time_function=time_function,
                                                  verbose=verbose)

    """ Utility Methods """

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

    def reset_model(self, model_type, controller_type, struct_type, **kwargs):
        self.model = model_type(self.alphabet_size,
                                self.read_size,
                                self.alphabet_size,
                                controller_type=controller_type,
                                struct_type=struct_type,
                                **kwargs)

    def _init_alphabet(self, null):
        x_length = len(self.length_fns)
        alphabet = {self._get_char(i): i for i in xrange(x_length)}
        alphabet["#"] = x_length
        return alphabet

    # def _evaluate_step(self, x, y, a, j):
    #     # TODO: There is an inconsistency in the dimensions of a and y.
    #     valid_y = y[:, j]
    #     valid_a = a
    #     _, valid_y_ = torch.max(valid_a, 1)

    #     total = len(valid_a)
    #     correct = len(torch.nonzero(valid_y_ == valid_y))
    #     loss = self.criterion(valid_a, valid_y)
    #     return loss, correct, total

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
        y_strings = [x_string[1:] for x_string in x_strings]
        
        x_var = self.sentences_to_one_hot(self.max_length, *x_strings)
        y_var = self.sentences_to_codes(self.max_length, *y_strings)
        return x_var, y_var

    def _get_x_string(self, n):
        x_string = []
        for i, length_fn in enumerate(self.length_fns):
            length = length_fn(n)
            x_string.extend(self._get_char(i) for _ in xrange(length))
        return x_string

    """ Data Visualization """

    @property
    def generic_example(self):
        """
        The string for visualizations.
        """
        return self._get_x_string(5)

