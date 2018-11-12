from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from tasks.base import Task
from stacknn_utils.vector_ops import array_map
from stacknn_utils.testcase import testcase, test_module


class NaturalTask(Task):

    """A task for loading and training on raw data; i.e. not generated data.

    It is assumed that the data is in CSV format. It is read into a pandas
    DataFrame and then used to train a model.
    """

    # TODO(lambdaviking): Write this class here.

    def __init__(self, train_filename, test_filename, data_reader, **args):
        self._train_filename = train_filename
        self._test_filename = test_filename
        self._data_reader = data_reader
        super(NaturalTask, self).__init__(**args)

    def reset_model(self, model_type, controller_type, struct_type):
        # TODO(lambdaviking): Implement this.
        pass

    def get_data(self):
        train_x, train_y = self._data_reader.read_x_and_y(self.train_filename)
        test_x, test_y = self._data_reader.read_x_and_y(self.test_filename)

        max_length = self._data_reader.max_x_length
        pad = lambda line: np.pad(line, max_length, "constant")
        train_x = array_map(pad, train_x)
        test_x = array_map(pad, test_x)

        print(train_x)

        # TODO: Map words to integers.

    def _init_alphabet(self, null):
        """Task doesn't fit desired design pattern for this method."""
        return None

    def _evaluate_step(self, x, y, a, j):
        pass

    def generic_example(self):
        pass


@testcase(DataFrameTask)
def test_get_data():
    from stacknn_utils.data_readers import ByLineDatasetReader, linzen_line_consumer
    filename = "../data/linzen/rnn_arg_simple/numpred.test.5"
    data_reader = ByLineDatasetReader(linzen_line_consumer)
    task = NaturalTask(filename, filename, data_reader)
    task.get_data()

