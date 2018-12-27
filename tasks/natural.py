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
from stacknn_utils import overrides


class NaturalTask(Task):

    """A task for loading and training on raw data; i.e. not generated data.

    It is assumed that the data is in CSV format. It is read into a pandas
    DataFrame and then used to train a model.
    """

    
    class Params(Task.Params):

        """New parameters for a task that loads a natural language dataset.

        See Task.Params for information about inherited parameters.

        Attributes:
            train_path: Path (from root directory of project) to a training
                file.
            test_path: Path (from root directory of project) to a test file.
            data_reader: Class specifying how to read the data from these files.
            max_num_embeddings: An upper bound on the number of words in the
                dataset.
            max_num_output_classes: An upper bound on the number of output
                classes in the dataset.
        """

        def __init__(self, train_path, test_path, data_reader, num_labels, **kwargs):
            self.train_path = train_path
            self.test_path = test_path
            self.data_reader = data_reader
            self.num_labels = num_labels
            self.embedding_dim = kwargs.get("embedding_dim", 300)
            super(NaturalTask.Params, self).__init__(**kwargs)


    def __init__(self, params):
        self.word_to_i = None
        self.label_to_i = None
        super(NaturalTask, self).__init__(params)

    @property
    def input_size(self):
        return self.params.embedding_dim

    @property
    def output_size(self):
        return self.params.num_labels + 1

    @property
    def num_embeddings(self):
        return len(self.word_to_i) + 1

    def get_data(self):
        train_x, train_y = self.data_reader.read_x_and_y(self.train_path)
        test_x, test_y = self.data_reader.read_x_and_y(self.test_path)

        # This field is referenced in the training logic.
        self.max_x_length = self._get_max_sentence_length(train_x, test_x)

        flatten = lambda li: (item for sublist in li for item in sublist)
        vocab_x, self.word_to_i = self._get_vocab_and_lookup_table(flatten(train_x), flatten(test_x))
        vocab_y, self.label_to_i = self._get_vocab_and_lookup_table(train_y, test_y)

        self.train_x = self._get_x_tensor(train_x)
        self.train_y = self._get_y_tensor(train_x, train_y)
        self.test_x = self._get_x_tensor(test_x)
        self.test_y = self._get_y_tensor(test_x, test_y)

        print(train_x[0])
        print(self.train_x[0])

        print(train_y[0])
        print(self.train_y[0])

        self.embedding = nn.Embedding(self.num_embeddings, self.input_size)
        print("Embedding layer initialized.")

        assert self.params.num_labels >= len(self.label_to_i), \
            "More labels found in data than specified in config."

        print("Train x shape:", self.train_x.shape)
        print("Train y shape:", self.train_y.shape)

    @staticmethod
    def _get_vocab_and_lookup_table(*datasets):
        vocab = set()
        for dataset in datasets:
            vocab.update(dataset)
        lookup_table = {word: i + 1 for i, word in enumerate(vocab)}
        return vocab, lookup_table

    @staticmethod
    def _get_max_sentence_length(*datasets):
        return max(len(sent) for dataset in datasets for sent in dataset)

    def _get_x_tensor(self, sents):
        x_tensor = torch.zeros(len(sents), self.max_x_length)
        x_tensor = x_tensor.long() # Type must be long to work properly with embeddings.
        for i, sent in enumerate(sents):
            for j, word in enumerate(sent):
                x_tensor[i, j] = self.word_to_i[word]
        return x_tensor

    def _get_y_tensor(self, sents, labels):
        y_tensor = torch.zeros(len(sents), self.max_x_length)
        y_tensor = y_tensor.long() # Make type long for evaluation logic.
        for i, label in enumerate(labels):
            predict_index = len(sents[i]) - 1
            y_tensor[i, predict_index] = self.label_to_i[label]
        return y_tensor

    @overrides(Task)
    def _evaluate_step(self, x, y, a, j):
        indices = (y[:, j] != 0)
        valid_a = a[indices.view(-1)].view(-1, self.output_size)
        valid_y = y[:, j][indices]

        if len(valid_a) == 0:
            return None, None, None

        _, valid_y_ = torch.max(valid_a, 1)

        total = len(valid_a)
        correct = len(torch.nonzero((valid_y_ == valid_y).data))
        # TODO(#19): Use binary cross entropy here.
        loss = self.criterion(valid_a, valid_y)
        return loss, correct, total
