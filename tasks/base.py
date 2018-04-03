from __future__ import division

from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim


class Task(object):
    """
    Abstract class for stack NN tasks.
    """
    __metaclass__ = ABCMeta

    def __init__(self,
                 batch_size=10,
                 criterion=nn.CrossEntropyLoss(),
                 cuda=False,
                 epochs=100,
                 learning_rate=0.1,
                 max_x_length=10,
                 max_y_length=10,
                 model=None,
                 read_size=1,
                 verbose=True):
        """

        :param batch_size:
        :param criterion:
        :param cuda:
        :param epochs:
        :param learning_rate:
        :param max_x_length:
        :param max_y_length:
        :param model:
        :param read_size:
        :param verbose:
        """
        self.max_x_length = max_x_length
        self.max_y_length = max_y_length

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.read_size = read_size
        self.cuda = cuda
        self.epochs = epochs

        self.model = model
        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.verbose = verbose

        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

    """ Experiments """

    def run_experiment(self):
        self._print_experiment_start()
        self.get_data()
        for epoch in xrange(self.epochs):
            self.run_epoch(epoch)

        return

    def run_epoch(self, epoch):
        """

        :param epoch:
        :return:
        """
        self._print_epoch_start(epoch)
        self._shuffle_training_data()
        self.train()
        self.evaluate(epoch)

    def _shuffle_training_data(self):
        """
        Shuffles the training data.

        :return: None
        """
        num_examples = len(self.train_x)
        shuffled_indices = torch.randperm(num_examples)
        self.train_x = self.train_x[shuffled_indices]
        self.train_y = self.train_y[shuffled_indices]
        return

    def _print_experiment_start(self):
        """

        :return:
        """
        if not self.verbose:
            return

        print "Learning Rate: " + str(self.learning_rate)
        print "Batch Size: " + str(self.batch_size)
        print "Read Size: " + str(self.read_size)
        return

    def _print_epoch_start(self, epoch):
        """

        :param epoch:
        :return:
        """
        if not self.verbose:
            return

        print "\n-- Epoch " + str(epoch) + " --\n"
        return

    """ Model Training """

    def train(self):
        """

        :return: None
        """
        if self.train_x is None or self.train_y is None:
            raise ValueError("Missing training data")

        self.model.train()

        last_trial = len(self.train_x.data) - self.batch_size
        for batch, i in enumerate(xrange(0, last_trial, self.batch_size)):
            x = self.train_x[i:i + self.batch_size, :, :]
            y = self.train_y[i:i + self.batch_size, :]
            self.model.init_stack(self.batch_size)
            self._evaluate_batch(x, y, batch, i)

    def evaluate(self, epoch):
        """

        :return:
        """
        if self.test_x is None or self.test_y is None:
            raise ValueError("Missing testing data")

        self.model.eval()
        self.model.init_stack(len(self.test_x.data))
        self._evaluate_batch(self.test_x, self.test_y, "Epoch " + str(epoch), 0)

    def _evaluate_batch(self, x, y, batch, i):
        """

        :param x:
        :param y:
        :param batch:
        :param i:
        :return:
        """
        batch_loss = 0.
        batch_correct = 0
        batch_total = 0

        # Read the input from left to right and evaluate the output
        for j in xrange(self.max_x_length):
            a = self.model.forward(x[:, j, :])
            loss, correct, total = self._evaluate_step(x, y, a, j)
            if loss is None or correct is None or total is None:
                continue

            batch_loss += loss
            batch_correct += correct
            batch_total += total

        # Update the model parameters
        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        # Log the results
        if type(batch) is not int or batch % 10 == 0:
            self._print_batch_summary(batch, batch_loss, batch_correct, batch_total)

    @abstractmethod
    def _evaluate_step(self, x, y, a, j):
        """
        Computes the loss, number of guesses correct, and total number
        of guesses when reading the jth symbol of the input string.

        :param x: The input training data

        :param y: The output training data

        :param a: The output of the neural network

        :type j: int
        :param j: The position of the input string being read

        :rtype: tuple
        :return: The loss, number of correct guesses, and total number of guesses
        """
        raise NotImplementedError("Missing implementation for _evaluate_batch")

    def _print_batch_summary(self, batch, batch_loss, batch_correct, batch_total):
        """

        :param batch:
        :param batch_loss:
        :param batch_correct:
        :param batch_total:

        :return: None
        """
        if not self.verbose:
            return

        if type(batch) is int:
            message = "Batch {}: ".format(batch)
        else:
            message = str(batch) + ": "

        loss = sum(batch_loss.data) / self.batch_size
        accuracy = (batch_correct * 1.0) / batch_total
        message += "Loss = {:.4f}, Accuracy = {:.2f}".format(loss, accuracy)

        print message
        return

    """ Data Generation """

    @abstractmethod
    def get_data(self):
        """
        Generates or loads training and testing datasets
        for this task.

        :return: None
        """
        raise NotImplementedError("Missing implementation for get_data")
