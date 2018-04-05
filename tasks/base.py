from __future__ import division

from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim


class Task(object):
    """
    Abstract class for creating experiments that train and evaluate a
    neural network model with a neural stack or queue. To create a
    custom task, create a class inheriting from this one that overrides
    the constructor self.__init__ and the functions self.get_data and
    self._evaluate_step.
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
        Constructor for the Task object.

        :type batch_size: int
        :param batch_size: The number of trials in each batch

        :type criterion: nn.modules.loss._Loss
        :param criterion: The error function used for training the model

        :type cuda: bool
        :param cuda: If True, CUDA functionality will be used

        :type epochs: int
        :param epochs: The number of training epochs that will be
            performed when executing an experiment

        :type learning_rate: float
        :param learning_rate: The learning rate used for training

        :type max_x_length: int
        :param max_x_length: The maximum length of an input to the
            neural net

        :type max_y_length: int
        :param max_y_length: The maximum length of a neural net output

        :type model:
        :param model: The machine learning model used in this
            experiment, specified by a choice of controller and neural
            data structure

        :type read_size: int
        :param read_size: The length of the vectors stored on the neural
            data structure

        :type verbose: bool
        :param verbose: If True, the progress of the experiment will be
            displayed in the console
        """
        self.max_x_length = max_x_length
        self.max_y_length = max_y_length

        # Hyperparameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.read_size = read_size

        # Model settings
        self.model = model
        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate)

        # Runtime settings
        self.cuda = cuda
        self.verbose = verbose

        # Training and testing data
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

        return

    """ Experiments """

    def run_experiment(self):
        """
        Runs an experiment that evaluates the performance of the model
        described by this Task. In the experiment, a number of training
        epochs are run, and each epoch is divided into batches. The
        number of epochs is self.epochs, and the number of examples in
        each batch is self.batch_size. Loss and accuracy are computed
        for each batch and each epoch.

        :return: None
        """
        self._print_experiment_start()
        self.get_data()
        for epoch in xrange(self.epochs):
            self.run_epoch(epoch)

        return

    def run_epoch(self, epoch):
        """
        Trains the model on all examples in the training data set.
        Training examples are divided into batches. The number of
        examples in each batch is given by self.batch_size.

        :type epoch: int
        :param epoch: The name of the current epoch

        :return: None
        """
        self._print_epoch_start(epoch)
        self._shuffle_training_data()
        self.train()
        self.evaluate(epoch)

        return

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
        Prints information about this Task's hyperparameters at the
        start of each experiment.

        :return: None
        """
        if not self.verbose:
            return

        print "Learning Rate: " + str(self.learning_rate)
        print "Batch Size: " + str(self.batch_size)
        print "Read Size: " + str(self.read_size)

        return

    def _print_epoch_start(self, epoch):
        """
        Prints a header with the epoch name at the beginning of each
            epoch.

        :type epoch: int
        :param epoch: The name of the current epoch

        :return: None
        """
        if not self.verbose:
            return

        print "\n-- Epoch " + str(epoch) + " --\n"

        return

    """ Model Training """

    def train(self):
        """
        Trains the model given by self.model for an epoch using the data
        given by self.train_x and self.train_y.

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
            self._evaluate_batch(x, y, batch, True)

        return

    def evaluate(self, epoch):
        """
        Evaluates the model given by self.model using the testing data
        given by self.test_x and self.test_y.

        :return: None
        """
        if self.test_x is None or self.test_y is None:
            raise ValueError("Missing testing data")

        self.model.eval()
        self.model.init_stack(len(self.test_x.data))
        self._evaluate_batch(self.test_x, self.test_y, epoch, False)

        return

    def _evaluate_batch(self, x, y, name, is_batch):
        """
        Computes the loss and accuracy for one batch or epoch. If a
        batch is being considered, the parameters of self.model are
        updated according to self.criterion.

        :param x: The training input data for this batch

        :param y: The training output data for this batch

        :type name: int
        :param name: The name of this batch or epoch

        :type is_batch: bool
        :param is_batch: If True, a batch is being considered;
            otherwise, an epoch is being considered

        :return: None
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
        if is_batch:
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

        # Log the results
        self._print_batch_summary(name, is_batch, batch_loss, batch_correct,
                                  batch_total)

        return

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
        :return: The loss, number of correct guesses, and total number
            of guesses
        """
        raise NotImplementedError("Missing implementation for _evaluate_step")

    def _print_batch_summary(self, name, is_batch, batch_loss, batch_correct,
                             batch_total):
        """
        Reports the loss and accuracy to the console at the end of an
        epoch or at the end of every tenth batch.

        :type name: int
        :param name: The name of this batch or epoch

        :type is_batch: bool
        :param is_batch: If True, this function is being called during a
            batch; otherwise, it is being called during an epoch

        :param batch_loss: The total loss incurred during the batch or
            epoch

        :type batch_correct: int
        :param batch_correct: The number of correct predictions made
            during this batch or epoch

        :type batch_total: int
        :param batch_total: The total number of predictions made during
            this batch or epoch

        :return: None
        """
        if not self.verbose:
            return
        elif is_batch and name % 10 != 0:
            return

        if is_batch:
            message = "Batch {}: ".format(name)
            loss = sum(batch_loss.data) / self.batch_size
        else:
            message = "Epoch {}: ".format(name)
            loss = sum(batch_loss.data) / len(self.train_x)

        accuracy = (batch_correct * 1.0) / batch_total
        message += "Loss = {:.4f}, Accuracy = {:.2f}".format(loss, accuracy)
        print message

        return

    """ Data Generation """

    @abstractmethod
    def get_data(self):
        """
        Populates self.train_x, self.train_y, self.test_x, and
        self.test_y by generating or loading data sets for training and
        testing.

        :return: None
        """
        raise NotImplementedError("Missing implementation for get_data")
