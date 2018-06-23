from __future__ import division

from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim

from models import VanillaController
from models.base import AbstractController
from models.networks.feedforward import LinearSimpleStructNetwork
from structs.simple import Stack


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
                 learning_rate=0.01,
                 l2_weight=.01,
                 max_x_length=10,
                 max_y_length=10,
                 model=None,
                 model_type=VanillaController,
                 network_type=LinearSimpleStructNetwork,
                 read_size=1,
                 struct_type=Stack,
                 time_function=(lambda t: t),
                 save_path=None,
                 load_path=None,
                 verbose=True):

        """
        Constructor for the Task object.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch

        :type criterion: nn.modules.loss._Loss
        :param criterion: The error function used for training the model

        :type cuda: bool
        :param cuda: If True, CUDA functionality will be used

        :type epochs: int
        :param epochs: The number of training epochs that will be
            performed when executing an experiment

        :type learning_rate: float
        :param learning_rate: The learning rate used for training

        :type l2_weight: float
        :param l2_weight: The amount of l2 regularization used for
            training

        :type max_x_length: int
        :param max_x_length: The maximum length of an input to the
            neural net

        :type max_y_length: int
        :param max_y_length: The maximum length of a neural net output

        :type model: AbstractController
        :param model: The model that will be trained and evaluated.
            This parameter is being kept for compatibility with older
            code. Please use the model_type parameter instead in order
            to automatically instantiate models

        :type model_type: type
        :param model_type: The type of Controller that will be trained
            and evaluated

        :type network_type: type
        :param network_type: The type of neural network that will drive
            the Controller

        :type read_size: int
        :param read_size: The length of the vectors stored on the neural
            data structure

        :type struct_type: type
        :param struct_type: The type of neural data structure that will
            be used by the model

        :type time_function: function
        :param time_function: A function mapping the length of an input
            to the number of computational steps the network will
            perform on that input

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
        self.time_function = time_function
        self.save_path = save_path

        # Model settings (compatibility)
        if model is None:
            self.model = None
            self.reset_model(model_type, network_type, struct_type)
        else:
            self.model = model

        if load_path:
            self.model.load_state_dict(torch.load(load_path))

        # Backpropagation settings
        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate,
                                    weight_decay=l2_weight)

        # Runtime settings
        self.cuda = cuda
        self.verbose = verbose

        # Training and testing data
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

    @abstractmethod
    def reset_model(self, model_type, network_type, struct_type):
        """
        Instantiates a neural network model of a given type that is
        compatible with this Task.

        :type model_type: type
        :param model_type: A type from the models package. Please pass
            the desired model's *type* to this parameter, not an
            instance thereof

        :return: None
        """
        raise NotImplementedError("Missing implementation for construct_model")

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
        if self.save_path:
            torch.save(self.model.state_dict(), self.save_path)

    def _shuffle_training_data(self):
        """
        Shuffles the training data.

        :return: None
        """
        num_examples = len(self.train_x)
        shuffled_indices = torch.randperm(num_examples)
        self.train_x = self.train_x[shuffled_indices]
        self.train_y = self.train_y[shuffled_indices]

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

    """ Model Training """

    def train(self):
        """
        Trains the model given by self.model for an epoch using the data
        given by self.train_x and self.train_y.

        :return: None
        """
        if self.model is None:
            raise ValueError("Missing model")
        if self.train_x is None or self.train_y is None:
            raise ValueError("Missing training data")

        self.model.train()

        last_trial = len(self.train_x.data) - self.batch_size + 1
        for batch, i in enumerate(xrange(0, last_trial, self.batch_size)):
            x = self.train_x[i:i + self.batch_size, :, :]
            y = self.train_y[i:i + self.batch_size, :]
            self.model.init_controller(self.batch_size, x)
            self._evaluate_batch(x, y, batch, True)

    def evaluate(self, epoch):
        """
        Evaluates the model given by self.model using the testing data
        given by self.test_x and self.test_y.

        :return: None
        """
        if self.test_x is None or self.test_y is None:
            raise ValueError("Missing testing data")

        self.model.eval()
        self.model.init_controller(len(self.test_x.data), self.test_x)
        self._evaluate_batch(self.test_x, self.test_y, epoch, False)

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
        num_steps = self.time_function(self.max_x_length)
        for j in xrange(num_steps):
            self.model()
        for j in xrange(self.max_x_length):
            a = self.model.read_output()
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

    """ Reporting """

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
        elif is_batch and name % 10 != 9:
            return

        if is_batch:
            message = "Batch {}: ".format(name)
            loss = sum(batch_loss.data) / self.batch_size
        else:
            message = "Epoch {} Test: ".format(name)
            loss = sum(batch_loss.data) / self.test_x.size(0)

        accuracy = (batch_correct * 1.0) / batch_total
        message += "Loss = {:.4f}, Accuracy = {:.2f}".format(loss, accuracy)
        print message
