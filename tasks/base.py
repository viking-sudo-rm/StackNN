from __future__ import division

from abc import ABCMeta, abstractmethod, abstractproperty

from copy import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from models import VanillaModel
from controllers.feedforward import LinearSimpleStructController
from stacknn_utils import *
from structs.simple import Stack


class Task(object):
    """
    Abstract class for creating experiments that train and evaluate a
    neural network model with a neural stack or queue.
    """
    __metaclass__ = ABCMeta

    def __init__(self,
                 batch_size=10,
                 clipping_norm=None,
                 criterion=nn.CrossEntropyLoss(),
                 cuda=False,
                 epochs=100,
                 early_stopping_steps=5,
                 hidden_size=10,
                 learning_rate=0.01,
                 load_path=None,
                 l2_weight=.01,
                 max_x_length=10,
                 max_y_length=10,
                 model_type=VanillaModel,
                 controller_type=LinearSimpleStructController,
                 null=u"#",
                 read_size=2,
                 reg_weight=1.,
                 save_path=None,
                 struct_type=Stack,
                 time_function=(lambda t: t),
                 verbose=True):

        """
        Constructor for the Task object.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch

        :type clipping_norm:
        :param clipping_norm:

        :type criterion: nn.modules.loss._Loss
        :param criterion: The error function used for training the model

        :type cuda: bool
        :param cuda: If True, CUDA functionality will be used

        :type epochs: int
        :param epochs: The number of training epochs that will be
            performed when executing an experiment

        :type early_stopping_steps: int
        :param early_stopping_steps: the number of epochs that training can 
            continue without an improvement in loss

        :type hidden_size: int
        :param hidden_size: The size of state vectors

        :type learning_rate: float
        :param learning_rate: The learning rate used for training

        :type load_path: str
        :param load_path: The neural network will be initialized to a
            saved controller located in this path. If load_path is set to
            None, then the controller will be initialized to an empty state

        :type l2_weight: float
        :param l2_weight: The amount of l2 regularization used for
            training

        :type max_x_length: int
        :param max_x_length: The maximum length of an input to the
            neural net

        :type max_y_length: int
        :param max_y_length: The maximum length of a neural net output

        :type model_type: type
        :param model_type: The type of Model that will be trained
            and evaluated

        :type controller_type: type
        :param controller_type: The type of neural network that will drive
            the Model

        :type null: unicode
        :param null: The "null" symbol used in this Task

        :type read_size: int
        :param read_size: The length of the vectors stored on the neural
            data structure

        :type save_path: str
        :param save_path: If this param is not set to None, then the
            neural network will be saved to the path specified by this
            save_path

        :type struct_type: type
        :param struct_type: The type of neural data structure that will
            be used by the model

        :type time_function: function
        :param time_function: A function mapping the length of an input
            to the number of computational steps the controller will
            perform on that input

        :type verbose: bool
        :param verbose: If True, the progress of the experiment will be
            displayed in the console
        """
        self.max_x_length = max_x_length
        self.max_y_length = max_y_length
        self.clipping_norm = clipping_norm

        # Hyperparameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_steps = early_stopping_steps
        self.learning_rate = learning_rate
        self.read_size = read_size
        self.time_function = time_function
        if not (isinstance(save_path, str) or isinstance(save_path, unicode)):
            self.save_path = None
        elif save_path.endswith(".dat"):
            self.save_path = save_path
        else:
            self.save_path = save_path + ".dat"

        # Model settings
        self.null = null
        self.alphabet = self._init_alphabet(null)
        self.code_to_word = {c: w for w, c in self.alphabet.iteritems()}
        self.alphabet_size = len(self.alphabet)

        self.model = None
        self.reset_model(model_type, controller_type, struct_type,
                         hidden_size=hidden_size, reg_weight=reg_weight)

        if load_path:
            self.model.load_state_dict(torch.load(load_path))
            self._has_trained_model = True
        else:
            self._has_trained_model = False

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

        # Reporting
        self._logging = False
        self._logged_x_text = None
        self._logged_y_text = None
        self._logged_a = None
        self._logged_loss = None
        self._logged_correct = None
        self._curr_log_index = 0

        self.batch_acc = None
    @abstractmethod
    def reset_model(self, model_type, controller_type, struct_type):
        """
        Instantiates a neural network model of a given type that is
        compatible with this Task. This function must set self.model to
        an instance of model_type.

        :type model_type: type
        :param model_type: The type of the Model used in this Task

        :type controller_type: type
        :param controller_type: The type of the Controller that will perform
            the neural network computations

        :type struct_type: type
        :param struct_type: The type of neural data structure that this
            Model will operate

        :return: None
        """
        raise NotImplementedError("Missing implementation for construct_model")

    """ Data """

    @abstractmethod
    def get_data(self):
        """
        Populates self.train_x, self.train_y, self.test_x, and
        self.test_y by generating or loading data sets for training and
        testing.

        :return: None
        """
        raise NotImplementedError("Missing implementation for get_data")

    @abstractmethod
    def _init_alphabet(self, null):
        """
        Creates the alphabet over which strings in this Task are
        defined.

        :type null: unicode
        :param null: A special "null" symbol

        :rtype: dict
        :return: A dict mapping each alphabet symbol to a unique number
        """
        raise NotImplementedError("Missing implementation for _init_alphabet")

    def sentences_to_one_hot(self, max_length, *sentences):
        """
        Converts one or more sentences to one-hot representation.

        :type max_length: int
        :param max_length: The maximum length of a sentence

        :type sentences: list
        :param sentences: A list of sentences

        :rtype: Variable
        :return: A Variable containing each sentence of sentences in
            one-hot representation. Each sentence is padded to length
            max_length with null symbols. Entry x[i, j, k] of the output
            Variable x represents the kth component of the one-hot
            representation of the jth word of the ith sentence
        """
        s_codes = [[self.alphabet[w] for w in s] for s in sentences]
        num_strings = len(s_codes)

        # Initialize output to NULLs
        null_code = self.alphabet[self.null]
        x = torch.FloatTensor(num_strings, max_length, self.alphabet_size)
        x[:, :, :].fill_(0)
        x[:, :, null_code].fill_(1)

        # Fill in values
        for i, s in enumerate(s_codes):
            for j, w in enumerate(s):
                x[i, j, :] = self.one_hot(w, self.alphabet_size)

        return Variable(x)

    def sentences_to_codes(self, max_length, *sentences):
        """
        Converts one or more sentences to numerical representation based
        on self.alphabet.

        :type max_length: int
        :param max_length: The maximum length of a sentence

        :type sentences: list
        :param sentences: A list of sentences

        :rtype: Variable
        :return: A Variable containing each sentence of sentences in
            numerical representation. Each sentence is padded to length
            max_length with null symbols. Entry y[i, j] of the output
            Variable y is the number representing the jth word of the
            ith sentence
        """
        s_codes = [[self.alphabet[w] for w in s] for s in sentences]
        num_strings = len(s_codes)

        # Initialize output to NULLs
        null_code = self.alphabet[self.null]
        y = torch.LongTensor(num_strings, max_length)
        y.fill_(null_code)

        # Fill in values
        for i, s in enumerate(s_codes):
            for j, w in enumerate(s):
                y[i, j] = w

        return Variable(y)

    def one_hot_to_sentences(self, max_length, one_hots):
        """
        Converts a one_hot tensor to sentence form.

        :type max_length: int
        :param max_length: If this param is set to a positive number,
            then the sentences produced will be padded to this length
            with NULL symbols. Otherwise, trailing NULL symbols will be
            removed

        :type one_hots: Variable
        :param one_hots: A Variable containing a number of sentences in
            one-hot representation. See self.sentences_to_one_hot for
            the format of the Variable. If the vector corresponding to a
            word contains numbers other than 0 or 1, then the largest
            component will be treated as a 1, and all other components
            will be treated as 0s

        :rtype: list
        :return: The list of sentences represented in one_hots
        """
        _, codes_var = torch.max(one_hots, 2)
        codes_array = codes_var.data.numpy()
        return self._codes_array_to_sentences(max_length, codes_array)

    def codes_to_sentences(self, max_length, codes):
        """
        Converts a numerical tensor to sentence form.

        :type max_length: int
        :param max_length: If this param is set to a positive number,
            then the sentences produced will be padded to this length
            with NULL symbols. Otherwise, trailing NULL symbols will be
            removed

        :type codes: Variable
        :param codes: A Variable containing a number of sentences in
            numerical representation based on self.alphabet

        :rtype: list
        :return: The list of sentences represented in codes
        """
        return self._codes_array_to_sentences(max_length, codes.data.numpy())

    @staticmethod
    def text_to_sentences(*lines):
        """
        Converts lines of text to sentence objects.

        :type lines: str
        :param lines: Each line is a " "-delimited string representing a
            sentence

        :rtype: list
        :return: A list containing the lines in sentence form
        """
        return [l.strip().split(" ") for l in lines]

    @staticmethod
    def sentences_to_text(*sentences):
        """
        Converts sentences to lines of text.

        :type sentences: list
        :param sentences: One or more sentences

        :rtype: list
        :return: A list of " "-delimited strings representing the
            sentences
        """
        return [" ".join(s) for s in sentences]

    @staticmethod
    def one_hot(number, size):
        """
        Computes the following one-hot encoding:
            0 -> [1., 0., 0., ..., 0.]
            1 -> [0., 1., 0., ..., 0.]
            2 -> [0., 0., 1., ..., 0.]
        etc.

        :type number: int
        :param number: A number

        :type size: int
        :param size: The number of dimensions of the one-hot vector.
            There should be at least one dimension corresponding to each
            possible value for number

        :rtype: torch.FloatTensor
        :return: The one-hot encoding of number
        """
        one_hot_tensor = torch.zeros([size])
        one_hot_tensor[number] = 1.
        return one_hot_tensor
        # return torch.FloatTensor([float(i == number) for i in xrange(size)])

    def _codes_array_to_sentences(self, max_length, codes_array):
        """
        Converts a numerical NumPy array to sentence form.

        :type max_length: int
        :param max_length: If this param is set to a positive number,
            then the sentences produced will be padded to this length
            with NULL symbols. Otherwise, trailing NULL symbols will be
            removed

        :type codes_array: np.ndarray
        :param codes_array: A NumPy array containing a number of
            sentences in numerical representation based on self.alphabet

        :rtype: list
        :return: The list of sentences represented in codes
        """
        if max_length > 0:
            codes = [list(s) for s in codes_array[:, :max_length]]
        else:
            codes = [list(s) for s in codes_array]
            null_code = self.alphabet[self.null]
            for i, s in enumerate(codes):
                trailing_null_ind = -1
                for j, c in enumerate(s):
                    if c == null_code and trailing_null_ind < 0:
                        trailing_null_ind = j
                    elif c != null_code:
                        trailing_null_ind = -1

                codes[i] = s[:trailing_null_ind]

        return [[self.code_to_word[c] for c in s] for s in codes]

    """ Backpropagation """

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
            self.model.init_model(self.batch_size, x)
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
        self.model.init_model(len(self.test_x.data), self.test_x)
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
        # TODO: Verify counts and total.
        num_steps = self.time_function(self.max_x_length)
        for j in xrange(num_steps):
            self.model()
        for j in xrange(self.max_x_length):
            a = self.model.read_output()
            self._log_prediction(a)
            loss, correct, total = self._evaluate_step(x, y, a, j)
            if loss is None or correct is None or total is None:
                continue

            batch_loss += loss
            batch_correct += correct
            batch_total += total

        # Get regularization loss term.
        if loss is not None:
            losses = self.model.get_and_reset_reg_loss()
            loss += torch.sum(losses)

        # Update the model parameters.
        if is_batch:
            self.optimizer.zero_grad()
            batch_loss.backward()

            if self.clipping_norm:
                nn.utils.clip_grad_norm(self.model.parameters(),
                                        self.clipping_norm)

            self.optimizer.step()

        # Log the results.
        self._print_batch_summary(name, is_batch, batch_loss, batch_correct,
                                  batch_total)

        # Make the accuracy accessible for early stopping.
        self.batch_acc = batch_correct / batch_total

    @abstractmethod
    def _evaluate_step(self, x, y, a, j):
        """
        Computes the loss, number of guesses correct, and total number
        of guesses when reading the jth symbol of the input string.

        :type x: Variable
        :param x: The input data, represented as a 3D tensor. For each
            i and j, x[i, j, :] is the jth symbol of the ith sentence of
            the batch, represented as a one-hot vector

        :type y: Variable
        :param y: The output data, represented as a 2D matrix. For each
            i and j, y[i, j] is the (j + 1)st symbol of the ith sentence
            of the batch, represented numerically according to
            self.alphabet

        :type a: Variable
        :param a: The output of the neural network after reading the jth
            word of the sentence, represented as a 1D vector

        :type j: int
        :param j: The jth word of a sentence is being read by the neural
            controller when this function is called

        :rtype: tuple
        :return: The loss, number of correct guesses, and number of
            total guesses after reading the jth word of the sentence
        """
        raise NotImplementedError("Missing implementation for _evaluate_step")

    """ Training Mode """

    def run_experiment(self):
        """
        Runs an experiment that evaluates the performance of the model
        described by this Task. In the experiment, a number of training
        epochs are run, and each epoch is divided into batches. The
        number of epochs is self.epochs, and the number of examples in
        each batch is self.batch_size. Loss and accuracy are computed
        for each batch and each epoch. Early stopping takes place if accuracy
        on the dev set does not show improvement for self.early_stopping_steps.

        :return: None
        """
        self._print_experiment_start()
        self.get_data()
        no_improvement_batches = 0
        best_acc = 0.
        for epoch in xrange(self.epochs):
            self.run_epoch(epoch)
#            print best_acc, self.batch_acc, no_improvement_batches
            if self.batch_acc <= best_acc:
                no_improvement_batches += 1
            else:
                best_acc = self.batch_acc
                no_improvement_batches = 0
            if no_improvement_batches == self.early_stopping_steps:
                break
        self._has_trained_model = True

        return {
            "best_acc": best_acc,
            "final_acc": self.batch_acc,
        }

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

        print "Starting {} Experiment".format(type(self).__name__)
        self.model.print_experiment_start()
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
        if self.verbose:
            print "\n-- Epoch {} of {} --\n".format(epoch, self.epochs - 1)

    """ Testing Mode """

    def run_test(self, data_file, log_file=None):
        """
        Evaluates the model based on testing data loaded from a file.

        :type data_file: str
        :param data_file: A CSV file containing the testing data

        :type log_file: str
        :param log_file: If a filename is provided, then the input,
            correct output, and output predicted by the controller for each
            example are saved to the path provided

        :return: None
        """
        if not self._has_trained_model:
            testing_mode_no_model_warning()
        self._load_testing_data(data_file)
        if log_file is not None:
            check_extension(log_file, "csv")
            self.start_log()
            self.reset_log()
        self.evaluate(-1)
        self.stop_log()
        self.export_log(log_file)

    def trace_step(self, x, step=True):
        """
        Steps through the neural network's computation. The controller will
        read an input and produce an output. At each time step, a
        summary of the controller's state and actions will be printed to
        the console.

        :type x: str
        :param x: A single input string in text form

        :type step: bool
        :param step: If True, the user will need to press Enter in the
            console after each computation step

        :return: None
        """
        x_sent = self.text_to_sentences(x)
        x_var = self.sentences_to_one_hot(self.max_x_length, *x_sent)
        x_code = self.sentences_to_codes(self.max_y_length, *x_sent)
        num_steps = self.time_function(len(x_sent))

        print "Begin computation on input " + x
        if step:
            raw_input("Press Enter to continue\n")

        self.model.trace_step(x_var, num_steps, step=step)

        # Get the output of the controller
        self.test_x = x_var
        self.test_y = x_code
        self.reset_log()
        self.start_log()
        for j in xrange(self.max_x_length):
            a = self.model.read_output()
            self._log_prediction(a)
        self.stop_log()

        a_sent = self.codes_to_sentences(self.max_y_length, self._logged_a)
        a_text = self.sentences_to_text(*a_sent)[0]

        print "Input: " + x
        print "Output: " + a_text

    def trace_console(self, step=True):
        """
        Allows the user to call trace_step using inputs entered in the
        console.

        :type step: bool
        :param step: If True, the user will need to press Enter in the
            console after each computation step

        :return: None
        """
        x = "x"
        while x != "":
            print ""
            x = raw_input("Please enter an input, or enter nothing to quit.\n")
            x = x.strip()
            if x != "":
                self.trace_step(x, step=step)

    def _load_testing_data(self, filename):
        """
        Loads a testing dataset from a file and saves it to self.test_x
        and self.test_y. See self._load_data_from_file for the format of
        the data file.

        :type filename: str
        :param filename: A CSV file containing the data

        :return: None
        """
        xs, ys = self._load_data_from_file(filename)
        self.test_x = xs
        self.test_y = ys

    def _load_data_from_file(self, filename):
        """
        Converts data stored in a CSV file to PyTorch Variables. Each
        line of the CSV file should contain two items: a testing input
        and its corresponding output, in that order. Input and output
        examples should be represented as " "-delimited strings. For
        example, the following is a possible data file for ReverseTask.

            0 1 0 0 1 2 2 2 2 2,2 2 2 2 2 1 0 0 1 0
            0 0 1 1 2 2 2 2 2 2,2 2 2 2 1 1 0 0 2 2
            1 0 0 0 0 2 2 2 2 2,2 2 2 2 2 0 0 0 0 1

        :type filename: str
        :param filename: A CSV file containing the data

        :rtype: tuple
        :return: Variables containing the input and output data,
            respectively
        """
        check_extension(filename, "csv")

        f = open(filename, "r")
        raw_strings = f.readlines()
        f.close()

        if len(raw_strings) % 2 != 0:
            raise ValueError("The file must have an even number of lines!")

        x_text = [line.split(",")[0].strip() for line in raw_strings]
        y_text = [line.split(",")[1].strip() for line in raw_strings]

        x_sentences = Task.text_to_sentences(*x_text)
        y_sentences = Task.text_to_sentences(*y_text)

        x_var = self.sentences_to_one_hot(self.max_x_length, *x_sentences)
        y_var = self.sentences_to_codes(self.max_y_length, *y_sentences)

        self._logged_x_text = x_text
        self._logged_y_text = y_text

        return x_var, y_var

    def _print_test_start(self):
        """
        Prints information about this Task's hyperparameters at the
        start of each test.

        :return: None
        """
        print "Starting Test"
        print "Read Size: " + str(self.read_size)

    """ Reporting """

    def start_log(self):
        """
        Sets self._logging to True, so that data will be logged the next
        time self.run_test is called. For each item in self.test_x and
        self.test_y, the neural network's predicted output will be
        recorded.

        :return: None
        """
        self._logging = True

    def stop_log(self):
        """
        Sets self._logging to False, so that data will no longer be
        logged.

        :return: None
        """
        self._logging = False

    def reset_log(self):
        """
        Resets the logged predictions to a blank state. The testing data
        are not reset.

        :return: None
        """
        num_strings = self.test_y.size(0)
        self._curr_log_index = 0
        self._logged_a = Variable(torch.zeros(num_strings, self.max_y_length))

    def export_log(self, filename):
        """
        Saves the logged testing inputs, correct outputs, and predicted
        outputs to a file.

        :type filename: str
        :param filename: A CSV file to save the logged data to

        :return: None
        """
        if filename is None:
            return
        check_extension(filename, "csv")

        a_sent = self.codes_to_sentences(self.max_y_length, self._logged_a)
        a_text = self.sentences_to_text(*a_sent)
        num_strings = len(a_text)

        f = open(filename, "w")
        f.write("Input,Correct Output,Predicted Output\n")
        for i in xrange(num_strings):
            line = ",".join([self._logged_x_text[i], self._logged_y_text[i],
                             a_text[i]]) + "\n"
            f.write(line)
        f.close()

    def _log_prediction(self, a):
        """
        Records a predicted output of the neural network.

        :type a: Variable
        :param a: The predicted output of the neural network. The value
            passed to this param should be a Variable containing the
            controller's prediction for the jth symbol of each string in
            the current testing batch, for some j.

        :return: None
        """
        if self._logging:
            _, y_pred = torch.max(a, 1)
            self._logged_a[:, self._curr_log_index] = y_pred
            self._curr_log_index += 1

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

        :type batch_loss: Variable
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
        elif name >= 0:
            message = "Epoch {} Test: ".format(name)
        else:
            message = "Test Results: "
        loss = batch_loss.data.item()

        accuracy = 100. * (batch_correct * 1.0) / batch_total
        message += "Loss = {:.4f}, Accuracy = {:.1f}%".format(loss, accuracy)
        print message

    @abstractproperty
    def generic_example(self):
        """Get the example input for creating visualizations."""
        raise NotImplementedError("Abstract property generic_example not implemented.")

    @staticmethod
    def from_config_dict(config_dict):
        """Create a new task instance from a config dict."""
        if "task" not in config_dict:
            return ValueError("Config dictionary does not contain a task.")
        config_dict = copy(config_dict)
        task_type = config_dict["task"]
        del config_dict["task"]
        return task_type(**config_dict)
