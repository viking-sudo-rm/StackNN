from __future__ import division

from abc import ABCMeta

import torch
import torch.nn as nn
from torch.autograd import Variable

from tasks.base import FormalTask
from models import VanillaModel
from controllers.feedforward import LinearSimpleStructController
from structs import Stack


class LanguageModelingTask(FormalTask):
    """
    Abstract class for language modelling (word prediction) tasks. In a
    LanguageModelingTask, the neural network must read each word of the
    input sentence and predict the next word. The user may specify a set
    of words such that the controller is only evaluated on predictions made
    when the correct next word is drawn from that set.

    This abstract class implements self._evaluate_step. Subclasses need
    to implement functions relating to data generation.

    Note that a BufferedModel will always be able to perform a
    LanguageModelingTask with perfect accuracy because it can simply
    output nothing during the first time step and then copy the input.
    """


    class Params(FormalTask.Params):

        """Parameters object for a LanguageModelingTask.

        New parameters are listed below.

        Attributes:
            to_predict: Set or list of unicode characters that should be
                predicted and used in accuracy computation.
            include_unpredicted_symbols_in_loss: If True, non-null symbols that
                are not in to_predict will contribute to loss.
            max_length: The maximum sentence length.
            mask_null: If True, null characters will always be ignored.
        """

        def __init__(self, to_predict, **kwargs):
            self.to_predict = to_predict
            self.include_unpredicted_symbols_in_loss = kwargs.get("include_unpredicted_symbols_in_loss", False)
            self.max_length = kwargs.get("max_length", 25)
            self.mask_null = kwargs.get("mask_null", True)
            super(LanguageModelingTask.Params, self).__init__(**kwargs)
            self.criterion = kwargs.get("criterion", nn.CrossEntropyLoss(reduction="none"))
            self.max_x_length = self.max_length
            self.max_y_length = self.max_length


    def __init__(self, params):
        super(LanguageModelingTask, self).__init__(params)
        self.to_predict_code = [self.alphabet[c] for c in self.to_predict]

    def _evaluate_step(self, x, y, a, j):
        """
        Computes the loss, number of guesses correct, and total number
        of guesses when reading the jth symbol of the input string. If
        the correct answer for a prediction does not appear in
        self.to_predict, then we consider the loss for that prediction
        to be 0.

        :type x: Variable
        :param x: The input data

        :type y: Variable
        :param y: The output data

        :type a: Variable
        :param a: The output of the neural network after reading the jth
            word of the sentence, represented as a 2D vector. For each
            i, a[i, :] is the controller's prediction for the (j + 1)st
            word of the sentence, in one-hot representation

        :type j: int
        :param j: The jth word of a sentence is being read by the neural
            controller when this function is called

        :rtype: tuple
        :return: The loss, number of correct guesses, and number of
            total guesses after reading the jth word of the sentence
        """
        _, y_pred = torch.max(a, 1)

        if self.mask_null:
            # Mask out the null stuff for loss calculation.
            null = self.alphabet[self.null]
            valid_x = (y[:, j] != null).float()
        else:
            # Include the null indices while calculating loss.
            valid_x = torch.ones_like(y[:, j]).float()

        # If we shouldn't include unpredicted symbols in the loss, zero them
        # out.
        if not self.include_unpredicted_symbols_in_loss:
            for k in xrange(len(valid_x)):
                if y[k, j].data.item() not in self.to_predict_code:
                    valid_x[k] = 0

        # Compute the loss.
        loss = valid_x * self.criterion(a, y[:, j])

        # If we should include unpredicted terms in the loss, then we now need
        # to mask them out for prediction and accuracy calculation.
        if self.include_unpredicted_symbols_in_loss:
            to_predict_x = valid_x.data.clone()
            for k in xrange(len(valid_x)):
                if y[k, j].data.item() not in self.to_predict_code:
                    to_predict_x[k] = 0
        else:
            to_predict_x = valid_x
        
        # Compute the accuracy over indices of interest.
        correct_trials = (y_pred == y[:, j]).type(torch.FloatTensor)
        correct = sum(to_predict_x * correct_trials.data)
        total = sum(to_predict_x)
        return loss.sum(), correct, total
