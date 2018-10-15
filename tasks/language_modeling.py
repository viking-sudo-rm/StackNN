from __future__ import division

from abc import ABCMeta

import torch
import torch.nn as nn
from torch.autograd import Variable

from base import Task
from models import VanillaModel
from controllers.feedforward import LinearSimpleStructController
from structs import Stack


class LanguageModelingTask(Task):
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
    __metaclass__ = ABCMeta

    def __init__(self,
                 to_predict,
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
                 max_length=25,
                 model_type=VanillaModel,
                 controller_type=LinearSimpleStructController,
                 null=u"#",
                 read_size=2,
                 save_path=None,
                 struct_type=Stack,
                 time_function=(lambda t: t),
                 verbose=True):
        """
        Constructor for the LanguageModelingTask object.

        :type to_predict: list
        :param to_predict: The words that will be predicted in this task

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch

        :type criterion: nn.modules.loss._Loss
        :param criterion: The error function used for training the model

        :type cuda: bool
        :param cuda: If True, CUDA functionality will be used

        :type epochs: int
        :param epochs: The number of training epochs that will be
            performed when executing an experiment

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

        :type max_length: int
        :param max_length: The maximum length of a sentence that will
            appear in the input training and testing data

        :type model_type: type
        :param model_type: The type of Model that will be trained
            and evaluated

        :type controller_type: type
        :param controller_type: The type of neural network that will drive
            the Model

        :type null: unicode
        :param null: The "null" symbol used in this CFGTask

        :type read_size: int
        :param read_size: The length of the vectors stored on the neural
            data structure

        :type save_path: str
        :param save_path: If this param is not set to None, then the
            neural network will be saved to the path specified by this
            save_path

        :type struct_type: type
        :param struct_type: The type of neural data structure that will
            be used by the Model

        :type time_function: function
        :param time_function: A function mapping the length of an input
            to the number of computational steps the controller will
            perform on that input

        :type verbose: bool
        :param verbose: If True, the progress of the experiment will be
            displayed in the console
        """
        super_class = super(LanguageModelingTask, self)
        super_class.__init__(batch_size=batch_size,
                             clipping_norm=clipping_norm,
                             criterion=criterion,
                             cuda=cuda,
                             epochs=epochs,
                             early_stopping_steps=early_stopping_steps,
                             hidden_size=hidden_size,
                             learning_rate=learning_rate,
                             load_path=load_path,
                             l2_weight=l2_weight,
                             max_x_length=max_length,
                             max_y_length=max_length,
                             model_type=model_type,
                             null=null,
                             controller_type=controller_type,
                             read_size=read_size,
                             save_path=save_path,
                             struct_type=struct_type,
                             time_function=time_function,
                             verbose=verbose)

        self.to_predict_code = [self.alphabet[c] for c in to_predict]

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

        # Mask out the null stuff for loss calculation.
        null = self.alphabet[self.null]
        valid_x = (y[:, j] != null).type(torch.FloatTensor)
        loss = valid_x * self.criterion(a, y[:, j])

        # Mask out uninteresting stuff for accuracy calculation.
        to_predict_x = valid_x.data.clone()
        for k in xrange(len(valid_x)):
            if y[k, j].data.item() not in self.to_predict_code:
                to_predict_x[k] = 0
        
        # Compute the accuracy over indices of interest.
        correct_trials = (y_pred == y[:, j]).type(torch.FloatTensor)
        correct = sum(to_predict_x * correct_trials.data)
        total = sum(to_predict_x.data)
        return loss.sum(), correct, total
