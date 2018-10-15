"""
Word prediction tasks based on CFGs. In each task in this module, the
neural network will read a sentence (sequence of words) and predict the
next word. We may specify a list of words that must be predicted by the
neural network. For example, if we specify that the controller must predict
verbs, then we only evaluate it based on the predictions made when the
correct answer is a verb.

Tasks are distinguished from one another based on how input and output
data are generated for training and testing. In all tasks below, the
data are generated by some sort of context-free grammar.
"""
# TODO: Make another version of CFGTask for PCFGs
from __future__ import division

import random
from abc import ABCMeta

import nltk.grammar as gr
import torch
import torch.nn as nn
from nltk.parse.generate import generate
from torch.autograd import Variable

from tasks.language_modeling import LanguageModellingTask
from models import VanillaModel
from controllers.feedforward import LinearSimpleStructController
from structs import Stack


class CFGTask(LanguageModellingTask):
    """
    In this task, the input and output data used for training and
    evaluation are based on examples uniformly sampled from a set of
    sentences generated by a deterministic context-free grammar.
    """

    def __init__(self,
                 grammar,
                 to_predict,
                 sample_depth,
                 batch_size=10,
                 clipping_norm=None,
                 criterion=nn.CrossEntropyLoss(reduction='none'),
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
                 test_set_size=100,
                 time_function=(lambda t: t),
                 train_set_size=800,
                 verbose=True):
        """
        Constructor for the CFGTask object.

        :type grammar: gr.CFG
        :param grammar: The context-free grammar that will generate
            training and testing data

        :type to_predict: list
        :param to_predict: The words that will be predicted in this task

        :type sample_depth: int
        :param sample_depth: The maximum depth to which sentences will
            be sampled from the grammar

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

        :type test_set_size: int
        :param test_set_size: The number of examples to include in the
            testing data

        :type time_function: function
        :param time_function: A function mapping the length of an input
            to the number of computational steps the controller will
            perform on that input

        :type train_set_size: int
        :param train_set_size: The number of examples to include in the
            training data

        :type verbose: bool
        :param verbose: If True, the progress of the experiment will be
            displayed in the console
        """
        self.grammar = grammar

        super(CFGTask, self).__init__(to_predict,
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
                                      max_length=max_length,
                                      model_type=model_type,
                                      null=null,
                                      controller_type=controller_type,
                                      read_size=read_size,
                                      save_path=save_path,
                                      struct_type=struct_type,
                                      time_function=time_function,
                                      verbose=verbose)

        self.sample_depth = sample_depth
        print "Sample depth: %d" % sample_depth
        self.max_length = max_length
        print "Max length: %d" % max_length

        self.train_set_size = train_set_size
        self.test_set_size = test_set_size

        self.sample_strings = self.generate_sample_strings()
        print "{} strings generated".format(len(self.sample_strings))
        if len(self.sample_strings) > 0:
            max_sample_length = max([len(x) for x in self.sample_strings])
        else:
            max_sample_length = 0
        print "Maximum sample length: " + str(max_sample_length)
        print "Maximum input length: " + str(self.max_x_length)

    def reset_model(self, model_type, controller_type, struct_type, **kwargs):
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
        self.model = model_type(self.alphabet_size, self.read_size,
                                self.alphabet_size,
                                controller_type=controller_type,
                                struct_type=struct_type,
                                **kwargs)

    def _init_alphabet(self, null):
        """
        Creates an encoding of a CFG's terminal symbols as numbers.

        :type null: unicode
        :param null: A string representing "null"

        :rtype: dict
        :return: A dict associating each terminal of the grammar with a
            unique number. The highest number represents "null"
        """
        rhss = [r.rhs() for r in self.grammar.productions()]
        rhs_symbols = set()
        rhs_symbols.update(*rhss)
        rhs_symbols = set(x for x in rhs_symbols if gr.is_terminal(x))

        alphabet = {x: i for i, x in enumerate(rhs_symbols)}
        alphabet[null] = len(alphabet)

        return alphabet

    """ Data Generation """

    def get_data(self):
        """
        Generates training and testing datasets for this task using the
        self.get_tensors method.

        :return: None
        """
        self.train_x, self.train_y = self.get_tensors(self.train_set_size)

        self.test_x, self.test_y = self.get_tensors(self.test_set_size)

        return

    def generate_sample_strings(self, remove_duplicates=True):
        """
        Generates all strings from self.grammar up to the depth
        specified by self.depth. Duplicates may optionally be removed.

        :type remove_duplicates: bool
        :param remove_duplicates: If True, duplicates will be removed

        :rtype: list
        :return: A list of strings generated by self.grammar
        """
        generator = generate(self.grammar, depth=self.sample_depth)
        if remove_duplicates:
            return [list(y) for y in set(tuple(x) for x in generator)]
        else:
            return list(generator)

    def get_tensors(self, num_tensors):
        """
        Generates a dataset for this task. Each input consists of a
        sentence generated by self.grammar. Each output consists of a
        list of words such that the jth word is the correct prediction
        the neural network should make after having read the jth input
        word. In this case, the correct prediction is the next word.

        Input words are represented in one-hot encoding. Output words
        are represented numerically according to self.code_for. Each
        sentence is truncated to a fixed length of self.max_length. If
        the sentence is shorter than this length, then it is padded with
        "null" symbols. The dataset is represented as two tensors, x and
        y; see self._evaluate_step for the structures of these tensors.

        :type num_tensors: int
        :param num_tensors: The number of sentences to include in the
            dataset

        :rtype: tuple
        :return: A Variable containing the input dataset and a Variable
            containing the output dataset
        """
        x_raw = [self.get_random_sample_string() for _ in xrange(num_tensors)]
        y_raw = [s[1:] for s in x_raw]

        x_var = self.sentences_to_one_hot(self.max_x_length, *x_raw)
        y_var = self.sentences_to_codes(self.max_y_length, *y_raw)

        return x_var, y_var

    def get_random_sample_string(self):
        """
        Randomly chooses a sentence from self.sample_strings with a
        uniform distribution.

        :rtype: list
        :return: A sentence from self.sample_strings
        """
        return random.choice(self.sample_strings)

    """ Data Visualization """

    @property
    def generic_example(self):
        """
        The string for visualizations.

        TODO: Make this a function of the grammar.
        """
        return [u"#"]


class CFGTransduceTask(CFGTask):
    """
    This task is like CFGTask, except that the controller receives symbols
    with even indices as input, and must predict the symbols with odd
    indices.
    """

    def get_tensors(self, num_tensors):
        """
        Generates a dataset for this task. Each input consists of a
        sentence generated by self.grammar. Each output consists of a
        list of words such that the jth word is the correct prediction
        the neural network should make after having read the jth input
        word. In this case, the correct prediction is the next word.

        Input words are represented in one-hot encoding. Output words
        are represented numerically according to self.code_for. Each
        sentence is truncated to a fixed length of self.max_length. If
        the sentence is shorter than this length, then it is padded with
        "null" symbols. The dataset is represented as two tensors, x and
        y; see self._evaluate_step for the structures of these tensors.

        :type num_tensors: int
        :param num_tensors: The number of sentences to include in the
            dataset

        :rtype: tuple
        :return: A Variable containing the input dataset and a Variable
            containing the output dataset
        """
        x_orig = [self.get_random_sample_string() for _ in xrange(num_tensors)]
        x_raw = [a[::2] for a in x_orig]
        y_raw = [a[1::2] for a in x_orig]

        x_var = self.sentences_to_one_hot(self.max_x_length, *x_raw)
        y_var = self.sentences_to_codes(self.max_y_length, *y_raw)

        return x_var, y_var
