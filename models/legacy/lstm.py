from __future__ import division

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

from models.base import Model
from controllers.recurrent import LSTMSimpleStructController
from structs.simple import Stack

from models.vanilla import VanillaModel

class LSTMModel(VanillaModel):
	"""
    A Model that uses a SimpleStruct as its data structure, and an
	LSTMSimpleStructController.
    """

	def __init__(self, input_size, read_size, output_size,
				 controller_type=LSTMSimpleStructController, struct_type=Stack):
		super(LSTMModel, self).__init__(input_size, read_size, output_size,
											controller_type=controller_type,
											struct_type=struct_type)

		return

	def init_struct(self, batch_size):
		super(LSTMModel, self).init_struct(batch_size)
		self._controller.init_hidden(batch_size)
		return
