from __future__ import division

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

from models.base import AbstractController
from shmetworks.recurrent import LSTMSimpleStructShmetwork
from structs.simple import Stack

from models.vanilla import VanillaController

class LSTMController(VanillaController):
	"""
    A Controller that uses a SimpleStruct as its data structure, and an
	LSTMSimpleStructShmetwork.
    """

	def __init__(self, input_size, read_size, output_size,
				 shmetwork_type=LSTMSimpleStructShmetwork, struct_type=Stack):
		super(LSTMController, self).__init__(input_size, read_size, output_size,
											shmetwork_type=shmetwork_type,
											struct_type=struct_type)

		return

	def init_struct(self, batch_size):
		super(LSTMController, self).init_struct(batch_size)
		self._shmetwork.init_hidden(batch_size)
		return
