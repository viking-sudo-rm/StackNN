from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod

import numpy as np
import scipy as sp


def linzen_line_consumer(line):
	"""Convert each line into an (x, y) tuple."""
	y, text = line.split("\t")
	return np.array(text.split(" ")), y


class ByLineDatasetReader(object):

	"""Class for reading raw data from the Linzen dataset."""

	def __init__(self, line_consumer):
		self._line_consumer = line_consumer

	def _generate_examples(self, filename):
		"""Generate a sequence of (x, y) pairs."""
		with open(filename) as file_in:
			lines = np.array(file_in.readlines())
		lines = sp.char.rstrip(lines)
		for line in lines:
			yield self._line_consumer(line)

	def read_x_and_y(self, filename):
		"""Read lists of input and output data from a file."""
		X, Y = [], []
		for x, y in self._generate_examples(filename):
			X.append(x)
			Y.append(y)
		return X, Y
