from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod

import scipy as sp


def linzen_line_consumer(line):
	"""Convert each line into an (x, y) tuple."""
	words = line.split(" ")
	return words[0], np.array(words[1:])


class ByLineDatasetReader(object):

	"""Class for reading raw data from the Linzen dataset."""

	def __init__(self, line_consumer, track_y_length=False):
		self._line_consumer = line_consumer
		self._max_x_length = 0
		self._max_y_length = 0 if track_y_length else -1

	def _generate_examples(self, filename):
		"""Generate a sequence of (x, y) pairs."""
		with open(filename) as file_in:
			lines = np.array(file_in.readlines())
		lines = sp.char.rstrip(lines)
		for line in lines:
			x, y = self._line_consumer(line)
			self._max_x_length = max(len(x), self._max_x_length)
			if self._max_y_length > -1:
				self._max_y_length = max(len(y), self._max_y_length)
			yield x, y

	def read_x_and_y(self, filename):
		"""Read lists of input and output data from a file."""
		X, Y = [], []
		for x, y in self._generate_examples(filename):
			X.append(x)
			Y.append(y)
		return X, Y

	def reset_counts(self):
		self._max_x_length = 0
		self._max_y_length = min(self._max_y_length, -1)

	@property
	def max_x_length(self):
		"""The length of the longest-seen input."""
		return self._max_x_length

	@property
	def max_y_length(self):
		"""The length of the longest-seen output."""
		return self._max_y_length
