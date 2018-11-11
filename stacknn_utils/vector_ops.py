import numpy as np


def array_map(f, arr):
	"""Map a vectorized function over an array."""
	return np.array([f(x) for x in arr])
