from __future__ import with_statement
from sklearn.preprocessing import LabelEncoder

class Dataset:

	def __init__(self, filename):
		with open(filename, "r") as fh:
			self.x, self.y = [], []
			for line in fh.readlines():
				y, x = line.strip().split("\t")
				self.x.append(x.split(" "))
				self.y.append(y)

	def getLexicon(self):
		""" Should use this function on all relevant datasets to build an encoder. """
		lexicon = set()
		for sent in self.x:
			lexicon |= set(sent)
		return lexicon

	def encode(self, encoder):
		""" Takes a LabelEncoder to convert tokens to numbers. """
		for i in xrange(len(self.x)):
			if i % 1000 == 0: print "{}/{}".format(i, len(self.x))
			self.x[i] = encoder.transform(self.x[i])
			self.y[i] = 0 if self.y[i] == "VBZ" else 1

print "loading dataset.."
dataset = Dataset("rnn_agr_simple/numpred.test.0")

print "getting lexicon.."
lexicon = dataset.getLexicon()

print "fitting.."
encoder = LabelEncoder()
encoder.fit(list(lexicon))

print "encoding.."
dataset.encode(encoder)
print dataset.x[:5]