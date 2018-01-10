from __future__ import with_statement
from sklearn.preprocessing import LabelEncoder

class Dataset:

	def __init__(self, filename):
		with open(filename, "r") as fh:
			self.x, self.y = [], []
			for line in fh.readlines():
				tokens = line.strip().split(" ")
				if len(tokens) < 2: continue
				self.x.append(tokens[1:])
				self.y.append(tokens[0])

	def getLexicon(self):
		""" Should use this function on all relevant datasets to build an encoder. """
		lexicon = set()
		for sent in self.x:
			lexicon |= set(sent)
		return lexicon

	def encode(self, encoder):
		""" Takes a LabelEncoder to convert tokens to numbers. """
		self.x = map(encoder.transform, self.x)

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