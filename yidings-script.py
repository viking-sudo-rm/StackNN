from __future__ import division

from models import VanillaController, BufferedController
from models.networks.feedforward import *
from tasks.cfg import CFGTask
from tasks.configs import *

vanilla = VanillaController
buffered = BufferedController

linear = LinearSimpleStructNetwork
lstm = LSTMSimpleStructNetwork
rnn = RNNSimpleStructNetwork

configs = dyck_config
del configs["task"]
configs["epochs"] = 1

task = CFGTask(**configs)

task.run_experiment()

trace_X, _ = task.get_tensors(1)
sentence = task.one_hot_to_sentences(trace_X.size(1),trace_X)
print task.sentences_to_text(*sentence)

if isinstance(task.model, buffered):
    task.model.trace(trace_X, 30)
else:
    task.model.trace(trace_X)
