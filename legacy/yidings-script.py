from __future__ import division

from models import VanillaModel, BufferedModel
from controllers import *
from tasks.cfg import CFGTask
from tasks.configs import *

vanilla = VanillaModel
buffered = BufferedModel

linear = LinearSimpleStructController
lstm = LSTMSimpleStructController
rnn = RNNSimpleStructController

configs = final_delayed_parity_config  # dyck_config
del configs["task"]
configs["epochs"] = 1
configs[
    "load_path"] = "stacknn-experiments0/delayed_parity-VanillaModel" \
                   "-LinearSimpleStructController-Stack/1.dat"
configs["model_type"] = vanilla
configs["controller_type"] = linear

task = DelayedXORTask(**configs)

# task.run_experiment()daefy

trace_X, _ = task.get_tensors(1)
sentence = task.one_hot_to_sentences(trace_X.size(1),trace_X)
print task.sentences_to_text(*sentence)

if isinstance(task.model, buffered):
    task.model.trace(trace_X, 30)
else:
    task.model.trace(trace_X)
