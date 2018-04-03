from __future__ import division

import torch.nn as nn
from nltk import CFG

from tasks.cfg import CFGTask

m = __import__("model-bare")

dyck_grammar = CFG.fromstring("""
S -> S S
S -> '(' S ')' | '(' ')' 
S -> '[' S ']' | '[' ']'
""")

read_size = 1
task = CFGTask(dyck_grammar,
               [u")", u"]"],
               5,
               max_length=25,
               learning_rate=.01,
               batch_size=10,
               read_size=2,
               epochs=30,
               model=m.FFController(5, read_size, 5))

task.run_experiment()
