from __future__ import division

import torch.nn as nn

from tasks.reverse import ReverseTask
from models.vanilla import Controller

read_size = 1
task = ReverseTask(min_length=1,
                   mean_length=10,
                   std_length=2,
                   max_length=12,
                   learning_rate=0.1,
                   batch_size=10,
                   read_size=read_size,
                   cuda=False,
                   epochs=100,
                   model=Controller(3, read_size, 3),
                   criterion=nn.CrossEntropyLoss(),
                   verbose=True)

task.run_experiment()
