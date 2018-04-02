from __future__ import division

import torch.nn as nn

from tasks.reverse import ReverseTask

m = __import__("model-bare")

learning_rate = 0.1
batch_size = 10
read_size = 1
epochs = 100

task = ReverseTask(min_length=1,
                   mean_length=10,
                   std_length=2,
                   max_length=12,
                   learning_rate=learning_rate,
                   batch_size=batch_size,
                   read_size=read_size,
                   cuda=False,
                   epochs=epochs,
                   model=m.FFController(3, 1, 3),
                   criterion=nn.CrossEntropyLoss(),
                   verbose=True)

task.run_experiment()
