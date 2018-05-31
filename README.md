# StackNN
A PyTorch implementation of differentiable stacks for use in neural networks. Inspired by https://arxiv.org/pdf/1506.02516.pdf.

## How to use

To train the stack model
on various tasks, here is what you need to know:

* `structs/stack.py` implements the stack data structure. You will probably not be interacting with this object directly.
* Classes in the `models` package implement various types of controller networks.
call `forward()` on every input and `init_stack()` whenever you want to
reset the stack between inputs. Since the model is implemented according to the standard PyTorch object-oriented paradigm, it might be useful to look at a PyTorch hello world example to see how to use it.

## Model

Please report any bugs in the GitHub issues tracker.

Some planned changes are:
* Initialize stack memory block to a parameterized constant size rather than concating repeatedly.
* Fix the LSTM controller (see [PyTorch documentation on LSTMs](http://pytorch.org/docs/master/nn.html)).

## Tasks

### String reversal

Use `reverse_experiment.py` to train a feed-forward controller network to do string reversal. The code generates a list of 800 Python strings on the alphabet {0, 1} with length normally distributed around 10. The task is as follows:

~~~~
i:       0 1 2 3 4 5 6 7
x:       1 1 0 1 - - - -
y:       - - - - 1 0 1 1
~~~~

In 10 epochs, the model tends to achieve 100% accuracy. Since the dataset it is learning is randomly generated each run, the model will sometimes get stuck around 60%. Note that these results were achieved before we modularized the code into the new object-oriented paradigm for future development. If you are unable to replicate these results, please let us know.

### Context-free language modelling

We also have an experiment that trains a context-free language model. This can be used to probe interesting questions about structure. For example, it can be used to predict closing parentheses in a Dijk language. On this task, our stack model converges to 100% accuracy.

### Tree automata evaluation

Yiding is working on implementing a task where the stack is used to evaluate the largest spanned constituent for strings according to a tree automata. This will let us train a network to evaluate Polish notion boolean formulae, which is an especially interesting novel task to try.

### Other tasks

As far as more linguistically interesting tasks, there's also a dataset for agreement in the
folder rnn_agr_simple. We discussed other tasks in CLAY meetings that I will write down here at some point.
