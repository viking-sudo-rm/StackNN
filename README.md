# StackNN
A PyTorch implementation of differentiable stacks for use in neural networks. Inspired by https://arxiv.org/pdf/1506.02516.pdf.

## Reporting bugs

Please report any bugs in the GitHub issues tracker.

## Models

Models implement the high-level controllers that interact with the stack. There are several different types of models, but the simplest one is, as the name implies, the `vanilla` one.

To use a model, call `forward()` on every input and `init_stack()` whenever you want to reset the stack between inputs.

## Data structures

* `structs.Stack` implements the stack data structure.
* `structs.Queue` implements the queue data structure.

## Tasks

Tasks can be run using run.py. For example:

~~~bash
python run.py ReverseTask
python run.py CFGTask --config agreement_config
~~~

### String reversal

The `ReverseTask` trains a feed-forward controller network to do string reversal. The code generates a list of 800 Python strings on the alphabet {0, 1} with length normally distributed around 10. The task is as follows:

~~~
i:       0 1 2 3 4 5 6 7
x:       1 1 0 1 - - - -
y:       - - - - 1 0 1 1
~~~

In 10 epochs, the model tends to achieve 100% accuracy. Since the dataset it is learning is randomly generated each run, the model will sometimes get stuck around 60%. Note that these results were achieved before we refactored some of the code to be more object-oriented. If you are unable to replicate these results, please let us know.

### Context-free language modelling

We also have an experiment that trains a context-free language model. This can be used to probe interesting questions about structure. For example, it can be used to predict closing parentheses in a Dijk language. On this task, our stack model converges to 100% accuracy.

### Tree automata evaluation

Yiding is working on implementing a task where the stack is used to evaluate the largest spanned constituent for strings according to a tree automata. This will let us train a network to evaluate Polish notion boolean formulae, which is an especially interesting novel task to try.

### Other tasks

As far as more linguistically interesting tasks, there's also a dataset for agreement in the
folder rnn_agr_simple. We discussed other tasks in CLAY meetings that I will write down here at some point.
