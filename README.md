# StackNN
A PyTorch implementation of differentiable stacks for use in neural networks. Inspired by [Grefenstette et al., 2015](https://arxiv.org/pdf/1506.02516.pdf).

Please report any bugs in the GitHub issues tracker.

## Models

Models implement the high-level controllers that interact with the stack. There are several different types of models, but the simplest one is, as the name implies, the `vanilla` one.

To use a model, call `model.forward()` on every input and `model.init_stack()` whenever you want to reset the stack between inputs.

## Data structures

* `structs.Stack` implements the differentiable stack data structure.
* `structs.Queue` implements the differentiable queue data structure.

The buffered models use read-only and write-only versions of the differentiable queue for their input and output buffers.

## Tasks

### String reversal

The `ReverseTask` trains a feed-forward controller network to do string reversal. The code generates a list of 800 Python strings on the alphabet {0, 1} with length normally distributed around 10. The task is as follows:

~~~
i:       0 1 2 3 4 5 6 7
x:       1 1 0 1 - - - -
y:       - - - - 1 0 1 1
~~~

In 5 epochs, the model tends to achieve 100% accuracy. To run the task for yourself, you can do:

~~~bash
python run.py ReverseTask
~~~

### Context-free language modelling

`CFGTask` can be used to train a context-free language model. Many interesting questions probing linguistic structure can be reduced to special cases of this general task. For example, the task can be used to predict closing parentheses in a Dijk language, which requires some notion of recursive depth. On this task, our stack model converges to 100% accuracy fairly quickly. You can run the Dijk task with:

~~~bash
python run.py CFGTask --config dijk_config
~~~

### Tree automata evaluation

Yiding is working on implementing a task where the stack is used to evaluate the largest spanned constituent for strings according to a tree automata. This will let us train a network to evaluate Polish notion boolean formulae.

### Other tasks

The data folder contains several real datasets that the stack can be applied to.
