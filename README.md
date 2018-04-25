# StackNN
A PyTorch implementation of differentiable stacks for use in neural networks. Inspired by https://arxiv.org/pdf/1506.02516.pdf.

## How to use

To train the stack model
on various tasks, here is what you need to know:

* `stack.py` implements the stack data structure.
* `model.py` implements a feed-forward controller network. You should
call `forward()` on every input and init_stack() whenever you want to
reset the stack between inputs. Since the model is implemented according to the standard PyTorch object-oriented paradigm, it might be useful to look at a PyTorch hello world example to see how to use it.

## The stack

It's possible that there are still bugs in `stack.py`, and there are definitely inefficiencies. The more pairs of eyes that read through the stack implementation, the better it gets.

See the latest CLAY email for an up-to-date list of tasks.

Some other things that can be done are:
* Initialize stack memory block to a parameterized constant size rather than concating repeatedly.
* Fix the LSTM (see [PyTorch documentation on LSTMs](http://pytorch.org/docs/master/nn.html)).
* Incorporate latest functionality from reverse.py into modularized version (tasks/reverse.py) -- namely the visualization of the stack.

## The controller

I am planning on trying a controller that reads the input from a pre-initialized buffer queue. This would allow the controller to learn epsilon transitions and also nicely parallels the structure of shift-reduce parsing. The resulting architecture seems fairly elegant and it should be easy enough to implement and train; I'm excited to see what performance is like.

## Tasks

### String reversal

In `reverse.py`, I train a feed-forward controller network to do string reversal. I generate a list of 800 Python strings on the alphabet {0, 1} with length normally distributed around 10. The task is as follows:

~~~~
i:       0 1 2 3 4 5 6 7
x:       1 1 0 1 - - - -
y:       - - - - 1 0 1 1
~~~~

In 10 epochs, the model tends to achieve 100% accuracy. Since the dataset it is learning is randomly generated each run, the model will sometimes get stuck in the 60%s.

### Context-free language modelling

We also have an experiment that trains a context-free language model. This can be used to probe interesting questions about structure. For example, it can be used to predict closing parentheses in a Dijk language. On this task, our stack model converges to 100% accuracy.

As far as more linguistically interesting tasks, there's also a dataset for agreement in the
folder rnn_agr_simple. We discussed other tasks in CLAY meetings that I will write down here at some point.

### Tree automata evaluation

Yiding is working on implementing a task where the stack is used to evaluate the largest spanned constituent for strings according to a tree automata. This will let us train a network to evaluate Polish notion boolean formulae, which is an especially interesting novel task to try.
