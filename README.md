# StackNN
A PyTorch implementation of differentiable stacks for use in neural networks. Inspired by https://arxiv.org/pdf/1506.02516.pdf.

## How to use

To train the stack model 
on various tasks, here is what you need to know:

* stack.py implements the stack data structure.
* model.py implements a feed-forward controller network. You should
call forward() on every input and init_stack() whenever you want to
reset the stack between inputs. Since the model is implemented in the
standard format for PyTorch models, it might be useful to look at a
PyTorch hello world example to see how to use it.

## Improving the model

It's possible that there are still bugs in stack.py, and there are definitely inefficiencies. The more pairs of eyes that read through the stack implementation, the better it gets.

Some specific things that can be done are:
* Initialize stack memory block to a parameterized constant size rather than concating repeatedly.
* Implement an LSTM controller network. This should be pretty simple using the built-in recurrent architectures in PyTorch (see [PyTorch documentation on LSTMs](http://pytorch.org/docs/master/nn.html)).
* Get rid of for loops in stack.py? Not sure how necessary this is, but could add some benefits towards parallelization.

## Tasks

In reverse.py, I train a feed-forward controller network to do string reversal. I generate a list of 800 Python strings on the alphabet {0, 1} with length normally distributed around 10. The task is as follows:

~~~~
i:       0 1 2 3 4 5 6 7
x:       1 1 0 1 - - - -
y:       - - - - 1 0 1 1
~~~~

In 30 epochs, the network tends to achieve over 95% test accuracy. Since the dataset it is learning is randomly generated each run, the model will sometimes get stuck.

As far as more linguistically interesting tasks, there's also a dataset for agreement in the
folder rnn_agr_simple. We discussed other tasks in CLAY meetings that I will write down here at some point.
