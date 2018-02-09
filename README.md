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

Due to the poor performance that the model achieves on string reversal (~66%), it's likely that there are still bugs in the implementation in stack.py. Having another pair of eyes to read through this implementation or suggest improvements would be immensely helpful.

It would also be nice if someone could write an LSTM controller network (see [PyTorch documentation on LSTMs](http://pytorch.org/docs/master/nn.html)).

## Tasks

I think it's a good idea to try to get the model
running on a single task like string reversal. For string reversal, you
don't need an annotated data set, but can just generate a list of Python strings on the alphabet {0, 1}. The strings should be
of varying length, and the task is as follows:

~~~~
i:       0 1 2 3 4 5 6 7
x:       1 1 0 1 - - - -
y:       - - - - 1 0 1 1
~~~~

As far as more linguistically interesting tasks, there's also a dataset for agreement in the
folder rnn_agr_simple.