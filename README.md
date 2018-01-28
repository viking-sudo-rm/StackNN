# StackNN
A PyTorch implementation of differentiable stacks for use in neural networks. Inspired by https://arxiv.org/pdf/1506.02516.pdf.

## How to use

For those of you who are interested in trying to train the stack model 
on various tasks, here is the information you need to know:

* stack.py implements the stack data structure.
* model.py implements a feed-forward controller network. You should
call forward() on every input and init_stack() whenever you want to
reset the stack between inputs. Since the model is implemented in the
standard format for PyTorch models, it might be useful to look at a
PyTorch hello world example to see how to use it.

I think it's a good idea to try to get the model
running on a single task like string copying. For string copying, you
don't need an annotated data set, but can just generate a list of
~10,000 Python strings on the alphabet {0, 1}. The strings should be
of varying length, and the task is as follows:

~~~~
i:       0 1 2 3 4 5 6 7
x:       1 0 0 1 - - - -
y:       - - - - 1 0 0 1
~~~~

For anyone who's interested in trying to get a more linguistically
interesting task working, there's also a dataset for agreement in the
folder rnn_agr_simple. Let me know if you are interested in working on
one of these tasks or if you find any issues with the code I've
written.
