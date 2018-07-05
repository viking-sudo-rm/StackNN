# StackNN
A PyTorch implementation of differentiable stacks for use in neural networks. Inspired by [Grefenstette et al., 2015](https://arxiv.org/pdf/1506.02516.pdf).

Please report any bugs in the GitHub issues tracker.

## Dependencies

Python 2.7 is supported. A possibly incomplete list of dependencies is:
* pytorch
* numpy
* matplotlib
* enum

## Models

Models implement the high-level controllers that use a stack for recurrent memory. You can think of these networks like LSTMs with a more sophisticated storage mechanism to pass data between time steps.

* `models.VanillaController` is the simplest controller network.
* `models.EmbeddingController` is a controller with an initial embedding layer.
* `models.BufferedController` implements the more complicated buffered architecture.

To use a model, call `model.forward()` on every input and `model.init_controller()` whenever you want to reset the stack between inputs. You can find example training logic in the `tasks` package.

## Data structures

* `structs.Stack` implements the differentiable stack data structure.
* `structs.Queue` implements the differentiable queue data structure.

The buffered models use read-only and write-only versions of the differentiable queue for their input and output buffers.

## Tasks

To run a task, specify a config name defined in `tasks.configs`:

~~~bash
python run.py reverse_config
~~~

You can pass a file path in which to save the model parameters:
~~~bash
python run.py reverse_config --savepath "saved_models/my_run_parameters"
~~~
Parameters are saved at the end of each epoch.

You can also pass a file path to load model parameters from a previous run:
~~~bash
python run.py reverse_Config --loadpath "saved_models/previous_run"
~~~

### String reversal

The `ReverseTask` trains a feed-forward controller network to do string reversal. The code generates a list of 800 Python strings on the alphabet {0, 1} with length normally distributed around 10. The task is as follows:

~~~
i:       0 1 2 3 4 5 6 7
x:       1 1 0 1 - - - -
y:       - - - - 1 0 1 1
~~~

By 10 epochs, the model tends to achieve 100% accuracy. To run the task for yourself, you can do:

~~~bash
python run.py reverse_config
~~~

### Context-free language modelling

`CFGTask` can be used to train a context-free language model. Many interesting questions probing linguistic structure can be reduced to special cases of this general task. For example, the task can be used to predict closing parentheses in a Dyck language (matching parentheses), which requires some notion of recursive depth. On this task, our stack model converges to 100% accuracy fairly quickly. You can run the Dyck task with:

~~~bash
python run.py dyck_config
~~~

### Evaluation tasks

We also have a class for evaluation tasks. These are tasks where output i can be succintly expressed as some function of inputs 0, .., i. Some applications of this are evaluation of parity and reverse polish boolean formulae.

### Real datasets

The data folder contains several real datasets that the stack can be trained on. We should implement a task for reading in these datasets.
