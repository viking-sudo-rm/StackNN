# StackNN
A PyTorch implementation of several differentiable data structures for use in recurrent neural networks. The code in this project is associated with [Context-Free Transductions with Neural Stacks](https://arxiv.org/abs/1809.02836), which will appear at the Analyzing and Interpreting Neural Networks for NLP workshop at EMNLP 2018.

A differentiable data structure is a version of a conventional data structure whose interface can be connected to a neural network. Our stacks, queues, and dequeues are inspired by the formalism presented by [Grefenstette et al., 2015](https://arxiv.org/pdf/1506.02516.pdf). We also implement several different models using these structures and tasks that the models can be trained on. See the paper for more information.

## Running a demo

There are several experiment configurations pre-defined in [configs.py](configs.py). To train a model on one of these configs, do:

```shell
python run.py CONFIG_NAME
```

For example, to train a model on the string reversal task:

```shell
python run.py final_reverse_config
```

In addition to experiment config, [run.py](run.py) takes several flags:
* `--model`: Model type (`BufferedModel` or `VanillaModel`)
* `--controller`: Controller type (`LinearSimpleStructController`, `LSTMSimpleStructController`, etc.)
* `--struct`: Struct type (`Stack`, `NullStruct`, etc.)
* `--savepath`: Path for saving a trained model
* `--loadpath`: Path for loading a model

## Documentation

You can find auto-generated documentation [here](https://stacknn.readthedocs.io/en/latest/index.html).

## Contributing

This project is managed by [Computational Linguistics at Yale](http://clay.yale.edu/). We welcome contributions from outside in the form of pull requests. Please report any bugs in the GitHub issues tracker.

## Citations

Please cite our paper:

```
@article{hao2018context,
  title={Context-Free Transductions with Neural Stacks},
  author={Hao, Yiding and Merrill, William and Angluin, Dana and Frank, Robert and Amsel, Noah and Benz, Andrew and Mendelsohn, Simon},
  journal={arXiv preprint arXiv:1809.02836},
  year={2018}
}
```

## Dependencies

Python 2.7 with PyTorch 0.4.1 is supported. A possibly incomplete list of dependencies is:
* PyTorch
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

### String reversal

The `ReverseTask` trains a feed-forward controller network to do string reversal. The code generates a list of 800 Python strings on the alphabet {0, 1} with length normally distributed around 10. The task is as follows:

~~~
i:       0 1 2 3 4 5 6 7
x:       1 1 0 1 - - - -
y:       - - - - 1 0 1 1
~~~

By 10 epochs, the model tends to achieve 100% accuracy. The config for this task is called `final_reverse_config`.

### Context-free language modelling

`CFGTask` can be used to train a context-free language model. Many interesting questions probing linguistic structure can be reduced to special cases of this general task. For example, the task can be used to predict closing parentheses in a Dyck language (matching parentheses), which requires some notion of recursive depth. On this task, our stack model converges to 100% accuracy fairly quickly. The config for this task is called `final_dyck_config`.

### Evaluation tasks

We also have a class for evaluation tasks. These are tasks where output i can be succintly expressed as some function of inputs 0, .., i. Some applications of this are evaluation of parity and reverse polish boolean formulae.

### Real datasets

The data folder contains several real datasets that the stack can be trained on. We should implement a task for reading in these datasets.
