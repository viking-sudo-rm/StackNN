""" Run a task defined in tasks.

Args:
  * task: Name of a Task class defined in tasks.

Example usage:
  python run.py ReverseTask
  python run.py CFGTask --config dyck_config
  python run.py CFGTask --agreement_config

"""

import argparse

from tasks import *
from tasks.configs import *


def get_args():
    parser = argparse.ArgumentParser(description="Run a task and customize hyperparameters.")
    parser.add_argument("task", type=str)
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    if args.task not in globals():
        raise ValueError("Unknown task {}".format(args.task))
    task = globals()[args.task]
    if not issubclass(task, Task):
        raise ValueError("{} is not Task".format(args.task))
    
    if args.config is None:
        config = {}
    else:
        if args.config not in globals():
            raise ValueError("Unknown parameter configuration {}".format(args.config))
        config = globals()[args.config]
        if not isinstance(config, dict):
            raise ValueError("{} is not a dictionary".format(args.config))

    task(**config).run_experiment()
