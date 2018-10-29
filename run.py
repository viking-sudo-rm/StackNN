"""Run a task defined in tasks.

Example usage:
  python run.py reverse_config
  python run.py dyck_config
  python run.py dyck_config --model BufferedModel

"""

import argparse
from copy import copy

from tasks import Task
from models import *
from controllers import *
from structs import *
from configs import *
from visualization import *


def get_args():
    parser = argparse.ArgumentParser(
        description="Run a task and customize hyperparameters.")
    parser.add_argument("config", type=str)

    # Manually specified parameters override those in configs.
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--controller", type=str, default=None)
    parser.add_argument("--struct", type=str, default=None)
    parser.add_argument("--visualizer", type=str, default=None)

    # Path arguments for loading and saving models.
    parser.add_argument("--loadpath", type=str, default=None)
    parser.add_argument("--savepath", type=str, default=None)

    return parser.parse_args()


def get_object_from_arg(arg, superclass, default=None):
    """
    Verify that arg refers to an instance of superclass.

    If so, return the instance.
    Otherwise, throw an error.
    """
    if arg is None:
        return default
    if arg not in globals():
        raise ValueError("Invalid argument {}".format(arg))
    obj = globals()[arg]
    if not (isinstance(obj, superclass) or issubclass(obj, superclass)):
        raise TypeError("{} is not a {}".format(arg, str(superclass)))
    return obj


def main(config,
         model_type=None,
         controller_type=None,
         struct_type=None,
         visualizer_type=None,
         load_path=None,
         save_path=None):
    config = copy(config)

    if model_type is not None:
        config["model_type"] = model_type
    if controller_type is not None:
        config["controller_type"] = controller_type
    if struct_type is not None:
        config["struct_type"] = struct_type

    if load_path is not None:
        config["load_path"] = load_path
    if save_path is not None:
        config["save_path"] = save_path

    task = Task.from_config_dict(config)
    metrics = task.run_experiment()
    if visualizer_type is not None:
        visualizer = visualizer_type(task)
        visualizer.visualize_generic_example()
    return metrics


if __name__ == "__main__":
    args = get_args()
    print("Loading {}".format(args.config))
    config = get_object_from_arg(args.config, dict)
    model_type = get_object_from_arg(args.model, Model)
    controller_type = get_object_from_arg(args.controller, SimpleStructController)
    struct_type = get_object_from_arg(args.struct, Struct)
    visualizer_type = get_object_from_arg(args.visualizer, Visualizer)

    main(config, model_type, controller_type, struct_type, visualizer_type,
         args.loadpath, args.savepath)
