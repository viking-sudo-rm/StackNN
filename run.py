"""Run a task defined in tasks.

Example usage:
  python run.py reverse_config
  python run.py dyck_config
  python run.py dyck_config --controller BufferedController

"""

import argparse
from copy import copy

from models import *
from models.networks import *
from structs import *


def get_args():
    parser = argparse.ArgumentParser(
        description="Run a task and customize hyperparameters.")
    parser.add_argument("config", type=str)

    # Manually specified parameters override those in configs.
    parser.add_argument("--controller", type=str, default=None)
    parser.add_argument("--network", type=str, default=None)
    parser.add_argument("--struct", type=str, default=None)

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
         controller_type=None,
         network_type=None,
         struct_type=None,
         load_path=None,
         save_path=None):
    config = copy(config)

    task = config["task"]
    del config["task"]

    if controller_type is not None:
        config["model_type"] = controller_type
    if network_type is not None:
        config["network_type"] = network_type
    if struct_type is not None:
        config["struct_type"] = struct_type

    if load_path is not None:
        config["load_path"] = load_path
    if save_path is not None:
        config["save_path"] = save_path

    task(**config).run_experiment()


if __name__ == "__main__":
    args = get_args()
    print("Loading {}".format(args.config))
    config = get_object_from_arg(args.config, dict)
    controller_type = get_object_from_arg(args.controller, AbstractController)
    network_type = get_object_from_arg(args.network, SimpleStructNetwork)
    struct_type = get_object_from_arg(args.struct, Struct)

    main(config, controller_type, network_type, struct_type, args.loadpath,
         args.savepath)
