"""

Run a task defined in tasks.

Example usage:
  python run.py reverse_config
  python run.py dyck_config
  python run.py dyck_config --controller BufferedController

"""

import argparse

from models import *
from models.networks.base import SimpleStructNetwork
from models.networks.feedforward import LinearSimpleStructNetwork
from models.networks.recurrent import *
from tasks import *
from tasks.configs import *

def get_args():
    parser = argparse.ArgumentParser(
        description="Run a task and customize hyperparameters.")
    parser.add_argument("config", type=str)

    # Path arguments for loading and saving models.
    parser.add_argument("--loadpath", type=str, default=None)
    parser.add_argument("--savepath", type=str, default=None)

    # Manually specified parameters override those in configs.
    parser.add_argument("--controller", type=str, default=None)
    parser.add_argument("--network", type=str, default=None)

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


if __name__ == "__main__":

    args = get_args()
    print("Loading {} Config".format(args.config))

    # Parse config.
    config = get_object_from_arg(args.config, dict)
    config = dict(config) # Get copy of config.
    task = config["task"]
    del config["task"]

    controller = get_object_from_arg(args.controller, AbstractController)
    network = get_object_from_arg(args.network, SimpleStructNetwork)

    if controller is not None:
        config["model_type"] = controller
    if network is not None:
        config["network_type"] = network

    if args.loadpath is not None:
        config['load_path'] = args.loadpath
    if args.savepath is not None:
        config['save_path'] = args.savepath

    task(**config).run_experiment()
