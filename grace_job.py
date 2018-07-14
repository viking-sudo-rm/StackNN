"""Template for Grace job to automate experiments.

Should run parallel instances of this script to do different configs simultaneously.

"""

import os

from models import *
from models.networks import *
import run
from structs import *
from tasks.configs import *


N_TRIALS = 10

# TODO: These should be set by flags or something.
results_dir = "stacknn-experiments"

configs = [
    ("reverse", reverse_config),
    ("parity", parity_config),
    ("delayed_parity", delayed_parity_config),
    ("dyck", dyck_config),
    ("agreement", agreement_config),
    ("formula", formula_config),
]

controller_types = [
    VanillaController,
    BufferedController,
]

network_types = [
    LinearSimpleStructNetwork,
    RNNSimpleStructNetwork,
    LSTMSimpleStructNetwork,
    GRUSimpleStructNetwork,
]

struct_types = [
    Stack,
    NullStruct,
]


for config_name, config in configs:
    for controller_type in controller_types:
        for network_type in network_types:
            for struct_type in struct_types:

                experiment_name = "-".join(config_name, controller_type.__name__, network_type.__name__, struct_type.__name__)
                config_dir = os.path.join(results_dir, experiment_name)
                os.makedirs(config_dir)

                for i in range(N_TRIALS):
                    # TODO: Should export figures, results, logs here too.
                    save_path = os.path.join(config_dir, "%i.dat" % i)
                    run.main(config, controller_type, network_type, struct_type, save_path=save_path)