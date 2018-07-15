"""
Template for automating experiments on your own computer or on the Grace
high-performance computing cluster (https://research.computing.yale.edu/
support/hpc/clusters/grace)

You should run parallel instances of this script to do different configs
simultaneously.

INSTRUCTIONS: This script runs experimental trials for all conditions
specified in the configurations in lines 36 to 64. Please comment out
all items in the configurations except for those pertaining to the
trials you are running. Then, run this script from the terminal. Please
report the testing accuracy for the last epoch from each trial in the
"Task Results (Last Epoch)" sheet and upload the saved models, found in
the directory "stacknn-experiments," to the "StackNN Trained Models"
Google Drive folder.
"""

import os

import run
from models import *
from models.networks import *
from structs import Stack, NullStruct
from tasks.configs import *

n_trials = 2

# TODO: These should be set by flags or something.
results_dir = "stacknn-experiments"

"""
EXPERIMENTAL CONDITIONS: Please comment out all items below except for
those corresponding to the experimental trials that you are running.
"""

# Task config dicts
configs = [
    ("reverse", final_reverse_config),
    ("parity", final_parity_config),
    ("delayed_parity", final_delayed_parity_config),
    ("dyck", final_dyck_config),
    ("agreement", final_agreement_config),
    ("formula", final_formula_config)
]

# Vanilla vs. Buffered Controller
controller_types = [
    VanillaController,
    BufferedController,
]

# Linear vs. LSTM Network
network_types = [
    LinearSimpleStructNetwork,
    LSTMSimpleStructNetwork,
]

# Stack vs. no Stack
struct_types = [
    Stack,
    NullStruct,
]

""" PLEASE DO NOT EDIT BELOW THIS LINE """

for config_name, config in configs:
    for controller_type in controller_types:
        for network_type in network_types:
            for struct_type in struct_types:

                experiment_name = "-".join([config_name,
                                            controller_type.__name__,
                                            network_type.__name__,
                                            struct_type.__name__])
                config_dir = os.path.join(results_dir, experiment_name)
                os.makedirs(config_dir)

                for i in xrange(n_trials):
                    # TODO: Should export figures, results, logs here too.
                    save_path = os.path.join(config_dir, "%i.dat" % i)
                    run.main(config, controller_type, network_type,
                             struct_type, save_path=save_path)
