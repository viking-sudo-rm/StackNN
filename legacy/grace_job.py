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
from __future__ import print_function

import os

import run
from models import *
from controllers import *
from stacknn_utils import FileLogger as Logger
from structs import Stack, NullStruct
from tasks.configs import *

n_trials = 10

# TODO: These should be set by flags or something.
results_dir = "stacknn-experiments"

"""
EXPERIMENTAL CONDITIONS: Please comment out all items below except for
those corresponding to the experimental trials that you are running.
"""

# Task config dicts
configs = [
    # ("reverse", final_reverse_config),
    # ("parity", final_parity_config),
    # ("delayed_parity", final_delayed_parity_config),
    # ("dyck", final_dyck_config),
    ("agreement", final_agreement_config),
<<<<<<< HEAD
    # ("formula", final_formula_config)
=======
    ("agreement10", final_agreement_config_10),
    ("formula", final_formula_config)
>>>>>>> 280c714949b6d67d317b456b3ffeb1bc1830fb48
]

# Vanilla vs. Buffered Model
model_types = [
    VanillaModel,
    # BufferedModel,
]

# Linear vs. LSTM Controller
controller_types = [
    LinearSimpleStructController,
    LSTMSimpleStructController,
]

# Stack vs. no Stack
struct_types = [
    Stack,
    # NullStruct,
]

""" PLEASE DO NOT EDIT BELOW THIS LINE """

output_file_name = "-".join([c[0] for c in configs] +
                            [c.__name__ for c in model_types] +
                            [n.__name__ for n in controller_types] +
                            [s.__name__ for s in struct_types])
output_file_name = "stacknn-experiments/log-" + output_file_name + ".txt"
logger = Logger(output_file_name)

for config_name, config in configs:
    for model_type in model_types:
        for controller_type in controller_types:
            for struct_type in struct_types:

                experiment_name = "-".join([config_name,
                                            model_type.__name__,
                                            controller_type.__name__,
                                            struct_type.__name__])
                config_dir = os.path.join(results_dir, experiment_name)
                os.makedirs(config_dir)
                final_accs = []

                for i in xrange(n_trials):
                    # TODO: Should export figures, results, logs here too.
                    save_path = os.path.join(config_dir, "%i.dat" % i)
                    results = run.main(config, model_type, controller_type,
                                       struct_type, save_path=save_path)
                    final_accs.append(results["final_acc"])

                print("Trial accuracies:", ["{:.1f}".format(acc) for acc in final_accs])

del logger
