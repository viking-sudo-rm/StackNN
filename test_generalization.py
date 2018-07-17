import os
import re

from models import VanillaController, BufferedController
from models.networks import LinearSimpleStructNetwork, LSTMSimpleStructNetwork
from stacknn_utils import StringLogger as Logger
from structs import Stack, NullStruct
from tasks.configs import *

controller_names = {
    "VanillaController": "Vanilla",
    "BufferedController": "Buffered"
}

network_names = {
    "LinearSimpleStructNetwork": "Linear",
    "LSTMSimpleStructNetwork": "LSTM"
}

controller_objects = {
    "Vanilla": VanillaController,
    "Buffered": BufferedController
}

network_objects = {
    "Linear": LinearSimpleStructNetwork,
    "LSTM": LSTMSimpleStructNetwork
}

struct_objects = {
    "Stack": Stack,
    "NullStruct": NullStruct
}

config_dicts = {
    "agreement": testing_agreement_config,
    "agreement10": testing_agreement_config_10,
    "delayed_parity": testing_delayed_parity_config,
    "dyck": testing_dyck_config,
    "formula": testing_formula_config,
    "parity": testing_parity_config,
    "reverse": testing_reverse_config
}


def parse_folder_name(fn):
    print fn
    components = fn.split("-")

    return (components[0],
            controller_names[components[1]],
            network_names[components[2]],
            components[3])


if __name__ == "__main__":
    # Do all the testing
    folders = next(os.walk("stacknn-experiments"))[1]
    logger = Logger()

    print "Begin testing"

    for folder in folders:
        conditions = parse_folder_name(folder)
        task_name, controller_name, network_name, struct_name = conditions

        controller = controller_objects[controller_name]
        network = network_objects[network_name]
        struct = struct_objects[struct_name]

        testing_data = "data/testing/final/" + task_name + ".csv"
        task_configs = config_dicts[task_name]

        print "Conditions:" + ",".join(conditions)

        for i in xrange(10):
            print "Trial {}".format(i)
            filename = "stacknn-experiments/" + folder + "/" + str(i) + ".dat"
            if os.path.exists(filename) and os.path.isfile(filename):
                task_type = task_configs["task"]
                configs = {k: task_configs[k] for k in task_configs if
                           k != "task"}

                configs["load_path"] = filename
                configs["model_type"] = controller
                configs["network_type"] = network
                configs["struct_type"] = struct

                task = task_type(**configs)
                task.run_test(testing_data)

    print "Testing complete!"

    # Parse the logs
    string_io = logger.logger
    del logger
    log = string_io.getvalue().split("\n")

    condition_re = r"^Conditions:[a-zA-Z]+,[a-zA-Z]+,[a-zA-Z]+,[a-zA-Z]+$"
    trial_re = r"^Trial \d$"
    result_re = r"^Test Results: Loss = [\d\.]+, Accuracy = [\d\.]+%$"
    result_sub_re = r"^Test Results: Loss = [\d\.]+, Accuracy = "

    results = []
    curr_condition = []
    curr_trials = []
    curr_trial = -1
    for line in log:
        if re.match(condition_re, line):
            # Start a new row
            results.append(curr_condition + curr_trials)
            curr_condition = re.sub(r"^Conditions:", "", line).split(",")
            curr_trials = ["" for _ in xrange(10)]
        elif re.match(trial_re, line):
            # Update current trial number
            curr_trial = int(line[-1])
        elif re.match(result_re, line):
            # Save the current trial result
            result = float(re.sub(result_sub_re, "", line)[:-1]) / 100.
            curr_trials[curr_trial] = result

    # Save logs as a CSV file
    f = open("stacknn-experiments/generalization_results.csv", "w")

    f.write("Task,Controller,Network,Struct,")
    f.write(",".join(["Trial " + str(i) for i in xrange(10)]))
    f.write("\n")

    for r in results:
        f.write(",".join([str(c).capitalize() for c in r]))
        f.write("\n")

    f.close()
