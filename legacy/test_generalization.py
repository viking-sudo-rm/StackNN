import os
import re
import time

from models import VanillaModel, BufferedModel
from controllers import LinearSimpleStructController, LSTMSimpleStructController
from stacknn_utils import StringLogger as Logger
from structs import Stack, NullStruct
from tasks.configs import *

model_names = {
    "VanillaModel": "Vanilla",
    "BufferedModel": "Buffered"
}

controller_names = {
    "LinearSimpleStructController": "Linear",
    "LSTMSimpleStructController": "LSTM"
}

model_objects = {
    "Vanilla": VanillaModel,
    "Buffered": BufferedModel
}

controller_objects = {
    "Linear": LinearSimpleStructController,
    "LSTM": LSTMSimpleStructController
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
            model_names[components[1]],
            controller_names[components[2]],
            components[3])


if __name__ == "__main__":
    # Do all the testing
    folders = next(os.walk("stacknn-experiments"))[1]
    logger = Logger()

    print "Begin testing"
    start_time = time.time()

    for folder in folders:
        folder_start_time = time.time()
        conditions = parse_folder_name(folder)
        task_name, model_name, controller_name, struct_name = conditions

        model = model_objects[model_name]
        controller = controller_objects[controller_name]
        struct = struct_objects[struct_name]

        testing_data = "data/testing/final/" + task_name + ".csv"
        task_configs = config_dicts[task_name]

        print "Conditions:" + ",".join(conditions)

        for i in xrange(10):
            trial_start_time = time.time()
            print "Trial {}".format(i)
            filename = "stacknn-experiments/" + folder + "/" + str(i) + ".dat"
            if os.path.exists(filename) and os.path.isfile(filename):
                task_type = task_configs["task"]
                configs = {k: task_configs[k] for k in task_configs if
                           k != "task"}

                configs["load_path"] = filename
                configs["model_type"] = model
                configs["controller_type"] = controller
                configs["struct_type"] = struct

                log_filename = filename[:-4] + "_log.csv"

                task = task_type(**configs)
                task.run_test(testing_data, log_file=log_filename)

            end_time = time.time()
            print "Trial time: {:.3f} seconds".format(end_time -
                                                      trial_start_time)

        end_time = time.time()
        print "Condition time: {:.3f} seconds".format(end_time -
                                                      folder_start_time)

    print "Testing complete!"
    end_time = time.time()
    print "Time elapsed: {:.3f} seconds".format(end_time - start_time)

    # Parse the logs
    string_io = logger.logger
    del logger
    log = string_io.getvalue().split("\n")

    condition_re = r"^Conditions:.+,.+,.+,.+$"
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

    results.append(curr_condition + curr_trials)

    # Save logs as a CSV file
    f = open("stacknn-experiments/generalization_results.csv", "w")

    f.write("Task,Model,Controller,Struct,")
    f.write(",".join(["Trial " + str(i) for i in xrange(10)]))
    f.write("\n")

    for r in results:
        f.write(",".join([str(c).capitalize() for c in r]))
        f.write("\n")

    f.close()
