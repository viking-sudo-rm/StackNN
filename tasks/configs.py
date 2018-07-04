""" Defines Task parameters for notable experiment configurations.

Notes:
    * A "task" MUST be specified in the config dictionary.
    * Other values are optional and task-specific. They are used to specify values for parameters in a Task constructor.

A config can be run with:

    python run.py CONFIG_NAME

"""

from formalisms.cfg import *
from models.networks.recurrent import LSTMSimpleStructNetwork, RNNSimpleStructNetwork, GRUSimpleStructNetwork
from tasks import *


# Dyck language task.
dyck_config = {
    "task": CFGTask,
    "grammar": dyck_grammar,
    "to_predict": [u")", u"]"],
    "sample_depth": 5,
}

# Reverse task.
reverse_config = {
    "task": ReverseTask,
}

# Reverse task formulated as CFG.
reverse_cfg = {
    "task": CFGTask,
    "grammar": reverse_grammar,
    "to_predict": [u"a1", u"b1"],
    "sample_depth": 12,
}

# Agreement grammar task.
agreement_config = {
    "task": CFGTask,
    "grammar": agreement_grammar,
    "to_predict": [u"Auxsing", u"Auxplur"],
    "sample_depth": 8,
}

# IDK what this is.
xor_clipping = {
    "clipping_norm": 0.5,
}

# Parity evaluation task with BufferedController.
parity_config = {
    "task": XORTask,
    "read_size": 6,
    "epochs": 30,
    # "time_function": lambda t: 2 * t,
}

