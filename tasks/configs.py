""" Defines Task parameters for notable experiment configurations.

Notes:
    * A "task" MUST be specified in the config dictionary.
    * Other values are optional and task-specific. They are used to specify values for parameters in a Task constructor.
    * Network_type, model_type, and struct_type should not be specified here. They can be set with command-line arguments.

A config can be run with:
    python run.py CONFIG_NAME

"""

from formalisms.cfg import *
from models.networks.recurrent import LSTMSimpleStructNetwork, RNNSimpleStructNetwork, GRUSimpleStructNetwork
from tasks import *
from structs import *


# 1) Reverse task.
reverse_config = {
    "task": ReverseTask,
}

# 2) XOR/parity evaluation task.
parity_config = {
    "task": XORTask,
    "read_size": 6,
}

# 3) Delayed XOR/parity evaluation task.
delayed_parity_config = {
    # TODO: Specify this.
}

# 4) Dyck language modeling task.
dyck_config = {
    "task": CFGTask,
    "grammar": dyck_grammar,
    "to_predict": [u")", u"]"],
    "sample_depth": 5,
}

# 5) Agreement grammar task.
agreement_config = {
    "task": CFGTask,
    "grammar": agreement_grammar,
    "to_predict": [u"Auxsing", u"Auxplur"],
    "sample_depth": 8,
}

# 6) Reverse Polish notation formula task.
formula_config = {
    "task": CFGTransduceTask,
    "grammar": exp_eval_grammar,
    "sample_depth": 6,
    "to_predict": [u"0", u"1"]
}


# =====================================================
# The following configs are not included in the paper.


# Reverse task formulated as CFG.
reverse_cfg = {
    "task": CFGTask,
    "grammar": reverse_grammar,
    "to_predict": [u"a1", u"b1"],
    "sample_depth": 12,
}

# Unambiguous agreement grammar task.
unambig_agreement_config = {
    "task": CFGTask,
    "grammar": unambig_agreement_grammar,
    "to_predict": [u"Auxsing", u"Auxplur"],
    "sample_depth": 16,
}

# Buffered parity evaluation with t steps.
parity_config_t = {
    "task": XORTask,
    "read_size": 6,
    "time_function": lambda t: t,
}
