"""
Defines Task parameters for notable experiment configurations.

Notes:
    * A "task" MUST be specified in the config dictionary.
    * Other values are optional and task-specific. They are used to
      specify values for parameters in a Task constructor.
    * Network_type, model_type, and struct_type should not be specified
      here. They can be set with command-line arguments.

A config can be run with:
    python run.py CONFIG_NAME
"""

from formalisms.cfg import *
from tasks import *

""" Configs for the Final Paper """

# 1) Reverse task.
final_reverse_config = {
    "task": ReverseTask,
    "epochs": 100,
    "early_stopping_steps": 5,
    "read_size": 2
}

# 2) XOR/parity evaluation task.
final_parity_config = {
    "task": XORTask,
    "epochs": 100,
    "early_stopping_steps": 5,
    "read_size": 6
}

# 3) Delayed XOR/parity evaluation task.
final_delayed_parity_config = {
    "task": DelayedXORTask,
    "epochs": 100,
    "early_stopping_steps": 5,
    "read_size": 6
}

# 4) Dyck language modeling task.
final_dyck_config = {
    "task": CFGTask,
    "epochs": 100,
    "early_stopping_steps": 5,
    "grammar": dyck_grammar_2,
    "to_predict": [u")", u"]"],
    "sample_depth": 6,
    "read_size": 2
}

# 5) Agreement grammar task.
final_agreement_config = {
    "task": CFGTask,
    "epochs": 100,
    "early_stopping_steps": 5,
    "grammar": unambig_agreement_grammar,
    "to_predict": [u"Auxsing", u"Auxplur"],
    "sample_depth": 16,
    "read_size": 2
}

# 5b) Agreement grammar task with longer early stopping 
final_agreement_config_10 = {
    "task": CFGTask,
    "epochs": 100,
    "early_stopping_steps": 10,
    "grammar": unambig_agreement_grammar,
    "to_predict": [u"Auxsing", u"Auxplur"],
    "sample_depth": 16,
    "read_size": 2
}
    
# 6) Reverse Polish notation formula task.
final_formula_config = {
    "task": CFGTransduceTask,
    "epochs": 100,
    "early_stopping_steps": 5,
    "grammar": exp_eval_grammar,
    "to_predict": [u"0", u"1"],
    "sample_depth": 6,
    "read_size": 2,
    "max_length": 32
}

""" Configs Not Included in the Paper """

old_inal_dyck_config = {
    "task": CFGTask,
    "epochs": 100,
    "early_stopping_steps": 5,
    "grammar": dyck_grammar,
    "to_predict": [u")", u"]"],
    "sample_depth": 5,
    "read_size": 2
}

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
