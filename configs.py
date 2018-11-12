"""
Defines Task parameters for notable experiment configurations.

Notes:
    * A "task" MUST be specified in the config dictionary.
    * Other values are optional and task-specific. They are used to
      specify values for parameters in a Task constructor.
    * Controller_type, model_type, and struct_type should not be specified
      here. They can be set with command-line arguments.

A config can be run with:
    python run.py CONFIG_NAME
"""

from formalisms.cfg import *
from stacknn_utils.data_readers import *
from tasks import *
from torch.nn import CrossEntropyLoss

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
    "read_size": 2,
    "criterion": CrossEntropyLoss(reduction="none")
}

# 5) Agreement grammar task.
final_agreement_config = {
    "task": CFGTask,
    "epochs": 100,
    "early_stopping_steps": 5,
    "grammar": unambig_agreement_grammar,
    "to_predict": [u"Auxsing", u"Auxplur"],
    "sample_depth": 16,
    "read_size": 2,
    "criterion": CrossEntropyLoss(reduction="none")
}

# 5b) Agreement grammar task with longer early stopping 
final_agreement_config_10 = {
    "task": CFGTask,
    "epochs": 100,
    "early_stopping_steps": 10,
    "grammar": unambig_agreement_grammar,
    "to_predict": [u"Auxsing", u"Auxplur"],
    "sample_depth": 16,
    "read_size": 2,
    "criterion": CrossEntropyLoss(reduction="none")
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
    "max_length": 32,
    "criterion": CrossEntropyLoss(reduction="none")
}

# 7) Reverse with deletion task.
final_reverse_deletion_config = {
    "task": ReverseDeletionTask,
    "epochs": 1,
    "early_stopping_steps": 5,
    "read_size": 2,
    "num_symbols": 4
}

""" Testing Configs """

# 1) Reverse task.
testing_reverse_config = {
    "task": ReverseTask,
    "epochs": 1,
    "early_stopping_steps": 5,
    "read_size": 2,
    "min_length": 1,
    "max_length": 24,
    "mean_length": 20,
    "std_length": 4.
}

# 2) XOR/parity evaluation task.
testing_parity_config = {
    "task": XORTask,
    "epochs": 1,
    "early_stopping_steps": 5,
    "read_size": 6,
    "str_length": 24
}

# 3) Delayed XOR/parity evaluation task.
testing_delayed_parity_config = {
    "task": DelayedXORTask,
    "epochs": 1,
    "early_stopping_steps": 5,
    "read_size": 6,
    "str_length": 24
}

# 4) Dyck language modeling task.
testing_dyck_config = {
    "task": CFGTask,
    "epochs": 1,
    "early_stopping_steps": 5,
    "grammar": dyck_grammar_2,
    "to_predict": [u")", u"]"],
    "sample_depth": 5,
    "read_size": 2,
    "max_length": 128,
}

# 5) Agreement grammar task.
testing_agreement_config = {
    "task": CFGTask,
    "epochs": 1,
    "early_stopping_steps": 5,
    "grammar": unambig_agreement_grammar,
    "to_predict": [u"Auxsing", u"Auxplur"],
    "sample_depth": 5,
    "read_size": 2,
    "max_length": 64,
}

# 5b) Agreement grammar task with longer early stopping
testing_agreement_config_10 = {
    "task": CFGTask,
    "epochs": 1,
    "early_stopping_steps": 10,
    "grammar": unambig_agreement_grammar,
    "to_predict": [u"Auxsing", u"Auxplur"],
    "sample_depth": 5,
    "read_size": 2,
    "max_length": 64,
}

# 6) Reverse Polish notation formula task.
testing_formula_config = {
    "task": CFGTransduceTask,
    "epochs": 1,
    "early_stopping_steps": 5,
    "grammar": exp_eval_grammar,
    "to_predict": [u"0", u"1"],
    "sample_depth": 5,
    "read_size": 2,
    "max_length": 48,
}

# 7) Reverse with deletion task.
testing_reverse_deletion_config = {
    "task": ReverseDeletionTask,
    "epochs": 1,
    "early_stopping_steps": 5,
    "read_size": 2,
    "num_symbols": 4,
    "min_length": 1,
    "max_length": 24,
    "mean_length": 20,
    "std_length": 4.
}

""" Configs Not Included in the Paper """

old_final_dyck_config = {
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

# 1) Reverse task that runs really quickly.
quick_reverse_config = {
    "task": ReverseTask,
    "epochs": 1,
    "early_stopping_steps": 5,
    "read_size": 2
}

""" Configs for Will's senior thesis. """

# 1) Extreme reverse task.
extreme_reverse_config = {
    "task": ReverseTask,
    "hidden_size": 8,
    "mean_length": 50,
    "max_length": 80,
    "std_length": 5,
    "epochs": 300,
    "early_stopping_steps": 5,
    "read_size": 2
}

# 2) a^nb^n.
anbn_config = {
    "task": OrderedCountingTask,
    "length_fns": [lambda n: n, lambda n: n],
    "hidden_size": 1,
}

# 3) a^nb^{2n}.
anb2n_config = {
    "task": OrderedCountingTask,
    "length_fns": [lambda n: n, lambda n: 2 * n],
    "hidden_size": 1,
}

"""Tasks using datasets."""

linzen_agreement_config = {
    "task": NaturalTask,
    "train_filename": "../data/linzen/rnn_arg_simple/numpred.test.5",
    "test_filename": "../data/linzen/rnn_arg_simple/numpred.test.5",
    "data_reader": ByLineDatasetReader(linzen_line_consumer),
}
