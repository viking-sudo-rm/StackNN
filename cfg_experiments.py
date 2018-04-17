from __future__ import division

from nltk import CFG

from models.vanilla import Controller as FFStackController
from tasks.cfg import CFGTask

dyck_grammar = CFG.fromstring("""
    S -> S S
    S -> '(' S ')' | '(' ')'
    S -> '[' S ']' | '[' ']'
""")

dyck_task = CFGTask(grammar=dyck_grammar,
                    to_predict=[u")", u"]"],
                    sample_depth=5,
                    model_type=FFStackController)

reverse_grammar = CFG.fromstring("""
    S -> "a" S "a1"
    S -> "b" S "b1"
    S -> "c"
""")

reverse_task = CFGTask(grammar=reverse_grammar,
                       to_predict=[u"a1", u"b1"],
                       sample_depth=12,
                       model_type=FFStackController)

agreement_grammar = CFG.fromstring("""
    S -> NPsing "Auxsing"
    S -> NPplur "Auxplur"
    NP -> NPsing
    NP -> NPplur
    NPsing -> NPsing PP
    NPplur -> NPplur PP
    NPsing -> NPsing Relsing
    NPplur -> NPplur Relplur
    NPsing -> "Det" "Nsing"
    NPplur -> "Det" "Nplur"
    PP -> "Prep" NP
    Relsing -> "Rel" "Auxsing" VP
    Relsing -> Relobj
    Relplur -> "Rel" "Auxplur" VP
    Relplur -> Relobj
    Relobj -> "Rel" NPsing "Auxsing" "Vtrans"
    Relobj -> "Rel" NPplur "Auxplur" "Vtrans"
    VP -> "Vintrans"
    VP -> "Vtrans" NP
""")

agreement_task = CFGTask(grammar=agreement_grammar,
                         to_predict=[u"Auxsing", u"Auxplur"],
                         sample_depth=8,
                         model_type=FFStackController)

agreement_task.run_experiment()
