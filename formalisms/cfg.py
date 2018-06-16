""" Utility module defining various context-free grammars.


Example usage:
  CFGTask(**formalisms.cfg.dyck_task_parameters).run_experiment()

"""

from __future__ import division

from nltk import CFG


dyck_grammar = CFG.fromstring("""
    S -> S S
    S -> '(' S ')' | '(' ')'
    S -> '[' S ']' | '[' ']'
""")


reverse_grammar = CFG.fromstring("""
    S -> "a" S "a1"
    S -> "b" S "b1"
    S -> "c"
""")


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
