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

dyck_grammar_2 = CFG.fromstring("""
    S -> S T | T S | T
    T -> '(' T ')' | '(' ')'
    T -> '[' T ']' | '[' ']'
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

unambig_agreement_grammar = CFG.fromstring("""
    S -> NPsing "Auxsing"
    S -> NPplur "Auxplur"
    NP -> NPsing
    NP -> NPplur
    NPsing -> "Det" "Nsing" PP
    NPplur -> "Det" "Nplur" PP
    NPsing -> "Det" "Nsing" Relsing
    NPplur -> "Det" "Nplur" Relplur
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

""" XOR Grammars """

exp_eval_grammar = CFG.fromstring("""
    S -> Strue | Sfalse
    Strue -> "T" "1" 
    Strue -> Strue Strue "and" "1"
    Strue -> Strue Strue "or" "1" | Strue Sfalse "or" "1"
    Strue -> Sfalse Strue "or" "1"
    Sfalse -> "F" "0"
    Sfalse -> Sfalse Sfalse "or" "0" 
    Sfalse -> Sfalse Sfalse "and" "0" | Sfalse Strue "and" "0"
    Sfalse -> Strue Sfalse "and" "0"
""")

xor_exp_eval_grammar = CFG.fromstring("""
    S -> Strue | Sfalse
    Strue -> "T" "1" 
    Strue -> Strue Sfalse "xor" "1"
    Strue -> Sfalse Strue "xor" "1"
    Sfalse -> "F" "0"
    Sfalse -> Sfalse Sfalse "xor" "0" 
    Sfalse -> Strue Strue "xor" "0"
""")

xor_string_grammar = CFG.fromstring("""
    S -> "0" "a" S | "1" "b" T | "0" "a" | "1" "b"
    T -> "0" "b" T | "1" "a" S | "0" "b" | "1" "a"
""")

padded_xor_string_grammar = CFG.fromstring("""
    S -> "0" "x" "a" S | "1" "x" "b" T | "0" "x" "a" | "1" "x" "b"
    T -> "0" "x" "b" T | "1" "x" "a" S | "0" "x" "b" | "1" "x" "a"
""")
