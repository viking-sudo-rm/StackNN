"""
Generate a list of random sentences of a given derivation depth from a
context-free grammar. Code by Dana Angluin. The main function is
random_sentences.
"""
from __future__ import division

from depth_generate import *


def random_cfg_test(count, depth, gr, savepath):
    """
    Calls random_sentences to generate a number of
    random sentences with derivations of depth at
    most the given amount from the grammar gr,
    and saves them as a test file in savepath.
    Format: each line is
    input,output
    where input is the generated string and output
    is the generated string with first symbol removed.
    """
    print "number of sentences: " + str(count)
    print "nltk depth: " + str(depth + 1)  # nltk depth is 1+depth

    sentences = random_sentences(count, depth, gr)
    print "maximum length sentence in test set"
    print max([len(x) for x in sentences])
    with open(savepath, "w") as f:
        for sentence in sentences:
            predict_sentence = sentence[1:]
            f.write(" ".join(sentence))
            f.write(",")
            f.write(" ".join(predict_sentence))
            f.write("\n")


if __name__ == "__main__":
    random.seed(552)

    dyck_grammar_2 = CFG.fromstring("""
        S -> S T | T S | T
        T -> '(' T ')' | '(' ')'
        T -> '[' T ']' | '[' ']'
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

    # to save typing
    dgr2 = dyck_grammar_2
    uagr = unambig_agreement_grammar
    eegr = exp_eval_grammar

    random_cfg_test(1000, 11, dgr2, "../data/testing/final/dyck.csv")
    random_cfg_test(1000, 31, uagr, "../data/testing/final/agreement.csv")
