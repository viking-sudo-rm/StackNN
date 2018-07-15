"""
Generate a list of random sentences of a given derivation depth from a
context-free grammar. Code by Dana Angluin. The main function is
random_sentences.
"""
from __future__ import division

import random

import nltk.grammar
from nltk import CFG
from nltk.parse.generate import generate


def random_sentences(count, depth, gr):
    """
    Generates a number of random sentences using derivations of at most
    a given depth.

    :type count: int
    :param count: The number of sentences to generate

    :type depth: int
    :param depth: The maximum derivation depth of a generated sentence

    :type gr: CFG
    :param gr: A context-free grammar to generate from

    :type: list
    :return: The generated sentences
    """
    nonterminals = nonterminals_from_grammar(gr)
    gr_table = make_table(depth, gr)
    sentences = [random_from_form([gr.start()], depth, gr_table,
                                  nonterminals, gr) for _ in xrange(count)]
    return sentences


def make_table(depth, gr):
    """
    For each production p of a context-free grammar and each number k,
    this function computes the number of terminal strings whose
    derivations
        - invoke the production p as the first step and
        - have depth at most k.

    :type depth: int
    :param depth: The maximum possible value of k (see above)

    :type gr: CFG
    :param gr: A context-free grammar

    :rtype: dict
    :return: For each production p and number k, the return dict maps
        the tuple (p, k) to the number described above
    """
    productions = gr.productions()
    table = {}
    for prod in productions:
        table[(prod, 0)] = 0
        if is_terminal_production(prod, gr):
            table[(prod, 1)] = 1
        else:
            table[(prod, 1)] = 0
    for k in xrange(2, depth + 1):
        for prod in productions:
            table[(prod, k)] = count_production_depth(prod, k, table, gr)
    return table


""" Helper Functions """


def remove_duplicates(lst):
    """
    Removes duplicates from a given list.

    :type lst: list
    :param lst: A list

    :rtype: list
    :return: lst, but with duplicates removed
    """
    return list(set(lst))


def all_not_in(lst1, lst2):
    """
    Determines whether a list contains an element of another list.

    :type lst1: list
    :param lst1: A list

    :type lst2: list
    :param lst2: Another list

    :rtype: bool
    :return: True if no element of lst1 is in list2, False otherwise
    """
    return set(lst1).isdisjoint(lst2)


def select_from_dist(prob):
    """
    This function chooses a random number according to a given
    probability distribution.

    :type prob: list
    :param prob: A probability distribution represented as a list. For
        each number i, prob[i] is the probability that this function
        returns i. For example, if [.2, .3, .5] is passed to this
        parameter, then there is a 20% chance of returning 0, a 30%
        chance of returning 1, and a 50% chance of returning 2.

    :rtype: int
    :return: The number chosen
    """
    x = random.random()
    cumulative = 0.0
    for i in xrange(len(prob)):
        if x <= cumulative + prob[i]:
            return i
        else:
            cumulative += prob[i]
    return len(prob)


def all_lhs_from_grammar(gr):
    """
    Finds the left-hand sides of all productions in a context-free
    grammar. The return value may contain duplicates.

    :type gr: CFG
    :param gr: A context-free grammar

    :rtype: list
    :return: All the nonterminals appearing in the left-hand side of a
        production of gr
    """
    return [prod.lhs() for prod in gr.productions()]


def nonterminals_from_grammar(gr):
    """
    Finds the left-hand sides of all productions in a context-free
    grammar. The return value does not contain duplicates.

    :type gr: CFG
    :param gr: A context-free grammar

    :rtype: list
    :return: All the nonterminals appearing in the left-hand side of a
        production of gr
    """
    return remove_duplicates(all_lhs_from_grammar(gr))


def is_terminal_production(prod, gr):
    """
    Determins whether or not a rule contains nonterminals in its right-
    hand side.

    :type prod: nltk.grammar.Production
    :param prod: A production

    :type gr: CFG
    :param gr: A CFG

    :rtype: bool
    :return: True if there are no nonterminals in the right-hand side of
        prod, False otherwise
    """
    return all(nltk.grammar.is_terminal(n) for n in prod.rhs())


def count_nonterminal_depth(nonterminal, depth, table, gr):
    """
    A helper function for make_table. This function computes the number
    of terminal strings generable from a nonterminal using a derivation
    of at most a given depth. This function assumes that the number for
    the previous depth has already been computed.

    :type nonterminal: nltk.grammar.Nonterminal
    :param nonterminal: The nonterminal from which generable strings are
        considered

    :type depth: int
    :param depth: The maximum depth of derivations considered

    :type table: dict
    :param table: A table containing the results obtained from this
        function for the previous depth. The format of this table is the
        same as the return value of make_table

    :type gr: CFG
    :param gr: The grammar whose generable strings are being considered

    :rtype: int
    :return: The number of strings generable by gr from nonterminal
        using derivations at most depth-many layers deep
    """
    nt_prods = gr.productions(lhs=nonterminal)
    total = 0
    for prod in nt_prods:
        total += table[(prod, depth)]
    return total


def count_production_depth(prod, depth, table, gr):
    """
    A helper function for make_table. This function computes the number
    of terminal strings generable using a derivation of at most a given
    depth that invokes a given production as its first step. This
    function assumes that the number for the previous depth has already
    been computed.

    :type prod: nltk.grammar.Production
    :param prod: The production invoked during the first step of the
        derivations considered

    :type depth: int
    :param depth: The maximum depth of derivations considered

    :type table: dict
    :param table: A table containing the results obtained from this
        function for the previous depth. The format of this table is the
        same as the return value of make_table

    :type gr: CFG
    :param gr: The grammar whose generable strings are being considered

    :rtype: int
    :return: The number of strings generable by gr using derivations at
        most depth-many layers deep using prod as their first step
    """
    result = 1
    for item in prod.rhs():
        if nltk.grammar.is_nonterminal(item):
            result *= count_nonterminal_depth(item, depth - 1, table, gr)
    return result


def choose_production(nt, depth, table, gr):
    """
    Randomly chooses a production with a given left-hand side nt. The
    probability of each production p is the proportion of strings
    generable from nt with derivations of at most a given depth that
    have derivations invoking p in the first step.

    :type nt: nltk.grammar.Nonterminal
    :param nt: The left-hand side of the production chosen

    :type depth: int
    :param depth: The maximum depth of derivations considered

    :type table: dict
    :param table: A table computed by make_table (see make_table and
        count_production_depth)

    :type gr: CFG
    :param gr: The context-free grammar from which the productions are
        drawn

    :rtype: nltk.grammar.Production
    :return: The chosen production
    """
    prods = gr.productions(lhs=nt)
    counts = []
    total = 0.
    for prod in prods:
        count = count_production_depth(prod, depth, table, gr)
        counts += [count]
        total += count
    counts = [i / total for i in counts]
    k = select_from_dist(counts)
    return prods[k]


def random_from_form(form, depth, table, nonterminals, gr):
    """
    Generates a random terminal string generable from a list of
    terminals and nonterminals using a derivation of at most a given
    depth.

    :type form: list
    :param form: A list of terminals and nonterminals, from which the
        return value is derived

    :type depth: int
    :param depth: The maximum depth of derivations considered

    :type table: dict
    :param table: A table computed by make_table (see make_table and
        count_production_depth)

    :type nonterminals: list
    :param nonterminals: Only nonterminals appearing in this list will
        be expanded

    :type gr: CFG
    :param gr: A context-free grammar

    :rtype: list
    :return: The generated terminal string, in sentence format
    """
    if depth <= 0:
        return form
    if all_not_in(form, nonterminals):
        return form
    else:
        result_lst = []
        for item in form:
            if item in nonterminals:
                prod = choose_production(item, depth, table, gr)
                result_lst += prod.rhs()
            else:
                result_lst += [item]
    return random_from_form(result_lst, depth - 1, table, nonterminals, gr)


if __name__ == "__main__":
    ######################################################
    # some test grammars

    dyck_grammar = CFG.fromstring("""
        S -> S S
        S -> '(' S ')' | '(' ')'
        S -> '[' S ']' | '[' ']'
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
    dgr = dyck_grammar
    uagr = unambig_agreement_grammar
    eegr = exp_eval_grammar

    # comparisons of table calculations and reported sample sizes

    # Note: the generate function from nltk uses a notion of
    # depth that is 1 more than that used above!

    # NB: the dyck_grammar is NOT unambiguous (S -> S S)
    dgr_table = make_table(6, dgr)
    print "dyck_grammar for 4 from count_nonterminal_depth"
    print count_nonterminal_depth(dgr.start(), 4, dgr_table, dgr)
    print "nltk generate: number of sentences for dyck grammar at depth = 5"
    print len(list(generate(dgr, depth=5)))
    print "The dyck_grammar is ambiguous!"

    # unambig_agreement_grammar
    # this agrees with the count for depth = 16 in generate
    uagr_table = make_table(15, uagr)
    print "unambig_agreement_grammar for 15 from count_nonterminal_depth"
    print count_nonterminal_depth(uagr.start(), 15, uagr_table, uagr)

    # exp_eval_grammar
    # this agrees with the count for depth = 6 in generate
    eegr_table = make_table(5, eegr)
    print "exp_eval_grammar for 5 from count_nonterminal_depth"
    print count_nonterminal_depth(eegr.start(), 5, eegr_table, eegr)

    print "number of nltk depth = 7 sentences from dyck_grammar"
    print count_nonterminal_depth(dgr.start(), 6, dgr_table, dgr)

    print "a random sentence nltk depth = 7 from dyck_grammar"
    print random_sentences(1, 6, dgr)

    sentences100 = random_sentences(100, 6, dgr)
    print "maximum length among 100 nltk depth = 7 sentences from dyck_grammar"
    print max([len(x) for x in sentences100])
