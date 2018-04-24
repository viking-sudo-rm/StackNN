"""
Classes for various kinds of tree automata. A tree automaton consists of
a list of transitions, a set of final states, and in the case of a top-
down tree automaton, an initial state. Throughout this module, automaton
states are represented as Nonterminal objects from nltk.grammar, while
tree labels are represented as unicode strings. State transitions are
represented as Production objects from nltk.grammar; see
check_is_transition and BUTA for more details.

For more information about tree automata in general, please see the TATA
book by Comon et al.: http://tata.gforge.inria.fr/
"""
from itertools import product, chain, islice
from numbers import Number

import nltk.grammar as gr
from nltk.tree import Tree

from trees import get_root_label, Tree


def check_is_nonterminal(*nts):
    """
    Asserts that all of one or more objects are Nonterminals.

    :param nts: An object, which may or may not be a Nonterminal

    :return: None
    """
    for nt in nts:
        if not gr.is_nonterminal(nt):
            raise TypeError("{} must be a nonterminal".format({}))
    return


def check_is_transition(*ps):
    """
    Asserts that all of one or more Productions are transitions.

    :type ps: Production
    :param ps: One or more Productions

    :return: None
    """
    for p in ps:
        if not is_transition(p):
            raise ValueError("{} must be a transition".format(p))
    return


def check_type(obj, t):
    """
    Asserts that an object has a certain type.

    :param obj: An object

    :type t: Type
    :param t: A type

    :return: None
    """
    if not isinstance(obj, t):
        raise TypeError("{} must be of type {}".format(obj, t))
    return


def is_transition(p):
    """
    Checks to see if a Production object is a transition. A transition
    is a Production in which the right-hand side must begin with a
    terminal. See BUTA for the interpretation of a transition.

    :type p: gr.Production
    :param p: A production

    :rtype: bool
    :return: True if p is a transition, False otherwise
    """
    check_type(p, gr.Production)
    return len(p.rhs()) > 0 and gr.is_terminal(p.rhs()[0])


class BUTA(object):
    """
    A non-deterministic bottom-up tree automaton (BUTA). A BUTA reads a
    tree from bottom to top. Each node of the tree is assigned a state
    based on its label and the states assigned to its children. The
    transitions of a BUTA are represented as Production objects of the
    following form:

        Q -> "a" Q1 Q2 ... Qn.

    The interpretation of a transition is that a node labelled "a" is
    assigned state Q if its n-many children are assigned states Q1, Q2,
    ..., Qn, respectively.
    """

    def __init__(self, transitions, finals):
        """
        Constructor for a BUTA. Note that it is not necessary to specify
        a start state, since this information can be encoded using
        transitions of the form Q -> "a" for each label symbol "a."

        :type transitions: set
        :param transitions: The set of transitions

        :type finals: set
        :param finals: The set of accept states
        """
        check_is_nonterminal(*finals)
        check_is_transition(*transitions)
        self.finals = finals
        self._transitions = transitions

    @staticmethod
    def fromstring(transitions, *finals):
        """
        Constructs a BUTA from a string representation of the
        transitions.

        :type transitions: str
        :param transitions: The transitions of the tree automaton, in
            string representation

        :type finals: unicode
        :param finals: The accept states of the tree automaton

        :rtype: BUTA
        :return: The BUTA described by the parameters
        """
        _, rules = gr.read_grammar(transitions, gr.standard_nonterm_parser)
        return BUTA(rules, set(gr.Nonterminal(nt) for nt in finals))

    """ Transition Function """

    def transitions(self, lhs=None, label=None):
        """
        Public accessor for self._transitions.

        :type lhs: gr.Nonterminal
        :param lhs: If lhs is not set to None, then only transitions
            with lhs on the left-hand side will be returned

        :type label: unicode
        :param label: If label is set to none, then only transitions
            whose right-hand sides begin with label will be returned

        :rtype: set
        :return: The set of transitions with the specified lhs and label
        """
        if lhs is None and label is None:
            return self._transitions
        elif label is None:
            return set(t for t in self._transitions if t.lhs() == lhs)
        elif lhs is None:
            return set(t for t in self._transitions if t.rhs()[0] == label)
        else:
            return set(t for t in self._transitions if t.lhs() == lhs and
                       t.rhs()[0] == label)

    def _transition(self, symbol, *children):
        """
        Evaluates the transition function of this BUTA.

        :type symbol: unicode
        :param symbol: The label of a node

        :type children: gr.Nonterminal
        :param children: The states of the children of the node

        :rtype: set
        :return: The set of states that could be assigned to the node
        """
        rhs = (symbol,) + tuple(children)
        return set(t.lhs() for t in self._transitions if t.rhs() == rhs)

    def _inverse_transition(self, state):
        """
        Evaluates the inverse of the transition funtion of this BUTA.

        :type state: gr.Nonterminal
        :param state: A state

        :rtype: set
        :return: A set of pairs, each of which contains the symbol and
            the sequence of child states that transition to state
        """
        transitions = self.transitions(lhs=state)
        return set((t.rhs()[0], t.rhs()[1:]) for t in transitions)

    """ Parsing and Recognition """

    def parse(self, tree):
        """
        Given a tree, this function computes the states assigned to each
        node of the tree (i.e., the "parse" of the tree). If this BUTA
        is nondeterministic, a tree may have more than one parse.

        :type tree: Tree
        :param tree: A tree

        :rtype: generator
        :return: The possible parses of tree according to this BUTA.
            Each parse is represented as a tree in which each node is
            labelled with its state according to this BUTA
        """
        if type(tree) is not Tree:
            for q in self._transition(tree):
                yield q
        else:
            symbol = get_root_label(tree)
            parsed_children = [set(self.parse(t)) for t in tree]
            for pc in product(*parsed_children):
                child_states = tuple(get_root_label(t) for t in pc)
                for q in self._transition(symbol, *child_states):
                    yield Tree(q, pc)

    def recognize(self, tree):
        """
        Checks whether or not a tree is accepted by this BUTA.

        :type tree: Tree
        :param tree: A tree

        :rtype: bool
        :return: True if this BUTA accepts tree; False otherwise
        """
        root_states = set(p.label() for p in self.parse(tree))
        return not root_states.isdisjoint(self.finals)

    """ Generation """

    def generate(self, states=None, depth=10, n=None):
        """
        Generates all trees up to a certain depth or number that are
        assigned one or more given states by this BUTA.

        :type states: set
        :param states: The root node of every tree generated is assigned
            a state from this set by this BUTA. If states is not
            specified, it will be the set of final states of this BUTA
            by default

        :type depth: int
        :param depth: The maximum height of a generated tree

        :type n: int
        :param n: The maximum number of trees to generate

        :rtype: generator
        :return: A generator that produces every tree satisfying the
            above description
        """
        check_type(depth, Number)
        if states is None:
            states = self.finals
        else:
            for q in states:
                check_is_nonterminal(q)

        gen = chain.from_iterable([self._generate_all((q,), depth)
                                   for q in states])
        if n is not None:
            gen = islice(gen, n)

        for t in gen:
            yield t[0]

    def _generate_all(self, states, depth):
        """
        Generates all trees up to a certain depth that are assigned a
        given state by this BUTA.

        :type states: tuple
        :param states: A tuple of states

        :type depth: int
        :param depth: The maximum height of a generated tree

        :rtype: generator
        :return: This generator produces a series of tuples of trees.
            The ith element of each tuple is a tree whose root node is
            assigned the state states[i] by this BUTA. This generator
            produces all possible tuples of this form
        """
        for q in states:
            check_is_nonterminal(q)
        if depth <= 0:
            return

        if len(states) == 0:
            yield ()
        elif len(states) == 1:
            for label, child_states in self._inverse_transition(states[0]):
                if len(child_states) == 0:
                    yield (label,)
                else:
                    for t in self._generate_all(child_states, depth - 1):
                        yield (Tree(label, t),)
        else:
            for t in self._generate_all(states[0:1], depth):
                for s in self._generate_all(states[1:], depth):
                    yield t + s
