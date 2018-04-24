"""
A script to demonstrate how to use the BUTA class from tree_automata to
create bottom-up tree automata.
"""
from nltk.grammar import Nonterminal

from tree_automata import BUTA
from trees import Tree, reverse_polish

"""
         1         2         3         4         5         6         7
        9012345678901234567890123456789012345678901234567890123456789012
"""
# Create a tree automaton recognizing true boolean expressions
a = BUTA.fromstring("""
    T -> '1'
    T -> 'not' F
    T -> 'and' T T
    T -> 'or' T T
    T -> 'or' T F
    T -> 'or' F T
    F -> '0'
    F -> 'not' T
    F -> 'and' T F
    F -> 'and' F T
    F -> 'and' F F
    F -> 'or' F F
""", u'T')

# Create some boolean expressions
t1 = Tree(u"or", [u"1", u"0"])  # true
t2 = Tree(u"not", [t1])  # false
t3 = Tree(u"and", [u"1", u"1"])  # true
t4 = Tree(u"or", [t3, u"0"])  # true
t5 = Tree(u"and", [t2, t4])  # false
t6 = Tree(u"not", [t5])  # true

# Use the BUTA to evaluate some expressions
print "{} is {}".format(t5, a.recognize(t5))
print "{} is {}".format(t6, a.recognize(t6))

# Use the BUTA to parse an expression
print "Parse:"
print " ".join(reverse_polish(t6))
for p in a.parse(t6):
    print " ".join([str(l) for l in reverse_polish(p)])

# Use the BUTA to generate some expressions
qt = Nonterminal(u"T")
qf = Nonterminal(u"F")
for e in a.generate(states={qt, qf}, depth=4, n=10):
    print "{} is {}".format(str(e), a.recognize(e))
