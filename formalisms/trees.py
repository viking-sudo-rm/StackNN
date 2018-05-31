"""
Helper functions for working with trees.
"""
from nltk.tree import Tree as NLTKTree


def get_root_label(tree):
    """
    Finds the label of the root node of a tree.

    :param tree: A tree

    :return: The label of the root node of tree
    """
    if type(tree) is not Tree:
        return tree
    else:
        return tree.label()


def polish(tree):
    """
    Computes the Polish representation of a tree.

    :type tree: Tree
    :param tree: A tree

    :rtype: list
    :return: The Polish representation of tree
    """
    if type(tree) is not Tree:
        return [tree]
    else:
        return [tree.label()] + [x for t in tree for x in polish(t)]


def reverse_polish(tree):
    """
    Computes the reverse-Polish representation of a tree.

    :type tree: Tree
    :param tree: A tree

    :rtype: list
    :return: The reverse-Polish representation of tree
    """
    if type(tree) is not Tree:
        return [tree]
    else:
        return [x for t in tree for x in reverse_polish(t)] + [tree.label()]


class Tree(NLTKTree):
    """
    A wrapper for the Tree class from nltk.tree, which adds additional
    functionality.
    """

    def __hash__(self):
        return hash((self._label, tuple(self[:])))

    def polish(self):
        return polish(self)

    def reverse_polish(self):
        return reverse_polish(self)
