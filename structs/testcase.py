from __future__ import print_function
import sys
import traceback
import StringIO
import torch
from torch.autograd import Variable


class testcase(object):

    """""
    Decorator marking a function as a test case.

    Usage:
        @testcase(structs.Stack)
        def test_push(...):
            ...
            assert CONDITION

    More example test cases in structs.tests.

    """

    def __init__(self, struct_type, always_print=False):
        self._struct_type = struct_type
        self._always_print = always_print

    @property
    def struct_type(self):
        return self._struct_type

    def __call__(self, test):
        name = self._rename(test, self.struct_type)
        def wrap_test():
            try:
                print("Running {}..".format(name), end="")
                stdout = StringIO.StringIO()
                old_stdout = sys.stdout
                sys.stdout = stdout
                test()
            except AssertionError:
                sys.stdout = old_stdout
                print(" FAILED!")
                if test.__doc__ is not None:
                    print("Documentation:")
                    print(test.__doc__)
                output = stdout.getvalue()
                if output:
                    print("Stdout:")
                    print(stdout.getvalue().strip())
                traceback.print_exc()
            else:
                sys.stdout = old_stdout
                print(" PASSED!")
                if self._always_print:
                    print("Stdout:")
                    print(stdout.getvalue().strip())
            finally:
                stdout.close()
        test.__name__ = name
        wrap_test.__name__ = name
        wrap_test._is_test_case = True
        return wrap_test

    @staticmethod
    def _rename(f, struct_type):
        return "{}::{}".format(struct_type.__name__, f.__name__)


def type_has_tests(struct_type):
    return hasattr(struct_type, "__tests__")


def test_module(module):
    """ Run all the tests defined within a class, module, or dictionary. """
    if isinstance(module, dict):
        d = module
    elif hasattr(module, "__dict__"):
        d = module.__dict__
    else:
        raise ValueError("{} is not a class, module, or dictionary".format(module))
    for obj in d.values():
        if getattr(obj, "_is_test_case", False):
            obj()


def is_close(a, b):
    diff = a - b
    if isinstance(diff, Variable) or isinstance(diff, torch.Tensor):
        abs_fn = torch.abs
    else:
        abs_fn = abs
    return abs_fn(a - b) <= .0001
