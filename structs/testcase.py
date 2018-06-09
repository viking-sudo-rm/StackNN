from __future__ import print_function
import traceback

class testcase(object):

    """""
    Decorator marking a function as a test case.

    Usage:
        @testcase(structs.Stack)
        def test_push(...):
            ...

    More example test cases in structs.tests.

    """

    def __init__(self, struct_type):
        self._struct_type = struct_type

    @property
    def struct_type(self):
        return self._struct_type

    def __call__(self, test):
        name = self._rename(test, self.struct_type)
        def wrap_test():
            try:
                print("Running {}..".format(name), end="")
                test()
            except AssertionError:
                print(" FAILED!")
                if test.__doc__ is not None:
                    print("Documentation:")
                    print(test.__doc__)
                traceback.print_exc()
            else:
                print(" PASSED!")
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
