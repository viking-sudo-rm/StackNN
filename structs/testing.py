from __future__ import print_function
import traceback

class testcase(object):

    """""
    Decorator marking a function as a test case.

    Usage:
        @testcase(structs.Stack)
        def test_push(...):
            ...

    Example test case is shown below

    """

    _case_map = {}

    def __init__(self, struct_type):
        self._struct_type = struct_type

    @property
    def struct_type(self):
        return self._struct_type

    def __call__(self, f):
        self._register(f)
        return f

    def _register(self, f):
        if self.struct_type not in self._case_map:
            self._case_map[self.struct_type] = []
        f.__name__ = self._rename(f, self.struct_type)
        self._case_map[self.struct_type].append(f)

    @staticmethod
    def _rename(f, struct_type):
        return "test<{}::{}>".format(struct_type.__name__, f.__name__)

    @classmethod
    def run_all(cls, struct_type):
        for f in cls._case_map[struct_type]:
            try:
                print("Running {}..".format(f.__name__))
                f()
            except AssertionError:
                print("FAILED!")
                print("Test documentation:")
                print(f.__doc__)
                print("Traceback:")
                traceback.print_exc()
            else:
                print("PASSED!")


@testcase(int)
def test_addition():
    """ Does 2 + 2 = 4? """
    assert 2 + 2 == 4

if __name__ == "__main__":
    testcase.run_all(int)