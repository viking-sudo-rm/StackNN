from stacknn_utils.testcase import testcase, test_module

import run
import configs

import structs.tests
import structs.regularization

# Configs that should be run automatically for e23 testing.
e2e_test_configs = [
    configs.testing_reverse_config,
    configs.testing_parity_config,
    configs.testing_delayed_parity_config,
    configs.testing_dyck_config,
    configs.testing_agreement_config,
    # configs.testing_agreement_config_10,
    configs.testing_formula_config,
    configs.testing_reverse_deletion_config,
]

@testcase(run.main, [[config] for config in e2e_test_configs])
def test_main(config):
    """ Run several test tasks end-to-end. """
    result = run.main(config)
    assert isinstance(result, dict), \
        "Return value should be a dictionary."


def main():
    """ Run all the tests in the whole program. """

    print("=" * 80)
    print("DATA STRUCTURE TESTS")
    test_module(structs.tests)
    test_module(structs.regularization)

    print("=" * 80)
    print("END-TO-END TESTS")
    test_module(globals())


if __name__ == '__main__':
    main()
