import warnings


def unused_init_param(param_name, arg_value, obj):
    """
    Displays a warning message saying that a constructor for an object
    has received an argument value for an unused parameter.

    :type param_name: str
    :param param_name: The name of the unused parameter

    :param arg_value: The argument value passed to the constructor

    :param obj: The object being instantiated

    :return: None
    """
    if arg_value is not None:
        class_name = type(obj).__name__
        msg = "Parameter {} is set to {}, ".format(param_name, arg_value)
        msg += "but it is not used in {}.".format(class_name)
        warnings.warn(msg, RuntimeWarning)
