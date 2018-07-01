import warnings


def check_extension(filename, extension, strict=False):
    """

    :param filename:
    :param extension:
    :param strict:

    :return: None
    """
    if not extension.startswith("."):
        extension = "." + extension

    if not filename.endswith(extension):
        msg = "{} is not of the expected format {}".format(filename, extension)
        if strict:
            raise TypeError(msg)
        else:
            warnings.warn(msg, RuntimeWarning)
