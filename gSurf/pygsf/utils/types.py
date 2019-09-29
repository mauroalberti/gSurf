# -*- coding: utf-8 -*-


def check_type(var, name, expected_type):
    """
    Checks the type of the variable.

    :param var:
    :param name:
    :param expected_type:
    :return:
    """

    if not (isinstance(var, expected_type)):
        raise Exception("{} should be {} but {} got ".format(name, expected_type, type(var)))


def check_optional_type(var, name, expected_type):
    """
    Checks the type of the optional variable.

    :param var:
    :param name:
    :param expected_type: Any
    :return:
    """

    if var:
        if not (isinstance(var, expected_type)):
            raise Exception("{} should be {} but got {}".format(name, expected_type, type(var)))


