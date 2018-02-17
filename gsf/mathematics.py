# -*- coding: utf-8 -*-

from math import isnan, isinf


def is_number(s: str) -> bool:
    """
    Check if string can be converted to number.

    @param  s:  parameter to check.
    @type  s:  string

    @return:  boolean, whether string can be converted to a number (float).

    Example:
      >>> is_number("1.0")
      True
      >>> is_number("1")
      True
      >>> is_number(u"-10")
      True
      >>> is_number("one")
      False
      >>> is_number("1e-10")
      True
      >>> is_number("")
      False
    """

    try:
        float(s)
    except:
        return False
    else:
        return True


def almost_zero(an_val: float, tolerance: float = 1e-10) -> bool:
    """
    Check if a value for which abs can be used, is near zero.

    :param an_val: an abs-compatible object
    :param tolerance: the tolerance value
    :return: Boolean

      >>> almost_zero(1)
      False
      >>> almost_zero(1e-9)
      False
      >>> almost_zero(1e-11)
      True
    """

    return abs(an_val) <= tolerance


def are_close(a: float, b: float, rtol: float = 1e-012, atol: float = 1e-12, equal_nan: bool = False, equal_inf: bool = False) -> bool:
    """
    Mimics math.isclose from Python 3.5 (see: https://docs.python.org/3.5/library/math.html)

    Example:
      >>> are_close(1.0, 1.0)
      True
      >>> are_close(1.0, 1.000000000000001)
      True
      >>> are_close(1.0, 1.0000000001)
      False
      >>> are_close(0.0, 0.0)
      True
      >>> are_close(0.0, 0.000000000000001)
      True
      >>> are_close(0.0, 0.0000000001)
      False
      >>> are_close(100000.0, 100000.0)
      True
      >>> are_close(100000.0, 100000.0000000001)
      True
      >>> are_close(float('nan'), float('nan'))
      False
      >>> are_close(float('nan'), 1000000)
      False
      >>> are_close(1.000000000001e300, 1.0e300)
      False
      >>> are_close(1.0000000000001e300, 1.0e300)
      True
      >>> are_close(float('nan'), float('nan'), equal_nan=True)
      True
      >>> are_close(float('inf'), float('inf'))
      False
      >>> are_close(float('inf'), 1.0e300)
      False
      >>> are_close(float('inf'), float('inf'), equal_inf=True)
      True
    """

    # nan cases
    if equal_nan and isnan(a) and isnan(b):
        return True
    elif isnan(a) or isnan(b):
        return False

    # inf cases
    if equal_inf and isinf(a) and a > 0 and isinf(b) and b > 0:
        return True
    elif equal_inf and isinf(a) and a < 0 and isinf(b) and b < 0:
        return True
    elif isinf(a) or isinf(b):
        return False

    # regular case
    return abs(a - b) <= max(rtol * max(abs(a), abs(b)), atol)


def _test():
    import doctest
    return doctest.testmod()


if __name__ == '__main__':
    _test()
