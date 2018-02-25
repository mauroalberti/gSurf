# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from .mathematics import are_close


def arrays_are_close(a_array, b_array, rtol=1e-012, atol=1e-12, equal_nan=False, equal_inf=False):
    """
    Check for equivalence between two numpy arrays.

    :param a_array: numpy array
    :param b_array: numpy array
    :param rtol: relative tolerance
    :param atol: absolute tolerance
    :param equal_nan: consider nan values equivalent or not
    :param equal_inf: consider inf values equivalent or not
    :return: Boolean

    Example:
      >>> arrays_are_close(np.array([1,2,3]), np.array([1,2,3]))
      True
      >>> arrays_are_close(np.array([[1,2,3], [4, 5, 6]]), np.array([1,2,3]))
      False
      >>> arrays_are_close(np.array([[1,2,3], [4,5,6]]), np.array([[1,2,3], [4,5,6]]))
      True
      >>> arrays_are_close(np.array([[1,2,np.nan], [4,5,6]]), np.array([[1,2,np.nan], [4,5,6]]))
      False
      >>> arrays_are_close(np.array([[1,2,np.nan], [4,5,6]]), np.array([[1,2,np.nan], [4,5,6]]), equal_nan=True)
      True
    """
    if a_array.shape != b_array.shape:
        return False

    are_equal = []
    for a, b in np.nditer([a_array, b_array]):
        are_equal.append(are_close(a.item(0), b.item(0), rtol=rtol, atol=atol, equal_nan=equal_nan, equal_inf=equal_inf))

    return all(are_equal)


def point_solution(a_array, b_array):
    """
    finds a non-unique solution
    for a set of linear equations
    """

    try:
        return np.linalg.lstsq(a_array, b_array)[0]
    except:
        return None, None, None


def xyz_svd(xyz_array):
    """
    Calculates the SVD solution given a Numpy array.

    # modified after: 
    # http://stackoverflow.com/questions/15959411/best-fit-plane-algorithms-why-different-results-solved
    """

    try:
        result = np.svd(xyz_array)
    except:
        result = None

    return dict(result=result)


def to_floats(iterable_obj):
    """
    Converts an iterable object storing float-compatible values to a list of floats.

    :param iterable_obj:
    :return: list of Floats

    Example:
      >>> to_floats([1, 2, 3])
      [1.0, 2.0, 3.0]
    """

    return [float(item) for item in iterable_obj]


if __name__ == "__main__":

    import doctest
    doctest.testmod()
