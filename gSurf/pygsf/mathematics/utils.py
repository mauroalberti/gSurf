import math
import numbers
from typing import Tuple

from .defaults import MIN_VECTOR_MAGNITUDE


def normXYZ(x: numbers.Real, y: numbers.Real, z: numbers.Real) -> Tuple:
    """
    Normalize numeric values.

    :param x: x numeric value
    :param y: y numeric value
    :param z: z numeric value
    :return: the magnitude and a tuple of three float values
    """

    # input vals checks

    vals = [x, y, z]
    if not all(map(lambda val: isinstance(val, numbers.Real), vals)):
        raise Exception("Input values must be integer or float")
    elif not all(map(math.isfinite, vals)):
        raise Exception("Input values must be finite")

    mag = math.sqrt(x*x + y*y + z*z)

    if mag <= MIN_VECTOR_MAGNITUDE:
        norm_xyz = None
    else:
        norm_xyz = x/mag, y/mag, z/mag

    return mag, norm_xyz

