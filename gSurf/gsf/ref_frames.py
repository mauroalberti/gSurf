# -*- coding: utf-8 -*-

import numpy as np

from .mathematics import almost_zero, are_close
from .geometry import Vect


class RefFrame(object):

    def __init__(self, versor_x, versor_y):
        """
        Constructor.

        :param versor_x: Vect instance, representing x axis orientation
        :param versor_y: Vect instance, representing y axis orientation
        """

        assert versor_x.is_near_unit
        assert versor_y.is_near_unit

        assert versor_x.is_suborthogonal(versor_y)

        self._x = versor_x
        self._y = versor_y

    @property
    def x(self):
        """
        Return the x vector,

        :return: Vect instance

        Example:
          >>> RefFrame(Vect(1,0,0), Vect(0,1,0)).x
          Vect(1.0000, 0.0000, 0.0000)
        """

        return self._x

    @property
    def y(self):
        """
        Return the y vector.

        :return: Vect instance

        Example:
          >>> RefFrame(Vect(1,0,0), Vect(0,1,0)).y
          Vect(0.0000, 1.0000, 0.0000)
        """

        return self._y

    @property
    def z(self):
        """
        Return the z vector.

        :return: Vect instance

        Example:
          >>> RefFrame(Vect(1,0,0), Vect(0,1,0)).z
          Vect(0.0000, 0.0000, 1.0000)
        """

        return self.x.vp(self.y)

if __name__ == "__main__":

    import doctest
    doctest.testmod()
