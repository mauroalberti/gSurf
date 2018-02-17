# -*- coding: utf-8 -*-

from math import sqrt, degrees, acos
import numpy as np

from .default_parameters import *
from .mathematics import are_close
from .geometry import Vect
from .arrays import arrays_are_close


class Quaternion(object):
    """
    Quaternion class.
    """

    def __init__(self, w=np.nan, x=np.nan, y=np.nan, z=np.nan):
        """
        Construct a Quaternion instance.

        Example;
          >>> Quaternion(1, 0, 1, 0)
          Quaternion(1.00000, 0.00000, 1.00000, 0.00000)
          >>> Quaternion()
          Quaternion(nan, nan, nan, nan)
        """

        self.q = np.array([w, x, y, z], dtype=np.float64)

    def __repr__(self):

        return "Quaternion({:.5f}, {:.5f}, {:.5f}, {:.5f})".format(self.q[0], self.q[1], self.q[2], self.q[3])

    def components(self):
        """
        Returns the quaternion components as a float tuple.

        :return: tuple of 4 float values

        Example:
          >>> Quaternion(0, 1, 0, 0).components()
          (0.0, 1.0, 0.0, 0.0)
        """

        return self.q[0], self.q[1], self.q[2], self.q[3]

    @property
    def scalar(self):
        """
        Return the scalar component of a quaternion.

        :return: Float value

        Example:
          >>> Quaternion(1, 2, 0, 3).scalar
          1.0
          >>> Quaternion(0.0, 4.6, 6.2, 3.1).scalar
          0.0
        """

        return self.q[0]

    @property
    def vector(self):
        """
        Return the vector component of the quaternion.

        :return: Vect

        Example:
          >>> Quaternion(0.1, 1.2, 3.1, 0.9).vector
          Vect(1.2000, 3.1000, 0.9000)
          >>> Quaternion(6.1, 4.9, 1.03, 5.12).vector
          Vect(4.9000, 1.0300, 5.1200)
        """

        return Vect.from_array(self.q[1:])

    @classmethod
    def from_class(cls, w, x, y, z):
        """
        Class method for quaternion construction.

        :param w: q0, float
        :param x: q1, float
        :param y: q2, float
        :param z: q3, float
        :return: Quaternion instance
        """

        obj = cls()
        q = np.array([w, x, y, z], dtype=np.float64)
        obj.q = q

        return obj

    @classmethod
    def from_array(cls, a):
        """
        Class method to construct a quaternion from a numpy 1x4 array.

        Example:
          >>> Quaternion.from_array(np.array([1, 0, 1, 0]))
          Quaternion(1.00000, 0.00000, 1.00000, 0.00000)
          >>> Quaternion.from_array(np.array([7.65, -12.34, -1.0, 2.234]))
          Quaternion(7.65000, -12.34000, -1.00000, 2.23400)
        """

        assert a.size == 4

        obj = cls()
        obj.q = a.astype(np.float64)

        return obj

    @classmethod
    def from_vect(cls, vect):
        """
        Class method to construct a quaternion from a vector.

        :param vect: Vector instance
        :return: Quaternion instance

        Example:
          >>> Quaternion.from_vect(Vect(1, 0, 3))
          Quaternion(0.00000, 1.00000, 0.00000, 3.00000)
        """

        w, x, y, z = 0, vect.x, vect.y, vect.z

        return Quaternion.from_class(w, x, y, z)

    @classmethod
    def from_rot_matr(cls, matr):
        """
        Class method to construct a quaternion from a 3x3 rotation matrix.
        """

        q0 = sqrt(1 + matr[0, 0] + matr[1, 1] + matr[2, 2]) / 2.0
        q1 = sqrt(1 + matr[0, 0] - matr[1, 1] - matr[2, 2]) / 2.0
        q2 = sqrt(1 - matr[0, 0] + matr[1, 1] - matr[2, 2]) / 2.0
        q3 = sqrt(1 - matr[0, 0] - matr[1, 1] + matr[2, 2]) / 2.0

        q0q1 = (matr[2, 1] - matr[1, 2]) / 4.0
        q0q2 = (matr[0, 2] - matr[2, 0]) / 4.0
        q0q3 = (matr[1, 0] - matr[0, 1]) / 4.0
        q1q2 = (matr[0, 1] + matr[1, 0]) / 4.0
        q1q3 = (matr[0, 2] + matr[2, 0]) / 4.0
        q2q3 = (matr[1, 2] + matr[2, 1]) / 4.0

        if (3 * q0) > (q1 + q2 + q3):

            q1 = q0q1 / q0
            q2 = q0q2 / q0
            q3 = q0q3 / q0

        elif (3 * q1) > (q0 + q2 + q3):

            q0 = q0q1 / q1
            q2 = q1q2 / q1
            q3 = q1q3 / q1

        elif (3 * q2) > (q0 + q1 + q3):

            q0 = q0q2 / q2
            q1 = q1q2 / q2
            q3 = q2q3 / q2

        else:

            q0 = q0q3 / q3
            q1 = q1q3 / q3
            q2 = q2q3 / q3

        w, x, y, z = q0, q1, q2, q3

        return Quaternion.from_class(w, x, y, z)

    @classmethod
    def zero(cls):
        """
        Class method to construct a zero quaternion.

        Example:
          >>> Quaternion.zero()
          Quaternion(0.00000, 0.00000, 0.00000, 0.00000)
        """

        w, x, y, z = 0, 0, 0, 0

        return Quaternion.from_class(w, x, y, z)

    @classmethod
    def identity(cls):
        """
        Class method to construct an identity quaternion (i.e., zero-rotation).

        Example:
          >>> Quaternion.identity()
          Quaternion(1.00000, 0.00000, 0.00000, 0.00000)
        """

        w, x, y, z = 1, 0, 0, 0

        return Quaternion.from_class(w, x, y, z)

    @classmethod
    def i(cls):
        """
        Class method to construct the i elementary quaternion.

        Example:
          >>> Quaternion.i()
          Quaternion(0.00000, 1.00000, 0.00000, 0.00000)
        """

        w, x, y, z = 0, 1, 0, 0

        return Quaternion.from_class(w, x, y, z)

    @classmethod
    def j(cls):
        """
        Class method to construct the j elementary quaternion.

        Example:
          >>> Quaternion.j()
          Quaternion(0.00000, 0.00000, 1.00000, 0.00000)
        """

        w, x, y, z = 0, 0, 1, 0

        return Quaternion.from_class(w, x, y, z)

    @classmethod
    def k(cls):
        """
        Class method to construct the k elementary quaternion.

        Example:
          >>> Quaternion.k()
          Quaternion(0.00000, 0.00000, 0.00000, 1.00000)
        """

        w, x, y, z = 0, 0, 0, 1

        return Quaternion.from_class(w, x, y, z)

    def __eq__(self, another):
        """
        Quaternion equality.

        :param another: a Quaternion instance
        :return: Boolean

        Example:
          >>> Quaternion(1, 1, 3, 0) == Quaternion(0, 7, -2, 4)
          False
          >>> Quaternion(1.0, 1.0, 3.0, 0.0) == Quaternion(1.0, 1.0, 3.0, 0.0)
          True
          >>> Quaternion(1.0, 1.0, 3.0, np.nan) == Quaternion(1.0, 1.0, 3.0, np.nan)
          True
          >>> Quaternion(1.0, 1.0, 3.0, 0.0) == Quaternion(1.0, 1.0, 3.0, -1.0e-20)
          False
        """

        return ((self.q == another.q) | (np.isnan(self.q) & np.isnan(another.q))).all()

    def __ne__(self, another):
        """
        Quaternion inequality.

        :param another: a Quaternion instance
        :return: Boolean

        Example:
          >>> Quaternion(1, 1, 3, 0) != Quaternion(0, 7, -2, 4)
          True
          >>> Quaternion(1.0, 1.0, 3.0, np.nan) != Quaternion(1.0, 1.0, 3.0, np.nan)
          False
        """

        return not (self == another)

    def __add__(self, another):
        """
        Quaternion sum.

        :param another: Quaternion instance.
        :return: Quaternion instance.

        Example:
          >>> Quaternion(1, 1, 3, 0) + Quaternion(0, 7, -2, 4)
          Quaternion(1.00000, 8.00000, 1.00000, 4.00000)
          >>> Quaternion(2, 1, np.nan, 3) + Quaternion(3, 2, -2, 1)
          Quaternion(5.00000, 3.00000, nan, 4.00000)
        """

        return Quaternion.from_array(self.q + another.q)

    def __sub__(self, another):
        """
        Quaternion difference.

        :param another: Quaternion instance.
        :return: Quaternion instance.

        Example:
          >>> Quaternion(1, 1, 3, 0) - Quaternion(0, 7, -2, 4)
          Quaternion(1.00000, -6.00000, 5.00000, -4.00000)
          >>> Quaternion(np.inf, 1, 3, np.inf) - Quaternion(np.nan, np.nan, -1, 4)
          Quaternion(nan, nan, 4.00000, inf)
        """

        return Quaternion.from_array(self.q - another.q)

    def scalar_mult(self, val):
        """
        Multiplication of a quaternion by a scalar value.

        :param val: Integer or Float
        :return: Quaternion instance

        Example:
          >>> Quaternion(1, 1, 3, 0).scalar_mult(4)
          Quaternion(4.00000, 4.00000, 12.00000, 0.00000)
          >>> Quaternion(1.9, -1.2, 3.6, 4.1).scalar_mult(2)
          Quaternion(3.80000, -2.40000, 7.20000, 8.20000)
        """

        return Quaternion.from_array(self.q * val)

    def __neg__(self):
        """
        Negative of quaternion.

        :return: Quaternion instance.

        Example:
          >>> - Quaternion(1, 1, 3, 0)
          Quaternion(-1.00000, -1.00000, -3.00000, -0.00000)
          >>> - Quaternion(1.9, -1.2, 3.6, 4.1)
          Quaternion(-1.90000, 1.20000, -3.60000, -4.10000)
        """

        return self.scalar_mult(-1)

    def quater_mult(self, another):
        """
        Quaternion multiplication.
        Examples are taken from Kuipers, 2002, chp. 5.

        :param another: Quaternion instance .
        :return: Quaternion instance.

        Example:
          >>> Quaternion(3, 1, -2, 1).quater_mult(Quaternion(2, -1, 2, 3))
          Quaternion(8.00000, -9.00000, -2.00000, 11.00000)
        """
        
        a = + (self.q[0] * another.q[0]) \
            - (self.q[1] * another.q[1]) \
            - (self.q[2] * another.q[2]) \
            - (self.q[3] * another.q[3])
        
        b = + (self.q[0] * another.q[1]) \
            + (self.q[1] * another.q[0]) \
            + (self.q[2] * another.q[3]) \
            - (self.q[3] * another.q[2])

        c = + (self.q[0] * another.q[2]) \
            - (self.q[1] * another.q[3]) \
            + (self.q[2] * another.q[0]) \
            + (self.q[3] * another.q[1])
        
        d = + (self.q[0] * another.q[3]) \
            + (self.q[1] * another.q[2]) \
            - (self.q[2] * another.q[1]) \
            + (self.q[3] * another.q[0])
                
        return Quaternion(a, b, c, d)

    def vector_mult(self, vect):
        """
        Quaternion multiplication by a vector.

        :param vect: Vect instance .
        :return: Quaternion instance.
        """

        return self.quater_mult(Quaternion.from_vect(vect))

    def __mul__(self, another):
        """
        Wrapper for quaternion multiplication.
        Some examples are taken from Kuipers, 2002, chp. 5.

        :param another: Quaternion or Float/Integer
        :return: Quaternion instance

        Example:
          >>> Quaternion(1, 1, 3, 0) * 3
          Quaternion(3.00000, 3.00000, 9.00000, 0.00000)
          >>> Quaternion(3, 1, -2, 1) * Quaternion(2, -1, 2, 3)
          Quaternion(8.00000, -9.00000, -2.00000, 11.00000)
          >>> Quaternion(1, 1, 3, 0) * Quaternion(1, 0, 0, 0)
          Quaternion(1.00000, 1.00000, 3.00000, 0.00000)
          >>> Quaternion.identity() * Vect(1, 3, 2)
          Quaternion(0.00000, 1.00000, 3.00000, 2.00000)
        """

        if isinstance(another, (float, int)):
            return self.scalar_mult(another)
        elif isinstance(another, Vect):
            return self.vector_mult(another)
        elif isinstance(another, Quaternion):
            return self.quater_mult(another)
        else:
            raise QuaternionException("Multiplicand is not number or quaternion")

    @property
    def conjugate(self):
        """
        Quaternion conjugate.

        :return: Quaternion instance.

        Example:
          >>> Quaternion(1, 1, 3, 0).conjugate
          Quaternion(1.00000, -1.00000, -3.00000, -0.00000)
          >>> Quaternion(2.0, 0.0, -3.3, 17.09).conjugate
          Quaternion(2.00000, -0.00000, 3.30000, -17.09000)
          >>> Quaternion(2.0, 0.0, np.nan, 17.09).conjugate
          Quaternion(2.00000, -0.00000, nan, -17.09000)
        """
        
        a = + self.q[0]
        b = - self.q[1]
        c = - self.q[2]
        d = - self.q[3]

        return Quaternion(a, b, c, d)

    def sqrd_norm(self):
        """
        Squared norm of a quaternion.

        :return: Float value

        Example:
          >>> Quaternion(1, 0, 0, 0).sqrd_norm()
          1.0
          >>> Quaternion(1, 1, 0, 2).sqrd_norm()
          6.0
          >>> Quaternion(2, -1, 2, 3).sqrd_norm()
          18.0
          >>> Quaternion(2, np.nan, 2, 3).sqrd_norm()
          nan
        """

        return self.q[0]**2 + self.q[1]**2 + self.q[2]**2 + self.q[3]**2

    def __abs__(self):
        """
        Quaternion absolute value.

        :return: Float value

        Example:
          >>> abs(Quaternion(1, 0, 0, 0))
          1.0
          >>> are_close(abs(Quaternion(2, -1, 2, 3)), sqrt(18.0))
          True
        """

        return sqrt(self.sqrd_norm())

    @property
    def norm(self):
        """
        The norm of the quaternion.
        Equivalent to its absolute value.

        :return: Float value

        Example:
          >>> Quaternion(1, 0, 0, 0).norm
          1.0
        """

        return abs(self)

    @property
    def inverse(self):
        """
        Quaternion inverse.

        :return: Quaternion instance.

        Example:
          >>> Quaternion(0, 1, 0, 0).inverse
          Quaternion(0.00000, -1.00000, -0.00000, -0.00000)
          >>> Quaternion(3.2, 2.4, 7.18, 4.3).inverse * Quaternion(3.2, 2.4, 7.18, 4.3)
          Quaternion(1.00000, 0.00000, 0.00000, 0.00000)
        """

        return self.conjugate / self.sqrd_norm()

    def is_normalized(self):
        """
        Check if a quaternion is unitary.

        :return: Boolean

        Example:
          >>> Quaternion(0, 1, 0, 0).is_normalized()
          True
          >>> Quaternion(1, 4, 0, -4).is_normalized()
          False
        """

        return abs(1.0 - sqrt(self.sqrd_norm())) < quat_normaliz_tolerance

    def scalar_div(self, denominator):
        """
        Division of a quaternion by a scalar.

        :param denominator: Float value
        :return: Quaternion instance.

        Example:
          >>> Quaternion(1, 1, 3, 0).scalar_div(3)
          Quaternion(0.33333, 0.33333, 1.00000, 0.00000)
          >>> Quaternion(1, 1, 3, 0).scalar_div(1e-11)
          Traceback (most recent call last):
          ...
          QuaternionException: Quaternion division by almost zero value
        """

        if abs(denominator) < quat_division_tolerance:
            raise QuaternionException("Quaternion division by almost zero value")
        else:
            return Quaternion.from_array(self.q / denominator)

    def quater_div(self, another):
        """
        Quaternion division by another quaternion.

        :param another: Quaternion instance
        :return: Quaternion instance
        """

        return self * another.conjugate / another.sqrd_norm()

    def __truediv__(self, another):
        """
        Wrapper for quaternion division.
        This is only compatible with Python 3.

        Example:
          >>> Quaternion(1, 1, 3, 0) / 3
          Quaternion(0.33333, 0.33333, 1.00000, 0.00000)
          >>> Quaternion(1, 1, 3, 0) / Quaternion(1, 1, 3, 0)
          Quaternion(1.00000, 0.00000, 0.00000, 0.00000)
        """

        if isinstance(another, (int, float)):
            return self.scalar_div(another)
        elif isinstance(another, Quaternion):
            return self.quater_div(another)
        else:
            raise QuaternionException("Denominator is not number or quaternion")

    def normalize(self):
        """
        Normalize a quaternion.

        :return: Quaternion instance.

        Example:
          >>> Quaternion(0, 4, 0, 0).normalize()
          Quaternion(0.00000, 1.00000, 0.00000, 0.00000)
          >>> Quaternion(0, 4, 0, 8).normalize()
          Quaternion(0.00000, 0.44721, 0.00000, 0.89443)
          >>> are_close(abs(Quaternion(0.2, 17.9, -2.7, 4.3).normalize()), 1.0)
          True
          >>> Quaternion(0.696, 0.322, -0.152, 0.624).normalize()
          Quaternion(0.69580, 0.32191, -0.15196, 0.62382)
        """

        return self / sqrt(self.sqrd_norm())

    def is_close_to(self, another, rtol=1e-012, atol=1e-12, equal_nan=False, equal_inf=False):
        """
        Check for quaternion equivalence.

        :param another: Quaternion instance
        :param rtol: relative tolerance (float)
        :param atol: absolute tolerance (float)
        :param equal_nan: nan values are considered equal to themselves (boolean)
        :param equal_inf: inf values are considered equal to themselves (boolean)
        :return: Boolean.

        Example:
          >>> Quaternion(1, 2, 3, 4).is_close_to(Quaternion(1, 2, 3, 4))
          True
          >>> Quaternion(1, 2, 3, 4).is_close_to(Quaternion(1, 2.01, 3, 4))
          False
          >>> Quaternion(1, 2, 3, 4).is_close_to(Quaternion(1, 2.01, 3, 4), atol=1e-1)
          True
          >>> Quaternion(1, 2, 3, np.nan).is_close_to(Quaternion(1, 2, 3, np.nan), equal_nan=True)
          True
        """

        return arrays_are_close(self.q, another.q, rtol, atol, equal_nan, equal_inf)

    def rotation_angle(self):
        """
        Calculate the rotation angle associated with a normalized quaternion.
        Formula from p. 710 in Kagan, Y. Y., 1991. 3-D rotation of double-couple earthquake sources.

        :return: Float

        Quaternion case for Kagan, 1991, p.712:
          >>> are_close(Quaternion(0.696, 0.322, -0.152, 0.624).rotation_angle(), 91.8182771683)
          True
          >>> are_close(Quaternion(0.62471, 0.32267, 0.69465, 0.15195).rotation_angle(), 102.67846140868497)
          True
        """

        return 2 * degrees(acos(self.normalize().scalar))

    def to_rotation_matrix(self):
        """
        Computes the rotation matrix from the quaternion components.
        Formula as in:
        - Eq. 3.5 in Salamin, E., 1979. Application of quaternions to computation with rotations.
        - Eq. 10 in Kagan, Y. Y., 1991. 3-D rotation of double-couple earthquake sources.
        - Eq. 32 in Kagan, Y. Y., 2008. On geometric complexity of earthquake focal zone and fault system.

        :return: 3x3 numpy array
        """

        q0, q1, q2, q3 = self.normalize().components()

        q0q0 = q0 * q0
        q0q1 = q0 * q1
        q0q2 = q0 * q2
        q0q3 = q0 * q3

        q1q1 = q1 * q1
        q1q2 = q1 * q2
        q1q3 = q1 * q3

        q2q2 = q2 * q2
        q2q3 = q2 * q3

        q3q3 = q3 * q3

        a11 = q0q0 + q1q1 - q2q2 - q3q3
        a12 = 2*(q1q2 - q0q3)
        a13 = 2*(q0q2 + q1q3)

        a21 = 2*(q0q3 + q1q2)
        a22 = q0q0 - q1q1 + q2q2 - q3q3
        a23 = 2*(q2q3 - q0q1)

        a31 = 2*(q1q3 - q0q2)
        a32 = 2*(q0q1 + q2q3)
        a33 = q0q0 - q1q1 - q2q2 + q3q3

        return np.array([(a11, a12, a13),
                         (a21, a22, a23),
                         (a31, a32, a33)])


class QuaternionException(Exception):
    """
    Exception for Quaternion calculation.
    """

    pass


if __name__ == "__main__":

    import doctest
    doctest.testmod()
