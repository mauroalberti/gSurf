# -*- coding: utf-8 -*-


from __future__ import division

from math import sqrt, sin, cos, radians, acos, atan, atan2, degrees
import numpy as np

from typing import Dict, Tuple, List

from .default_parameters import *
from .mathematics import are_close
from .arrays import point_solution, arrays_are_close


class Point(object):
    """
    Cartesian point.
    Dimensions: 3D (space) + time
    """

    def __init__(self, x=np.nan, y=np.nan, z=np.nan, t=np.nan):
        """
        Construct a Point instance given 2, 3 or 4 float values.
        """

        self._p = np.array([x, y, z, t], dtype=np.float64)

    def __repr__(self):

        return "Point({:.4f}, {:.4f}, {:.4f}, {:.4f})".format(self.x, self.y, self.z, self.t)

    @classmethod
    def from_array(cls, a):
        """
        Class method to construct a point from a numpy 1x4 array.

        Example:
          >>> Point.from_array(np.array([1, 0, 1]))
          Point(1.0000, 0.0000, 1.0000, nan)
        """

        assert 3 <= a.size <= 4

        obj = cls()

        b = a.astype(np.float64)
        if b.size == 3:
            c = np.append(b, [np.nan])
        else:
            c = b
        obj._p = c

        return obj

    @property
    def v(self):
        """
        Return values as array

        Example:
          >>> arrays_are_close(Point(1, 0, 0).v, np.array([ 1.,  0.,  0., np.nan]), equal_nan=True)
          True
        """

        return self._p

    @property
    def x(self):
        """
        Return x value

        Example:
          >>> Point(1.5, 1, 1).x
          1.5
        """

        return self.v[0]

    @property
    def y(self):
        """
        Return y value

        Example:
          >>> Point(1.5, 3.0, 1).y
          3.0
        """
        return self.v[1]

    @property
    def z(self):
        """
        Return z value

        Example:
          >>> Point(1.5, 3.2, 41.).z
          41.0
        """
        return self.v[2]

    @property
    def t(self):
        """
        Return time value

        Example:
          >>> Point(1.5, 3.2, 41., 22.).t
          22.0
          >>> Point(1, 0, 0).t
          nan
        """
        return self.v[3]

    def clone(self):
        """
        Clone the point.

        Example:
          >>> Point(1, 1, 1).clone()
          Point(1.0000, 1.0000, 1.0000, nan)
        """

        return Point.from_array(self.v)

    def __sub__(self, another):
        """Return point difference

        Example:
          >>> Point(1., 1., 1.) - Point(1., 1., 1.)
          Point(0.0000, 0.0000, 0.0000, nan)
          >>> Point(1., 1.) - Point(1., 1.)
          Point(0.0000, 0.0000, nan, nan)
        """

        return Point.from_array(self.v - another.v)

    def __abs__(self):
        """
        Point distance from frame origin.

        Example:
          >>> abs(Point(3, 4, 0))
          5.0
          >>> abs(Point(0, 12, 5))
          13.0
          >>> abs(Point(0, 12, np.nan))
          nan
        """

        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def dist_3d(self, another):
        """
        Calculate Euclidean spatial distance between two points.

        Examples:
          >>> Point(1., 1., 1.).dist_3d(Point(4., 5., 1,))
          5.0
          >>> Point(1, 1, 1, 4).dist_3d(Point(4, 5, 1, 14))
          5.0
          >>> Point(1, np.nan, 1, 4).dist_3d(Point(4, 5, 1, 14))
          nan
        """

        return abs(self - another)

    def dist_2d(self, another):
        """
        Calculate horizontal (2D) distance between two points.

        Examples:
          >>> Point(1., 1., 1.).dist_2d(Point(4., 5., 7.))
          5.0
          >>> Point(1., np.nan, 1.).dist_2d(Point(4., np.nan, 7.))
          nan
        """

        return sqrt((self.x - another.x) ** 2 + (self.y - another.y) ** 2)

    def space_coincident(self, another, tolerance=MIN_SEPARATION_THRESHOLD):
        """
        Check spatial coincidence of two points

        Example:
          >>> Point(1., 0., -1.).space_coincident(Point(1., 1.5, -1.))
          False
          >>> Point(1., 0., 0.).space_coincident(Point(1., 0., 0.))
          True
          >>> Point(1., 0., 0.).space_coincident(Point(1., 0., np.nan))
          False
        """

        distance_2d = self.dist_2d(another)
        if np.isnan(distance_2d) or distance_2d > tolerance:
            return False

        distance_3d = self.dist_3d(another)
        if np.isnan(distance_3d) or distance_3d > tolerance:
            return False

        return True

    def translate(self, sx=0.0, sy=0.0, sz=0.0, st=0.0):
        """
        Create a new point shifted by given amount from the self instance.

        Example:
          >>> Point(1, 1, 1).translate(0.5, 1., 1.5)
          Point(1.5000, 2.0000, 2.5000, nan)
       """

        return Point(self.x + sx, self.y + sy, self.z + sz, self.t + st)

    def vect_offset(self, displ_vect):
        """
        Create a new point from the self, with offsets defined by a vector.

        Example:
          >>> Point(1, 2, 0).vect_offset(Vect(10, 5, 0))
          Point(11.0000, 7.0000, 0.0000, nan)
        """

        return Point(self.x + displ_vect.x,
                     self.y + displ_vect.y,
                     self.z + displ_vect.z,
                     self.t)

    @property
    def vector(self):
        """
        Create a vector based on the point coordinates

        Example:
          >>> Point(1, 1, 0, 5).vector
          Vect(1.0000, 1.0000, 0.0000)
        """

        return Vect(self.x, self.y, self.z)

    def delta_time(self, another):
        """
        Calculate the time difference between two points

        Example:
          >>> Point(1,1,1,4).delta_time(Point(1,1,2,5))
          1.0
          >>> Point(1,1,1,4).delta_time(Point(1,1,2,np.nan))
          nan
        """

        return another.t - self.t

    def speed(self, another):
        """
        Calculate the speed required to displace self to another.

        Example:
          >>> Point(1, 1, 1, 4).speed(Point(4, 5, 1, 14))
          0.5
          >>> Point(1, 1, 1, 4).speed(Point(4, 5, 1, 4))
          inf
          >>> Point(1, 1, 1, 4).speed(Point(1, 1, 1, 4))
          nan
          >>> Point(1, 1, 1, 4).speed(Point(4, 5, 1))
          nan
        """

        delta_s = self.dist_3d(another)
        delta_t = self.delta_time(another)
        if delta_t == 0.0:
            if delta_s == 0.0:
                return np.nan
            else:
                return np.Infinity
        else:
            return delta_s / delta_t


class Vect(object):
    """
    Cartesian vector, 3D
    Right-handed rectangular Cartesian coordinate system (ENU):
    x axis -> East
    y axis -> North
    z axis -> Up
    """

    def __init__(self, x=0, y=0, z=0):
        """
        Vect constructor.

        Example;
          >>> Vect(1, 0, 1)
          Vect(1.0000, 0.0000, 1.0000)
          >>> Vect(1, np.nan, 1)
          Traceback (most recent call last):
          ...
          AssertionError: all vector components must be finite
          >>> Vect(1, 0, np.inf)
          Traceback (most recent call last):
          ...
          AssertionError: all vector components must be finite
          >>> Vect(0, 0, 0)
          Vect(0.0000, 0.0000, 0.0000)
        """

        assert np.isfinite(x) and np.isfinite(y) and np.isfinite(z), "all vector components must be finite"
        self._v = np.array([x, y, z], dtype=np.float64)

    @classmethod
    def from_array(cls, a):
        """
        Class method to construct a vector from a numpy 1x3 array.

        Example:
          >>> Vect.from_array(np.array([1, 0, 1]))
          Vect(1.0000, 0.0000, 1.0000)
        """

        obj = cls()

        assert a.size == 3
        b = a.astype(np.float64)
        obj._v = b
        return obj

    @property
    def v(self):
        """
        Return the vector values as array

        Example:
          >>> arrays_are_close(Vect(1, 1, 0).v, np.array([1., 1., 0.]))
          True
        """

        return self._v

    @property
    def x(self) -> float:
        """
        Return x value

        Example:
          >>> Vect(1, 2, 0).x
          1.0
        """

        return self.v[0]

    @property
    def y(self) -> float:
        """
        Return y value

        Example:
          >>> Vect(1, 2, 0).y
          2.0
        """

        return self.v[1]

    @property
    def z(self) -> float:
        """
        Return z value

        Example:
          >>> Vect(1, 2, 0).z
          0.0
        """

        return self.v[2]

    def components(self) -> Tuple[float, float, float]:
        """
        Returns the vector components as a tuple of three values.

        :return: the vector components (x, y, z).
        :rtype: a tuple of three floats.

        Examples:
          >>> Vect(1, 0, 3).components()
          (1.0, 0.0, 3.0)
        """

        return self.x, self.y, self.z

    @property
    def valid(self) -> bool:
        """
        Check if the Vect instance components are not all zero.

        :return: Boolean

        Example:
          >>> Vect(1, 2, 0).valid
          True
          >>> Vect(0.0, 0.0, 0.0).valid
          False
        """

        return not self.is_near_zero

    @property
    def len_2d(self):
        """
        2D Vector magnitude.

        Example:
          >>> Vect(3, 4).len_2d
          5.0
          >>> Vect(12, 5, 3).len_2d
          13.0
        """

        return sqrt(self.x * self.x + self.y * self.y)

    @property
    def len_3d(self):
        """
        3D Vector magnitude.

        Example:
          >>> Vect(12, 0, 5).len_3d
          13.0
          >>> Vect(3, 0, 4).len_3d
          5.0
        """

        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    @property
    def is_near_zero(self):
        """
        Check if the Vect instance lenght is near zero.

        :return: Boolean

        Example:
          >>> Vect(1, 2, 0).is_near_zero
          False
          >>> Vect(0.0, 0.0, 0.0).is_near_zero
          True
        """

        return are_close(self.len_3d, 0)

    @property
    def is_near_unit(self):
        """
        Check if the Vect instance lenght is near unit.

        :return: Boolean

        Example:
          >>> Vect(1, 2, 0).is_near_unit
          False
          >>> Vect(0.0, 1.0, 0.0).is_near_unit
          True
        """

        return are_close(self.len_3d, 1)

    def __sub__(self, another):
        """
        Return vector difference.

        Example:
          >>> Vect(1., 1., 1.) - Vect(1., 1., 1.)
          Vect(0.0000, 0.0000, 0.0000)
          >>> Vect(0., 1., 4.) - Vect(7., 3., 1.)
          Vect(-7.0000, -2.0000, 3.0000)
        """

        return Vect.from_array(self.v - another.v)

    def __eq__(self, another):
        """
        Return True if vectors are equal.

        Example:
          >>> Vect(1., 1., 1.) == Vect(1, 1, 1)
          True
        """

        return (self - another).len_3d < MIN_VECTOR_MAGN_DIFF

    def __ne__(self, another):
        """
        Return False if vectors are equal.

        Example:
          >>> Vect(1., 1., 1.) != Vect(0., 0., 0.)
          True
        """

        return not (self == another)

    def __repr__(self):

        return "Vect({:.4f}, {:.4f}, {:.4f})".format(self.x, self.y, self.z)

    def __add__(self, another):
        """
        Sum of two vectors.

        Example:
          >>> Vect(1, 0, 0) + Vect(0, 1, 1)
          Vect(1.0000, 1.0000, 1.0000)
          >>> Vect(1, 1, 1) + Vect(-1, -1, -1)
          Vect(0.0000, 0.0000, 0.0000)
        """

        return Vect.from_array(self.v + another.v)

    def clone(self):
        """
        Clone the vector.

        Example:
          >>> Vect(1, 1, 1).clone()
          Vect(1.0000, 1.0000, 1.0000)
        """
        return Vect.from_array(self.v)

    def scale(self, scale_factor):
        """
        Create a scaled vector.

        Example;
          >>> Vect(1,0,1).scale(2.5)
          Vect(2.5000, 0.0000, 2.5000)
          >>> Vect(1,0,1).scale(np.nan)
          Traceback (most recent call last):
          ...
          AssertionError: scale factor for vector must be finite
          >>> Vect(1,0,1).scale(np.inf)
          Traceback (most recent call last):
          ...
          AssertionError: scale factor for vector must be finite
        """

        assert np.isfinite(scale_factor), "scale factor for vector must be finite"
        return Vect.from_array(self.v * scale_factor)

    def versor(self):
        """
        Calculate versor in xyz space.

        Example:
          >>> Vect(5, 0, 0).versor()
          Vect(1.0000, 0.0000, 0.0000)
          >>> Vect(0, 0, -1).versor()
          Vect(0.0000, 0.0000, -1.0000)
          >>> Vect(0, 0, 0).versor()
          Traceback (most recent call last):
          ...
          AssertionError: vector must be valid (not zero-valued)
        """

        assert self.valid, "vector must be valid (not zero-valued)"
        return self.scale(1.0 / self.len_3d)

    def to_2d(self):
        """
        Create equivalent vector in xy space.

        Example:
          >>> Vect(5, 16, 43).to_2d()
          Vect(5.0000, 16.0000, 0.0000)
        """

        return Vect(self.x, self.y, 0.0)

    def versor_2d(self):
        """
        Create 2D versor equivalent of the current vector

        :return: Vector

        Example:
          >>> Vect(7, 0, 10).versor_2d()
          Vect(1.0000, 0.0000, 0.0000)
        """

        return self.to_2d().versor()

    def invert(self):
        """
        Create a new vector with inverted direction.

        Examples:
          >>> Vect(1, 1, 1).invert()
          Vect(-1.0000, -1.0000, -1.0000)
          >>> Vect(2, -1, 4).invert()
          Vect(-2.0000, 1.0000, -4.0000)
        """

        return self.scale(-1)

    @property
    def is_upward(self):
        """
        Check that a vector is upward-directed.
         
        Example:
          >>> Vect(0,0,1).is_upward
          True
          >>> Vect(0,0,-0.5).is_upward
          False
        """

        return self.z > 0.0

    @property
    def is_downward(self):
        """
        Check that a vector is downward-directed.

        Example:
          >>> Vect(0,0,1).is_downward
          False
          >>> Vect(0,0,-0.5).is_downward
          True
        """

        return self.z < 0.0

    def upward(self):
        """
        Calculate a new upward-pointing vector.

        Example:
          >>> Vect(1, 1, 1).upward()
          Vect(1.0000, 1.0000, 1.0000)
          >>> Vect(-1, -1, -1).upward()
          Vect(1.0000, 1.0000, 1.0000)
        """

        if self.z < 0.0:
            return self.scale(-1.0)
        else:
            return self.clone()

    def downward(self):
        """
        Calculate a new vector downward-pointing.

        Example:
          >>> Vect(1, 1, 1).downward()
          Vect(-1.0000, -1.0000, -1.0000)
          >>> Vect(-1, -1, -1).downward()
          Vect(-1.0000, -1.0000, -1.0000)
        """

        if self.z > 0.0:
            return self.scale(-1.0)
        else:
            return self.clone()

    @property
    def slope(self):
        """
        Slope of a vector expressed as degrees.
        Positive when vector is downward pointing or horizontal,
        negative when upward pointing.

        Example:
          >>> Vect(1, 0, -1).slope
          45.0
          >>> Vect(1, 0, 1).slope
          -45.0
          >>> Vect(0, 1, 0).slope
          0.0
          >>> Vect(0, 0, 1).slope
          -90.0
          >>> Vect(0, 0, -1).slope
          90.0
          >>> Vect(0, 0, 0).slope
          Traceback (most recent call last):
          ...
          AssertionError: vector must be valid (not zero-valued)
        """

        assert self.valid, "vector must be valid (not zero-valued)"

        hlen = self.len_2d
        if hlen == 0.0:
            if self.z > 0.:
                return -90.
            elif self.z < 0.:
                return 90.
            else:
                raise Exception("Zero-valued vector")
        else:
            slope = - degrees(atan(self.z / self.len_2d))
            if abs(slope) > MIN_SCALAR_VALUE:
                return slope
            else:
                return 0.

    def gvect(self):
        """
        Calculate the geological vector parallel to the Vect instance.
        Trend range: [0°, 360°[
        Plunge range: [-90°, 90°], with negative values for upward-pointing
        geological axes and positive values for downward-pointing axes.

        Examples:
          >>> Vect(1, 1, 1).gvect()
          GVect(045.00, -35.26)
          >>> Vect(0, 1, 1).gvect()
          GVect(000.00, -45.00)
          >>> Vect(1, 0, 1).gvect()
          GVect(090.00, -45.00)
          >>> Vect(0, 0, 1).gvect()
          GVect(000.00, -90.00)
          >>> Vect(0, 0, -1).gvect()
          GVect(000.00, +90.00)
          >>> Vect(-1, 0, 0).gvect()
          GVect(270.00, +00.00)
          >>> Vect(0, -1, 0).gvect()
          GVect(180.00, +00.00)
          >>> Vect(-1, -1, 0).gvect()
          GVect(225.00, +00.00)
          >>> Vect(0, 0, 0).gvect()
          Traceback (most recent call last):
          ...
          AssertionError: vector must be valid (not zero-valued)
        """

        assert self.valid, "vector must be valid (not zero-valued)"

        plunge = self.slope  # upward pointing -> negative value, downward -> positive

        unit_vect = self.versor()
        if unit_vect.y == 0. and unit_vect.x == 0:
            trend = 0.
        else:
            trend = (90. - degrees(atan2(unit_vect.y, unit_vect.x))) % 360.

        return GVect(trend, plunge)

    def gaxis(self):
        """
        Calculate the geological axis parallel to the Vect instance.
        Trend range: [0°, 360°[
        Plunge range: [-90°, 90°], with negative values for upward-pointing
        geological axes and positive values for downward-pointing axes.

        Examples:
          >>> Vect(0, 1, 1).gaxis()
          GAxis(000.00, -45.00)
          >>> Vect(1, 0, 1).gaxis()
          GAxis(090.00, -45.00)
          >>> Vect(0, 0, 1).gaxis()
          GAxis(000.00, -90.00)
          >>> Vect(0, 0, -1).gaxis()
          GAxis(000.00, +90.00)
          >>> Vect(-1, 0, 0).gaxis()
          GAxis(270.00, +00.00)
          >>> Vect(0, -1, 0).gaxis()
          GAxis(180.00, +00.00)
          >>> Vect(-1, -1, 0).gaxis()
          GAxis(225.00, +00.00)
          >>> Vect(0, 0, 0).gaxis()
          Traceback (most recent call last):
          ...
          AssertionError: vector must be valid (not zero-valued)
        """

        assert self.valid, "vector must be valid (not zero-valued)"

        return self.gvect().as_axis()

    def sp(self, another):
        """
        Vector scalar __mul__.

        Examples:
          >>> Vect(1, 0, 0).sp(Vect(1, 0, 0))
          1.0
          >>> Vect(1, 0, 0).sp(Vect(0, 1, 0))
          0.0
          >>> Vect(1, 0, 0).sp(Vect(-1, 0, 0))
          -1.0
        """

        return self.x * another.x + self.y * another.y + self.z * another.z

    def cos_angle(self, another):
        """
        Return the cosine of the angle between two vectors.

        Examples:
          >>> Vect(1,0,0).cos_angle(Vect(0,0,1))
          0.0
          >>> Vect(1,0,0).cos_angle(Vect(-1,0,0))
          -1.0
          >>> Vect(1,0,0).cos_angle(Vect(1,0,0))
          1.0
          >>> Vect(0, 0, 0).cos_angle(Vect(1,0,0))
          Traceback (most recent call last):
          ...
          AssertionError: vector must be valid (not zero-valued)
          >>> Vect(1, 0, 0).cos_angle(Vect(0,0,0))
          Traceback (most recent call last):
          ...
          AssertionError: vector must be valid (not zero-valued)
        """

        assert self.valid, "vector must be valid (not zero-valued)"
        assert another.valid, "vector must be valid (not zero-valued)"

        val = self.sp(another) / (self.len_3d * another.len_3d)
        if val > 1.0:
            return 1.0
        elif val < -1.0:
            return -1.0
        else:
            return val

    def angle(self, another):
        """
        Calculate angle between two vectors, as degrees
        in 0° - 180° range.

        Example:
          >>> Vect(1, 0, 0).angle(Vect(0, 0, 1))
          90.0
          >>> Vect(1, 0, 0).angle(Vect(-1, 0, 0))
          180.0
          >>> Vect(0, 0, 1).angle(Vect(0, 0, -1))
          180.0
          >>> Vect(1, 1, 1).angle(Vect(1, 1,1 ))
          0.0
          >>> Vect(0, 0, 0).angle(Vect(1,0,0))
          Traceback (most recent call last):
          ...
          AssertionError: vector must be valid (not zero-valued)
          >>> Vect(1, 0, 0).angle(Vect(0,0,0))
          Traceback (most recent call last):
          ...
          AssertionError: vector must be valid (not zero-valued)
        """

        assert self.valid, "vector must be valid (not zero-valued)"
        assert another.valid, "vector must be valid (not zero-valued)"

        return degrees(acos(self.cos_angle(another)))

    def almost_parallel(self, another, angle_tolerance=VECTOR_ANGLE_THRESHOLD):
        """
        Check that two Vect are sub-parallel,

        :param another: a Vect instance
        :param angle_tolerance: the maximum allowed divergence angle (in degrees)
        :return: Boolean

        Examples:
          >>> Vect(1, 0, 0).almost_parallel(Vect(1, 0, 0))
          True
          >>> Vect(1, 0, 0).almost_parallel(Vect(0, 0, 1))
          False
          >>> Vect(1, 0, 0).almost_parallel(Vect(-1, 0, 0))
          False
          >>> Vect(0, 0, 0).almost_parallel(Vect(1,0,0))
          Traceback (most recent call last):
          ...
          AssertionError: vector must be valid (not zero-valued)
          >>> Vect(1, 0, 0).almost_parallel(Vect(0,0,0))
          Traceback (most recent call last):
          ...
          AssertionError: vector must be valid (not zero-valued)
        """

        assert self.valid, "vector must be valid (not zero-valued)"
        assert another.valid, "vector must be valid (not zero-valued)"

        return self.angle(another) <= angle_tolerance

    def is_suborthogonal(self, another):
        """
        Check whether two vectors are sub-orhogonal.

        :param another:
        :return: Boolean

        Example:
          >>> Vect(1, 0, 0).is_suborthogonal(Vect(0, 1, 0))
          True
          >>> Vect(1, 0, 0).is_suborthogonal(Vect(0, 1, 1))
          True
          >>> Vect(1, 0, 0).is_suborthogonal(Vect(0, 0.9999999999999, 0))
          True
        """

        return are_close(0, self.cos_angle(another))

    def vp(self, another):
        """
        Vector __mul__.

        Examples:
          >>> Vect(1, 0, 0).vp(Vect(0, 1, 0))
          Vect(0.0000, 0.0000, 1.0000)
          >>> Vect(1, 0, 0).vp(Vect(1, 0, 0))
          Vect(0.0000, 0.0000, 0.0000)
          >>> (Vect(1, 0, 0).vp(Vect(-1, 0, 0))).is_near_zero
          True
        """

        return Vect.from_array(np.cross(self.v, another.v))

    def by_matrix(self, array3x3):
        """
        Matrix multiplication of a vector.

        """

        return Vect.from_array(array3x3.dot(self.v))


def plng2colatTop(plunge: float) -> float:
    """
    Calculates the colatitude angle from the top.

    :param plunge: an angle from -90° (upward-pointing) to 90° (downward-pointing)
    :type plunge: float
    :return: the colatitude angle
    :rtype: float

    Examples:
      >>> plng2colatTop(90)
      180.0
      >>> plng2colatTop(45)
      135.0
      >>> plng2colatTop(0)
      90.0
      >>> plng2colatTop(-45)
      45.0
      >>> plng2colatTop(-90)
      0.0
    """

    return 90.0 + plunge


def plng2colatBottom(plunge: float) -> float:
    """
    Calculates the colatitude angle from the bottom.

    :param plunge: an angle from -90° (upward-pointing) to 90° (downward-pointing)
    :type plunge: float
    :return: the colatitude angle
    :rtype: float

    Examples:
      >>> plng2colatBottom(90)
      0.0
      >>> plng2colatBottom(45)
      45.0
      >>> plng2colatBottom(0)
      90.0
      >>> plng2colatBottom(-45)
      135.0
      >>> plng2colatBottom(-90)
      180.0
    """

    return 90.0 - plunge


class GVect(object):
    """
    Geological vector.
    Defined by trend and plunge (both in degrees):
     - trend: [0.0, 360.0[ clockwise, from 0 (North):
     - plunge: [-90.0, 90.0].
    """

    def __init__(self, src_trend, src_plunge):
        """
        Geological vector constructor.
        src_trend: Trend range: [0.0, 360.0[ clockwise, from 0 (North)
        src_plunge: Plunge: [-90.0, 90.0],
        negative value: upward pointing axis, positive values: downward axis;
            
        Example:
          >>> GVect(120.2, -27.4)
          GVect(120.20, -27.40)
          >>> GVect(315.0, -0.0)
          GVect(315.00, -00.00)
          >>> GVect(23, 40.0)
          GVect(023.00, +40.00)
       """

        # assert -90.0 <= float(src_plunge) <= 90.0, "plunge must be between -90° and +90° (comprised)"
        self._trend = float(src_trend) % 360.0
        self._plunge = float(src_plunge)

    @property
    def tr(self):
        """
        Return trend of the geological direction.
        Range is [0, 360[

        Example:
          >>> GVect(420, -17).tr
          60.0
          >>> GVect(-20, 49).tr
          340.0
        """

        return self._trend

    @property
    def pl(self):
        """
        Return plunge of the geological direction.
        Range is [-90, 90]

        Example:
          >>> GVect(420, -17).pl
          -17.0
        """

        return self._plunge

    @property
    def tp(self):
        """
        Return trend and plunge of the geological direction.

        Example:
          >>> GVect(-90, -45).tp
          (270.0, -45.0)
        """

        return self.tr, self.pl

    def __repr__(self):

        return "GVect({:06.2f}, {:+06.2f})".format(*self.tp)

    @property
    def colatitude_north(self) -> float:
        """
        Calculates the colatitude from the North (top).

        :return: an angle between 0 and 180 (in degrees).
        :rtype: float

        Examples:
          >>> GVect(320, 90).colatitude_north
          180.0
          >>> GVect(320, 45).colatitude_north
          135.0
          >>> GVect(320, 0).colatitude_north
          90.0
          >>> GVect(320, -45).colatitude_north
          45.0
          >>> GVect(320, -90).colatitude_north
          0.0
        """

        return plng2colatTop(self.pl)

    @property
    def colatitude_south(self) -> float:
        """
        Calculates the colatitude from the South (bottom).

        :return: an angle between 0 and 180 (in degrees).
        :rtype: float

        Examples:
          >>> GVect(320, 90).colatitude_south
          0.0
          >>> GVect(320, 45).colatitude_south
          45.0
          >>> GVect(320, 0).colatitude_south
          90.0
          >>> GVect(320, -45).colatitude_south
          135.0
          >>> GVect(320, -90).colatitude_south
          180.0
        """

        return plng2colatBottom(self.pl)

    def copy(self):
        """
        Return a copy of the GVect instance.
        
        Example:
          >>> GVect(10, 20).copy()
          GVect(010.00, +20.00)
        """

        return GVect(*(self.tp))

    def opposite(self):
        """
        Return the opposite GVect.
        
        Example:
          >>> GVect(0, 30).opposite()
          GVect(180.00, -30.00)
          >>> GVect(315, 10).opposite()
          GVect(135.00, -10.00)
          >>> GVect(135, 0).opposite()
          GVect(315.00, -00.00)
        """

        trend = (self.tr + 180.) % 360.
        plunge = -self.pl

        return GVect(trend, plunge)

    def versor(self):
        """
        Return the unit vector corresponding to the geological vector.

        Examples:
          >>> GVect(0, 90).versor()
          Vect(0.0000, 0.0000, -1.0000)
          >>> GVect(0, -90).versor()
          Vect(0.0000, 0.0000, 1.0000)
          >>> GVect(90, 90).versor()
          Vect(0.0000, 0.0000, -1.0000)
          
        """

        north_coord = cos(radians(self.pl)) * cos(radians(self.tr))
        east_coord = cos(radians(self.pl)) * sin(radians(self.tr))
        down_coord = sin(radians(self.pl))

        return Vect(east_coord, north_coord, -down_coord)

    @property
    def is_upward(self):
        """
        Check whether the instance is pointing upward or horizontal.

        Examples:
          >>> GVect(10, 15).is_upward
          False
          >>> GVect(257.4, 0.0).is_upward
          False
          >>> GVect(90, -45).is_upward
          True
        """

        return self.versor().z > 0.0

    @property
    def is_downward(self):
        """
        Check whether the instance is pointing downward or horizontal.

        Examples:
          >>> GVect(10, 15).is_downward
          True
          >>> GVect(257.4, 0.0).is_downward
          False
          >>> GVect(90, -45).is_downward
          False
        """

        if self.versor().z < 0.0:
            return True
        else:
            return False

    def upward(self):
        """
        Return upward-point geological vector.

        Examples:
          >>> GVect(90, -45).upward().almost_parallel(GVect(90.0, -45.0))
          True
          >>> GVect(180, 45).upward().almost_parallel(GVect(0.0, -45.0))
          True
          >>> GVect(0, 0).upward().almost_parallel(GVect(0.0, 0.0))
          True
          >>> GVect(0, 90).upward().almost_parallel(GVect(180.0, -90.0))
          True
          >>> GVect(90, -45).upward().almost_parallel(GVect(90.0, -35.0))
          False
          >>> GVect(180, 45).upward().almost_parallel(GVect(10.0, -45.0))
          False
          >>> GVect(0, 0).upward().almost_parallel(GVect(170.0, 0.0))
          False
          >>> GVect(0, 90).upward().almost_parallel(GVect(180.0, -80.0))
          False
        """

        if not self.is_downward:
            return self.copy()
        else:
            return self.opposite()

    def downward(self):
        """
        Return downward-pointing geological vector.

        Examples:
          >>> GVect(90, -45).downward().almost_parallel(GVect(270.0, 45.0))
          True
          >>> GVect(180, 45).downward().almost_parallel(GVect(180.0, 45.0))
          True
          >>> GVect(0, 0).downward().almost_parallel(GVect(180.0, 0.0))
          False
          >>> GVect(0, 90).downward().almost_parallel(GVect(0.0, 90.0))
          True
          >>> GVect(90, -45).downward().almost_parallel(GVect(270.0, 35.0))
          False
          >>> GVect(180, 45).downward().almost_parallel(GVect(170.0, 45.0))
          False
          >>> GVect(0, 0).downward().almost_parallel(GVect(180.0, 10.0))
          False
          >>> GVect(0, 90).downward().almost_parallel(GVect(0.0, 80.0))
          False
        """

        if not self.is_upward:
            return self.copy()
        else:
            return self.opposite()

    def angle(self, another):
        """
        Calculate angle (in degrees) between the two GVect instances.
        Range is 0°-180°.

        Examples:
          >>> are_close(GVect(0, 90).angle(GVect(90, 0)), 90)
          True
          >>> are_close(GVect(0, 0).angle(GVect(270, 0)), 90)
          True
          >>> are_close(GVect(0, 0).angle(GVect(0, 0)), 0)
          True
          >>> are_close(GVect(0, 0).angle(GVect(180, 0)), 180)
          True
          >>> are_close(GVect(90, 0).angle(GVect(270, 0)), 180)
          True
        """

        return self.versor().angle(another.versor())

    def almost_parallel(self, another, angle_tolerance=VECTOR_ANGLE_THRESHOLD):
        """
        Check that two GVect are sub-parallel,

        :param another: a GVect instance
        :param angle_tolerance: the maximum allowed divergence angle (in degrees)
        :return: Boolean

        Examples:
          >>> GVect(0, 90).almost_parallel(GVect(90, 0))
          False
          >>> GVect(0, 0).almost_parallel(GVect(0, 1e-6))
          True
          >>> GVect(0, 90).almost_parallel(GVect(180, 0))
          False
          >>> GVect(0, 90).almost_parallel(GVect(0, -90))
          False
        """

        return self.angle(another) <= angle_tolerance

    def normal_gplane(self):
        """
        Return the geological plane that is normal to the geological vector.

        Examples:
          >>> GVect(0, 45).normal_gplane()
          GPlane(180.00, +45.00)
          >>> GVect(0, -45).normal_gplane()
          GPlane(000.00, +45.00)
          >>> GVect(0, 90).normal_gplane()
          GPlane(180.00, +00.00)
        """

        down_axis = self.downward()
        dipdir = (down_axis.tr + 180.0) % 360.0
        dipangle = 90.0 - down_axis.pl

        return GPlane(dipdir, dipangle)

    def common_plane(self, another):
        """
        Calculate GPlane instance defined by the two GVect instances.

        Examples:
          >>> GVect(0, 0).common_plane(GVect(90, 0)).almost_parallel(GPlane(180.0, 0.0))
          True
          >>> GVect(0, 0).common_plane(GVect(90, 90)).almost_parallel(GPlane(90.0, 90.0))
          True
          >>> GVect(45, 0).common_plane(GVect(135, 45)).almost_parallel(GPlane(135.0, 45.0))
          True
          >>> GVect(315, 45).common_plane(GVect(135, 45)).almost_parallel(GPlane(225.0, 90.0))
          True
          >>> GVect(0, 0).common_plane(GVect(90, 0)).almost_parallel(GPlane(180.0, 10.0))
          False
          >>> GVect(0, 0).common_plane(GVect(90, 90)).almost_parallel(GPlane(90.0, 80.0))
          False
          >>> GVect(45, 0).common_plane(GVect(135, 45)).almost_parallel(GPlane(125.0, 45.0))
          False
          >>> GVect(315, 45).common_plane(GVect(135, 45)).almost_parallel(GPlane(225.0, 80.0))
          False
        """

        normal_versor = self.versor().vp(another.versor())
        return normal_versor.gvect().normal_gplane()

    def as_axis(self):
        """
        Create GAxis instance with the same attitude as the self instance.

        Example:
          >>> GVect(220, 32).as_axis()
          GAxis(220.00, +32.00)
        """

        return GAxis(*self.tp)

    def normal_gvect(self, another):
        """
        Calculate the GVect instance that is normal to the two provided sources.
        Angle between sources must be larger than MIN_ANGLE_DEGR_DISORIENTATION,
        otherwise a SubparallelLineationException will be raised.
        
        Example:
          >>> GVect(0, 0).normal_gvect(GVect(4, 0))
          Traceback (most recent call last):
          ...
          SubparallelLineationException: Sources must not be sub- or anti-parallel
          >>> GVect(0, 0).normal_gvect(GVect(179, 0))
          Traceback (most recent call last):
          ...
          SubparallelLineationException: Sources must not be sub- or anti-parallel
          >>> GVect(0, 0).normal_gvect(GVect(5.1, 0))
          GVect(000.00, +90.00)
          >>> GVect(90, 45).normal_gvect(GVect(90, 0))
          GVect(180.00, +00.00)
        """

        if not MIN_ANGLE_DEGR_DISORIENTATION <= self.angle(another) <= 180. - MIN_ANGLE_DEGR_DISORIENTATION:
            raise SubparallelLineationException("Sources must not be sub- or anti-parallel")

        return self.versor().vp(another.versor()).gvect()


class GAxis(GVect):
    """
    Geological axis.
    While GAxis is non-directional, the geological vector (GVect) is directional.
    Defined by trend and plunge (both in degrees):
     - trend: [0.0, 360.0[ clockwise, from 0 (North):
     - plunge: [-90.0, 90.0].
    """

    def __init__(self, src_trend, src_plunge):

        super(GAxis, self).__init__(src_trend, src_plunge)

    def __repr__(self):

        return "GAxis({:06.2f}, {:+06.2f})".format(*self.tp)

    def as_gvect(self):
        """
        Create GVect instance with the same attitude as the self instance.
        
        Example:
          >>> GAxis(220, 32).as_gvect()
          GVect(220.00, +32.00)
        """

        return GVect(*self.tp)

    def as_versor(self):
        """
        Create a unit Vect instance with the same attitude as the self instance.

        Example:
          >>> GAxis(90, 0).as_versor()
          Vect(1.0000, 0.0000, -0.0000)
          >>> GAxis(0, 45).as_versor()
          Vect(0.0000, 0.7071, -0.7071)
          >>> GAxis(0, 90).as_versor()
          Vect(0.0000, 0.0000, -1.0000)
          >>> GAxis(270, -90).as_versor()
          Vect(-0.0000, -0.0000, 1.0000)
        """

        return GVect(*self.tp).versor()

    def angle(self, another):
        """
        Calculate angle (in degrees) between the two GAxis instances.
        Range: 0.0 - 90.0

        Examples:
          >>> are_close(GAxis(0, 90).angle(GAxis(90, 0)), 90)
          True
          >>> are_close(GAxis(0, 0).angle(GAxis(270, 0)), 90)
          True
          >>> are_close(GAxis(0, 0).angle(GAxis(0, 0)), 0)
          True
          >>> are_close(GAxis(0, 0).angle(GAxis(180, 0)), 0)
          True
          >>> are_close(GAxis(0, 0).angle(GAxis(179, 0)), 1)
          True
          >>> are_close(GAxis(0, -90).angle(GAxis(0, 90)), 0)
          True
          >>> are_close(GAxis(90, 0).angle(GAxis(315, 0)), 45)
          True
        """

        angle_vers = self.versor().angle(another.versor())
        return min(angle_vers, 180. - angle_vers)

    def almost_parallel(self, another, tolerance_angle=VECTOR_ANGLE_THRESHOLD):
        """
        Check that two GAxis are sub-parallel

        :param another: a GAxis instance
        :param tolerance_angle: the maximum allowed divergence angle (in degrees)
        :return: Boolean

         Examples:
          >>> GAxis(0, 90).almost_parallel(GAxis(90, 0))
          False
          >>> GAxis(0, 0).almost_parallel(GAxis(0, 1.e-6))
          True
          >>> GAxis(0, 0).almost_parallel(GAxis(180, 0))
          True
          >>> GAxis(90, 0).almost_parallel(GAxis(270, 0))
          True
          >>> GAxis(0, 90).almost_parallel(GAxis(0, -90))
          True
        """

        return self.angle(another) <= tolerance_angle

    def is_suborthogonal(self, another, angle_tolerance=VECTOR_ANGLE_THRESHOLD):
        """
        Check that two GAxis are sub-orthogonal

        :param another: a GAxis instance
        :param angle_tolerance: the maximum allowed divergence angle (in degrees) from orthogonality
        :return: Boolean

         Examples:
          >>> GAxis(0, 90).is_suborthogonal(GAxis(90, 0))
          True
          >>> GAxis(0, 0).is_suborthogonal(GAxis(0, 1.e-6))
          False
          >>> GAxis(0, 0).is_suborthogonal(GAxis(180, 0))
          False
          >>> GAxis(90, 0).is_suborthogonal(GAxis(270, 89.5))
          True
          >>> GAxis(0, 90).is_suborthogonal(GAxis(0, 0.5))
          True
        """

        return self.angle(another) >= 90.0 - angle_tolerance

    def normal_gplane(self):
        """
        Calculate the geological plane that is normal to the given GAxis instance.
        
        Example:
          >>> GAxis(0, 90).normal_gplane().almost_parallel(GPlane(180.00, 0.00))
          True
          >>> GAxis(0, 0).normal_gplane().almost_parallel(GPlane(0.00, 90.00))
          True
          >>> GAxis(45, 45).normal_gplane().almost_parallel(GPlane(225.00, 45.00))
          True
          >>> GAxis(0, 90).normal_gplane().almost_parallel(GPlane(170.00, 10.00))
          False
          >>> GAxis(0, 0).normal_gplane().almost_parallel(GPlane(20.00, 90.00))
          False
          >>> GAxis(45, 45).normal_gplane().almost_parallel(GPlane(215.00, 45.00))
          False
        """

        return self.as_gvect().normal_gplane()

    @property
    def is_upward(self):
        """
        Check whether the instance is pointing upward.

        Examples:
          >>> GAxis(10, 15).is_upward
          False
          >>> GAxis(257.4, 0.0).is_upward
          False
          >>> GAxis(90, -45).is_upward
          True
        """

        return self.as_gvect().is_upward

    @property
    def is_downward(self):
        """
        Check whether the instance is pointing downward.

        Examples:
          >>> GAxis(10, 15).is_downward
          True
          >>> GAxis(257.4, 0.0).is_downward
          False
          >>> GAxis(90, -45).is_downward
          False
        """

        return self.as_gvect().is_downward

    def is_subhorizontal(self, max_dip_angle=DIP_ANGLE_THRESHOLD):
        """
        Check whether the instance is almost horizontal.

        Examples:
          >>> GAxis(10, 15).is_subhorizontal()
          False
          >>> GAxis(257, 2).is_subhorizontal()
          True
          >>> GAxis(90, -5).is_subhorizontal()
          False
        """

        return abs(self.pl) < max_dip_angle

    def upward(self):
        """
        Return upward-point geological axis.

        Examples:
          >>> GAxis(90, -45).upward().almost_parallel(GAxis(90.0, -45.0))
          True
          >>> GAxis(180, 45).upward().almost_parallel(GAxis(0.0, -45.0))
          True
          >>> GAxis(0, 0).upward().almost_parallel(GAxis(180.0, 0.0))
          True
          >>> GAxis(0, 90).upward().almost_parallel(GAxis(180.0, -90.0))
          True
          >>> GAxis(90, -45).upward().almost_parallel(GAxis(80.0, -45.0))
          False
          >>> GAxis(180, 45).upward().almost_parallel(GAxis(10.0, -45.0))
          False
          >>> GAxis(0, 0).upward().almost_parallel(GAxis(180.0, 10.0))
          False
          >>> GAxis(0, 90).upward().almost_parallel(GAxis(180.0, -80.0))
          False
        """

        return self.as_gvect().upward().as_axis()

    def downward(self):
        """
        Return downward-pointing geological vector.

        Examples:
          >>> GAxis(90, -45).downward().almost_parallel(GAxis(270, 45))
          True
          >>> GAxis(180, 45).downward().almost_parallel(GAxis(180, 45))
          True
          >>> GAxis(0, 0).downward().almost_parallel(GAxis(180, 0))
          True
          >>> GAxis(0, 90).downward().almost_parallel(GAxis(0, 90))
          True
          >>> GAxis(0, 0).downward().almost_parallel(GAxis(170, 0))
          False
          >>> GAxis(0, 90).downward().almost_parallel(GAxis(0, 80))
          False
        """

        return self.as_gvect().downward().as_axis()

    def common_plane(self, another):
        """
        Calculate GPlane instance defined by the two GAxis instances.

        Examples:
          >>> GAxis(0, 0).common_plane(GAxis(90, 0)).almost_parallel(GPlane(180.0, 0.0))
          True
          >>> GAxis(0, 0).common_plane(GAxis(90, 90)).almost_parallel(GPlane(90.0, 90.0))
          True
          >>> GAxis(45, 0).common_plane(GAxis(135, 45)).almost_parallel(GPlane(135.0, 45.0))
          True
          >>> GAxis(315, 45).common_plane(GAxis(135, 45)).almost_parallel(GPlane(225.0, 90.0))
          True
          >>> GAxis(45, 0).common_plane(GAxis(135, 45)).almost_parallel(GPlane(135.0, 39.0))
          False
          >>> GAxis(315, 45).common_plane(GAxis(135, 45)).almost_parallel(GPlane(225.0, 80.0))
          False
        """

        return self.as_gvect().common_plane(another.as_gvect())

    def normal_vect(self, another):
        """
        Calculates the vector that is normal
        to two GAxis instances.

        :param another: GAxis instance
        :return: a unit Vect instance.

        Example:
          >>> GAxis(90, 0).normal_vect(GAxis(0, 0))
          Vect(0.0000, 0.0000, 1.0000)
        """

        return self.as_versor().vp(another.as_versor())

    def normal_gaxis(self, another):
        """
        Calculate the GAxis instance that is perpendicular to the two provided.
        The two source GAxis must not be subparallel (threshold is MIN_ANGLE_DEGR_DISORIENTATION),
        otherwise a SubparallelLineationException will be raised.
        
        Example:
          >>> GAxis(0, 0).normal_gaxis(GAxis(4, 0))
          Traceback (most recent call last):
          ...
          SubparallelLineationException: Sources must not be sub- or anti-parallel
          >>> GAxis(0, 0).normal_gaxis(GAxis(180, 0))
          Traceback (most recent call last):
          ...
          SubparallelLineationException: Sources must not be sub- or anti-parallel
          >>> GAxis(90, 0).normal_gaxis(GAxis(180, 0))
          GAxis(000.00, +90.00)
          >>> GAxis(90, 45).normal_gaxis(GAxis(180, 0))
          GAxis(270.00, +45.00)
          >>> GAxis(270, 45).normal_gaxis(GAxis(180, 90)).almost_parallel(GAxis(180, 0))
          True
        """

        return self.as_gvect().normal_gvect(another.as_gvect()).as_axis()


class Plane(object):
    """
    Cartesian plane.
    Expressed by equation:
    ax + by + cz + d = 0

    Note: Plane is locational - its position in space is defined.
    This contrast with GPlane, defined just by its attitude, but with undefined position

    """

    def __init__(self, a, b, c, d):

        self._a = float(a)
        self._b = float(b)
        self._c = float(c)
        self._d = float(d)

    @property
    def a(self):
        """
        Return a coefficient of a Plane instance.

        Example:
          >>> Plane(1, 0, 0, 2).a
          1.0
        """

        return self._a

    @property
    def b(self):
        """
        Return b coefficient of a Plane instance.

        Example:
          >>> Plane(1, 4, 0, 2).b
          4.0
        """

        return self._b

    @property
    def c(self):
        """
        Return a coefficient of a Plane instance.

        Example:
          >>> Plane(1, 0, 5.4, 2).c
          5.4
        """

        return self._c

    @property
    def d(self):
        """
        Return a coefficient of a Plane instance.

        Example:
          >>> Plane(1, 0, 0, 2).d
          2.0
        """

        return self._d

    @property
    def v(self):
        """
        Return coefficients of a Plane instance.

        Example:
          >>> Plane(1, 1, 7, -4).v
          (1.0, 1.0, 7.0, -4.0)
        """
        return self.a, self.b, self.c, self.d

    @classmethod
    def from_points(cls, pt1, pt2, pt3):
        """
        Create a Plane from three given Point instances.

        Example:
          >>> Plane.from_points(Point(0, 0, 0), Point(1, 0, 0), Point(0, 1, 0))
          Plane(0.0000, 0.0000, 1.0000, 0.0000)
          >>> Plane.from_points(Point(0, 0, 0), Point(0, 1, 0), Point(0, 0, 1))
          Plane(1.0000, 0.0000, 0.0000, 0.0000)
        """

        matr_a = np.array([[pt1.y, pt1.z, 1],
                           [pt2.y, pt2.z, 1],
                           [pt3.y, pt3.z, 1]])

        matr_b = - np.array([[pt1.x, pt1.z, 1],
                             [pt2.x, pt2.z, 1],
                             [pt3.x, pt3.z, 1]])

        matr_c = np.array([[pt1.x, pt1.y, 1],
                           [pt2.x, pt2.y, 1],
                           [pt3.x, pt3.y, 1]])

        matr_d = - np.array([[pt1.x, pt1.y, pt1.z],
                             [pt2.x, pt2.y, pt2.z],
                             [pt3.x, pt3.y, pt3.z]])

        return cls(np.linalg.det(matr_a),
                   np.linalg.det(matr_b),
                   np.linalg.det(matr_c),
                   np.linalg.det(matr_d))

    def __repr__(self):

        return "Plane({:.4f}, {:.4f}, {:.4f}, {:.4f})".format(*self.v)

    def nversor(self):
        """
        Return the versor normal to the cartesian plane.

        Examples:
          >>> Plane(0, 0, 5, -2).nversor()
          Vect(0.0000, 0.0000, 1.0000)
          >>> Plane(0, 7, 0, 5).nversor()
          Vect(0.0000, 1.0000, 0.0000)
        """

        return Vect(self.a, self.b, self.c).versor()

    def gplane_point(self):
        """
        Converts a cartesian plane into a geological plane
        and a point lying in the plane (non-unique solution).

        Examples:
          >>> gpl, pt = Plane(0, 0, 1, -1).gplane_point()
          >>> gpl
          GPlane(000.00, +00.00)
          >>> pt
          Point(0.0000, 0.0000, 1.0000, nan)
        """

        geol_plane = self.nversor().gvect().normal_gplane()
        point = Point(*point_solution(np.array([[self.a, self.b, self.c]]),
                                      np.array([-self.d])))
        return geol_plane, point

    def inters_versor(self, another):
        """
        Return intersection versor for two intersecting planes.

        >>> a = Plane(1, 0, 0, 0)
        >>> b = Plane(0, 0, 1, 0)
        >>> a.inters_versor(b)
        Vect(0.0000, -1.0000, 0.0000)
        """

        return self.nversor().vp(another.nversor()).versor()

    def inters_point(self, another):
        """
        Return point on intersection line (obviously non-unique solution)
        for two planes.

        >>> a = Plane(1, 0, 0, 0)
        >>> b = Plane(0, 0, 1, 0)
        >>> a.inters_point(b)
        Point(0.0000, 0.0000, 0.0000, nan)
        """

        # find a point lying on the intersection line (this is a non-unique solution)
        a = np.array([[self.a, self.b, self.c], [another.a, another.b, another.c]])
        b = np.array([-self.d, -another.d])
        x, y, z = point_solution(a, b)

        return Point(x, y, z)

    def is_point_in_plane(self, pt):
        """
          Check whether a point lie in a plane.

          >>> pl = Plane(0, 0, 1, 0)
          >>> pt = Point(0, 1, 0)
          >>> pl.is_point_in_plane(pt)
          True
        """

        if abs(self.a * pt.x + self.b * pt.y + self.c * pt.z + self.d) < MIN_SCALAR_VALUE:
            return True
        else:
            return False

    def angle(self, another):
        """
        Calculate angle (in degrees) between two planes.

        Examples:
          >>> Plane(1,0,0,0).angle(Plane(0,1,0,0))
          90.0
          >>> Plane(1,0,0,0).angle(Plane(1,0,1,0))
          45.0
          >>> Plane(1,0,0,0).angle(Plane(1,0,0,0))
          0.0
        """

        angle_degr = self.nversor().angle(another.nversor())
        if abs(angle_degr) < MIN_ANGLE_DEGR_VALUE:
            angle_degr = 0.0
        elif angle_degr > 90.0:
            angle_degr = 180.0 - angle_degr

        return angle_degr

    def almost_parallel(self, another, angle_tolerance=PLANE_ANGLE_THRESHOLD):
        """
        Check that two Plane are sub-parallel

        :param another: a Plane instance
        :param angle_tolerance: the maximum allowed divergence angle (in degrees)
        :return: Boolean

         Examples:
          >>> Plane(1,0,0,0).almost_parallel(Plane(1,0,0,0))
          True
          >>> Plane(1,0,0,0).almost_parallel(Plane(1,0,1,0))
          False
        """

        return self.angle(another) < angle_tolerance

class GPlane(object):
    """
    Geological plane.
    Defined by dip direction and dip angle (both in degrees):
     - dip direction: [0.0, 360.0[ clockwise, from 0 (North);
     - dip angle: [0, 90.0]: downward-pointing.
    """

    def __init__(self, src_azimuth, src_dip_angle, is_rhr_strike=False):
        """
        Geological plane constructor.

        @param  src_azimuth:  Azimuth of the plane (RHR strike or dip direction).
        @type  src_azimuth:  number or string convertible to float.
        @param  src_dip_angle:  Dip angle of the plane (0-90°).
        @type  src_dip_angle:  number or string convertible to float.
           
        @return:  GPlane.

        Example:
          >>> GPlane(0, 90)
          GPlane(000.00, +90.00)
          >>> GPlane(0, 90, is_rhr_strike=True)
          GPlane(090.00, +90.00)
          >>> GPlane(0, 90, True)
          GPlane(090.00, +90.00)
        """

        def rhrstrk2dd(rhr_strk):
            """Converts RHR strike value to dip direction value.

            Example:
                >>> rhrstrk2dd(285.5)
                15.5
            """

            return (rhr_strk + 90.0) % 360.0

        if is_rhr_strike:
            self._dipdir = rhrstrk2dd(src_azimuth)
        else:
            self._dipdir = src_azimuth % 360.0
        self._dipangle = float(src_dip_angle)

    @property
    def dd(self):
        """
        Return the dip direction of the geological plane.

        Example:
          >>> GPlane(34.2, 89.7).dd
          34.2
        """

        return self._dipdir

    @property
    def da(self):
        """
        Return the dip angle of the geological plane.

        Example:
          >>> GPlane(183, 77).da
          77.0

        """

        return self._dipangle

    @property
    def dda(self):
        """
        Return a tuple storing the dip direction and dip angle values of a geological plane.

        Example:
          >>> gp = GPlane(89.4, 17.2)
          >>> gp.dda
          (89.4, 17.2)
        """
        
        return self.dd, self.da

    @property
    def strike_rhr(self):
        """
        Return the strike according to the right-hand-rule.

        Examples:
          >>> GPlane(90, 45).strike_rhr
          0.0
          >>> GPlane(45, 89).strike_rhr
          315.0
          >>> GPlane(275, 38).strike_rhr
          185.0
          >>> GPlane(0, 38).strike_rhr
          270.0
        """

        return (self.dd - 90.0) % 360.0

    @property
    def strike_lhr(self):
        """
        Return the strike according to the left-hand-rule.

        Examples:
          >>> GPlane(90, 45).strike_lhr
          180.0
          >>> GPlane(45, 89).strike_lhr
          135.0
          >>> GPlane(275, 38).strike_lhr
          5.0
          >>> GPlane(0, 38).strike_lhr
          90.0
        """

        return (self.dd + 90.0) % 360.0

    def __repr__(self):

        return "GPlane({:06.2f}, {:+06.2f})".format(*self.dda)

    def normal(self):
        """
        Return the geological vector normal to the geological plane,
        pointing in the same direction as the geological plane.

        Example:
            >>> GPlane(90, 55).normal()
            GVect(090.00, -35.00)
            >>> GPlane(90, 90).normal()
            GVect(090.00, +00.00)
            >>> GPlane(90, 0).normal()
            GVect(090.00, -90.00)
        """
        
        trend = self.dd % 360.0
        plunge = self.da - 90.0

        return GVect(trend, plunge)

    def plane(self, point):
        """
        Given a GPlane instance and a provided Point instance,
        calculate the corresponding Plane instance.

        Example:
          >>> GPlane(0, 0).plane(Point(0, 0, 0))
          Plane(0.0000, 0.0000, 1.0000, -0.0000)
          >>> GPlane(90, 45).plane(Point(0, 0, 0))
          Plane(0.7071, 0.0000, 0.7071, -0.0000)
          >>> GPlane(0, 90).plane(Point(0, 0, 0))
          Plane(0.0000, 1.0000, -0.0000, -0.0000)
        """

        normal_versor = self.normal().versor()
        a, b, c = normal_versor.x, normal_versor.y, normal_versor.z
        d = - (a * point.x + b * point.y + c * point.z)
        return Plane(a, b, c, d)

    def angle(self, another):
        """
        Calculate angle (in degrees) between two geoplanes.
        Range is 0°-90°.

        >>> GPlane(100.0, 50.0).angle(GPlane(100.0, 50.0))
        0.0
        >>> GPlane(300.0, 10.0).angle(GPlane(300.0, 90.0))
        80.0
        >>> GPlane(90.0, 90.0).angle(GPlane(270.0, 90.0))
        0.0
        >>> are_close(GPlane(90.0, 10.0).angle(GPlane(270.0, 10.0)), 20.0)
        True
        >>> are_close(GPlane(90.0, 10.0).angle(GPlane(270.0, 30.0)), 40.0)
        True
        """

        return self.normal().as_axis().angle(another.normal().as_axis())

    def almost_parallel(self, another, angle_tolerance=PLANE_ANGLE_THRESHOLD):
        """
        Check that two GPlanes are sub-parallel

        :param another: a GPlane instance
        :param angle_tolerance: the maximum allowed divergence angle (in degrees)
        :return: Boolean

         Examples:
          >>> GPlane(0, 90).almost_parallel(GPlane(270, 90))
          False
          >>> GPlane(0, 90).almost_parallel(GPlane(180, 90))
          True
          >>> GPlane(0, 90).almost_parallel(GPlane(0, 0))
          False
          >>> GPlane(0, 0).almost_parallel(GPlane(0, 1e-6))
          True
          >>> GPlane(0, 0).almost_parallel(GPlane(0, 1.1))
          False
        """

        return self.angle(another) < angle_tolerance

    def rake_to_gvect(self, rake):
        """
        Calculate GVect given a GPlane instance and a rake value.
        The rake is defined according to the Aki and Richards, 1980 conventions:
        rake = 0° -> left-lateral
        rake = 90° -> reverse
        rake = +/- 180° -> right-lateral
        rake = -90° -> normal

        Examples:
          >>> GPlane(180, 45).rake_to_gvect(0.0)
          GVect(090.00, +00.00)
          >>> GPlane(180, 45).rake_to_gvect(90.0)
          GVect(000.00, -45.00)
          >>> GPlane(180, 45).rake_to_gvect(-90.0)
          GVect(180.00, +45.00)
          >>> GPlane(180, 45).rake_to_gvect(180.0).almost_parallel(GVect(270.00, -00.00))
          True
          >>> GPlane(180, 45).rake_to_gvect(-180.0)
          GVect(270.00, +00.00)
        """

        rk = radians(rake)
        strk = radians(self.strike_rhr)
        dip = radians(self.da)

        x = cos(rk)*sin(strk)-sin(rk)*cos(dip)*cos(strk)
        y = cos(rk)*cos(strk)+sin(rk)*cos(dip)*sin(strk)
        z = sin(rk) * sin(dip)

        return Vect(x, y, z).gvect()


class SubparallelLineationException(Exception):
    """
    Exception for subparallel GAxis/GVect instances.
    """

    pass


if __name__ == "__main__":

    import doctest
    doctest.testmod()
