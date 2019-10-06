# -*- coding: utf-8 -*-


import itertools

from enum import Enum

import numbers

from math import *
import random

from array import array
from typing import List

from ...utils.lists import *

from ...mathematics.statistics import *
from ...mathematics.utils import *
from ..transformations.quaternions import *
from ..vectors import *
from .defaults import *
from .direct_utils import *


class Point(object):
    """
    Cartesian point.
    Dimensions: 4D (space-time)
    """

    def __init__(
        self,
        x: numbers.Real,
        y: numbers.Real,
        z: numbers.Real = 0.0,
        t: numbers.Real = 0.0,
        epsg_cd: numbers.Integral = -1):
        """
        Construct a Point instance.

        :param x: point x coordinate.
        :type x: numbers.Real.
        :param y: point y coordinate.
        :type y: numbers.Real.
        :param z: point z coordinate.
        :type z: numbers.Real.
        :param t: point time coordinate.
        :type t: numbers.Real.
        :param epsg_cd: CRS EPSG code.
        :type epsg_cd: numbers.Integral.
        """

        vals = [x, y, z, t]
        if any(map(lambda val: not isinstance(val, numbers.Real), vals)):
            raise Exception("Input values must be integer or float type")
        if not all(map(math.isfinite, vals)):
            raise Exception("Input values must be finite")

        if not isinstance(epsg_cd, numbers.Integral):
            raise Exception("Input EPSG value must be integer")
        if not math.isfinite(epsg_cd):
            raise Exception("Input EPSG value must be finite")

        self._x = float(x)
        self._y = float(y)
        self._z = float(z)
        self._t = float(t)
        self._crs = Crs(epsg_cd)

    @classmethod
    def fromVect(cls,
        vect: Vect) -> 'Point':
        """

        :param vect:
        :return:
        """

        return cls(
            x=vect.x,
            y=vect.y,
            z=vect.z,
            epsg_cd=vect.epsg()
        )

    @property
    def x(self) -> numbers.Real:
        """
        Return the x coordinate of the current point.

        :return: x coordinate.
        :rtype: numbers.Real

        Examples:
          >>> Point(4, 3, 7, epsg_cd=4326).x
          4.0
          >>> Point(-0.39, 3, 7).x
          -0.39
        """

        return self._x

    @property
    def y(self) -> numbers.Real:
        """
        Return the y coordinate of the current point.

        :return: y coordinate.
        :rtype: numbers.Real

        Examples:
          >>> Point(4, 3, 7, epsg_cd=4326).y
          3.0
          >>> Point(-0.39, 17.42, 7).y
          17.42
        """

        return self._y

    @property
    def z(self) -> numbers.Real:
        """
        Return the z coordinate of the current point.

        :return: z coordinate.
        :rtype: numbers.Real

        Examples:
          >>> Point(4, 3, 7, epsg_cd=4326).z
          7.0
          >>> Point(-0.39, 17.42, 8.9).z
          8.9
        """

        return self._z

    @property
    def t(self) -> numbers.Real:
        """
        Return the time coordinate of the current point.

        :return: time coordinate.
        :rtype: numbers.Real

        Examples:
          >>> Point(4, 3, 7, epsg_cd=4326).t
          0.0
          >>> Point(-0.39, 17.42, 8.9, 4112).t
          4112.0
        """

        return self._t

    @property
    def crs(self) -> Crs:
        """
        Return the current point CRS.

        :return: the CRS code as a Csr instance.
        :rtype: Crs.
        """

        return self._crs

    def epsg(self) -> numbers.Integral:
        """
        Returns the EPSG code of the point.

        :return: the CRS code.
        :rtype: numbers.Integral.
        """

        return self.crs.epsg()

    def __iter__(self):
        """
        Return the elements of a Point.

        :return:

        Examples;
          >>> x, y, z, t, epsg_cd = Point(1,1)
          >>> x == 1
          True
          >>> y == 1
          True

        """

        return (i for i in self.a())

    def __repr__(self) -> str:

        return "Point({:.4f}, {:.4f}, {:.4f}, {:.4f}, {})".format(self.x, self.y, self.z, self.t, self.epsg())

    def __eq__(self,
        another: 'Point'
    ) -> bool:
        """
        Return True if objects are equal.

        :param another: another point.
        :type another: Point.
        :raise: Exception.

        Example:
          >>> Point(1., 1., 1.) == Point(1, 1, 1)
          True
          >>> Point(1., 1., 1., epsg_cd=4326) == Point(1, 1, 1, epsg_cd=4326)
          True
          >>> Point(1., 1., 1., epsg_cd=4326) == Point(1, 1, -1, epsg_cd=4326)
          False
        """

        if not isinstance(another, Point):
            raise Exception("Another instance must be a Point")

        return all([
            self.x == another.x,
            self.y == another.y,
            self.z == another.z,
            self.t == another.t,
            self.crs == another.crs])

    def __ne__(self,
        another: 'Point'
    ) -> bool:
        """
        Return False if objects are equal.

        Example:
          >>> Point(1., 1., 1.) != Point(0., 0., 0.)
          True
          >>> Point(1., 1., 1., epsg_cd=4326) != Point(1, 1, 1)
          True
        """

        return not (self == another)

    def a(self) -> Tuple[numbers.Real, numbers.Real, numbers.Real, numbers.Real, numbers.Integral]:
        """
        Return the individual values of the point.

        :return: double array of x, y, z values

        Examples:
          >>> Point(4, 3, 7, epsg_cd=4326).a()
          (4.0, 3.0, 7.0, 0.0, 4326)
        """

        return self.x, self.y, self.z, self.t, self.crs.epsg()

    def __add__(self, another: 'Point') -> 'Point':
        """
        Sum of two points.

        :param another: the point to add
        :type another: Point
        :return: the sum of the two points
        :rtype: Point
        :raise: Exception

        Example:
          >>> Point(1, 0, 0, epsg_cd=2000) + Point(0, 1, 1, epsg_cd=2000)
          Point(1.0000, 1.0000, 1.0000, 0.0000, 2000)
          >>> Point(1, 1, 1, epsg_cd=2000) + Point(-1, -1, -1, epsg_cd=2000)
          Point(0.0000, 0.0000, 0.0000, 0.0000, 2000)
        """

        check_type(another, "Second point", Point)

        check_crs(self, another)

        x0, y0, z0, t0, epsg_cd = self
        x1, y1, z1, t1, _ = another

        return Point(
            x=x0+x1,
            y=y0+y1,
            z=z0+z1,
            t=t0+t1,
            epsg_cd=epsg_cd
        )

    def __sub__(self, another: 'Point') -> 'Point':
        """Subtract two points.

        :param another: the point to subtract
        :type another: Point
        :return: the difference between the two points
        :rtype: Point
        :raise: Exception

        Example:
          >>> Point(1., 1., 1., epsg_cd=2000) - Point(1., 1., 1., epsg_cd=2000)
          Point(0.0000, 0.0000, 0.0000, 0.0000, 2000)
          >>> Point(1., 1., 3., epsg_cd=2000) - Point(1., 1., 2.2, epsg_cd=2000)
          Point(0.0000, 0.0000, 0.8000, 0.0000, 2000)
        """

        check_type(another, "Second point", Point)

        check_crs(self, another)

        x0, y0, z0, t0, epsg_cd = self
        x1, y1, z1, t1, _ = another

        return Point(
            x=x0 - x1,
            y=y0 - y1,
            z=z0 - z1,
            t=t0 - t1,
            epsg_cd=epsg_cd
        )

    def clone(self) -> 'Point':
        """
        Clone a point.

        :return: a new point.
        :rtype: Point.
        """

        return Point(*self.a())

    def toXYZ(self) -> Tuple[numbers.Real, numbers.Real, numbers.Real]:
        """
        Returns the spatial components as a tuple of three values.

        :return: the spatial components (x, y, z).
        :rtype: a tuple of three floats.

        Examples:
          >>> Point(1, 0, 3).toXYZ()
          (1.0, 0.0, 3.0)
        """

        return self.x, self.y, self.z

    def toXYZT(self) -> Tuple[numbers.Real, numbers.Real, numbers.Real, numbers.Real]:
        """
        Returns the spatial and time components as a tuple of four values.

        :return: the spatial components (x, y, z) and the time component.
        :rtype: a tuple of four floats.

        Examples:
          >>> Point(1, 0, 3).toXYZT()
          (1.0, 0.0, 3.0, 0.0)
        """

        return self.x, self.y, self.z, self.t

    def toArray(self) -> 'np.array':
        """
        Return a Numpy array representing the point values (without the crs code).

        :return: Numpy array

        Examples:
          >>> np.allclose(Point(1, 2, 3).toArray(), np.array([ 1., 2., 3., 0.]))
          True
        """

        return np.asarray(self.toXYZT())

    def pXY(self) -> 'Point':
        """
        Projection on the x-y plane

        :return: projected object instance

        Examples:
          >>> Point(2, 3, 4).pXY()
          Point(2.0000, 3.0000, 0.0000, 0.0000, -1)
        """

        return Point(self.x, self.y, 0.0, self.t, self.epsg())

    def pXZ(self) -> 'Point':
        """
        Projection on the x-z plane

        :return: projected object instance

        Examples:
          >>> Point(2, 3, 4).pXZ()
          Point(2.0000, 0.0000, 4.0000, 0.0000, -1)
        """

        return Point(self.x, 0.0, self.z, self.t, self.epsg())

    def pYZ(self) -> 'Point':
        """
        Projection on the y-z plane

        :return: projected object instance

        Examples:
          >>> Point(2, 3, 4).pYZ()
          Point(0.0000, 3.0000, 4.0000, 0.0000, -1)
        """

        return Point(0.0, self.y, self.z, self.t, self.epsg())

    def deltaX(self,
        another: 'Point'
    ) -> Optional[numbers.Real]:
        """
        Delta between x components of two Point Instances.

        :return: x coordinates difference value.
        :rtype: optional numbers.Real.
        :raise: Exception

        Examples:
          >>> Point(1, 2, 3, epsg_cd=32632).deltaX(Point(4, 7, 1, epsg_cd=32632))
          3.0
          >>> Point(1, 2, 3, epsg_cd=4326).deltaX(Point(4, 7, 1))
          Traceback (most recent call last):
          ....
          Exception: checked Point instance has -1 EPSG code but 4326 expected
        """

        check_crs(self, another)

        return another.x - self.x

    def deltaY(self,
        another: 'Point'
    ) -> Optional[numbers.Real]:
        """
        Delta between y components of two Point Instances.

        :return: y coordinates difference value.
        :rtype: optional numbers.Real.

        Examples:
          >>> Point(1, 2, 3, epsg_cd=32632).deltaY(Point(4, 7, 1, epsg_cd=32632))
          5.0
          >>> Point(1, 2, 3, epsg_cd=4326).deltaY(Point(4, 7, 1))
          Traceback (most recent call last):
          ...
          Exception: checked Point instance has -1 EPSG code but 4326 expected
        """

        check_crs(self, another)

        return another.y - self.y

    def deltaZ(self,
        another: 'Point'
    ) -> Optional[numbers.Real]:
        """
        Delta between z components of two Point Instances.

        :return: z coordinates difference value.
        :rtype: optional numbers.Real.

        Examples:
          >>> Point(1, 2, 3, epsg_cd=32632).deltaZ(Point(4, 7, 1, epsg_cd=32632))
          -2.0
          >>> Point(1, 2, 3, epsg_cd=4326).deltaZ(Point(4, 7, 1))
          Traceback (most recent call last):
          ....
          Exception: checked Point instance has -1 EPSG code but 4326 expected
        """

        check_crs(self, another)

        return another.z - self.z

    def deltaT(self,
        another: 'Point'
    ) -> numbers.Real:
        """
        Delta between t components of two Point Instances.

        :return: difference value
        :rtype: numbers.Real

        Examples:
          >>> Point(1, 2, 3, 17.3).deltaT(Point(4, 7, 1, 42.9))
          25.599999999999998
        """

        return another.t - self.t

    def dist3DWith(self,
        another: 'Point'
    ) -> numbers.Real:
        """
        Calculate Euclidean spatial distance between two points.
        TODO: consider case of polar CRS

        :param another: another Point instance.
        :type another: Point.
        :return: the distance (when the two points have the same CRS).
        :rtype: numbers.Real.
        :raise: Exception.

        Examples:
          >>> Point(1., 1., 1., epsg_cd=32632).dist3DWith(Point(4., 5., 1, epsg_cd=32632))
          5.0
          >>> Point(1, 1, 1, epsg_cd=32632).dist3DWith(Point(4, 5, 1, epsg_cd=32632))
          5.0
          >>> Point(1, 1, 1).dist3DWith(Point(4, 5, 1))
          5.0
        """

        check_type(another, "Point", Point)

        check_crs(self, another)

        return sqrt((self.x - another.x) ** 2 + (self.y - another.y) ** 2 + (self.z - another.z) ** 2)

    def dist2DWith(self,
        another: 'Point'
    ) -> numbers.Real:
        """
        Calculate horizontal (2D) distance between two points.
        TODO: consider case of polar CRS

        :param another: another Point instance.
        :type another: Point.
        :return: the 2D distance (when the two points have the same CRS).
        :rtype: numbers.Real.
        :raise: Exception.

        Examples:
          >>> Point(1., 1., 1., epsg_cd=32632).dist2DWith(Point(4., 5., 7., epsg_cd=32632))
          5.0
          >>> Point(1., 1., 1., epsg_cd=32632).dist2DWith(Point(4., 5., 7.))
          Traceback (most recent call last):
          ...
          Exception: checked Point instance has -1 EPSG code but 32632 expected
        """

        check_type(another, "Second point", Point)

        check_crs(self, another)

        return sqrt((self.x - another.x) ** 2 + (self.y - another.y) ** 2)

    def scale(self,
        scale_factor: numbers.Real
    ) -> 'Point':
        """
        Create a scaled object.
        Note: it does not make sense for polar coordinates.
        TODO: manage polar coordinates cases OR deprecate and remove - after dependency check.

        Example;
          >>> Point(1, 0, 1).scale(2.5)
          Point(2.5000, 0.0000, 2.5000, 0.0000, -1)
          >>> Point(1, 0, 1).scale(2.5)
          Point(2.5000, 0.0000, 2.5000, 0.0000, -1)
        """

        x, y, z = self.x * scale_factor, self.y * scale_factor, self.z * scale_factor
        return Point(x, y, z, self.t, self.epsg())

    def invert(self) -> 'Point':
        """
        Create a new object with inverted direction.
        Note: it depends on scale method, that could be deprecated/removed.

        Examples:
          >>> Point(1, 1, 1).invert()
          Point(-1.0000, -1.0000, -1.0000, 0.0000, -1)
          >>> Point(2, -1, 4).invert()
          Point(-2.0000, 1.0000, -4.0000, 0.0000, -1)
        """

        return self.scale(-1)

    def reflect_vertical(self) -> 'Point':
        """
        Reflect a point along a vertical axis.

        :return: reflected point.
        :rtype: Point

        Examples:
          >>> Point(1,1,1).reflect_vertical()
          Point(-1.0000, -1.0000, 1.0000, 0.0000, -1)
        """

        x, y, z, t, epsg_cd = self

        return Point(
            x=-x,
            y=-y,
            z=z,
            t=t,
            epsg_cd=epsg_cd
        )

    def isCoinc(self,
        another: 'Point',
        tolerance: numbers.Real = MIN_SEPARATION_THRESHOLD
    ) -> bool:
        """
        Check spatial coincidence of two points

        :param another: the point to compare.
        :type another: Point.
        :param tolerance: the maximum allowed distance between the two points.
        :type tolerance: numbers.Real.
        :return: whether the two points are coincident.
        :rtype: bool.
        :raise: Exception.

        Example:
          >>> Point(1., 0., -1.).isCoinc(Point(1., 1.5, -1.))
          False
          >>> Point(1., 0., 0., epsg_cd=32632).isCoinc(Point(1., 0., 0., epsg_cd=32632))
          True
          >>> Point(1.2, 7.4, 1.4, epsg_cd=32632).isCoinc(Point(1.2, 7.4, 1.4))
          Traceback (most recent call last):
          ...
          Exception: checked Point instance has -1 EPSG code but 32632 expected
          >>> Point(1.2, 7.4, 1.4, epsg_cd=4326).isCoinc(Point(1.2, 7.4, 1.4))
          Traceback (most recent call last):
          ...
          Exception: checked Point instance has -1 EPSG code but 4326 expected
        """

        check_type(another, "Second point", Point)

        check_crs(self, another)

        return self.dist3DWith(another) <= tolerance

    def already_present(self,
        pt_list: List['Point'],
        tolerance: numbers.Real = MIN_SEPARATION_THRESHOLD
    ) -> Optional[bool]:
        """
        Determines if a point is already in a given point list, using an optional distance separation,

        :param pt_list: list of points. May be empty.
        :type pt_list: List of Points.
        :param tolerance: optional maximum distance between near-coincident point pair.
        :type tolerance: numbers.Real.
        :return: True if already present, False otherwise.
        :rtype: optional boolean.
        """

        for pt in pt_list:
            if self.isCoinc(pt, tolerance=tolerance):
                return True
        return False

    def shift(self,
        sx: numbers.Real,
        sy: numbers.Real,
        sz: numbers.Real
    ) -> Optional['Point']:
        """
        Create a new object shifted by given amount from the self instance.

        Example:
          >>> Point(1, 1, 1, epsg_cd=32632).shift(0.5, 1., 1.5)
          Point(1.5000, 2.0000, 2.5000, 0.0000, 32632)
          >>> Point(1, 2, -1, epsg_cd=32632).shift(0.5, 1., 1.5)
          Point(1.5000, 3.0000, 0.5000, 0.0000, 32632)
       """

        return Point(self.x + sx, self.y + sy, self.z + sz, self.t, self.epsg())

    def shiftByVect(self,
        v: Vect
    ) -> 'Point':
        """
        Create a new point shifted from the self instance by given vector.

        :param v: the shift vector.
        :type v: Vect.
        :return: the shifted point.
        :rtype: Point.
        :raise: Exception

        Example:
          >>> Point(1, 1, 1, epsg_cd=32632).shiftByVect(Vect(0.5, 1., 1.5, epsg_cd=32632))
          Point(1.5000, 2.0000, 2.5000, 0.0000, 32632)
          >>> Point(1, 2, -1, epsg_cd=32632).shiftByVect(Vect(0.5, 1., 1.5, epsg_cd=32632))
          Point(1.5000, 3.0000, 0.5000, 0.0000, 32632)
       """

        check_crs(self, v)

        x, y, z, t, epsg_cd = self

        sx, sy, sz = v.toXYZ()

        return Point(x + sx, y + sy, z + sz, t, epsg_cd)

    def asVect(self) -> 'Vect':
        """
        Create a vector based on the point coordinates

        Example:
          >>> Point(1, 1, 0).asVect()
          Vect(1.0000, 1.0000, 0.0000, EPSG: -1)
          >>> Point(0.2, 1, 6).asVect()
          Vect(0.2000, 1.0000, 6.0000, EPSG: -1)
        """

        return Vect(self.x, self.y, self.z, self.epsg())

    def rotate(self,
        rotation_axis: 'RotationAxis',
        center_point: 'Point' = None
        ) -> 'Point':
        """
        Rotates a point.
        :param rotation_axis:
        :param center_point:
        :return: the rotated point
        :rtype: Point

        Examples:
          >>> pt = Point(0,0,1,10, 32633)
          >>> rot_axis = RotationAxis(0,0,90)
          >>> center_pt = Point(0,0,0.5, 0, 32633)
          >>> pt.rotate(rotation_axis=rot_axis, center_point=center_pt)
          Point(0.5000, 0.0000, 0.5000, 10.0000, 32633)
          >>> center_pt = Point(0,0,1, 0, 32633)
          >>> pt.rotate(rotation_axis=rot_axis, center_point=center_pt)
          Point(0.0000, 0.0000, 1.0000, 10.0000, 32633)
          >>> center_pt = Point(0,0,2, 0, 32633)
          >>> pt.rotate(rotation_axis=rot_axis, center_point=center_pt)
          Point(-1.0000, 0.0000, 2.0000, 10.0000, 32633)
          >>> rot_axis = RotationAxis(0,0,180)
          >>> pt.rotate(rotation_axis=rot_axis, center_point=center_pt)
          Point(-0.0000, 0.0000, 3.0000, 10.0000, 32633)
          >>> pt.rotate(rotation_axis=rot_axis)
          Point(0.0000, 0.0000, -1.0000, 10.0000, 32633)
          >>> pt = Point(1,1,1,5)
          >>> rot_axis = RotationAxis(0,90,90)
          >>> pt.rotate(rotation_axis=rot_axis)
          Point(1.0000, -1.0000, 1.0000, 5.0000, -1)
          >>> rot_axis = RotationAxis(0,90,180)
          >>> pt.rotate(rotation_axis=rot_axis)
          Point(-1.0000, -1.0000, 1.0000, 5.0000, -1)
          >>> center_pt = Point(1,1,1)
          >>> pt.rotate(rotation_axis=rot_axis, center_point=center_pt)
          Point(1.0000, 1.0000, 1.0000, 5.0000, -1)
          >>> center_pt = Point(2,2,10)
          >>> pt.rotate(rotation_axis=rot_axis, center_point=center_pt)
          Point(3.0000, 3.0000, 1.0000, 5.0000, -1)
          >>> pt = Point(1,1,2,7.5)
          >>> rot_axis = RotationAxis(135,0,180)
          >>> center_pt = Point(0,0,1)
          >>> pt.rotate(rotation_axis=rot_axis, center_point=center_pt)
          Point(-1.0000, -1.0000, 0.0000, 7.5000, -1)
        """

        _, _, _, t, epsg_cd = self

        if not center_point:

            center_point = Point(
                x=0.0,
                y=0.0,
                z=0.0,
                t=0.0,
                epsg_cd=epsg_cd
            )

        check_type(center_point, "Center point", Point)

        check_crs(self, center_point)

        p_diff = self - center_point

        p_vect = p_diff.asVect()

        rot_vect = rotVectByAxis(
            v=p_vect,
            rot_axis=rotation_axis
        )

        x, y, z, epsg_cd = rot_vect

        rot_pt = Point(
            x=x,
            y=y,
            z=z,
            t=t,
            epsg_cd=epsg_cd
        )

        transl_pt = center_point + rot_pt

        return transl_pt

    @classmethod
    def random(cls,
        lower_boundary: float = -MAX_SCALAR_VALUE,
        upper_boundary: float =  MAX_SCALAR_VALUE):
        """
        Creates a random point.

        :return: random point
        :rtype: Point
        """

        vals = [random.uniform(lower_boundary, upper_boundary) for _ in range(3)]
        return cls(*vals)


class CPlane(object):
    """
    Cartesian plane.
    Expressed by equation:
    ax + by + cz + d = 0

    Note: CPlane is locational - its position in space is defined.
    This contrast with Plane, defined just by its attitude, but with undefined position

    """

    def __init__(self, a: numbers.Real, b: numbers.Real, c: numbers.Real, d: numbers.Real, epsg_cd: numbers.Integral = -1):

        if not isinstance(a, numbers.Real):
            raise Exception("Input value a must be float or int but is {}".format(type(a)))
        if not isinstance(b, numbers.Real):
            raise Exception("Input value b must be float or int but is {}".format(type(b)))
        if not isinstance(c, numbers.Real):
            raise Exception("Input value c must be float or int but is {}".format(type(c)))
        if not isinstance(d, numbers.Real):
            raise Exception("Input value d must be float or int but is {}".format(type(d)))
        if not isinstance(epsg_cd, numbers.Integral):
            raise Exception("Input value epsg_cd must be int but is {}".format(type(epsg_cd)))

        norm = sqrt(a*a + b*b + c*c)
        self._a = float(a) / norm
        self._b = float(b) / norm
        self._c = float(c) / norm
        self._d = float(d) / norm
        self._crs = Crs(epsg_cd)

    def a(self) -> numbers.Real:
        """
        Return a coefficient of a CPlane instance.

        Example:
          >>> CPlane(1, 0, 0, 2).a()
          1.0
        """

        return self._a

    def b(self) -> numbers.Real:
        """
        Return b coefficient of a CPlane instance.

        Example:
          >>> CPlane(1, 4, 0, 2).b()
          0.9701425001453319
        """

        return self._b

    def c(self) -> numbers.Real:
        """
        Return a coefficient of a CPlane instance.

        Example:
          >>> CPlane(1, 0, 5.4, 2).c()
          0.9832820049844602
        """

        return self._c

    def d(self) -> numbers.Real:
        """
        Return a coefficient of a CPlane instance.

        Example:
          >>> CPlane(1, 0, 0, 2).d()
          2.0
        """

        return self._d

    @property
    def crs(self) -> Crs:
        """
        Returns the Crs instance.

        :return: EPSG code.
        :rtype: Crs

        Example:
        """

        return self._crs

    def epsg(self) -> numbers.Integral:
        """
        Returns the EPSG code.

        :return: EPSG code.
        :rtype: numbers.Integral

        Example:
        """

        return self.crs.epsg()

    def v(self) -> Tuple[numbers.Real, numbers.Real, numbers.Real, numbers.Real, numbers.Integral]:
        """
        Return coefficients of a CPlane instance.

        Example:
          >>> CPlane(1, 1, 7, -4).v()
          (0.14002800840280097, 0.14002800840280097, 0.9801960588196068, -0.5601120336112039, -1)
        """

        return self.a(), self.b(), self.c(), self.d(), self.epsg()

    @classmethod
    def fromPoints(cls, pt1, pt2, pt3) -> 'CPlane':
        """
        Create a CPlane from three given Point instances.

        Example:
          >>> CPlane.fromPoints(Point(0, 0, 0), Point(1, 0, 0), Point(0, 1, 0))
          CPlane(0.0000, 0.0000, 1.0000, 0.0000, -1)
          >>> CPlane.fromPoints(Point(0, 0, 0, epsg_cd=4326), Point(1, 0, 0, epsg_cd=4326), Point(0, 1, 0, epsg_cd=4326))
          CPlane(0.0000, 0.0000, 1.0000, 0.0000, 4326)
          >>> CPlane.fromPoints(Point(0, 0, 0, epsg_cd=4326), Point(0, 1, 0, epsg_cd=4326), Point(0, 0, 1, epsg_cd=4326))
          CPlane(1.0000, 0.0000, 0.0000, 0.0000, 4326)
          >>> CPlane.fromPoints(Point(1,2,3), Point(2,3,4), Point(-1,7,-2))
          CPlane(-0.7956, 0.2387, 0.5569, -1.3524, -1)
        """

        if not (isinstance(pt1, Point)):
            raise Exception("First input point should be Point but is {}".format(type(pt1)))

        if not (isinstance(pt2, Point)):
            raise Exception("Second input point should be Point but is {}".format(type(pt2)))

        if not (isinstance(pt3, Point)):
            raise Exception("Third input point should be Point but is {}".format(type(pt3)))

        check_crs(pt2, pt1)

        check_crs(pt3, pt1)

        matr_a = np.array(
            [[pt1.y, pt1.z, 1],
             [pt2.y, pt2.z, 1],
             [pt3.y, pt3.z, 1]])

        matr_b = - np.array(
            [[pt1.x, pt1.z, 1],
             [pt2.x, pt2.z, 1],
             [pt3.x, pt3.z, 1]])

        matr_c = np.array(
            [[pt1.x, pt1.y, 1],
             [pt2.x, pt2.y, 1],
             [pt3.x, pt3.y, 1]])

        matr_d = - np.array(
            [[pt1.x, pt1.y, pt1.z],
             [pt2.x, pt2.y, pt2.z],
             [pt3.x, pt3.y, pt3.z]])

        return cls(
            np.linalg.det(matr_a),
            np.linalg.det(matr_b),
            np.linalg.det(matr_c),
            np.linalg.det(matr_d),
            epsg_cd=pt1.epsg())

    def __repr__(self):

        return "CPlane({:.4f}, {:.4f}, {:.4f}, {:.4f}, {:d})".format(*self.v(), self.crs)

    def normVersor(self) -> Vect:
        """
        Return the versor normal to the cartesian plane.

        Examples:
          >>> CPlane(0, 0, 5, -2).normVersor()
          Vect(0.0000, 0.0000, 1.0000, EPSG: -1)
          >>> CPlane(0, 7, 0, 5, epsg_cd=32632).normVersor()
          Vect(0.0000, 1.0000, 0.0000, EPSG: 32632)
        """

        return Vect(self.a(), self.b(), self.c(), epsg_cd=self.epsg()).versor()

    def toPoint(self) -> Point:
        """
        Returns a point lying in the plane (non-unique solution).

        Examples:
          >>> CPlane(0, 0, 1, -1).toPoint()
          Point(0.0000, 0.0000, 1.0000, 0.0000, -1)
        """

        point = Point(
            *pointSolution(
                np.array([[self.a(), self.b(), self.c()]]),
                np.array([-self.d()])),
            epsg_cd=self.epsg())

        return point

    def intersVersor(self, another) -> Optional[Vect]:
        """
        Return intersection versor for two intersecting planes.
        Return None for not intersecting planes.

        :param another: another Cartesian plane.
        :type another: CPlane.
        :return: the intersection line as a vector.
        :rtype: Optional[Vect].
        :raise: Exception.

        Examples:
          >>> a = CPlane(1, 0, 0, 0, epsg_cd=2000)
          >>> b = CPlane(0, 0, 1, 0, epsg_cd=2000)
          >>> a.intersVersor(b)
          Vect(0.0000, -1.0000, 0.0000, EPSG: 2000)
          >>> b = CPlane(-1, 0, 0, 0, epsg_cd=2000)  # parallel plane, no intersection
          >>> a.intersVersor(b) is None
          True
        """

        check_type(another, "Input Cartesian plane", CPlane)

        check_crs(self, another)

        return self.normVersor().vCross(another.normVersor()).versor()

    def intersPoint(self,
            another) -> Optional[Point]:
        """
        Return point on intersection line (non-unique solution)
        for two planes.

        :param another: the second cartesian plane
        :type another: CPlane
        :return: the optional instersection point
        :rtype: Optional[Point]
        :raise: Exception

        Examples:
          >>> a = CPlane(1, 0, 0, 0, epsg_cd=32632)
          >>> b = CPlane(0, 0, 1, 0, epsg_cd=32632)
          >>> a.intersPoint(b)
          Point(0.0000, 0.0000, 0.0000, 0.0000, 32632)
          >>> b = CPlane(-1, 0, 0, 0, epsg_cd=32632)  # parallel plane, no intersection
          >>> a.intersPoint(b) is None
        """

        check_type(another, "Second plane", CPlane)

        check_crs(self, another)

        # find a point lying on the intersection line (this is a non-unique solution)

        a = np.array([[self.a(), self.b(), self.c()], [another.a(), another.b(), another.c()]])
        b = np.array([-self.d(), -another.d()])
        x, y, z = pointSolution(a, b)

        if x is not None and y is not None and z is not None:
            return Point(x, y, z, epsg_cd=self.epsg())
        else:
            return None

    def pointDistance(self,
        pt: Point
    ) -> numbers.Real:
        """
        Calculate the distance between a point and the cartesian plane.
        Distance expression:
        distance = a * x1 + b * y1 + c * z1 + d
        where a, b, c and d are plane parameters of the plane equation:
         a * x + b * y + c * z + d = 0
        and x1, y1, and z1 are the point coordinates.

        :param pt: the point to calculate distance with.
        :type pt: Point.
        :return: the distance value.
        :rtype: numbers.Real.
        :raise: Exception.

        Examples:
          >>> cpl = CPlane(0, 0, 1, 0, epsg_cd=32632)
          >>> pt = Point(0, 0, 1, epsg_cd=32632)
          >>> cpl.pointDistance(pt)
          1.0
          >>> pt = Point(0, 0, 0.5, epsg_cd=32632)
          >>> cpl.pointDistance(pt)
          0.5
          >>> pt = Point(0, 0, -0.5, epsg_cd=32632)
          >>> cpl.pointDistance(pt)
          -0.5
          >>> pt = Point(10, 20, 0.0, epsg_cd=32632)
          >>> cpl.pointDistance(pt)
          0.0
          >>> cpl = CPlane(0, 0, 1, 0, epsg_cd=32632)
          >>> pt = Point(10, 20, 0.0)
          >>> cpl.pointDistance(pt)
          Traceback (most recent call last):
          ...
          Exception: checked Point instance has -1 EPSG code but 32632 expected
        """

        check_type(pt, "Input point", Point)

        check_crs(self, pt)

        return self.a() * pt.x + self.b() * pt.y + self.c() * pt.z + self.d()

    def isPointInPlane(self,
        pt: Point
    ) -> bool:
        """
        Check whether a point lies in the current plane.

        :param pt: the point to check.
        :type pt: Point.
        :return: whether the point lies in the current plane.
        :rtype: bool.
        :raise: Exception.

        Examples:
          >>> pl = CPlane(0, 0, 1, 0)
          >>> pt = Point(0, 1, 0)
          >>> pl.isPointInPlane(pt)
          True
          >>> pl = CPlane(0, 0, 1, 0, epsg_cd=32632)
          >>> pt = Point(0, 1, 0, epsg_cd=32632)
          >>> pl.isPointInPlane(pt)
          True
        """

        check_type(pt, "Input point", Point)

        check_crs(self, pt)

        if abs(self.pointDistance(pt)) < MIN_SEPARATION_THRESHOLD:
            return True
        else:
            return False

    def angle(self,
        another: 'CPlane'
    ) -> numbers.Real:
        """
        Calculate angle (in degrees) between two planes.

        :param another: the CPlane instance to calculate angle with.
        :type another: CPlane.
        :return: the angle (in degrees) between the two planes.
        :rtype: numbers.Real.
        :raise: Exception.

        Examples:
          >>> CPlane(1,0,0,0).angle(CPlane(0,1,0,0))
          90.0
          >>> CPlane(1,0,0,0, epsg_cd=32632).angle(CPlane(0,1,0,0, epsg_cd=32632))
          90.0
          >>> CPlane(1,0,0,0, epsg_cd=32632).angle(CPlane(1,0,1,0, epsg_cd=32632))
          45.0
          >>> CPlane(1,0,0,0, epsg_cd=32632).angle(CPlane(1,0,0,0, epsg_cd=32632))
          0.0
          >>> CPlane(1,0,0,0, epsg_cd=32632).angle(CPlane(1,0,0,0))
          Traceback (most recent call last):
          ...
          Exception: checked CPlane instance has -1 EPSG code but 32632 expected
        """

        check_type(another, "Second Cartesian plane", CPlane)

        check_crs(self, another)

        angle_degr = self.normVersor().angle(another.normVersor())

        if angle_degr > 90.0:
            angle_degr = 180.0 - angle_degr

        return angle_degr


class Segment:
    """
    Segment is a geometric object defined by the straight line between
    two vertices.
    """

    def __init__(self, start_pt: Point, end_pt: Point):
        """
        Creates a segment instance provided the two points have the same CRS code.

        :param start_pt: the start point.
        :type: Point.
        :param end_pt: the end point.
        :type end_pt: Point.
        :return: the new segment instance if both points have the same crs.
        :raises: CRSCodeException.
        """

        check_type(start_pt, "Start point", Point)

        check_type(end_pt, "End point", Point)

        check_crs(start_pt, end_pt)

        if start_pt.dist3DWith(end_pt) == 0.0:
            raise Exception("Segment point distance must be greater than zero")

        self._start_pt = start_pt.clone()
        self._end_pt = end_pt.clone()
        # self._crs = Crs(start_pt.epsg())

    def __repr__(self) -> str:
        """
        Represents a Segment instance.

        :return: the Segment representation.
        :rtype: str.
        """

        return "Segment(start_pt={}, end_pt={})".format(
            self.start_pt,
            self.end_pt
        )

    @property
    def start_pt(self) -> Point:

        return self._start_pt

    @property
    def end_pt(self) -> Point:

        return self._end_pt

    @property
    def crs(self) -> Crs:

        return self.start_pt.crs

    def epsg(self) -> numbers.Integral:

        return self.crs.epsg()

    def __iter__(self):
        """
        Return the elements of a Segment, i.e., start and end point.
        """

        return (i for i in [self.start_pt, self.end_pt])

    def clone(self) -> 'Segment':

        return Segment(self._start_pt, self._end_pt)

    def increasing_x(self) -> 'Segment':

        if self.end_pt.x < self.start_pt.x:
            return Segment(self.end_pt, self.start_pt)
        else:
            return self.clone()

    def x_range(self) -> Tuple[numbers.Real, numbers.Real]:

        if self.start_pt.x < self.end_pt.x:
            return self.start_pt.x, self.end_pt.x
        else:
            return self.end_pt.x, self.start_pt.x

    def y_range(self) -> Tuple[numbers.Real, numbers.Real]:

        if self.start_pt.y < self.end_pt.y:
            return self.start_pt.y, self.end_pt.y
        else:
            return self.end_pt.y, self.start_pt.y

    def z_range(self) -> Tuple[numbers.Real, numbers.Real]:

        if self.start_pt.z < self.end_pt.z:
            return self.start_pt.z, self.end_pt.z
        else:
            return self.end_pt.z, self.start_pt.z

    def delta_x(self) -> numbers.Real:

        return self.end_pt.x - self.start_pt.x

    def delta_y(self) -> numbers.Real:

        return self.end_pt.y - self.start_pt.y

    def delta_z(self) -> numbers.Real:
        """
        Z delta between segment end point and start point.

        :return: numbers.Real.
        """

        return self.end_pt.z - self.start_pt.z

    def delta_t(self) -> numbers.Real:
        """
        T delta between segment end point and start point.

        :return: numbers.Real.
        """

        return self.end_pt.t - self.start_pt.t

    def length2D(self) -> numbers.Real:
        """
        Returns the horizontal length of the segment.

        :return: the horizontal length of the segment.
        :rtype: numbers.Real.
        """

        return self.start_pt.dist2DWith(self.end_pt)

    def length3D(self) -> numbers.Real:

        return self.start_pt.dist3DWith(self.end_pt)

    def deltaZS(self) -> Optional[numbers.Real]:
        """
        Calculates the delta z - delta s ratio of a segment.

        :return: optional numbers.Real.
        """

        len2d = self.length2D()

        if len2d == 0.0:
            return None

        return self.delta_z() / len2d

    def slope_rad(self) -> Optional[numbers.Real]:
        """
        Calculates the slope in radians of the segment.
        Positive is downward point, negative upward pointing.

        :return: optional numbers.Real.
        """

        delta_zs = self.deltaZS()

        if delta_zs is None:
            return None
        else:
            return - math.atan(delta_zs)

    def vector(self) -> Vect:

        return Vect(self.delta_x(),
                    self.delta_y(),
                    self.delta_z(),
                    epsg_cd=self.epsg())

    def antivector(self) -> Vect:
        """
        Returns the vector pointing from the segment end to the segment start.

        :return: the vector pointing from the segment end to the segment start.
        :rtype: Vect.
        """

        return self.vector().invert()

    def segment_2d_m(self) -> Optional[numbers.Real]:

        denom = self.end_pt.x - self.start_pt.x

        if denom == 0.0:
            return None

        return (self.end_pt.y - self.start_pt.y) / denom

    def segment_2d_p(self) -> Optional[numbers.Real]:

        s2d_m = self.segment_2d_m()

        if s2d_m is None:
            return None

        return self.start_pt.y - s2d_m * self.start_pt.x

    def intersection_2d_pt(self,
        another: 'Segment'
    ) -> Optional[Point]:
        """

        :param another:
        :return:
        """

        check_type(another, "Second segment", Segment)

        check_crs(self, another)

        s_len2d = self.length2D()
        a_len2d = another.length2D()

        if s_len2d == 0.0 or a_len2d == 0.0:
            return None

        if self.start_pt.x == self.end_pt.x:  # self segment parallel to y axis
            x0 = self.start_pt.x
            m1, p1 = another.segment_2d_m(), another.segment_2d_p()
            if m1 is None:
                return None
            y0 = m1 * x0 + p1
        elif another.start_pt.x == another.end_pt.x:  # another segment parallel to y axis
            x0 = another.start_pt.x
            m1, p1 = self.segment_2d_m(), self.segment_2d_p()
            if m1 is None:
                return None
            y0 = m1 * x0 + p1
        else:  # no segment parallel to y axis
            m0, p0 = self.segment_2d_m(), self.segment_2d_p()
            m1, p1 = another.segment_2d_m(), another.segment_2d_p()
            if m0 is None or m1 is None:
                return None
            x0 = (p1 - p0) / (m0 - m1)
            y0 = m0 * x0 + p0

        return Point(x0, y0, epsg_cd=self.epsg())

    def contains_pt(self,
        pt: Point
    ) -> bool:
        """
        Checks whether a point is contained in a segment.

        :param pt: the point for which to check containement.
        :return: bool.
        :raise: Exception.

        Examples:
          >>> segment = Segment(Point(0, 0, 0), Point(1, 0, 0))
          >>> segment.contains_pt(Point(0, 0, 0))
          True
          >>> segment.contains_pt(Point(1, 0, 0))
          True
          >>> segment.contains_pt(Point(0.5, 0, 0))
          True
          >>> segment.contains_pt(Point(0.5, 0.00001, 0))
          False
          >>> segment.contains_pt(Point(0.5, 0, 0.00001))
          False
          >>> segment.contains_pt(Point(1.00001, 0, 0))
          False
          >>> segment.contains_pt(Point(0.000001, 0, 0))
          True
          >>> segment.contains_pt(Point(-0.000001, 0, 0))
          False
          >>> segment.contains_pt(Point(0.5, 1000, 1000))
          False
          >>> segment = Segment(Point(0, 0, 0), Point(0, 1, 0))
          >>> segment.contains_pt(Point(0, 0, 0))
          True
          >>> segment.contains_pt(Point(0, 0.5, 0))
          True
          >>> segment.contains_pt(Point(0, 1, 0))
          True
          >>> segment.contains_pt(Point(0, 1.5, 0))
          False
          >>> segment = Segment(Point(0, 0, 0), Point(1, 1, 1))
          >>> segment.contains_pt(Point(0.5, 0.5, 0.5))
          True
          >>> segment.contains_pt(Point(1, 1, 1))
          True
          >>> segment = Segment(Point(1,2,3), Point(9,8,2))
          >>> segment.contains_pt(segment.pointAt(0.745))
          True
          >>> segment.contains_pt(segment.pointAt(1.745))
          False
          >>> segment.contains_pt(segment.pointAt(-0.745))
          False
          >>> segment.contains_pt(segment.pointAt(0))
          True
        """

        check_type(pt, "Point", Point)

        segment_length = self.length3D()
        length_startpt_pt = self.start_pt.dist3DWith(pt)
        length_endpt_pt = self.end_pt.dist3DWith(pt)

        return areClose(
            a=segment_length,
            b=length_startpt_pt + length_endpt_pt
        )

    def fast_2d_contains_pt(self,
        pt2d
    ) -> bool:
        """
        Deprecated. Use 'contains_pt'.

        to work properly, this function requires that the pt lies on the line defined by the segment
        """

        range_x = self.x_range
        range_y = self.y_range

        if range_x()[0] <= pt2d.x <= range_x()[1] or \
                range_y()[0] <= pt2d.y <= range_y()[1]:
            return True
        else:
            return False

    def pointAt(self,
        scale_factor: numbers.Real
    ) -> Point:
        """
        Returns a point aligned with the segment
        and lying at given scale factor, where 1 is segment length
        ans 0 is segment start.

        :param scale_factor: the scale factor, where 1 is the segment length.
        :type scale_factor: numbers.Real
        :return: Point at scale factor
        :rtype: Point

        Examples:
          >>> s = Segment(Point(0,0,0), Point(1,0,0))
          >>> s.pointAt(0)
          Point(0.0000, 0.0000, 0.0000, 0.0000, -1)
          >>> s.pointAt(0.5)
          Point(0.5000, 0.0000, 0.0000, 0.0000, -1)
          >>> s.pointAt(1)
          Point(1.0000, 0.0000, 0.0000, 0.0000, -1)
          >>> s.pointAt(-1)
          Point(-1.0000, 0.0000, 0.0000, 0.0000, -1)
          >>> s.pointAt(-2)
          Point(-2.0000, 0.0000, 0.0000, 0.0000, -1)
          >>> s.pointAt(2)
          Point(2.0000, 0.0000, 0.0000, 0.0000, -1)
          >>> s = Segment(Point(0,0,0), Point(0,0,1))
          >>> s.pointAt(0)
          Point(0.0000, 0.0000, 0.0000, 0.0000, -1)
          >>> s.pointAt(0.5)
          Point(0.0000, 0.0000, 0.5000, 0.0000, -1)
          >>> s.pointAt(1)
          Point(0.0000, 0.0000, 1.0000, 0.0000, -1)
          >>> s.pointAt(-1)
          Point(0.0000, 0.0000, -1.0000, 0.0000, -1)
          >>> s.pointAt(-2)
          Point(0.0000, 0.0000, -2.0000, 0.0000, -1)
          >>> s.pointAt(2)
          Point(0.0000, 0.0000, 2.0000, 0.0000, -1)
          >>> s = Segment(Point(0,0,0), Point(1,1,1))
          >>> s.pointAt(0.5)
          Point(0.5000, 0.5000, 0.5000, 0.0000, -1)
          >>> s = Segment(Point(0,0,0), Point(4,0,0))
          >>> s.pointAt(7.5)
          Point(30.0000, 0.0000, 0.0000, 0.0000, -1)
        """

        dx = self.delta_x() * scale_factor
        dy = self.delta_y() * scale_factor
        dz = self.delta_z() * scale_factor
        dt = self.delta_t() * scale_factor

        return Point(
            x=self.start_pt.x + dx,
            y=self.start_pt.y + dy,
            z=self.start_pt.z + dz,
            t=self.start_pt.t + dt,
            epsg_cd=self.epsg())

    def pointProjection(self,
        point: Point
    ) -> Point:
        """
        Return the point projection on the segment.

        Examples:
          >>> s = Segment(start_pt=Point(0,0,0), end_pt=Point(1,0,0))
          >>> p = Point(0.5, 1, 4)
          >>> s.pointProjection(p)
          Point(0.5000, 0.0000, 0.0000, 0.0000, -1)
          >>> s = Segment(start_pt=Point(0,0,0), end_pt=Point(4,0,0))
          >>> p = Point(7.5, 19.2, -14.72)
          >>> s.pointProjection(p)
          Point(7.5000, 0.0000, 0.0000, 0.0000, -1)
        """

        check_type(point, "Input point", Point)

        check_crs(self, point)

        other_segment = Segment(
            self.start_pt,
            point
        )
        
        scale_factor = self.vector().scalarProjection(other_segment.vector()) / self.length3D()
        return self.pointAt(scale_factor)

    def pointDistance(self,
        point: Point
    ) -> numbers.Real:
        """
        Returns the point distance to the segment.

        :param point: the point to calculate the distance with
        :type point: Point
        :return: the distance of the point to the segment
        :rtype: numbers.Real

        Examples:
          >>> s = Segment(Point(0,0,0), Point(0,0,4))
          >>> s.pointDistance(Point(-17.2, 0.0, -49,3))
          17.2
          >>> s.pointDistance(Point(-17.2, 1.22, -49,3))
          17.24321315764553
        """

        check_type(point, "Input point", Point)

        check_crs(self, point)

        point_projection = self.pointProjection(point)

        return point.dist3DWith(point_projection)

    def pointS(self,
        point: Point
    ) -> Optional[numbers.Real]:
        """
        Calculates the optional distance of the point along the segment.
        A zero value is for a point coinciding with the start point.
        Returns None if the point is not contained in the segment.

        :param point: the point to calculate the optional distance in the segment.
        :type point: Point
        :return: the the optional distance of the point along the segment.
        """

        check_type(point, "Input point", Point)

        check_crs(self, point)

        if not self.contains_pt(point):
            return None

        return self.start_pt.dist3DWith(point)

    def scale(self,
        scale_factor
    ) -> 'Segment':
        """
        Scale a segment by the given scale_factor.
        Start point does not change.

        :param scale_factor: the scale factor, where 1 is the segment length.
        :type scale_factor: numbers.Real
        :return: Point at scale factor
        :rtype: Point
        """

        end_pt = self.pointAt(scale_factor)

        return Segment(
            self.start_pt,
            end_pt)

    def densify2d_asSteps(self,
        densify_distance: numbers.Real
    ) -> array:
        """
        Defines the array storing the incremental lengths according to the provided densify distance.

        :param densify_distance: the step distance.
        :type densify_distance: numbers.Real.
        :return: array storing incremental steps, with the last step being equal to the segment length.
        :rtype: array.
        """

        if not isinstance(densify_distance, numbers.Real):
            raise Exception("Densify distance must be float or int")

        if not math.isfinite(densify_distance):
            raise Exception("Densify distance must be finite")

        if not densify_distance > 0.0:
            raise Exception("Densify distance must be positive")

        segment_length = self.length2D()

        s_list = []
        n = 0
        length = n * densify_distance

        while length < segment_length:
            s_list.append(length)
            n += 1
            length = n * densify_distance

        s_list.append(segment_length)

        return array('d', s_list)

    def densify2d_asPts(self,
        densify_distance
    ) -> List[Point]:
        """
        Densify a segment by adding additional points
        separated a distance equal to densify_distance.
        The result is no longer a Segment instance, instead it is a Line instance.

        :param densify_distance: the distance with which to densify the segment.
        :type densify_distance: numbers.Real.
        :return: the set of densified points.
        :rtype: List[Point].
        """

        if not isinstance(densify_distance, numbers.Real):
            raise Exception("Input densify distance must be float or integer")

        if not math.isfinite(densify_distance):
            raise Exception("Input densify distance must be finite")

        if densify_distance <= 0.0:
            raise Exception("Input densify distance must be positive")

        length2d = self.length2D()

        vect = self.vector()
        vers_2d = vect.versor2D()
        generator_vector = vers_2d.scale(densify_distance)

        pts = [self.start_pt]

        n = 0
        while True:
            n += 1
            new_pt = self.start_pt.shiftByVect(generator_vector.scale(n))
            distance = self.start_pt.dist2DWith(new_pt)
            if distance >= length2d:
                break
            pts.append(new_pt)

        pts.append(self.end_pt)

        return pts

    def densify2d_asLine(self,
        densify_distance
    ) -> 'Line':
        """
        Densify a segment by adding additional points
        separated a distance equal to densify_distance.
        The result is no longer a Segment instance, instead it is a Line instance.

        :param densify_distance: numbers.Real
        :return: Line
        """

        pts = self.densify2d_asPts(densify_distance=densify_distance)

        return Line(
            pts=pts)

    def vertical_plane(self) -> Optional[CPlane]:
        """
        Returns the vertical Cartesian plane containing the segment.

        :return: the vertical Cartesian plane containing the segment.
        :rtype: Optional[CPlane].
        """

        if self.length2D() == 0.0:
            return None

        # arbitrary point on the same vertical as end point

        section_final_pt_up = self.end_pt.shift(
            sx=0.0,
            sy=0.0,
            sz=1000.0)

        return CPlane.fromPoints(
            pt1=self.start_pt,
            pt2=self.end_pt,
            pt3=section_final_pt_up)

    def same_start(self,
        another: 'Segment',
        tol: numbers.Real = 1e-12
    ) -> bool:
        """
        Check whether the two segments have the same start point.

        :param another: a segment to check for.
        :type another: Segment.
        :param tol: tolerance for distance between points.
        :type tol: numbers.Real.
        :return: whether the two segments have the same start point.
        :rtype: bool.

        Examples:
          >>> s1 = Segment(Point(0,0,0), Point(1,0,0))
          >>> s2 = Segment(Point(0,0,0), Point(0,1,0))
          >>> s1.same_start(s2)
          True
        """

        return self.start_pt.isCoinc(
            another=another.start_pt,
            tolerance=tol
        )

    def same_end(self,
        another: 'Segment',
        tol: numbers.Real = 1e-12
    ) -> bool:
        """
        Check whether the two segments have the same end point.

        :param another: a segment to check for.
        :type another: Segment.
        :param tol: tolerance for distance between points.
        :type tol: numbers.Real.
        :return: whether the two segments have the same end point.
        :rtype: bool.

        Examples:
          >>> s1 = Segment(Point(0,0,0), Point(1,0,0))
          >>> s2 = Segment(Point(2,0,0), Point(1,0,0))
          >>> s1.same_end(s2)
          True
        """

        return self.end_pt.isCoinc(
            another=another.end_pt,
            tolerance=tol)

    def conn_to_other(self,
        another: 'Segment',
        tol: numbers.Real = 1e-12
    ) -> bool:
        """
        Check whether the first segment is sequentially connected to the second one.

        :param another: a segment to check for.
        :type another: Segment.
        :param tol: tolerance for distance between points.
        :type tol: numbers.Real.
        :return: whether the first segment is sequentially connected to the second one.
        :rtype: bool.

        Examples:
          >>> s1 = Segment(Point(0,0,0), Point(1,0,0))
          >>> s2 = Segment(Point(1,0,0), Point(2,0,0))
          >>> s1.conn_to_other(s2)
          True
        """

        return self.end_pt.isCoinc(
            another=another.start_pt,
            tolerance=tol)

    def other_connected(self,
        another: 'Segment',
        tol: numbers.Real = 1e-12
    ) -> bool:
        """
        Check whether the second segment is sequentially connected to the first one.

        :param another: a segment to check for.
        :type another: Segment.
        :param tol: tolerance for distance between points.
        :type tol: numbers.Real.
        :return: whether the second segment is sequentially connected to the first one.
        :rtype: bool.

        Examples:
          >>> s1 = Segment(Point(0,0,0), Point(1,0,0))
          >>> s2 = Segment(Point(-1,0,0), Point(0,0,0))
          >>> s1.other_connected(s2)
          True
        """

        return another.end_pt.isCoinc(
            another=self.start_pt,
            tolerance=tol)

    def segment_start_in(self,
        another: 'Segment'
    ) -> bool:
        """
        Check whether the second segment contains the first segment start point.

        :param another: a segment to check for.
        :type another: Segment.
        :return: whether the second segment contains the first segment start point.
        :rtype: bool.

        Examples:
          >>> s1 = Segment(Point(0,0,0), Point(1,0,0))
          >>> s2 = Segment(Point(-0.5,0,0), Point(0.5,0,0))
          >>> s1.segment_start_in(s2)
          True
          >>> s1 = Segment(Point(0,0,0), Point(1,1,1))
          >>> s1.segment_start_in(s2)
          True
          >>> s1 = Segment(Point(0,1,0), Point(1,1,1))
          >>> s1.segment_start_in(s2)
          False
          >>> s1 = Segment(Point(-1,-1,-1), Point(1,1,1))
          >>> s1.segment_start_in(s2)
          False
        """

        return another.contains_pt(self.start_pt)

    def segment_end_in(self,
        another: 'Segment'
    ) -> bool:
        """
        Check whether the second segment contains the first segment end point.

        :param another: a segment to check for.
        :type another: Segment.
        :return: whether the second segment contains the first segment end point.
        :rtype: bool.

        Examples:
          >>> s1 = Segment(Point(0,0,0), Point(1,0,0))
          >>> s2 = Segment(Point(-0.5,0,0), Point(0.5,0,0))
          >>> s1.segment_end_in(s2)
          False
          >>> s1 = Segment(Point(0,0,0), Point(1,1,1))
          >>> s1.segment_end_in(s2)
          False
          >>> s1 = Segment(Point(0,1,0), Point(1,1,1))
          >>> s2 = Segment(Point(1,1,1), Point(0.5,0,0))
          >>> s1.segment_end_in(s2)
          True
          >>> s1 = Segment(Point(-1,-1,3), Point(1,1,3))
          >>> s2 = Segment(Point(0,2,3), Point(2,0,3))
          >>> s1.segment_end_in(s2)
          True
        """

        return another.contains_pt(self.end_pt)

    def rotate(self,
        rotation_axis: 'RotationAxis',
        center_point: 'Point' = None
        ) -> 'Segment':
        """
        Rotates a segment.
        :param rotation_axis:
        :param center_point:
        :return: the rotated segment
        :rtype: Segment

        Examples:
        >>> seg = Segment(Point(0,0,0), Point(0,0,1))
        >>> rot_ax = RotationAxis(0, 0, 90)
        >>> seg.rotate(rot_ax)
        Segment(start_pt=Point(0.0000, 0.0000, 0.0000, 0.0000, -1), end_pt=Point(1.0000, 0.0000, 0.0000, 0.0000, -1))
        >>> rot_ax = RotationAxis(0, 0, 180)
        >>> seg.rotate(rot_ax)
        Segment(start_pt=Point(0.0000, 0.0000, 0.0000, 0.0000, -1), end_pt=Point(0.0000, 0.0000, -1.0000, 0.0000, -1))
        >>> centr_pt = Point(0,0,0.5)
        >>> seg.rotate(rotation_axis=rot_ax, center_point=centr_pt)
        Segment(start_pt=Point(-0.0000, 0.0000, 1.0000, 0.0000, -1), end_pt=Point(0.0000, 0.0000, 0.0000, 0.0000, -1))
        >>> seg = Segment(Point(0,0,0), Point(1,1,0))
        >>> centr_pt = Point(1,0,0)
        >>> rot_ax = RotationAxis(0, 90, 90)
        >>> seg.rotate(rotation_axis=rot_ax, center_point=centr_pt)
        Segment(start_pt=Point(1.0000, 1.0000, 0.0000, 0.0000, -1), end_pt=Point(2.0000, 0.0000, -0.0000, 0.0000, -1))
        >>> seg = Segment(Point(1,1,1), Point(0,0,0))
        >>> rot_ax = RotationAxis(135, 0, 180)
        >>> centr_pt = Point(0.5,0.5,0.5)
        >>> seg.rotate(rotation_axis=rot_ax, center_point=centr_pt)
        Segment(start_pt=Point(0.0000, 0.0000, 0.0000, 0.0000, -1), end_pt=Point(1.0000, 1.0000, 1.0000, 0.0000, -1))
        """

        start_pt, end_pt = self

        rotated_start_pt = start_pt.rotate(
            rotation_axis=rotation_axis,
            center_point=center_point
        )

        rotated_end_pt = end_pt.rotate(
            rotation_axis=rotation_axis,
            center_point=center_point
        )

        return Segment(
            start_pt=rotated_start_pt,
            end_pt=rotated_end_pt
        )

    @classmethod
    def random(cls,
        lower_boundary: float = -MAX_SCALAR_VALUE,
        upper_boundary: float = MAX_SCALAR_VALUE):
        """
        Creates a random segment.

        :return: random segment
        :rtype: Segment
        """

        return cls(
            start_pt=Point.random(lower_boundary, upper_boundary),
            end_pt=Point.random(lower_boundary, upper_boundary)
        )


class CLine:
    """
    Cartesian line.

    Defined by a point and a unit vector.
    """

    def __init__(self,
        point: Point,
        dir_vector: Vect):

        check_type(point, "Input point", Point)
        check_type(dir_vector, "Directional vector", Vect)

        check_crs(point, dir_vector)

        self._start_pt = point

        self._dir_vect = dir_vector.versor().upward()

    @property
    def start_pt(self) -> Point:
        """
        Returns the Cartesian line point.

        :return: the Cartesian line point
        :rtype: Point
        """

        return self._start_pt

    @property
    def end_pt(self) -> Point:
        """
        Returns the Cartesian line point.

        :return: the Cartesian line point
        :rtype: Point
        """

        return self.start_pt.shiftByVect(self.versor)

    @property
    def versor(self) -> Vect:
        """
        Returns .

        :return: the unit vector
        :rtype: Vect
        """

        return self._dir_vect

    def points(self)-> Tuple[Point, Point]:
        """
        Returns the CLine as a tuple of two points.

        :return: the CLine as a tuple of two points
        :rtype: Tuple[Point, Point]
        """

        return self.start_pt, self.end_pt

    def segment(self) -> Segment:
        """
        Returns the CLine as a segment.

        :return: the CLine as a segment
        :rtype: Segment
        """

        return Segment(
            start_pt=self.start_pt,
            end_pt=self.end_pt
        )

    @property
    def crs(self) -> Crs:
        """
        Returns the CRS.

        :return: the CRS
        :rtype: Crs
        """

        return self.start_pt.crs

    def epsg(self) -> numbers.Integral:
        """
        Returns the EPSG code.

        :return: the EPSG code
        :rtype: numbers.Integral
        """

        return self.crs.epsg()

    @classmethod
    def fromPoints(cls,
        first_pt: Point,
        second_pt: Point,
        tolerance=PRACTICAL_MIN_DIST
    ) -> 'CLine':
        """
        Creates a CLine instance from two distinct points.

        :param tolerance:
        :param first_pt: the first input point
        :type first_pt: Point
        :param second_pt: the second input point
        :type second_pt: Point
        :return: a new CLine instance
        :rtype: CLine
        """

        check_type(first_pt, "First point", Point)
        check_type(second_pt, "Second point", Point)

        check_crs(first_pt, second_pt)

        if first_pt.isCoinc(second_pt, tolerance=tolerance):
            raise Exception("The two input points are practically coincident")

        segment = Segment(
            start_pt=first_pt,
            end_pt=second_pt
        )

        return cls(
            point=first_pt,
            dir_vector=segment.vector()
        )

    @classmethod
    def fromSegment(cls,
        segment: Segment):
        """
        Creates a Cartesian line from a segment instance.

        :param segment: the segment to convert to Cartesian line
        :type segment: Segment
        :return: a new CLine
        :rtype: CLine
        """

        return cls.fromPoints(
            first_pt=segment.start_pt,
            second_pt=segment.end_pt
        )

    def shortest_segment_or_point(self,
        another: 'CLine',
        tol: numbers.Real = PRACTICAL_MIN_DIST
    ) -> Optional[Union[Segment, Point]]:

        """
        Calculates the optional shortest segment - or the intersection point - between two lines represented by two segments.

        Adapted from:
            http://paulbourke.net/geometry/pointlineplane/

        C code from:
            http://paulbourke.net/geometry/pointlineplane/lineline.c
    [
        typedef struct {
        double x,y,z;
        } XYZ;

        /*
        Calculate the line segment PaPb that is the shortest route between
        two lines P1P2 and P3P4. Calculate also the values of mua and mub where
          Pa = P1 + mua (P2 - P1)
          Pb = P3 + mub (P4 - P3)
        Return FALSE if no solution exists.
        */
        int LineLineIntersect(
        XYZ p1,XYZ p2,XYZ p3,XYZ p4,XYZ *pa,XYZ *pb,
        double *mua, double *mub)
        {
        XYZ p13,p43,p21;
        double d1343,d4321,d1321,d4343,d2121;
        double numer,denom;

        p13.x = p1.x - p3.x;
        p13.y = p1.y - p3.y;
        p13.z = p1.z - p3.z;
        p43.x = p4.x - p3.x;
        p43.y = p4.y - p3.y;
        p43.z = p4.z - p3.z;
        if (ABS(p43.x) < EPS && ABS(p43.y) < EPS && ABS(p43.z) < EPS)
          return(FALSE);
        p21.x = p2.x - p1.x;
        p21.y = p2.y - p1.y;
        p21.z = p2.z - p1.z;
        if (ABS(p21.x) < EPS && ABS(p21.y) < EPS && ABS(p21.z) < EPS)
          return(FALSE);

        d1343 = p13.x * p43.x + p13.y * p43.y + p13.z * p43.z;
        d4321 = p43.x * p21.x + p43.y * p21.y + p43.z * p21.z;
        d1321 = p13.x * p21.x + p13.y * p21.y + p13.z * p21.z;
        d4343 = p43.x * p43.x + p43.y * p43.y + p43.z * p43.z;
        d2121 = p21.x * p21.x + p21.y * p21.y + p21.z * p21.z;

        denom = d2121 * d4343 - d4321 * d4321;
        if (ABS(denom) < EPS)
          return(FALSE);
        numer = d1343 * d4321 - d1321 * d4343;

        *mua = numer / denom;
        *mub = (d1343 + d4321 * (*mua)) / d4343;

        pa->x = p1.x + *mua * p21.x;
        pa->y = p1.y + *mua * p21.y;
        pa->z = p1.z + *mua * p21.z;
        pb->x = p3.x + *mub * p43.x;
        pb->y = p3.y + *mub * p43.y;
        pb->z = p3.z + *mub * p43.z;

        return(TRUE);
        }

        :param another: the second Cartesian line.
        :type another: Cartesian line.
        :param tol: tolerance value for collapsing a segment into the midpoint.
        :type tol: numbers.Real
        :return: the optional shortest segment or an intersection point.
        :rtype: Optional[Union[Segment, Point]]
        """

        check_type(another, "Second Cartesian line", CLine)

        check_crs(self, another)

        epsg_cd = self.epsg()

        p1 = self.start_pt
        p2 = self.end_pt

        p3 = another.start_pt
        p4 = another.end_pt

        p13 = Point(
            x=p1.x - p3.x,
            y=p1.y - p3.y,
            z=p1.z - p3.z,
            epsg_cd=epsg_cd
        )

        p43 = Point(
            x=p4.x - p3.x,
            y=p4.y - p3.y,
            z=p4.z - p3.z,
            epsg_cd=epsg_cd
        )

        if p43.asVect().isAlmostZero:
            return None

        p21 = Point(
            x=p2.x - p1.x,
            y=p2.y - p1.y,
            z=p2.z - p1.z,
            epsg_cd=epsg_cd
        )

        if p21.asVect().isAlmostZero:
            return None

        d1343 = p13.x * p43.x + p13.y * p43.y + p13.z * p43.z
        d4321 = p43.x * p21.x + p43.y * p21.y + p43.z * p21.z
        d1321 = p13.x * p21.x + p13.y * p21.y + p13.z * p21.z
        d4343 = p43.x * p43.x + p43.y * p43.y + p43.z * p43.z
        d2121 = p21.x * p21.x + p21.y * p21.y + p21.z * p21.z

        denom = d2121 * d4343 - d4321 * d4321

        if fabs(denom) < MIN_SCALAR_VALUE:
            return None

        numer = d1343 * d4321 - d1321 * d4343

        mua = numer / denom
        mub = (d1343 + d4321 * mua) / d4343

        pa = Point(
            x=p1.x + mua * p21.x,
            y=p1.y + mua * p21.y,
            z=p1.z + mua * p21.z,
            epsg_cd=epsg_cd
        )

        pb = Point(
            x=p3.x + mub * p43.x,
            y=p3.y + mub * p43.y,
            z=p3.z + mub * p43.z,
            epsg_cd=epsg_cd
        )

        intersection = point_or_segment(
            point1=pa,
            point2=pb,
            tol=tol
        )

        return intersection

    def pointDistance(self,
        point: Point
    ) -> numbers.Real:
        """
        Returns the distance between a line and a point.

        Algorithm from Wolfram MathWorld: Point-Line Distance -- 3-Dimensional

        :param point: input point
        :type point: Point
        :return: the distance
        :rtype: numbers.Real

        Examples:
        """

        v2 = self.end_pt.asVect()
        v1 = self.start_pt.asVect()

        v0 = point.asVect()

        d = abs((v0 - v1).vCross(v0 - v2)) / abs(v2 - v1)

        return d


class Line(object):
    """
    A list of Point objects, all with the same CRS code.
    """

    def __init__(self, pts: Optional[List[Point]] = None, epsg_cd: numbers.Integral = -1):
        """
        Creates the Line instance, when all the provided points have the same CRS codes.

        :param pts: a list of points
        :type pts: List of Point instances.
        :param epsg_cd: the CRS code of the points.
        :type epsg_cd: numbers.Integral.
        :return: a Line instance.
        :rtype: Line.
        :raises: CRSCodeException.

        """

        if pts is None:
            pts = []

        for pt in pts:
            if not isinstance(pt, Point):
                raise Exception("All input data must be point")

        # when implicit (-1) EPSG line code, initialize it to that of the first point

        if pts and epsg_cd == -1:
            epsg_cd = pts[0].epsg()

        # check all points have the same CRS

        for ndx in range(len(pts)):
            pt = pts[ndx]
            if pt.epsg() != epsg_cd:
                raise Exception("All points must have the same '{}' EPSG code".format(epsg_cd))

        self._pts = [pt.clone() for pt in pts]
        self._crs = Crs(epsg_cd)

    @classmethod
    def fromPointList(cls, pt_list: List[List[numbers.Real]], epsg_cd: numbers.Integral = -1) -> 'Line':
        """
        Create a Line instance from a list of x, y and optional z values.

        Example:
          >>> Line.fromPointList([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
          Line with 3 points: (0.0000, 0.0000, 0.0000) ... (0.0000, 1.0000, 0.0000) - EPSG: -1
        """

        pts = []
        for vals in pt_list:
            if len(vals) == 2:
                pt = Point(
                    x=vals[0],
                    y=vals[1],
                    epsg_cd=epsg_cd)
            elif len(vals) == 3:
                pt = Point(
                    x=vals[0],
                    y=vals[1],
                    z=vals[2],
                    epsg_cd=epsg_cd)
            elif len(vals) == 3:
                pt = Point(
                    x=vals[0],
                    y=vals[1],
                    z=vals[2],
                    t=vals[3],
                    epsg_cd=epsg_cd)
            else:
                raise Exception("Point input values should be 2, 3 or 4. {} got ({}).".format(len(vals), vals))

            pts.append(pt)

        return cls(pts, epsg_cd=epsg_cd)

    def extract_pts(self):

        return self._pts

    def extract_pt(self, pt_ndx: numbers.Integral) -> Point:
        """
        Extract the point at index pt_ndx.

        :param pt_ndx: point index.
        :type pt_ndx: numbers.Integral.
        :return: the extracted Point instance.
        :rtype: Point.

        Examples:
        """

        return self._pts[pt_ndx]

    def pts(self):

        return [pt.clone() for pt in self._pts]

    def segment(self,
        ndx: numbers.Integral
    ) -> Segment:
        """
        Returns the segment at index ndx.

        :param ndx: the segment index.
        :type ndx: numbers.Integral
        :return: the segment
        :rtype: Segment
        """

        return Segment(
            start_pt=self.extract_pt(ndx),
            end_pt=self.extract_pt(ndx+1)
        )

    def intersectSegment(self,
        segment: Segment
    ) -> List[Optional[Union[Point, Segment]]]:
        """
        Calculates the possible intersection between the line and a provided segment.

        :param segment: the input segment
        :type segment: Segment
        :return: the possible intersections, points or segments
        :rtype: List[Optional[Union[Point, Segment]]]
        """

        check_type(segment, "Input segment", Segment)
        check_crs(self, segment)

        intersections = [intersect_segments(curr_segment, segment) for curr_segment in self]
        intersections = list(filter(lambda val: val is not None, intersections))

        return intersections

    @property
    def crs(self) -> Crs:

        return self._crs

    def epsg(self) -> numbers.Integral:

        return self.crs.epsg()

    def num_pts(self):

        return len(self._pts)

    def start_pt(self) -> Optional[Point]:
        """
        Return the first point of a Line or None when no points.

        :return: the first point or None.
        :rtype: optional Point instance.
        """

        if self.num_pts() >= 1:
            return self._pts[0].clone()
        else:
            return None

    def end_pt(self) -> Optional[Point]:
        """
        Return the last point of a Line or None when no points.

        :return: the last point or None.
        :rtype: optional Point instance.
        """

        if self.num_pts() >= 1:
            return self._pts[-1].clone()
        else:
            return None

    def __iter__(self):
        """
        Return the elements of a Line, i.e., its segments.
        """

        return (self.segment(i) for i in range(0, self.num_pts()-1))

    def __repr__(self) -> str:
        """
        Represents a Line instance as a shortened text.

        :return: a textual shortened representation of a Line instance.
        :rtype: str.
        """

        num_points = self.num_pts()
        epsg = self.epsg()

        if num_points == 0:
            txt = "Empty Line - EPSG: {}".format(epsg)
        else:
            x1, y1, z1, _, _ = self.start_pt()
            if num_points == 1:
                txt = "Line with unique point: {.4f}.{.4f},{.4f} - EPSG: {}".format(x1, y1, z1, epsg)
            else:
                x2, y2, z2, _, _ = self.end_pt()
                txt = "Line with {} points: ({:.4f}, {:.4f}, {:.4f}) ... ({:.4f}, {:.4f}, {:.4f}) - EPSG: {}".format(num_points, x1, y1, z1, x2, y2, z2, epsg)

        return txt

    def clone(self):

        return Line(
            pts=self._pts,
            epsg_cd=self.epsg()
        )

    def add_pt(self, pt) -> bool:
        """
        In-place transformation of the original Line instance
        by adding a new point at the end.

        :param pt: the point to add
        :type pt: Point.
        :return: status of addition. True when added, False otherwise.
        :rtype: bool.
        """

        if self.num_pts() == 0 and not self.crs.valid():
            self._crs = Crs(pt.epsg())

        if self.num_pts() > 0 and pt.crs != self.crs:
            return False

        self._pts.append(pt.clone())
        return True

    def add_pts(self, pt_list) -> numbers.Integral:
        """
        In-place transformation of the original Line instance
        by adding a new set of points at the end.

        :param pt_list: list of Points.
        :type pt_list: List of Point instances.
        :return: number of added points
        :rtype: numbers.Integral.
        """

        num_added = 0
        for pt in pt_list:
            success = self.add_pt(pt)
            if success:
                num_added += 1

        return num_added

    def x_list(self) -> List[numbers.Real]:

        return [pt.x for pt in self._pts]

    def y_list(self) -> List[numbers.Real]:

        return [pt.y for pt in self._pts]

    def z_list(self) -> List[numbers.Real]:

        return [pt.z for pt in self._pts]

    def t_list(self) -> List[numbers.Real]:

        return [pt.t for pt in self._pts]

    def z_array(self) -> np.array:

        return np.array(self.z_list())

    def xy_lists(self) -> Tuple[List[numbers.Real], List[numbers.Real]]:

        return self.x_list(), self.y_list()

    def x_min(self) -> Optional[numbers.Real]:

        return find_val(
            func=min,
            lst=self.x_list())

    def x_max(self) -> Optional[numbers.Real]:

        return find_val(
            func=max,
            lst=self.x_list())

    def y_min(self) -> Optional[numbers.Real]:

        return find_val(
            func=min,
            lst=self.y_list())

    def y_max(self) -> Optional[numbers.Real]:

        return find_val(
            func=max,
            lst=self.y_list())

    def z_stats(self) -> Dict:
        """
        Returns the line elevation statistics.

        :return: the statistics parameters: min, max, mean, var, std.
        :rtype: Dictionary of numbers.Real values.
        """

        return get_statistics(self.z_array())

    def z_min(self) -> Optional[numbers.Real]:

        return find_val(
            func=min,
            lst=self.z_list())

    def z_max(self) -> Optional[numbers.Real]:

        return find_val(
            func=max,
            lst=self.z_list())

    def z_mean(self) -> Optional[numbers.Real]:

        zs = self.z_list()
        return float(np.mean(zs)) if zs else None

    def z_var(self) -> Optional[numbers.Real]:

        zs = self.z_list()
        return float(np.var(zs)) if zs else None

    def z_std(self) -> Optional[numbers.Real]:

        zs = self.z_list()
        return float(np.std(zs)) if zs else None

    def remove_coincident_points(self) -> 'Line':
        """
        Remove coincident successive points

        :return: Line instance
        """

        new_line = Line(
            pts=self._pts[:1])

        for ndx in range(1, self.num_pts()):
            if not self._pts[ndx].isCoinc(new_line._pts[-1]):
                new_line.add_pt(self._pts[ndx])

        return new_line

    def as_segments(self):
        """
        Convert to a list of segments.

        :return: list of Segment objects
        """

        pts_pairs = zip(self._pts[:-1], self._pts[1:])

        segments = [Segment(pt_a, pt_b) for (pt_a, pt_b) in pts_pairs]

        return segments

    def densify_2d_line(self, sample_distance) -> 'Line':
        """
        Densify a line into a new line instance,
        using the provided sample distance.
        Returned Line instance has coincident successive points removed.

        :param sample_distance: numbers.Real
        :return: Line instance
        """

        if sample_distance <= 0.0:
            raise Exception("Sample distance must be positive. {} received".format(sample_distance))

        segments = self.as_segments()

        densified_line_list = [segment.densify2d_asLine(sample_distance) for segment in segments]

        densifyied_multiline = MultiLine(densified_line_list, epsg_cd=self.epsg())

        densifyied_line = densifyied_multiline.to_line()

        densifyied_line_wo_coinc_pts = densifyied_line.remove_coincident_points()

        return densifyied_line_wo_coinc_pts

    def join(self, another) -> 'Line':
        """
        Joins together two lines and returns the join as a new line without point changes,
        with possible overlapping points
        and orientation mismatches between the two original lines
        """

        return Line(self.pts() + another.pts())

    def length_3d(self) -> numbers.Real:

        length = 0.0
        for ndx in range(self.num_pts() - 1):
            length += self._pts[ndx].dist3DWith(self._pts[ndx + 1])
        return length

    def length_2d(self) -> numbers.Real:

        length = 0.0
        for ndx in range(self.num_pts() - 1):
            length += self._pts[ndx].dist2DWith(self._pts[ndx + 1])
        return length

    def step_delta_z(self) -> List[numbers.Real]:
        """
        Return the difference in elevation between consecutive points:
        z[ndx+1] - z[ndx]

        :return: a list of height differences.
        :rtype: list of floats.
        """

        delta_z = [0.0]

        for ndx in range(1, self.num_pts()):
            delta_z.append(self._pts[ndx].z - self._pts[ndx - 1].z)

        return delta_z

    def step_lengths_3d(self) -> List[numbers.Real]:
        """
        Returns the point-to-point 3D distances.
        It is the distance between a point and its previous one.
        The list has the same lenght as the source point list.

        :return: the individual 3D segment lengths.
        :rtype: list of floats.

        Examples:
        """

        step_length_list = [0.0]
        for ndx in range(1, self.num_pts()):
            length = self._pts[ndx].dist3DWith(self._pts[ndx - 1])
            step_length_list.append(length)

        return step_length_list

    def step_lengths_2d(self) -> List[numbers.Real]:
        """
        Returns the point-to-point 2D distances.
        It is the distance between a point and its previous one.
        The list has the same length as the source point list.

        :return: the individual 2D segment lengths.
        :rtype: list of floats.

        Examples:
        """

        step_length_list = [0.0]
        for ndx in range(1, self.num_pts()):
            length = self._pts[ndx].dist2DWith(self._pts[ndx - 1])
            step_length_list.append(length)

        return step_length_list

    def incremental_length_3d(self) -> List[numbers.Real]:
        """
        Returns the accumulated 3D segment lengths.

        :return: accumulated 3D segment lenghts
        :rtype: list of floats.
        """

        return list(itertools.accumulate(self.step_lengths_3d()))

    def incremental_length_2d(self) -> List[numbers.Real]:
        """
        Returns the accumulated 2D segment lengths.

        :return: accumulated 2D segment lenghts
        :rtype: list of floats.
        """

        return list(itertools.accumulate(self.step_lengths_2d()))

    def reversed(self) -> 'Line':
        """
        Return a Line instance with reversed point list.

        :return: a new Line instance.
        :rtype: Line.
        """

        new_line = self.clone()
        new_line._pts.reverse()  # in-place operation on new_line

        return new_line

    def slopes_degr(self) -> List[Optional[numbers.Real]]:
        """
        Calculates the slopes (in degrees) of each Line segment.
        The first value is the slope of the first segment.
        The last value, always None, is the slope of the segment starting at the last point.
        The number of elements is equal to the number of points in the Line.

        :return: list of slopes (degrees).
        :rtype: List[Optional[numbers.Real]].
        """

        lSlopes = []

        segments = self.as_segments()
        for segment in segments:
            vector = segment.vector()
            lSlopes.append(-vector.slope_degr())  # minus because vector convention is positive downward

        lSlopes.append(None)  # None refers to the slope of the Segment starting with the last point

        return lSlopes

    def slopes_stats(self) -> Dict:
        """
        Returns the line directional slope statistics.

        :return: the statistics parameters: min, max, mean, var, std.
        :rtype: Dictionary.
        """

        return get_statistics(self.slopes_degr())

    def abs_slopes_degr(self) -> List[Optional[numbers.Real]]:

        return [abs(val) for val in self.slopes_degr()]

    def abs_slopes_stats(self) -> Dict:
        """
        Returns the line absolute slopes statistics.

        :return: the statistics parameters: min, max, mean, var, std.
        :rtype: Dictionary.
        """

        return get_statistics(self.abs_slopes_degr())


class MultiLine(object):
    """
    MultiLine is a list of Line objects, each one with the same CRS code
    """

    def __init__(self, lines: Optional[List[Line]] = None, epsg_cd: numbers.Integral = -1):

        if lines is None:
            lines = []

        if lines and epsg_cd == -1:
            epsg_cd = lines[0].epsg()

        for ndx in range(len(lines)):
            if lines[ndx].epsg() != epsg_cd:
                raise Exception("Input line with index {} should have EPSG code {} but has {}".format(
                    ndx,
                    epsg_cd,
                    lines[ndx].epsg()
                ))

        self._lines = lines
        self._crs = Crs(epsg_cd)

    def lines(self):

        return self._lines

    @property
    def crs(self) -> Crs:

        return self._crs

    def epsg(self) -> numbers.Integral:

        return self._crs.epsg()

    def num_lines(self):

        return len(self.lines())

    def num_tot_pts(self) -> numbers.Integral:

        num_points = 0
        for line in self._lines:
            num_points += line.num_pts()

        return num_points

    def line(self, ln_ndx: numbers.Integral = 0) -> Optional[Line]:
        """
        Extracts a line from the multiline instance, based on the provided index.

        :return: Line instance or None when ln_ndx is out-of-range.
        :rtype: Optional[Line].
        """

        num_lines = self.num_lines()
        if num_lines == 0:
            return None

        if ln_ndx not in range(num_lines):
            return None

        return self.lines()[ln_ndx]

    def __iter__(self):
        """
        Return the elements of a MultiLine, i.e., its lines.
        """

        return (self.line(i) for i in range(0, self.num_lines()-1))

    def __repr__(self) -> str:
        """
        Represents a MultiLine instance as a shortened text.

        :return: a textual shortened representation of a MultiLine instance.
        :rtype: basestring.
        """

        num_lines = self.num_lines()
        num_tot_pts = self.num_tot_pts()
        epsg = self.epsg()

        txt = "MultiLine with {} line(s) and {} total point(s) - EPSG: {}".format(num_lines, num_tot_pts, epsg)

        return txt

    def __len__(self):
        """
        Return number of lines.

        :return: number of lines
        :rtype: numbers.Integral
        """

        return self.num_lines()

    def add_line(self, line) -> bool:
        """
        In-place addition of a Line instance (that is not cloned).

        :param line: the line to add.
        :type line: Line.
        :return: status of addition. True when added, False otherwise.
        :rtype: bool.
        """

        if self.num_lines() == 0 and not self.crs.valid():
            self._crs = line.crs

        if self.num_lines() > 0 and line.crs != self.crs:
            return False

        self._lines += [line]
        return True

    def clone(self) -> 'MultiLine':

        return MultiLine(
            lines=[line.clone() for line in self._lines],
            epsg_cd=self.epsg()
        )

    def x_min(self) -> Optional[numbers.Real]:

        if self.num_tot_pts() == 0:
            return None
        else:
            return float(np.nanmin([line.x_min() for line in self.lines()]))

    def x_max(self) -> Optional[numbers.Real]:

        if self.num_tot_pts() == 0:
            return None
        else:
            return float(np.nanmax([line.x_max() for line in self.lines()]))

    def y_min(self) -> Optional[numbers.Real]:

        if self.num_tot_pts() == 0:
            return None
        else:
            return float(np.nanmin([line.y_min() for line in self.lines()]))

    def y_max(self) -> Optional[numbers.Real]:

        if self.num_tot_pts() == 0:
            return None
        else:
            return float(np.nanmax([line.y_max() for line in self.lines()]))

    def z_min(self) -> Optional[numbers.Real]:

        if self.num_tot_pts() == 0:
            return None
        else:
            return float(np.nanmin([line.z_min() for line in self.lines()]))

    def z_max(self) -> Optional[numbers.Real]:

        if self.num_tot_pts() == 0:
            return None
        else:
            return float(np.nanmax([line.z_max() for line in self.lines()]))

    def is_continuous(self) -> bool:
        """
        Checks whether all lines in a multiline are connected.

        :return: whether all lines are connected.
        :rtype: bool.
        """

        if len(self._lines) <= 1:
            return False

        for line_ndx in range(len(self._lines) - 1):
            first = self._lines[line_ndx]
            second = self._lines[line_ndx + 1]
            if not analizeJoins(first, second):
                return False

        return True

    def is_unidirectional(self):

        for line_ndx in range(len(self.lines()) - 1):
            if not self.lines()[line_ndx].extract_pts()[-1].isCoinc(self.lines()[line_ndx + 1].extract_pts()[0]):
                return False

        return True

    def to_line(self):

        return Line([point for line in self._lines for point in line.extract_pts()], epsg_cd=self.epsg())

    def densify_2d_multiline(self, sample_distance):

        lDensifiedLines = []
        for line in self.lines():
            lDensifiedLines.append(line.densify_2d_line(sample_distance))

        return MultiLine(lDensifiedLines, self.epsg())

    def remove_coincident_points(self):

        cleaned_lines = []
        for line in self.lines():
            cleaned_lines.append(line.remove_coincident_points())

        return MultiLine(cleaned_lines, self.epsg())

    def intersectSegment(self,
        segment: Segment
    ) -> List[Optional[Union[Point, 'Segment']]]:
        """
        Calculates the possible intersection between the multiline and a provided segment.

        :param segment: the input segment
        :type segment: Segment
        :return: the possible intersections, points or segments
        :rtype: List[List[Optional[Union[Point, 'Segment']]]]
        """

        check_type(segment, "Input segment", Segment)
        check_crs(self, segment)

        intersections = []
        for line in self:
            intersections.extend(line.intersectSegment(segment))

        return intersections


class ParamLine3D(object):
    """
    parametric line
    srcPt: source Point
    l, m, n: line coefficients
    """

    def __init__(self, srcPt, l, m, n):

        for v in (l, m, n):
            if not (-1.0 <= v <= 1.0):
                raise Exception("Parametric line values must be in -1 to 1 range")

        self._srcPt = srcPt.clone()
        self._l = l
        self._m = m
        self._n = n

    def intersect_cartes_plane(self, cartes_plane) -> Optional[Point]:
        """
        Return intersection point between parametric line and Cartesian plane.

        :param cartes_plane: a Cartesian plane:
        :type cartes_plane: CPlane.
        :return: the intersection point between parametric line and Cartesian plane.
        :rtype: Point.
        :raise: Exception.
        """

        if not isinstance(cartes_plane, CPlane):
            raise Exception("Method argument should be a Cartesian plane but is {}".format(type(cartes_plane)))

        # line parameters
        x1, y1, z1 = self._srcPt.x, self._srcPt.y, self._srcPt.z
        l, m, n = self._l, self._m, self._n

        # Cartesian plane parameters
        a, b, c, d = cartes_plane.a, cartes_plane.b, cartes_plane.c, cartes_plane.d

        try:
            k = (a * x1 + b * y1 + c * z1 + d) / (a * l + b * m + c * n)
        except ZeroDivisionError:
            return None

        return Point(x1 - l * k,
                     y1 - m * k,
                     z1 - n * k)


class JoinTypes(Enum):
    """
    Enumeration for Line and Segment type.
    """

    START_START = 1  # start point coincident with start point
    START_END   = 2  # start point coincident with end point
    END_START   = 3  # end point coincident with start point
    END_END     = 4  # end point coincident with end point


def analizeJoins(first: Union[Line, Segment], second: Union[Line, Segment]) -> List[Optional[JoinTypes]]:
    """
    Analyze join types between two lines/segments.

    :param first: a line or segment.
    :type first: Line or Segment.
    :param second: a line or segment.
    :param second: Line or Segment.
    :return: a list of join types.
    :rtype: List[Optional[JoinTypes]].

    Examples:
      >>> first = Segment(Point(x=0,y=0, epsg_cd=32632), Point(x=1,y=0, epsg_cd=32632))
      >>> second = Segment(Point(x=1,y=0, epsg_cd=32632), Point(x=0,y=0, epsg_cd=32632))
      >>> analizeJoins(first, second)
      [<JoinTypes.START_END: 2>, <JoinTypes.END_START: 3>]
      >>> first = Segment(Point(x=0,y=0, epsg_cd=32632), Point(x=1,y=0, epsg_cd=32632))
      >>> second = Segment(Point(x=2,y=0, epsg_cd=32632), Point(x=3,y=0, epsg_cd=32632))
      >>> analizeJoins(first, second)
      []
    """

    join_types = []

    if first.start_pt.isCoinc(second.start_pt):
        join_types.append(JoinTypes.START_START)

    if first.start_pt.isCoinc(second.end_pt):
        join_types.append(JoinTypes.START_END)

    if first.end_pt.isCoinc(second.start_pt):
        join_types.append(JoinTypes.END_START)

    if first.end_pt.isCoinc(second.end_pt):
        join_types.append(JoinTypes.END_END)

    return join_types


class Plane(object):
    """
    Geological plane.
    Defined by dip direction and dip angle (both in degrees):
     - dip direction: [0.0, 360.0[ clockwise, from 0 (North);
     - dip angle: [0, 90.0]: downward-pointing.
    """

    def __init__(self, azim: numbers.Real, dip_ang: numbers.Real, is_rhr_strike: bool=False):
        """
        Geological plane constructor.

        :param  azim:  azimuth of the plane (RHR strike or dip direction).
        :type  azim:  number or string convertible to float.
        :param  dip_ang:  Dip angle of the plane (0-90°).
        :type  dip_ang:  number or string convertible to float.
        :param is_rhr_strike: if the source azimuth is RHR strike (default is False, i.e. it is dip direction)
        :return: the instantiated geological plane.
        :rtype: Plane.

        Example:
          >>> Plane(0, 90)
          Plane(000.00, +90.00)
          >>> Plane(0, 90, is_rhr_strike=True)
          Plane(090.00, +90.00)
          >>> Plane(0, 90, True)
          Plane(090.00, +90.00)
          >>> Plane(0, "90", True)
          Traceback (most recent call last):
          ...
          Exception: Source dip angle must be number
          >>> Plane(0, 900)
          Traceback (most recent call last):
          ...
          Exception: Dip angle must be between 0° and 90°
        """

        def rhrstrk2dd(rhr_strk):
            """Converts RHR strike value to dip direction value.

            Example:
                >>> rhrstrk2dd(285.5)
                15.5
            """

            return (rhr_strk + 90.0) % 360.0

        if not isinstance(azim, numbers.Real):
            raise Exception("Source azimuth must be number")
        if not isinstance(dip_ang, numbers.Real):
            raise Exception("Source dip angle must be number")
        if not isinstance(is_rhr_strike, bool):
            raise Exception("Source azimuth type must be boolean")

        if not (0.0 <= dip_ang <= 90.0):
            raise Exception("Dip angle must be between 0° and 90°")

        if is_rhr_strike:
            self._dipdir = rhrstrk2dd(azim)
        else:
            self._dipdir = azim % 360.0
        self._dipangle = float(dip_ang)

    @property
    def dd(self):
        """
        Return the dip direction of the geological plane.

        Example:
          >>> Plane(34.2, 89.7).dd
          34.2
        """

        return self._dipdir

    @property
    def da(self):
        """
        Return the dip angle of the geological plane.

        Example:
          >>> Plane(183, 77).da
          77.0

        """

        return self._dipangle

    @property
    def dda(self):
        """
        Return a tuple storing the dip direction and dip angle values of a geological plane.

        Example:
          >>> gp = Plane(89.4, 17.2)
          >>> gp.dda
          (89.4, 17.2)
        """

        return self.dd, self.da

    @property
    def rhrStrike(self):
        """
        Return the strike according to the right-hand-rule.

        Examples:
          >>> Plane(90, 45).rhrStrike
          0.0
          >>> Plane(45, 89).rhrStrike
          315.0
          >>> Plane(275, 38).rhrStrike
          185.0
          >>> Plane(0, 38).rhrStrike
          270.0
        """

        return (self.dd - 90.0) % 360.0

    @property
    def srda(self):
        """
        Return a tuple storing the right-hand-rule strike and dip angle values of a geological plane.

        Example:
          >>> Plane(100, 17.2).srda
          (10.0, 17.2)
          >>> Plane(10, 87).srda
          (280.0, 87.0)
        """

        return self.rhrStrike, self.da

    @property
    def lhrStrike(self):
        """
        Return the strike according to the left-hand-rule.

        Examples:
          >>> Plane(90, 45).lhrStrike
          180.0
          >>> Plane(45, 89).lhrStrike
          135.0
          >>> Plane(275, 38).lhrStrike
          5.0
          >>> Plane(0, 38).lhrStrike
          90.0
        """

        return (self.dd + 90.0) % 360.0

    @property
    def slda(self):
        """
        Return a tuple storing the left-hand-rule strike and dip angle values of a geological plane.

        Example:
          >>> Plane(100, 17.2).slda
          (190.0, 17.2)
          >>> Plane(10, 87).slda
          (100.0, 87.0)
        """

        return self.lhrStrike, self.da

    def __repr__(self):

        return "Plane({:06.2f}, {:+06.2f})".format(*self.dda)

    def rhrStrikeOrien(self):
        """
        Creates a OrienM instance that is parallel to the right-hand rule strike.

        :return: OrienM instance,

        Examples:
          >>> Plane(90, 45).rhrStrikeOrien()
          Direct(az: 0.00°, pl: 0.00°)
          >>> Plane(45, 17).rhrStrikeOrien()
          Direct(az: 315.00°, pl: 0.00°)
          >>> Plane(90, 0).rhrStrikeOrien()
          Direct(az: 0.00°, pl: 0.00°)
        """

        return Direct.fromAzPl(
            az=self.rhrStrike,
            pl=0.0)

    def lhrStrikeOrien(self):
        """
        Creates an Orientation instance that is parallel to the left-hand rule strike.

        :return: OrienM instance.

        Examples:
          >>> Plane(90, 45).lhrStrikeOrien()
          Direct(az: 180.00°, pl: 0.00°)
          >>> Plane(45, 17).lhrStrikeOrien()
          Direct(az: 135.00°, pl: 0.00°)
        """

        return Direct.fromAzPl(
            az=self.lhrStrike,
            pl=0.0)

    def dipDirOrien(self):
        """
        Creates a OrienM instance that is parallel to the dip direction.

        :return: OrienM instance.

        Examples:
          >>> Plane(90, 45).dipDirOrien()
          Direct(az: 90.00°, pl: 45.00°)
          >>> Plane(45, 17).dipDirOrien()
          Direct(az: 45.00°, pl: 17.00°)
        """

        return Direct.fromAzPl(
            az=self.dd,
            pl=self.da)

    def dipDirOppOrien(self):
        """
        Creates a OrienM instance that is anti-parallel to the dip direction.

        :return: OrienM instance.

        Examples:
          >>> Plane(90, 45).dipDirOppOrien()
          Direct(az: 270.00°, pl: -45.00°)
          >>> Plane(45, 17).dipDirOppOrien()
          Direct(az: 225.00°, pl: -17.00°)
        """

        return self.dipDirOrien().opposite()

    def mirrorVertPPlane(self):
        """
        Mirror a geological plane around a vertical plane
        creating a new one that has a dip direction opposite
        to the original one but with downward plunge.

        :return: geological plane
        :rtype: Plane

        Examples:
          >>> Plane(0, 45).mirrorVertPPlane()
          Plane(180.00, +45.00)
          >>> Plane(225, 80).mirrorVertPPlane()
          Plane(045.00, +80.00)
          >>> Plane(90, 90).mirrorVertPPlane()
          Plane(270.00, +90.00)
          >>> Plane(270, 0).mirrorVertPPlane()
          Plane(090.00, +00.00)
        """

        return Plane(
            azim=opposite_trend(self.dd),
            dip_ang=self.da)

    def normDirectFrwrd(self):
        """
        Return the direction normal to the geological plane,
        pointing in the same direction as the geological plane.

        Example:
            >>> Plane(90, 55).normDirectFrwrd()
            Direct(az: 90.00°, pl: -35.00°)
            >>> Plane(90, 90).normDirectFrwrd()
            Direct(az: 90.00°, pl: 0.00°)
            >>> Plane(90, 0).normDirectFrwrd()
            Direct(az: 90.00°, pl: -90.00°)
        """

        tr = self.dd % 360.0
        pl = self.da - 90.0

        return Direct.fromAzPl(
            az=tr,
            pl=pl)

    def normDirectBckwrd(self):
        """
        Return the direction normal to the geological plane,
        pointing in the opposite direction to the geological plane.

        Example:
            >>> Plane(90, 55).normDirectBckwrd()
            Direct(az: 270.00°, pl: 35.00°)
            >>> Plane(90, 90).normDirectBckwrd()
            Direct(az: 270.00°, pl: -0.00°)
            >>> Plane(90, 0).normDirectBckwrd()
            Direct(az: 270.00°, pl: 90.00°)
        """

        return self.normDirectFrwrd().opposite()

    def normDirectDown(self):
        """
        Return the direction normal to the geological plane and
        pointing downward.

        Example:
            >>> Plane(90, 55).normDirectDown()
            Direct(az: 270.00°, pl: 35.00°)
            >>> Plane(90, 90).normDirectDown()
            Direct(az: 90.00°, pl: 0.00°)
            >>> Plane(90, 0).normDirectDown()
            Direct(az: 270.00°, pl: 90.00°)
        """

        return self.normDirectFrwrd().downward()

    def normDirectUp(self):
        """
        Return the direction normal to the polar plane,
        pointing upward.

        Example:
            >>> Plane(90, 55).normDirectUp()
            Direct(az: 90.00°, pl: -35.00°)
            >>> Plane(90, 90).normDirectUp()
            Direct(az: 90.00°, pl: 0.00°)
            >>> Plane(90, 0).normDirectUp()
            Direct(az: 90.00°, pl: -90.00°)
        """

        return self.normDirectFrwrd().upward()

    def normDirect(self) -> 'Direct':
        """
        Wrapper to down_normal_gv.

        :return: downward-pointing Direct instance normal to the Plane self instance
        """

        return self.normDirectDown()

    def normAxis(self):
        """
        Normal Axis.

        :return: Axis normal to the Plane self instance
        """

        return self.normDirectDown().asAxis()

    def angle(self, another: 'Plane'):
        """
        Calculate angle (in degrees) between two geoplanes.
        Range is 0°-90°.

        Examples:
          >>> Plane(100.0, 50.0).angle(Plane(100.0, 50.0))
          0.0
          >>> Plane(300.0, 10.0).angle(Plane(300.0, 90.0))
          80.0
          >>> Plane(90.0, 90.0).angle(Plane(270.0, 90.0))
          0.0
          >>> areClose(Plane(90.0, 90.0).angle(Plane(130.0, 90.0)), 40)
          True
          >>> areClose(Plane(90, 70).angle(Plane(270, 70)), 40)
          True
          >>> areClose(Plane(90.0, 10.0).angle(Plane(270.0, 10.0)), 20.0)
          True
          >>> areClose(Plane(90.0, 10.0).angle(Plane(270.0, 30.0)), 40.0)
          True
        """

        if not isinstance(another, Plane):
            raise Exception("Second instance for angle is of {} type".format(type(another)))

        gpl_axis = self.normDirectFrwrd().asAxis()
        an_axis = another.normDirectFrwrd().asAxis()

        return gpl_axis.angle(an_axis)

    def isSubParallel(self, another, angle_tolerance: numbers.Real=PLANE_ANGLE_THRESHOLD):
        """
        Check that two GPlanes are sub-parallel

        :param another: a Plane instance
        :param angle_tolerance: the maximum allowed divergence angle (in degrees)
        :return: Boolean

         Examples:
          >>> Plane(0, 90).isSubParallel(Plane(270, 90))
          False
          >>> Plane(0, 90).isSubParallel(Plane(180, 90))
          True
          >>> Plane(0, 90).isSubParallel(Plane(0, 0))
          False
          >>> Plane(0, 0).isSubParallel(Plane(0, 1e-6))
          True
          >>> Plane(0, 0).isSubParallel(Plane(0, 1.1))
          False
        """

        return self.angle(another) < angle_tolerance

    def contains(self,
        direct: 'Direct',
        angle_tolerance: numbers.Real=PLANE_ANGLE_THRESHOLD
    ) -> bool:
        """
        Check that a plane contains a direction instance.

        :param direct: a Direct instance
        :param angle_tolerance: the tolerance angle
        :return: True or False

        Examples:
          >>> Plane(90, 0).contains(Direct.fromAzPl(60, 0))
          True
          >>> Plane(90, 0).contains(Axis.fromAzPl(60, 0))
          True
          >>> Plane(90, 0).contains(Direct.fromAzPl(60, 10))
          False
        """

        plane_norm = self.normAxis()

        return direct.isSubOrthog(plane_norm, angle_tolerance)

    def isSubOrthog(self, another, angle_tolerance: numbers.Real=PLANE_ANGLE_THRESHOLD):
        """
        Check that two GPlanes are sub-orthogonal.

        :param another: a Plane instance
        :param angle_tolerance: the maximum allowed divergence angle (in degrees)
        :return: Boolean

         Examples:
          >>> Plane(0, 90).isSubOrthog(Plane(270, 90))
          True
          >>> Plane(0, 90).isSubOrthog(Plane(180, 90))
          False
          >>> Plane(0, 90).isSubOrthog(Plane(0, 0))
          True
          >>> Plane(0, 0).isSubOrthog(Plane(0, 88))
          False
          >>> Plane(0, 0).isSubOrthog(Plane(0, 45))
          False
        """

        fst_axis = self.normDirect().asAxis()

        if isinstance(another, Plane):
            snd_gaxis = another.normDirect().asAxis()
        else:
            raise Exception("Not accepted argument type for isSubOrthog method")

        angle = fst_axis.angle(snd_gaxis)

        if isinstance(another, Plane):
            return angle > 90.0 - angle_tolerance
        else:
            return angle < angle_tolerance

    def rakeToDirect(self, rake):
        """
        Calculate the Direct instance given a Plane instance and a rake value.
        The rake is defined according to the Aki and Richards, 1980 conventions:
        rake = 0° -> left-lateral
        rake = 90° -> reverse
        rake = +/- 180° -> right-lateral
        rake = -90° -> normal

        Examples:
          >>> Plane(180, 45).rakeToDirect(0.0)
          Direct(az: 90.00°, pl: -0.00°)
          >>> Plane(180, 45).rakeToDirect(90.0)
          Direct(az: 0.00°, pl: -45.00°)
          >>> Plane(180, 45).rakeToDirect(-90.0)
          Direct(az: 180.00°, pl: 45.00°)
          >>> Plane(180, 45).rakeToDirect(180.0).isSubParallel(Direct.fromAzPl(270.00, 0.00))
          True
          >>> Plane(180, 45).rakeToDirect(-180.0)
          Direct(az: 270.00°, pl: 0.00°)
        """

        rk = radians(rake)
        strk = radians(self.rhrStrike)
        dip = radians(self.da)

        x = cos(rk) * sin(strk) - sin(rk) * cos(dip) * cos(strk)
        y = cos(rk) * cos(strk) + sin(rk) * cos(dip) * sin(strk)
        z = sin(rk) * sin(dip)

        return Direct.fromXYZ(x, y, z)

    def isVLowAngle(self, dip_angle_threshold: numbers.Real=angle_gplane_thrshld):
        """
        Checks if a geological plane is very low angle.

        :param dip_angle_threshold: the limit for the plane angle, in degrees
        :type dip_angle_threshold: numbers.Real.
        :return: bool flag indicating if it is very low angle

        Examples:
          >>> Plane(38.9, 1.2).isVLowAngle()
          True
          >>> Plane(38.9, 7.4).isVLowAngle()
          False
        """

        return self.da < dip_angle_threshold

    def isVHighAngle(self, dip_angle_threshold: numbers.Real=angle_gplane_thrshld):
        """
        Checks if a geological plane is very high angle.

        :param dip_angle_threshold: the limit for the plane angle, in degrees
        :type dip_angle_threshold: numbers.Real.
        :return: bool flag indicating if it is very high angle

        Examples:
          >>> Plane(38.9, 11.2).isVHighAngle()
          False
          >>> Plane(38.9, 88.4).isVHighAngle()
          True
        """

        return self.da > (90.0 - dip_angle_threshold)

    def toCPlane(self, pt):
        """
        Given a Plane instance and a provided Point instance,
        calculate the corresponding Plane instance.

        Example:
          >>> Plane(0, 0).toCPlane(Point(0, 0, 0))
          CPlane(0.0000, 0.0000, 1.0000, -0.0000, -1)
          >>> Plane(90, 45).toCPlane(Point(0, 0, 0))
          CPlane(0.7071, 0.0000, 0.7071, -0.0000, -1)
          >>> Plane(0, 90).toCPlane(Point(0, 0, 0))
          CPlane(0.0000, 1.0000, -0.0000, -0.0000, -1)
        """

        normal_versor = self.normDirectFrwrd().asVersor()
        a, b, c = normal_versor.x, normal_versor.y, normal_versor.z
        d = - (a * pt.x + b * pt.y + c * pt.z)
        return CPlane(a, b, c, d, epsg_cd=pt.epsg())

    def slope_x_dir(self) -> numbers.Real:
        """
        Calculate the slope of a given plane along the x direction.
        The plane orientation  is expressed following the geological convention.

        :return: the slope along the x direction
        :rtype: numbers.Real.

        Example:
        """
        return - sin(radians(self.dd)) * tan(radians(self.da))

    def slope_y_dir(self) -> numbers.Real:
        """
        Calculate the slope of a given plane along the y direction.
        The plane orientation  is expressed following the geological convention.

        :return: the slope along the y direction
        :rtype: numbers.Real.

        Example:
        """
        return - cos(radians(self.dd)) * tan(radians(self.da))

    def closure_plane_from_geo(self, src_pt: Point) -> Callable:
        """
        Closure that embodies the analytical formula for a given, non-vertical plane.
        This closure is used to calculate the z value from given horizontal coordinates (x, y).

        :param src_pt: Point_3D instance expressing a location point contained by the plane.
        :type src_pt: Point_3D.

        :return: lambda (closure) expressing an analytical formula for deriving z given x and y values.
        """

        x0 = src_pt.x
        y0 = src_pt.y
        z0 = src_pt.z

        # slope of the line parallel to the x axis and contained by the plane
        a = self.slope_x_dir()

        # slope of the line parallel to the y axis and contained by the plane
        b = self.slope_y_dir()

        return lambda x, y: a * (x - x0) + b * (y - y0) + z0


class Azim(object):
    """
    Azim class
    """

    def __init__(self,
        val: numbers.Real,
        unit: str = 'd'
    ):
        """
        Creates an azimuth instance.

        :param val: azimuth value
        :param unit: angle measurement unit, 'd' (default, stands for decimal degrees) or 'r' (stands for radians)

        Examples:
          >>> Azim(10)
          Azimuth(10.00°)
          >>> Azim(370)
          Azimuth(10.00°)
          >>> Azim(pi/2, unit='r')
          Azimuth(90.00°)
          >>> Azim("10")
          Traceback (most recent call last):
          ...
          Exception: Input azimuth value must be int/float
          >>> Azim(np.nan)
          Traceback (most recent call last):
          ...
          Exception: Input azimuth value must be finite
        """

        # unit check
        if unit not in ("d", "r"):
            raise Exception("Unit input must be 'd' or 'r'")

        if not (isinstance(val, numbers.Real)):
            raise Exception("Input azimuth value must be int/float")
        elif not isfinite(val):
            raise Exception("Input azimuth value must be finite")

        if unit == 'd':
            val = radians(val)

        self.a = val % (2*pi)

    @property
    def d(self
    ):
        """
        Returns the angle in decimal degrees.

        :return: angle in decimal degrees

        Example:
          >>> Azim(10).d
          10.0
          >>> Azim(pi/2, unit='r').d
          90.0
        """

        return degrees(self.a)

    @property
    def r(self
    ):
        """
        Returns the angle in radians.

        :return: angle in radians

        Example:
          >>> Azim(180).r
          3.141592653589793
        """

        return self.a

    @classmethod
    def fromXY(cls,
        x: numbers.Real,
        y: numbers.Real
    ) -> 'Azim':
        """
        Calculates azimuth given cartesian components.

        :param cls: class
        :param x: x component
        :param y: y component
        :return: Azimuth instance

        Examples:
          >>> Azim.fromXY(1, 1)
          Azimuth(45.00°)
          >>> Azim.fromXY(1, -1)
          Azimuth(135.00°)
          >>> Azim.fromXY(-1, -1)
          Azimuth(225.00°)
          >>> Azim.fromXY(-1, 1)
          Azimuth(315.00°)
          >>> Azim.fromXY(0, 0)
          Azimuth(0.00°)
          >>> Azim.fromXY(0, np.nan)
          Traceback (most recent call last):
          ...
          Exception: Input x and y values must be finite
          >>> Azim.fromXY("10", np.nan)
          Traceback (most recent call last):
          ...
          Exception: Input x and y values must be integer or float
        """

        # input vals checks
        vals = [x, y]
        if not all(map(lambda val: isinstance(val, numbers.Real), vals)):
            raise Exception("Input x and y values must be integer or float")
        elif not all(map(isfinite, vals)):
            raise Exception("Input x and y values must be finite")

        angle = atan2(x, y)
        return cls(angle, unit='r')

    def __repr__(self) -> str:

        return "Azimuth({:.2f}°)".format(self.d)

    def toXY(self
    ) -> Tuple[numbers.Real, numbers.Real]:
        """
        Converts an azimuth to x-y components.

        :return: a tuple storing x and y values:
        :type: tuple of two floats

        Examples:
          >>> apprFTuple(Azim(0).toXY())
          (0.0, 1.0)
          >>> apprFTuple(Azim(90).toXY())
          (1.0, 0.0)
          >>> apprFTuple(Azim(180).toXY())
          (0.0, -1.0)
          >>> apprFTuple(Azim(270).toXY())
          (-1.0, 0.0)
          >>> apprFTuple(Azim(360).toXY())
          (0.0, 1.0)
        """

        return sin(self.a), cos(self.a)


class Plunge(object):
    """
    Class representing a plunge
    """

    def __init__(self,
        val: numbers.Real,
        unit: str='d'
    ):
        """
        Creates a Plunge instance.

        :param val: plunge value
        :param unit: angle measurement unit, decimal degrees ('d') or radians ('r')

        Examples:
          >>> Plunge(10)
          Plunge(10.00°)
          >>> Plunge("10")
          Traceback (most recent call last):
          ...
          Exception: Input plunge value must be int/float
          >>> Plunge(np.nan)
          Traceback (most recent call last):
          ...
          Exception: Input plunge value must be finite
          >>> Plunge(-100)
          Traceback (most recent call last):
          ...
          Exception: Input value in degrees must be between -90° and 90°
         """

        # unit check
        if unit not in ('d', 'r'):
            raise Exception("Unit input must be 'd' (for degrees) or 'r' (for radians)")

        # val check
        if not (isinstance(val, numbers.Real)):
            raise Exception("Input plunge value must be int/float")
        elif not isfinite(val):
            raise Exception("Input plunge value must be finite")
        if unit == 'd' and not (-90.0 <= val <= 90.0):
            raise Exception("Input value in degrees must be between -90° and 90°")
        elif unit == 'r' and not (-pi/2 <= val <= pi/2):
            raise Exception("Input value in radians must be between -pi/2 and pi/2")

        if unit == 'd':
            val = radians(val)

        self.p = val

    @property
    def d(self):
        """
        Returns the angle in decimal degrees.

        :return: angle in decimal degrees

        Example:
          >>> Plunge(10).d
          10.0
          >>> Plunge(-pi/2, unit='r').d
          -90.0
        """

        return degrees(self.p)

    @property
    def r(self):
        """
        Returns the angle in radians.

        :return: angle in radians

        Example:
          >>> Plunge(90).r
          1.5707963267948966
          >>> Plunge(45).r
          0.7853981633974483
        """

        return self.p

    @classmethod
    def fromHZ(cls, h: numbers.Real, z: numbers.Real) -> 'Plunge':
        """
        Calculates plunge from h and z components.

        :param cls: class
        :param h: horizontal component (always positive)
        :param z: vertical component (positive upward)
        :return: Plunge instance

        Examples:
          >>> Plunge.fromHZ(1, 1)
          Plunge(-45.00°)
          >>> Plunge.fromHZ(1, -1)
          Plunge(45.00°)
          >>> Plunge.fromHZ(0, 1)
          Plunge(-90.00°)
          >>> Plunge.fromHZ(0, -1)
          Plunge(90.00°)
          >>> Plunge.fromHZ(-1, 0)
          Traceback (most recent call last):
          ...
          Exception: Horizontal component cannot be negative
          >>> Plunge.fromHZ(0, 0)
          Traceback (most recent call last):
          ...
          Exception: Input h and z values cannot be both zero
        """

        # input vals check

        vals = [h, z]
        if not all(map(lambda val: isinstance(val, numbers.Real), vals)):
            raise Exception("Input h and z values must be integer or float")
        elif not all(map(isfinite, vals)):
            raise Exception("Input h and z values must be finite")

        if h == 0.0 and z == 0.0:
            raise Exception("Input h and z values cannot be both zero")
        elif h < 0.0:
            raise Exception("Horizontal component cannot be negative")

        angle = atan2(-z, h)

        return cls(angle, unit='r')

    def __repr__(self) -> str:

        return "Plunge({:.2f}°)".format(self.d)

    def toHZ(self):

        """
        Converts an azimuth to h-z components.

        :return: a tuple storing h (horizontal) and z values:
        :type: tuple of two floats

        Examples:
          >>> apprFTuple(Plunge(0).toHZ())
          (1.0, 0.0)
          >>> apprFTuple(Plunge(90).toHZ())
          (0.0, -1.0)
          >>> apprFTuple(Plunge(-90).toHZ())
          (0.0, 1.0)
          >>> apprFTuple(Plunge(-45).toHZ(), ndec=6)
          (0.707107, 0.707107)
          >>> apprFTuple(Plunge(45).toHZ(), ndec=6)
          (0.707107, -0.707107)
        """

        return cos(self.p), -sin(self.p)

    @property
    def isUpward(self):
        """
        Check whether the instance is pointing upward or horizontal.

        Examples:
          >>> Plunge(10).isUpward
          False
          >>> Plunge(0.0).isUpward
          False
          >>> Plunge(-45).isUpward
          True
        """

        return self.r < 0.0

    @property
    def isDownward(self):
        """
        Check whether the instance is pointing downward or horizontal.

        Examples:
          >>> Plunge(15).isDownward
          True
          >>> Plunge(0.0).isDownward
          False
          >>> Plunge(-45).isDownward
          False
        """

        return self.r > 0.0


class Direct(object):
    """
    Class describing a direction, expressed as a polar direction.
    """

    def __init__(self, az: 'Azim', pl: 'Plunge'):
        """
        Creates a polar direction instance.

        :param az: the azimuth value
        :param pl: the plunge value
        """

        if not isinstance(az, Azim):
            raise Exception("First input value must be of type Azim")

        if not isinstance(pl, Plunge):
            raise Exception("Second input value must be of type Plunge")

        self._az = az
        self._pl = pl

    @property
    def d(self):
        """
        Returns azimuth and plunge in decimal degrees as a tuple.

        :return: tuple of azimuth and plunge in decimal degrees

        Example:
          >>> Direct.fromAzPl(100, 20).d
          (100.0, 20.0)
          >>> Direct.fromAzPl(-pi/2, -pi/4, unit='r').d
          (270.0, -45.0)
        """

        return self.az.d, self.pl.d

    @property
    def r(self):
        """
        Returns azimuth and plunge in radians as a tuple.

        :return: tuple of azimuth and plunge in radians

        Example:
          >>> Direct.fromAzPl(90, 45).r
          (1.5707963267948966, 0.7853981633974483)
        """

        return self.az.r, self.pl.r

    @property
    def az(self):
        """
        Returns the azimuth instance.

        :return: Azimuth
        """

        return self._az

    @property
    def pl(self):
        """
        Returns the plunge instance.

        :return: Plunge
        """

        return self._pl

    @classmethod
    def fromAzPl(cls, az: numbers.Real, pl: numbers.Real, unit='d'):
        """
        Class constructor from trend and plunge.

        :param az: trend value
        :param pl: plunge value
        :param unit: measurement unit, in degrees ('d') or radians ('r')
        :return: Orientation instance

        Examples:
          >>> Direct.fromAzPl(30, 40)
          Direct(az: 30.00°, pl: 40.00°)
          >>> Direct.fromAzPl(370, 80)
          Direct(az: 10.00°, pl: 80.00°)
          >>> Direct.fromAzPl(pi/2, pi/4, unit='r')
          Direct(az: 90.00°, pl: 45.00°)
          >>> Direct.fromAzPl(280, -100)
          Traceback (most recent call last):
          ...
          Exception: Input value in degrees must be between -90° and 90°
          >>> Direct.fromAzPl("10", 0)
          Traceback (most recent call last):
          ...
          Exception: Input azimuth value must be int/float
          >>> Direct.fromAzPl(100, np.nan)
          Traceback (most recent call last):
          ...
          Exception: Input plunge value must be finite
        """

        azim = Azim(az, unit=unit)
        plng = Plunge(pl, unit=unit)

        return cls(azim, plng)

    @classmethod
    def _from_xyz(cls, x: numbers.Real, y: numbers.Real, z: numbers.Real) -> 'Direct':
        """
        Private class constructor from three Cartesian values. Note: norm of components is unit.

        :param x: x component
        :param y: y component
        :param z: z component
        :return: Orientation instance
        """

        h = sqrt(x*x + y*y)

        az = Azim.fromXY(x, y)
        pl = Plunge.fromHZ(h, z)

        return cls(az, pl)

    @classmethod
    def fromXYZ(cls, x: numbers.Real, y: numbers.Real, z: numbers.Real) -> 'Direct':
        """
        Class constructor from three generic Cartesian values.

        :param x: x component
        :param y: y component
        :param z: z component
        :return: Orientation instance

        Examples:
          >>> Direct.fromXYZ(1, 0, 0)
          Direct(az: 90.00°, pl: -0.00°)
          >>> Direct.fromXYZ(0, 1, 0)
          Direct(az: 0.00°, pl: -0.00°)
          >>> Direct.fromXYZ(0, 0, 1)
          Direct(az: 0.00°, pl: -90.00°)
          >>> Direct.fromXYZ(0, 0, -1)
          Direct(az: 0.00°, pl: 90.00°)
          >>> Direct.fromXYZ(1, 1, 0)
          Direct(az: 45.00°, pl: -0.00°)
          >>> Direct.fromXYZ(0.5, -0.5, -0.7071067811865476)
          Direct(az: 135.00°, pl: 45.00°)
          >>> Direct.fromXYZ(-0.5, 0.5, 0.7071067811865476)
          Direct(az: 315.00°, pl: -45.00°)
          >>> Direct.fromXYZ(0, 0, 0)
          Traceback (most recent call last):
          ...
          Exception: Input components have near-zero values
        """

        mag, norm_xyz = normXYZ(x, y, z)

        if norm_xyz is None:
            raise Exception("Input components have near-zero values")

        return cls._from_xyz(*norm_xyz)

    @classmethod
    def fromVect(cls, vect: Vect) -> [None, 'Direct', 'Axis']:
        """
        Calculate the polar direction parallel to the Vect instance.
        Trend range: [0°, 360°[
        Plunge range: [-90°, 90°], with negative values for upward-pointing
        geological axes and positive values for downward-pointing axes.

        Examples:
          >>> Direct.fromVect(Vect(1, 1, 1))
          Direct(az: 45.00°, pl: -35.26°)
          >>> Direct.fromVect(Vect(0, 1, 1))
          Direct(az: 0.00°, pl: -45.00°)
          >>> Direct.fromVect(Vect(1, 0, 1))
          Direct(az: 90.00°, pl: -45.00°)
          >>> Direct.fromVect(Vect(0, 0, 1))
          Direct(az: 0.00°, pl: -90.00°)
          >>> Direct.fromVect(Vect(0, 0, -1))
          Direct(az: 0.00°, pl: 90.00°)
          >>> Direct.fromVect(Vect(-1, 0, 0))
          Direct(az: 270.00°, pl: -0.00°)
          >>> Direct.fromVect(Vect(0, -1, 0))
          Direct(az: 180.00°, pl: -0.00°)
          >>> Direct.fromVect(Vect(-1, -1, 0))
          Direct(az: 225.00°, pl: -0.00°)
          >>> Direct.fromVect(Vect(0, 0, 0))
          Traceback (most recent call last):
          ...
          Exception: Input components have near-zero values
        """

        x, y, z = vect.toXYZ()
        return cls.fromXYZ(x, y, z)

    def __repr__(self) -> str:

        return "Direct(az: {:.2f}°, pl: {:.2f}°)".format(*self.d)

    def toXYZ(self) -> Tuple[numbers.Real, numbers.Real, numbers.Real]:
        """
        Converts a direction to a tuple of x, y and z cartesian components (with unit norm).

        :return: tuple of x, y and z components.

        Examples:
          >>> az, pl = Azim(90), Plunge(0)
          >>> apprFTuple(Direct(az, pl).toXYZ())
          (1.0, 0.0, 0.0)
          >>> az, pl = Azim(135), Plunge(45)
          >>> apprFTuple(Direct(az, pl).toXYZ(), ndec=6)
          (0.5, -0.5, -0.707107)
          >>> az, pl = Azim(135), Plunge(0)
          >>> apprFTuple(Direct(az, pl).toXYZ(), ndec=6)
          (0.707107, -0.707107, 0.0)
          >>> az, pl = Azim(180), Plunge(45)
          >>> apprFTuple(Direct(az, pl).toXYZ(), ndec=6)
          (0.0, -0.707107, -0.707107)
          >>> az, pl = Azim(225), Plunge(-45)
          >>> apprFTuple(Direct(az, pl).toXYZ(), ndec=6)
          (-0.5, -0.5, 0.707107)
          >>> az, pl = Azim(270), Plunge(90)
          >>> apprFTuple(Direct(az, pl).toXYZ(), ndec=6)
          (0.0, 0.0, -1.0)
        """

        x, y = self.az.toXY()
        h, z = self.pl.toHZ()

        return x*h, y*h, z

    def copy(self):
        """
        Return a copy of the instance.

        Example:
          >>> Direct.fromAzPl(10, 20).copy()
          Direct(az: 10.00°, pl: 20.00°)
        """

        return self.__class__(self.az, self.pl)

    def opposite(self):
        """
        Return the opposite direction.

        Example:
          >>> Direct.fromAzPl(0, 30).opposite()
          Direct(az: 180.00°, pl: -30.00°)
          >>> Direct.fromAzPl(315, 10).opposite()
          Direct(az: 135.00°, pl: -10.00°)
          >>> Direct.fromAzPl(135, 0).opposite()
          Direct(az: 315.00°, pl: -0.00°)
        """

        az, pl = self.r

        az = (az + pi) % (2*pi)
        pl = -pl

        return self.__class__.fromAzPl(az, pl, unit='r')

    def mirrorHoriz(self):
        """
        Return the mirror Orientation using a horizontal plane.

        Example:
          >>> Direct.fromAzPl(0, 30).mirrorHoriz()
          Direct(az: 0.00°, pl: -30.00°)
          >>> Direct.fromAzPl(315, 10).mirrorHoriz()
          Direct(az: 315.00°, pl: -10.00°)
          >>> Direct.fromAzPl(135, 0).mirrorHoriz()
          Direct(az: 135.00°, pl: -0.00°)
        """

        az = self.az.r
        pl = -self.pl.r

        return self.__class__.fromAzPl(az, pl, unit='r')

    @property
    def colatNorth(self) -> numbers.Real:
        """
        Calculates the colatitude from the North (top).

        :return: an angle between 0 and 180 (in degrees).
        :rtype: numbers.Real.

        Examples:
          >>> Direct.fromAzPl(320, 90).colatNorth
          180.0
          >>> Direct.fromAzPl(320, 45).colatNorth
          135.0
          >>> Direct.fromAzPl(320, 0).colatNorth
          90.0
          >>> Direct.fromAzPl(320, -45).colatNorth
          45.0
          >>> Direct.fromAzPl(320, -90).colatNorth
          0.0
        """

        return plng2colatTop(self.pl.d)

    @property
    def colatSouth(self) -> numbers.Real:
        """
        Calculates the colatitude from the South (bottom).

        :return: an angle between 0 and 180 (in degrees).
        :rtype: numbers.Real.

        Examples:
          >>> Direct.fromAzPl(320, 90).colatSouth
          0.0
          >>> Direct.fromAzPl(320, 45).colatSouth
          45.0
          >>> Direct.fromAzPl(320, 0).colatSouth
          90.0
          >>> Direct.fromAzPl(320, -45).colatSouth
          135.0
          >>> Direct.fromAzPl(320, -90).colatSouth
          180.0
        """

        return plng2colatBottom(self.pl.d)

    def asVersor(self):
        """
        Return the unit vector corresponding to the Direct instance.

        Examples:
          >>> Direct.fromAzPl(0, 90).asVersor()
          Vect(0.0000, 0.0000, -1.0000, EPSG: -1)
          >>> Direct.fromAzPl(0, -90).asVersor()
          Vect(0.0000, 0.0000, 1.0000, EPSG: -1)
          >>> Direct.fromAzPl(90, 90).asVersor()
          Vect(0.0000, 0.0000, -1.0000, EPSG: -1)
        """

        az, pl = self.r
        cos_az, cos_pl = cos(az), cos(pl)
        sin_az, sin_pl = sin(az), sin(pl)
        north_coord = cos_pl * cos_az
        east_coord = cos_pl * sin_az
        down_coord = sin_pl

        return Vect(east_coord, north_coord, -down_coord)

    @property
    def isUpward(self):
        """
        Check whether the instance is pointing upward or horizontal.

        Examples:
          >>> Direct.fromAzPl(10, 15).isUpward
          False
          >>> Direct.fromAzPl(257.4, 0.0).isUpward
          False
          >>> Direct.fromAzPl(90, -45).isUpward
          True
        """

        return self.pl.isUpward

    @property
    def isDownward(self):
        """
        Check whether the instance is pointing downward or horizontal.

        Examples:
          >>> Direct.fromAzPl(10, 15).isDownward
          True
          >>> Direct.fromAzPl(257.4, 0.0).isDownward
          False
          >>> Direct.fromAzPl(90, -45).isDownward
          False
        """

        return self.pl.isDownward

    def upward(self):
        """
        Return upward-point geological vector.

        Examples:
          >>> Direct.fromAzPl(90, -45).upward().isSubParallel(Direct.fromAzPl(90.0, -45.0))
          True
          >>> Direct.fromAzPl(180, 45).upward().isSubParallel(Direct.fromAzPl(0.0, -45.0))
          True
          >>> Direct.fromAzPl(0, 0).upward().isSubParallel(Direct.fromAzPl(0.0, 0.0))
          True
          >>> Direct.fromAzPl(0, 90).upward().isSubParallel(Direct.fromAzPl(180.0, -90.0))
          True
          >>> Direct.fromAzPl(90, -45).upward().isSubParallel(Direct.fromAzPl(90.0, -35.0))
          False
          >>> Direct.fromAzPl(180, 45).upward().isSubParallel(Direct.fromAzPl(10.0, -45.0))
          False
          >>> Direct.fromAzPl(0, 0).upward().isSubParallel(Direct.fromAzPl(170.0, 0.0))
          False
          >>> Direct.fromAzPl(0, 90).upward().isSubParallel(Direct.fromAzPl(180.0, -80.0))
          False
        """

        if not self.isDownward:
            return self.copy()
        else:
            return self.opposite()

    def downward(self):
        """
        Return downward-pointing geological vector.

        Examples:
          >>> Direct.fromAzPl(90, -45).downward().isSubParallel(Direct.fromAzPl(270.0, 45.0))
          True
          >>> Direct.fromAzPl(180, 45).downward().isSubParallel(Direct.fromAzPl(180.0, 45.0))
          True
          >>> Direct.fromAzPl(0, 0).downward().isSubParallel(Direct.fromAzPl(180.0, 0.0))
          False
          >>> Direct.fromAzPl(0, 90).downward().isSubParallel(Direct.fromAzPl(0.0, 90.0))
          True
          >>> Direct.fromAzPl(90, -45).downward().isSubParallel(Direct.fromAzPl(270.0, 35.0))
          False
          >>> Direct.fromAzPl(180, 45).downward().isSubParallel(Direct.fromAzPl(170.0, 45.0))
          False
          >>> Direct.fromAzPl(0, 0).downward().isSubParallel(Direct.fromAzPl(180.0, 10.0))
          False
          >>> Direct.fromAzPl(0, 90).downward().isSubParallel(Direct.fromAzPl(0.0, 80.0))
          False
        """

        if not self.isUpward:
            return self.copy()
        else:
            return self.opposite()

    def isAbsDipWithin(self, min_val, max_val, min_val_incl=False, max_value_incl=True):
        """
        Check whether the absolute value of the dip angle of an Direct instance is intersect a given range
        (default: minimum value is not included, maximum value is included).

        :param min_val: the minimum dip angle, positive, domain: 0-90°.
        :param max_val: the maximum dip angle, positive, domain: 0-90°.
        :param min_val_incl: is minimum value included, boolean.
        :param max_value_incl: is maximum value included, boolean.
        :return: Boolean

        Examples:
          >>> Direct.fromAzPl(90, -45).isAbsDipWithin(30, 60)
          True
          >>> Direct.fromAzPl(120, 0).isAbsDipWithin(0, 60)
          False
          >>> Direct.fromAzPl(120, 0).isAbsDipWithin(0, 60, min_val_incl=True)
          True
          >>> Direct.fromAzPl(120, 60).isAbsDipWithin(0, 60)
          True
        """

        abs_dip = abs(self.pl.d)

        if abs_dip < min_val or abs_dip > max_val:
            return False
        elif abs_dip == min_val:
            if min_val_incl:
                return True
            else:
                return False
        elif abs_dip == max_val:
            if max_value_incl:
                return True
            else:
                return False
        else:
            return True

    def isSubHoriz(self, max_dip_angle=DIP_ANGLE_THRESHOLD):
        """
        Check whether the instance is almost horizontal.

        Examples:
          >>> Direct.fromAzPl(10, 15).isSubHoriz()
          False
          >>> Direct.fromAzPl(257, 2).isSubHoriz()
          True
          >>> Direct.fromAzPl(90, -5).isSubHoriz()
          False
        """

        return abs(self.pl.d) < max_dip_angle

    def isSubVert(self, min_dip_angle=90.0 - DIP_ANGLE_THRESHOLD):
        """
        Check whether the instance is almost vertical.

        Examples:
          >>> Direct.fromAzPl(10, 15).isSubVert()
          False
          >>> Direct.fromAzPl(257, 89).isSubVert()
          True
        """

        return abs(self.pl.d) > min_dip_angle

    def angle(self, another):
        """
        Calculate angle (in degrees) between the two Direct instances.
        Range is 0°-180°.

        Examples:
          >>> areClose(Direct.fromAzPl(0, 90).angle(Direct.fromAzPl(90, 0)), 90)
          True
          >>> areClose(Direct.fromAzPl(0, 0).angle(Direct.fromAzPl(270, 0)), 90)
          True
          >>> areClose(Direct.fromAzPl(0, 0).angle(Direct.fromAzPl(0, 0)), 0)
          True
          >>> areClose(Direct.fromAzPl(0, 0).angle(Direct.fromAzPl(180, 0)), 180)
          True
          >>> areClose(Direct.fromAzPl(90, 0).angle(Direct.fromAzPl(270, 0)), 180)
          True
        """

        angle_vers = self.asVersor().angle(another.asVersor())

        return angle_vers

    def isSubParallel(self, another, angle_tolerance=VECTOR_ANGLE_THRESHOLD):
        """
        Check that two Direct instances are sub-parallel,

        :param another: an Direct instance
        :param angle_tolerance: the maximum allowed divergence angle (in degrees)
        :return: Boolean

        Examples:
          >>> Direct.fromAzPl(0, 90).isSubParallel(Direct.fromAzPl(90, 0))
          False
          >>> Direct.fromAzPl(0, 0).isSubParallel(Direct.fromAzPl(0, 1e-6))
          True
          >>> Direct.fromAzPl(0, 90).isSubParallel(Direct.fromAzPl(180, 0))
          False
          >>> Direct.fromAzPl(0, 90).isSubParallel(Direct.fromAzPl(0, -90))
          False
        """

        fst_gvect = self

        snd_geoelem = another

        angle = fst_gvect.angle(snd_geoelem)

        if isinstance(another, Plane):
            return angle > (90.0 - angle_tolerance)
        else:
            return angle <= angle_tolerance

    def isSubAParallel(self, another, angle_tolerance=VECTOR_ANGLE_THRESHOLD):
        """
        Check that two Vect instances are almost anti-parallel,

        :param another: a Vect instance
        :param angle_tolerance: the maximum allowed divergence angle (in degrees)
        :return: Boolean

        Examples:
          >>> Direct.fromAzPl(0, 90).isSubAParallel(Direct.fromAzPl(90, -89.5))
          True
          >>> Direct.fromAzPl(0, 0).isSubAParallel(Direct.fromAzPl(180, 1e-6))
          True
          >>> Direct.fromAzPl(90, 45).isSubAParallel(Direct.fromAzPl(270, -45.5))
          True
          >>> Direct.fromAzPl(45, 90).isSubAParallel(Direct.fromAzPl(0, -90))
          True
          >>> Direct.fromAzPl(45, 72).isSubAParallel(Direct.fromAzPl(140, -38))
          False
        """

        return self.angle(another) > (180.0 - angle_tolerance)

    def isSubOrthog(self, another, angle_tolerance=VECTOR_ANGLE_THRESHOLD):
        """
        Check that two Direct instance are sub-orthogonal

        :param another: a Direct instance
        :param angle_tolerance: the maximum allowed divergence angle (in degrees) from orthogonality
        :return: Boolean

         Examples:
          >>> Direct.fromAzPl(0, 90).isSubOrthog(Direct.fromAzPl(90, 0))
          True
          >>> Direct.fromAzPl(0, 0).isSubOrthog(Direct.fromAzPl(0, 1.e-6))
          False
          >>> Direct.fromAzPl(0, 0).isSubOrthog(Direct.fromAzPl(180, 0))
          False
          >>> Direct.fromAzPl(90, 0).isSubOrthog(Direct.fromAzPl(270, 89.5))
          True
          >>> Direct.fromAzPl(0, 90).isSubOrthog(Direct.fromAzPl(0, 0.5))
          True
        """

        return 90.0 - angle_tolerance <= self.angle(another) <= 90.0 + angle_tolerance

    def normVersor(self, another):
        """
        Calculate the versor (Vect) defined by the vector product of two Direct instances.

        Examples:
          >>> Direct.fromAzPl(0, 0).normVersor(Direct.fromAzPl(90, 0))
          Vect(0.0000, 0.0000, -1.0000, EPSG: -1)
          >>> Direct.fromAzPl(45, 0).normVersor(Direct.fromAzPl(310, 0))
          Vect(0.0000, 0.0000, 1.0000, EPSG: -1)
          >>> Direct.fromAzPl(0, 0).normVersor(Direct.fromAzPl(90, 90))
          Vect(-1.0000, 0.0000, -0.0000, EPSG: -1)
          >>> Direct.fromAzPl(315, 45).normVersor(Direct.fromAzPl(315, 44.5)) is None
          True
        """

        if self.isSubParallel(another):
            return None
        else:
            return self.asVersor().vCross(another.asVersor()).versor()

    def normPlane(self) -> 'Plane':
        """
        Return the geological plane that is normal to the direction.

        Examples:
          >>> Direct.fromAzPl(0, 45).normPlane()
          Plane(180.00, +45.00)
          >>> Direct.fromAzPl(0, -45).normPlane()
          Plane(000.00, +45.00)
          >>> Direct.fromAzPl(0, 90).normPlane()
          Plane(180.00, +00.00)
        """

        down_orien = self.downward()
        dipdir = (down_orien.az.d + 180.0) % 360.0
        dipangle = 90.0 - down_orien.pl.d

        return Plane(dipdir, dipangle)

    def commonPlane(self, another):
        """
        Calculate Plane instance defined by the two Vect instances.

        Examples:
          >>> Direct.fromAzPl(0, 0).commonPlane(Direct.fromAzPl(90, 0)).isSubParallel(Plane(180.0, 0.0))
          True
          >>> Direct.fromAzPl(0, 0).commonPlane(Direct.fromAzPl(90, 90)).isSubParallel(Plane(90.0, 90.0))
          True
          >>> Direct.fromAzPl(45, 0).commonPlane(Direct.fromAzPl(135, 45)).isSubParallel(Plane(135.0, 45.0))
          True
          >>> Direct.fromAzPl(315, 45).commonPlane(Direct.fromAzPl(135, 45)).isSubParallel(Plane(225.0, 90.0))
          True
          >>> Direct.fromAzPl(0, 0).commonPlane(Direct.fromAzPl(90, 0)).isSubParallel(Plane(180.0, 10.0))
          False
          >>> Direct.fromAzPl(0, 0).commonPlane(Direct.fromAzPl(90, 90)).isSubParallel(Plane(90.0, 80.0))
          False
          >>> Direct.fromAzPl(45, 0).commonPlane(Direct.fromAzPl(135, 45)).isSubParallel(Plane(125.0, 45.0))
          False
          >>> Direct.fromAzPl(315, 45).commonPlane(Direct.fromAzPl(135, 45)).isSubParallel(Plane(225.0, 80.0))
          False
          >>> Direct.fromAzPl(315, 45).commonPlane(Direct.fromAzPl(315, 44.5)) is None
          True
        """

        normal_versor = self.normVersor(another)
        if normal_versor is None:
            return None
        else:
            return Direct.fromVect(normal_versor).normPlane()

    def asAxis(self):
        """
        Create Axis instance with the same attitude as the self instance.

        Example:
          >>> Direct.fromAzPl(220, 32).asAxis()
          Axis(az: 220.00°, pl: 32.00°)
        """

        return Axis(self.az, self.pl)

    def normDirect(self, another):
        """
        Calculate the instance that is normal to the two provided sources.
        Angle between sources must be larger than MIN_ANGLE_DEGR_DISORIENTATION,
        otherwise a SubparallelLineationException will be raised.

        Example:
          >>> Direct.fromAzPl(0, 0).normDirect(Direct.fromAzPl(0.5, 0)) is None
          True
          >>> Direct.fromAzPl(0, 0).normDirect(Direct.fromAzPl(179.5, 0)) is None
          True
          >>> Direct.fromAzPl(0, 0).normDirect(Direct.fromAzPl(5.1, 0))
          Direct(az: 0.00°, pl: 90.00°)
          >>> Direct.fromAzPl(90, 45).normDirect(Direct.fromAzPl(90, 0))
          Direct(az: 180.00°, pl: -0.00°)
        """

        if self.isSubAParallel(another):
            return None
        elif self.isSubParallel(another):
            return None
        else:
            return self.__class__.fromVect(self.normVersor(another))


class Axis(Direct):
    """
    Polar Axis. Inherits from Orientation
    """

    def __init__(self, az: Azim, pl: Plunge):

        super().__init__(az, pl)

    def __repr__(self):

        return "Axis(az: {:.2f}°, pl: {:.2f}°)".format(*self.d)

    def asDirect(self):
        """
        Create Direct instance with the same attitude as the self instance.

        Example:
          >>> Axis.fromAzPl(220, 32).asDirect()
          Direct(az: 220.00°, pl: 32.00°)
        """

        return Direct(self.az, self.pl)

    def normAxis(self, another):
        """
        Calculate the Axis instance that is perpendicular to the two provided.
        The two source Axis must not be subparallel (threshold is MIN_ANGLE_DEGR_DISORIENTATION),
        otherwise a SubparallelLineationException will be raised.

        Example:
          >>> Axis.fromAzPl(0, 0).normAxis(Axis.fromAzPl(0.5, 0)) is None
          True
          >>> Axis.fromAzPl(0, 0).normAxis(Axis.fromAzPl(180, 0)) is None
          True
          >>> Axis.fromAzPl(90, 0).normAxis(Axis.fromAzPl(180, 0))
          Axis(az: 0.00°, pl: 90.00°)
          >>> Axis.fromAzPl(90, 45).normAxis(Axis.fromAzPl(180, 0))
          Axis(az: 270.00°, pl: 45.00°)
          >>> Axis.fromAzPl(270, 45).normAxis(Axis.fromAzPl(180, 90)).isSubParallel(Axis.fromAzPl(180, 0))
          True
        """

        norm_orien = self.normDirect(another)
        if norm_orien is None:
            return None
        else:
            return norm_orien.asAxis()

    def angle(self, another):
        """
        Calculate angle (in degrees) between the two Axis instances.
        Range is 0°-90°.

        Examples:
          >>> areClose(Axis.fromAzPl(0, 90).angle(Axis.fromAzPl(90, 0)), 90)
          True
          >>> areClose(Axis.fromAzPl(0, 0).angle(Axis.fromAzPl(270, 0)), 90)
          True
          >>> areClose(Axis.fromAzPl(0, 0).angle(Axis.fromAzPl(0, 0)), 0)
          True
          >>> areClose(Axis.fromAzPl(0, 0).angle(Axis.fromAzPl(180, 0)), 0)
          True
          >>> areClose(Axis.fromAzPl(0, 0).angle(Axis.fromAzPl(179, 0)), 1)
          True
          >>> areClose(Axis.fromAzPl(0, -90).angle(Axis.fromAzPl(0, 90)), 0)
          True
          >>> areClose(Axis.fromAzPl(90, 0).angle(Axis.fromAzPl(315, 0)), 45)
          True
        """

        angle_vers = self.asVersor().angle(another.asVersor())

        return min(angle_vers, 180.0 - angle_vers)


class RotationAxis(object):
    """
    Rotation axis, expressed by an Orientation and a rotation angle.
    """

    def __init__(self, trend: numbers.Real, plunge: numbers.Real, rot_ang: numbers.Real):
        """
        Constructor.

        :param trend: Float/Integer
        :param plunge: Float/Integer
        :param rot_ang: Float/Integer

        Example:
        >> RotationAxis(0, 90, 120)
        RotationAxis(0.0000, 90.0000, 120.0000)
        """

        self.dr = Direct.fromAzPl(trend, plunge)
        self.a = float(rot_ang)

    @classmethod
    def fromQuater(cls, quat: Quaternion):
        """
        Calculates the Rotation Axis expressed by a quaternion.
        The resulting rotation asVect is set to point downward.
        Examples are taken from Kuipers, 2002, chp. 5.

        :return: RotationAxis instance.

        Examples:
          >>> RotationAxis.fromQuater(Quaternion(0.5, 0.5, 0.5, 0.5))
          RotationAxis(45.0000, -35.2644, 120.0000)
          >>> RotationAxis.fromQuater(Quaternion(sqrt(2)/2, 0.0, 0.0, sqrt(2)/2))
          RotationAxis(0.0000, -90.0000, 90.0000)
          >>> RotationAxis.fromQuater(Quaternion(sqrt(2)/2, sqrt(2)/2, 0.0, 0.0))
          RotationAxis(90.0000, -0.0000, 90.0000)
        """

        if abs(quat) < QUAT_MAGN_THRESH:

            rot_ang = 0.0
            rot_direct = Direct.fromAzPl(0.0, 0.0)

        elif areClose(quat.scalar, 1):

            rot_ang = 0.0
            rot_direct = Direct.fromAzPl(0.0, 0.0)

        else:

            unit_quat = quat.normalize()
            rot_ang = unit_quat.rotAngle()
            rot_direct = Direct.fromVect(unit_quat.vector())

        return RotationAxis(*rot_direct.d, rot_ang)

    @classmethod
    def fromDirect(cls, direct: Direct, angle: numbers.Real):
        """
        Class constructor from a Direct instance and an angle value.

        :param direct: a Direct instance
        :param angle: numbers.Real.
        :return: RotationAxis instance

        Example:
          >>> RotationAxis.fromDirect(Direct.fromAzPl(320, 12), 30)
          RotationAxis(320.0000, 12.0000, 30.0000)
          >>> RotationAxis.fromDirect(Direct.fromAzPl(315.0, -0.0), 10)
          RotationAxis(315.0000, -0.0000, 10.0000)
        """

        return RotationAxis(*direct.d, angle)

    @classmethod
    def fromVect(cls, vector: Vect, angle: numbers.Real):
        """
        Class constructor from a Vect instance and an angle value.

        :param vector: a Vect instance
        :param angle: float value
        :return: RotationAxis instance

        Example:
          >>> RotationAxis.fromVect(Vect(0, 1, 0), 30)
          RotationAxis(0.0000, -0.0000, 30.0000)
          >>> RotationAxis.fromVect(Vect(1, 0, 0), 30)
          RotationAxis(90.0000, -0.0000, 30.0000)
          >>> RotationAxis.fromVect(Vect(0, 0, -1), 30)
          RotationAxis(0.0000, 90.0000, 30.0000)
        """

        direct = Direct.fromVect(vector)

        return RotationAxis.fromDirect(direct, angle)

    def __repr__(self):

        return "RotationAxis({:.4f}, {:.4f}, {:.4f})".format(*self.dr.d, self.a)

    @property
    def rotAngle(self) -> float:
        """
        Returns the rotation angle of the rotation axis.

        :return: rotation angle (Float)

        Example:
          >>> RotationAxis(10, 15, 230).rotAngle
          230.0
        """

        return self.a

    @property
    def rotDirect(self) -> Direct:
        """
        Returns the rotation axis, expressed as a Direct.

        :return: Direct instance

        Example:
          >>> RotationAxis(320, 40, 15).rotDirect
          Direct(az: 320.00°, pl: 40.00°)
          >>> RotationAxis(135, 0, -10).rotDirect
          Direct(az: 135.00°, pl: 0.00°)
          >>> RotationAxis(45, 10, 10).rotDirect
          Direct(az: 45.00°, pl: 10.00°)
        """

        return self.dr

    @property
    def versor(self) -> Vect:
        """
        Return the versor equivalent to the Rotation geological asVect.

        :return: Vect
        """

        return self.dr.asVersor()

    def specular(self):
        """
        Derives the rotation axis with opposite asVect direction
        and rotation angle that is the complement to 360°.
        The resultant rotation is equivalent to the original one.

        :return: RotationAxis instance.

        Example
          >>> RotationAxis(90, 45, 320).specular()
          RotationAxis(270.0000, -45.0000, 40.0000)
          >>> RotationAxis(135, 0, -10).specular()
          RotationAxis(315.0000, -0.0000, 10.0000)
          >>> RotationAxis(45, 10, 10).specular()
          RotationAxis(225.0000, -10.0000, 350.0000)
        """

        gvect_opp = self.rotDirect.opposite()
        opposite_angle = (360.0 - self.rotAngle) % 360.0

        return RotationAxis.fromDirect(gvect_opp, opposite_angle)

    def compl180(self):
        """
        Creates a new rotation axis that is the complement to 180 of the original one.

        :return: RotationAxis instance.

        Example:
          >>> RotationAxis(90, 45, 120).compl180()
          RotationAxis(90.0000, 45.0000, 300.0000)
          >>> RotationAxis(117, 34, 18).compl180()
          RotationAxis(117.0000, 34.0000, 198.0000)
          >>> RotationAxis(117, 34, -18).compl180()
          RotationAxis(117.0000, 34.0000, 162.0000)
        """

        rot_ang = - (180.0 - self.rotAngle) % 360.0
        return RotationAxis.fromDirect(self.dr, rot_ang)

    def strictlyEquival(self, another, angle_tolerance: numbers.Real=VECTOR_ANGLE_THRESHOLD) -> bool:
        """
        Checks if two RotationAxis are almost equal, based on a strict checking
        of the Direct component and of the rotation angle.

        :param another: another RotationAxis instance, to be compared with
        :type another: RotationAxis
        :parameter angle_tolerance: the tolerance as the angle (in degrees)
        :type angle_tolerance: numbers.Real.
        :return: the equivalence (true/false) between the two compared RotationAxis
        :rtype: bool

        Examples:
          >>> ra_1 = RotationAxis(180, 10, 10)
          >>> ra_2 = RotationAxis(180, 10, 10.5)
          >>> ra_1.strictlyEquival(ra_2)
          True
          >>> ra_3 = RotationAxis(180.2, 10, 10.4)
          >>> ra_1.strictlyEquival(ra_3)
          True
          >>> ra_4 = RotationAxis(184.9, 10, 10.4)
          >>> ra_1.strictlyEquival(ra_4)
          False
        """

        if not self.dr.isSubParallel(another.dr, angle_tolerance):
            return False

        if not areClose(self.a, another.a, atol=1.0):
            return False

        return True

    def toRotQuater(self) -> Quaternion:
        """
        Converts the rotation axis to the equivalent rotation quaternion.

        :return: the rotation quaternion.
        :rtype: Quaternion
        """

        rotation_angle_rad = radians(self.a)
        rotation_vector = self.dr.asVersor()

        w = cos(rotation_angle_rad / 2.0)
        x, y, z = rotation_vector.scale(sin(rotation_angle_rad / 2.0)).toXYZ()

        return Quaternion(w, x, y, z).normalize()

    def toRotMatrix(self):
        """
        Derives the rotation matrix from the RotationAxis instance.

        :return: 3x3 numpy array
        """

        rotation_versor = self.versor
        phi = radians(self.a)

        l = rotation_versor.x
        m = rotation_versor.y
        n = rotation_versor.z

        cos_phi = cos(phi)
        sin_phi = sin(phi)

        a11 = cos_phi + ((l * l) * (1 - cos_phi))
        a12 = ((l * m) * (1 - cos_phi)) - (n * sin_phi)
        a13 = ((l * n) * (1 - cos_phi)) + (m * sin_phi)

        a21 = ((l * m) * (1 - cos_phi)) + (n * sin_phi)
        a22 = cos_phi + ((m * m) * (1 - cos_phi))
        a23 = ((m * n) * (1 - cos_phi)) - (l * sin_phi)

        a31 = ((l * n) * (1 - cos_phi)) - (m * sin_phi)
        a32 = ((m * n) * (1 - cos_phi)) + (l * sin_phi)
        a33 = cos_phi + ((n * n) * (1 - cos_phi))

        return np.array([(a11, a12, a13),
                         (a21, a22, a23),
                         (a31, a32, a33)])

    def toMinRotAxis(self):
        """
        Calculates the minimum rotation axis from the given quaternion.

        :return: RotationAxis instance.
        """

        return self if abs(self.rotAngle) <= 180.0 else self.specular()

    @classmethod
    def randomNaive(cls):
        """
        Naive method for creating a random RotationAxis instance.
        :return: random rotation axis (not uniformly distributed in the space)
        :rtype: RotationAxis
        """

        random_trend = random.uniform(0, 360)
        random_dip = random.uniform(-90, 90)
        random_rotation = random.uniform(0, 360)

        return cls(
            trend=random_trend,
            plunge=random_dip,
            rot_ang=random_rotation
        )


def sortRotations(rotation_axes: List[RotationAxis]) -> List[RotationAxis]:
    """
    Sorts a list or rotation axes, based on the rotation angle (absolute value),
    in an increasing order.

    :param rotation_axes: o list of RotationAxis objects.
    :return: the sorted list of RotationAxis

    Example:
      >>> rots = [RotationAxis(110, 14, -23), RotationAxis(42, 13, 17), RotationAxis(149, 87, 13)]
      >>> sortRotations(rots)
      [RotationAxis(149.0000, 87.0000, 13.0000), RotationAxis(42.0000, 13.0000, 17.0000), RotationAxis(110.0000, 14.0000, -23.0000)]
    """

    return sorted(rotation_axes, key=lambda rot_ax: abs(rot_ax.rotAngle))


def rotVectByAxis(
    v: Vect,
    rot_axis: RotationAxis
) -> Vect:
    """
    Rotates a vector.

    Implementation as in:
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    Faster formula:
    t = 2 q x v
    v' = v + q0 t + q x t
    cited as:
    Janota, A; Šimák, V; Nemec, D; Hrbček, J (2015).
    "Improving the Precision and Speed of Euler Angles Computation from Low-Cost Rotation Sensor Data".
    Sensors. 15 (3): 7016–7039. doi:10.3390/s150307016. PMC 4435132. PMID 25806874.

    :param v: the vector to rotate
    :type v: Vect
    :param rot_axis: the rotation axis
    :type rot_axis: RotationAxis
    :return: the rotated vector
    :rtype: Vect

    Examples:
      >>> v = Vect(1,0,1)
      >>> rotation = RotationAxis(0, -90, 90)
      >>> rotVectByAxis(v, rotation)
      Vect(0.0000, 1.0000, 1.0000, EPSG: -1)
      >>> rotation = RotationAxis(0, 90, 90)
      >>> rotVectByAxis(v, rotation)
      Vect(0.0000, -1.0000, 1.0000, EPSG: -1)
      >>> rotation = RotationAxis(0, -90, 180)
      >>> rotVectByAxis(v, rotation)
      Vect(-1.0000, 0.0000, 1.0000, EPSG: -1)
      >>> rotation = RotationAxis(0, -90, 270)
      >>> rotVectByAxis(v, rotation)
      Vect(-0.0000, -1.0000, 1.0000, EPSG: -1)
      >>> rotation = RotationAxis(90, 0, 90)
      >>> rotVectByAxis(v, rotation)
      Vect(1.0000, -1.0000, 0.0000, EPSG: -1)
      >>> rotation = RotationAxis(90, 0, 180)
      >>> rotVectByAxis(v, rotation)
      Vect(1.0000, 0.0000, -1.0000, EPSG: -1)
      >>> rotation = RotationAxis(90, 0, 270)
      >>> rotVectByAxis(v, rotation)
      Vect(1.0000, 1.0000, -0.0000, EPSG: -1)
      >>> rotation = RotationAxis(90, 0, 360)
      >>> rotVectByAxis(v, rotation)
      Vect(1.0000, 0.0000, 1.0000, EPSG: -1)
      >>> rotation = RotationAxis(0, -90, 90)
      >>> v = Vect(0,0,3)
      >>> rotVectByAxis(v, rotation)
      Vect(0.0000, 0.0000, 3.0000, EPSG: -1)
      >>> rotation = RotationAxis(90, -45, 180)
      >>> rotVectByAxis(v, rotation)
      Vect(3.0000, -0.0000, -0.0000, EPSG: -1)
      >>> v = Vect(0,0,3, epsg_cd=32633)
      >>> rotVectByAxis(v, rotation)
      Vect(3.0000, -0.0000, -0.0000, EPSG: 32633)
    """

    rot_quat = rot_axis.toRotQuater()
    q = rot_quat.vector(epsg_cd=v.epsg())

    t = q.scale(2).vCross(v)
    rot_v = v + t.scale(rot_quat.scalar) + q.vCross(t)

    return rot_v


def rotVectByQuater(quat: Quaternion, vect: Vect) -> Vect:
    """
    Calculates a rotated solution of a Vect instance given a normalized quaternion.
    Original formula in Ref. [1].
    Eq.6: R(qv) = q qv q(-1)

    :param quat: a Quaternion instance
    :param vect: a Vect instance
    :return: a rotated Vect instance

    Example:
      >>> q = Quaternion.i()  # rotation of 180° around the x axis
      >>> rotVectByQuater(q, Vect(0, 1, 0))
      Vect(0.0000, -1.0000, 0.0000, EPSG: -1)
      >>> rotVectByQuater(q, Vect(0, 1, 1))
      Vect(0.0000, -1.0000, -1.0000, EPSG: -1)
      >>> q = Quaternion.k()  # rotation of 180° around the z axis
      >>> rotVectByQuater(q, Vect(0, 1, 1))
      Vect(0.0000, -1.0000, 1.0000, EPSG: -1)
      >>> q = Quaternion.j()  # rotation of 180° around the y axis
      >>> rotVectByQuater(q, Vect(1, 0, 1))
      Vect(-1.0000, 0.0000, -1.0000, EPSG: -1)
    """

    q = quat.normalize()
    qv = Quaternion.fromVect(vect)

    rotated_v = q * (qv * q.inverse)

    return rotated_v.vector()


def point_or_segment(
        point1: Point,
        point2: Point,
        tol: numbers.Real = PRACTICAL_MIN_DIST
) -> Union[Point, Segment]:
    """
    Creates a point or segment based on the points distance.

    :param point1: first input point.
    :type point1: Point.
    :param point2: second input point.
    :type point2: Point.
    :param tol: distance tolerance between the two points.
    :type tol: numbers.Real.
    :return: point or segment based on their distance.
    :rtype: PointOrSegment.
    :raise: Exception.
    """

    check_type(point1, "First point", Point)
    check_type(point2, "Second point", Point)

    check_crs(point1, point2)

    if point1.dist3DWith(point2) <= tol:
        return Points([point1, point2]).nanmean_point()
    else:
        return Segment(
            start_pt=point1,
            end_pt=point2
        )


def intersect_segments(
    segment1: Segment,
    segment2: Segment,
    tol: numbers.Real = PRACTICAL_MIN_DIST
) -> Optional[Union[Point, Segment]]:
    """
    Determines the optional point or segment intersection between the segment pair.

    :param segment1: the first segment
    :type segment1: Segment
    :param segment2: the second segment
    :type segment2: Segment
    :param tol: the distance tolerance for collapsing a intersection segment into a point
    :type tol: numbers.Real
    :return: the optional point or segment intersection between the segment pair.
    :rtype: Optional[Union[Point, Segment]]

    Examples:
      >>> s2 = Segment(Point(0,0,0), Point(1,0,0))
      >>> s1 = Segment(Point(0,0,0), Point(1,0,0))
      >>> intersect_segments(s1, s2)
      Segment(start_pt=Point(0.0000, 0.0000, 0.0000, 0.0000, -1), end_pt=Point(1.0000, 0.0000, 0.0000, 0.0000, -1))
      >>> s1 = Segment(Point(-2,0,0), Point(-1,0,0))
      >>> intersect_segments(s1, s2) is None
      True
      >>> s1 = Segment(Point(-2,0,0), Point(0,0,0))
      >>> intersect_segments(s1, s2)
      Point(0.0000, 0.0000, 0.0000, 0.0000, -1)
      >>> s1 = Segment(Point(-2,0,0), Point(0.5,0,0))
      >>> intersect_segments(s1, s2)
      Segment(start_pt=Point(0.0000, 0.0000, 0.0000, 0.0000, -1), end_pt=Point(0.5000, 0.0000, 0.0000, 0.0000, -1))
      >>> s1 = Segment(Point(-2,0,0), Point(1,0,0))
      >>> intersect_segments(s1, s2)
      Segment(start_pt=Point(0.0000, 0.0000, 0.0000, 0.0000, -1), end_pt=Point(1.0000, 0.0000, 0.0000, 0.0000, -1))
      >>> s1 = Segment(Point(-2,0,0), Point(2,0,0))
      >>> intersect_segments(s1, s2)
      Segment(start_pt=Point(0.0000, 0.0000, 0.0000, 0.0000, -1), end_pt=Point(1.0000, 0.0000, 0.0000, 0.0000, -1))
      >>> s1 = Segment(Point(0,0,0), Point(0.5,0,0))
      >>> intersect_segments(s1, s2)
      Segment(start_pt=Point(0.0000, 0.0000, 0.0000, 0.0000, -1), end_pt=Point(0.5000, 0.0000, 0.0000, 0.0000, -1))
      >>> s1 = Segment(Point(0.25,0,0), Point(0.75,0,0))
      >>> intersect_segments(s1, s2)
      Segment(start_pt=Point(0.2500, 0.0000, 0.0000, 0.0000, -1), end_pt=Point(0.7500, 0.0000, 0.0000, 0.0000, -1))
      >>> s1 = Segment(Point(0.25,0,0), Point(1,0,0))
      >>> intersect_segments(s1, s2)
      Segment(start_pt=Point(0.2500, 0.0000, 0.0000, 0.0000, -1), end_pt=Point(1.0000, 0.0000, 0.0000, 0.0000, -1))
      >>> s1 = Segment(Point(0.25,0,0), Point(1.25,0,0))
      >>> intersect_segments(s1, s2)
      Segment(start_pt=Point(0.2500, 0.0000, 0.0000, 0.0000, -1), end_pt=Point(1.0000, 0.0000, 0.0000, 0.0000, -1))
      >>> s1 = Segment(Point(0,0,0), Point(1.25,0,0))
      >>> intersect_segments(s1, s2)
      Segment(start_pt=Point(0.0000, 0.0000, 0.0000, 0.0000, -1), end_pt=Point(1.0000, 0.0000, 0.0000, 0.0000, -1))
      >>> s1 = Segment(Point(1,0,0), Point(1.25,0,0))
      >>> intersect_segments(s1, s2)
      Point(1.0000, 0.0000, 0.0000, 0.0000, -1)
      >>> s2 = Segment(Point(0,0,0), Point(1,1,1))
      >>> s1 = Segment(Point(0.25,0.25,0.25), Point(0.75,0.75,0.75))
      >>> intersect_segments(s1, s2)
      Segment(start_pt=Point(0.2500, 0.2500, 0.2500, 0.0000, -1), end_pt=Point(0.7500, 0.7500, 0.7500, 0.0000, -1))
      >>> s1 = Segment(Point(0.25,0.25,0.25), Point(1.75,1.75,1.75))
      >>> intersect_segments(s1, s2)
      Segment(start_pt=Point(0.2500, 0.2500, 0.2500, 0.0000, -1), end_pt=Point(1.0000, 1.0000, 1.0000, 0.0000, -1))
      >>> s1 = Segment(Point(0.25,0.25,0.25), Point(1.75,0,1.75))
      >>> intersect_segments(s1, s2)
      Point(0.2500, 0.2500, 0.2500, 0.0000, -1)
      >>> s1 = Segment(Point(0.25,1,0.25), Point(0.75,0.75,0.75))
      >>> intersect_segments(s1, s2)
      Point(0.7500, 0.7500, 0.7500, 0.0000, -1)
      >>> s2 = Segment(Point(-1,-1,-1), Point(1,1,1))
      >>> s1 = Segment(Point(-1,1,1), Point(1,-1,-1))
      >>> intersect_segments(s1, s2)
      Point(-0.0000, 0.0000, 0.0000, 0.0000, -1)
    """

    check_type(segment1, "First segment", Segment)
    check_type(segment2, "Second segment", Segment)

    check_crs(segment1, segment2)

    s1_startpt_inside = segment1.segment_start_in(segment2)
    s2_startpt_inside = segment2.segment_start_in(segment1)

    s1_endpt_inside = segment1.segment_end_in(segment2)
    s2_endpt_inside = segment2.segment_end_in(segment1)

    elements = [s1_startpt_inside, s2_startpt_inside, s1_endpt_inside, s2_endpt_inside]

    if all(elements):
        return segment1.clone()

    if s1_startpt_inside and s1_endpt_inside:
        return segment1.clone()

    if s2_startpt_inside and s2_endpt_inside:
        return segment2.clone()

    if s1_startpt_inside and s2_startpt_inside:
        return point_or_segment(
            segment1.start_pt,
            segment2.start_pt,
            tol=tol
        )

    if s1_startpt_inside and s2_endpt_inside:
        return point_or_segment(
            segment1.start_pt,
            segment2.end_pt,
            tol = tol
        )

    if s1_endpt_inside and s2_startpt_inside:
        return point_or_segment(
            segment2.start_pt,
            segment1.end_pt,
            tol=tol
        )

    if s1_endpt_inside and s2_endpt_inside:
        return point_or_segment(
            segment1.end_pt,
            segment2.end_pt,
            tol=tol
        )

    if s1_startpt_inside:
        return segment1.start_pt.clone()

    if s1_endpt_inside:
        return segment1.end_pt.clone()

    if s2_startpt_inside:
        return segment2.start_pt.clone()

    if s2_endpt_inside:
        return segment2.end_pt.clone()

    cline1 = CLine.fromSegment(segment1)
    cline2 = CLine.fromSegment(segment2)

    shortest_segm_or_pt = cline1.shortest_segment_or_point(
        cline2,
        tol=tol
    )

    if not shortest_segm_or_pt:
        return None

    if not isinstance(shortest_segm_or_pt, Point):
        return None

    inters_pt = shortest_segm_or_pt

    if not segment1.contains_pt(inters_pt):
        return None

    if not segment2.contains_pt(inters_pt):
        return None

    return inters_pt


class Points:
    """
    Collection of points.
    """

    def __init__(self,
         points: List[Point],
         epsg_cd: numbers.Integral = None,
         crs_check: bool = True
):
        """

        :param points: list of points
        :type points: List[Point]
        :param epsg_cd: optional EPSG code
        :type epsg_cd: numbers.Integral
        :param crs_check: whether to check points crs
        :type crs_check: bool
        """

        for ndx, point in enumerate(points):

            check_type(point, "Input point {}".format(ndx), Point)

        if not epsg_cd:
            epsg_cd = points[0].epsg()

        if crs_check:

            for ndx, point in enumerate(points):

                if point.epsg() != epsg_cd:

                    raise Exception("Point {} has EPSG code {} but {} required".format(ndx, point.epsg(), epsg_cd))

        self._xs = np.array([p.x for p in points])
        self._ys = np.array([p.y for p in points])
        self._zs = np.array([p.z for p in points])
        self._ts = np.array([p.t for p in points])
        self._crs = Crs(epsg_cd)

    @property
    def xs(self):
        """
        The points x values.

        :return: points x values
        :rtype: float
        """

        return self._xs

    @property
    def ys(self):
        """
        The points y values.

        :return: points y values
        :rtype: float
        """

        return self._ys

    @property
    def zs(self):
        """
        The points z values.

        :return: points z values
        :rtype: float
        """

        return self._zs


    @property
    def ts(self):
        """
        The points t values.

        :return: points t values
        :rtype: float
        """

        return self._ts

    @property
    def crs(self) -> Crs:
        """
        The points CRS.

        :return: the points CRS
        :rtype: Crs
        """

        return self._crs

    def epsg(self) -> numbers.Integral:
        """
        The points EPSG code.

        :return: the points EPSG code
        :rtype: numbers.Integral
        """

        return self.crs.epsg()

    def nanmean_point(self) -> Point:
        """
        Returns the nan- excluded mean point of the collection.
        It is the mean point for a collection of point in a x-y-z frame (i.e., not lat-lon).

        :return: the nan- excluded mean point of the collection.
        :rtype: Point
        """

        return Point(
            x=np.nanmean(self.xs),
            y=np.nanmean(self.ys),
            z=np.nanmean(self.zs),
            t=np.nanmean(self.ts),
            epsg_cd=self.epsg()
        )


class Segments(list):
    """
    Collection of segments, inheriting from list.

    """

    def __init__(self, segments: List[Segment]):

        check_type(segments, "Segments", List)
        for el in segments:
            check_type(el, "Segment", Segment)

        super(Segments, self).__init__(segments)


class Lines(list):
    """
    Collection of lines, inheriting from list.

    """

    def __init__(self, lines: List[Line]):

        check_type(lines, "Lines", List)
        for el in lines:
            check_type(el, "Line", Line)

        super(Lines, self).__init__(lines)


class MultiLines(list):
    """
    Collection of multilines, inheriting from list.

    """

    def __init__(self, multilines: List[MultiLine]):

        check_type(multilines, "MultiLines", List)
        for el in multilines:
            check_type(el, "MultiLine", MultiLine)

        super(MultiLines, self).__init__(multilines)

