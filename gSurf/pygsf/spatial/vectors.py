# -*- coding: utf-8 -*-


import numbers

from .projections.crs import *
from ..utils.types import *

from ..mathematics.defaults import *
from ..mathematics.arrays import *


class Vect(object):
    """
    Cartesian 3D vector.
    Right-handed rectangular Cartesian coordinate system (ENU):
    x axis -> East
    y axis -> North
    z axis -> Up
    """

    def __init__(self,
        x: numbers.Real,
        y: numbers.Real,
        z: numbers.Real = 0.0,
        epsg_cd: numbers.Integral = -1):
        """
        Vect constructor.

        Example;
          >>> Vect(1, 0, 1)
          Vect(1.0000, 0.0000, 1.0000, EPSG: -1)
          >>> Vect(1, np.nan, 1)
          Traceback (most recent call last):
          ...
          Exception: Input values must be finite
          >>> Vect(1, 0, np.inf)
          Traceback (most recent call last):
          ...
          Exception: Input values must be finite
          >>> Vect(0, 0, 0, epsg_cd=32648)
          Vect(0.0000, 0.0000, 0.0000, EPSG: 32648)
          >>> Vect(2.2, -19.7, epsg_cd=32648)
          Vect(2.2000, -19.7000, 0.0000, EPSG: 32648)
        """

        vals = [x, y, z]

        if any(map(lambda val: not isinstance(val, numbers.Real), vals)):
            raise Exception("Input values must be integer of float")

        if not all(map(math.isfinite, vals)):
            raise Exception("Input values must be finite")

        self._a = np.array(vals, dtype=np.float64)
        self._crs = Crs(epsg_cd)

    def __abs__(self):
        """
        The abs of a vector.

        :return: numbers.Real
        """

        return self.len3D

    def __eq__(self, another: 'Vect') -> bool:
        """
        Return True if objects are equal.

        Example:
          >>> Vect(1., 1., 1.) == Vect(1, 1, 1)
          True
          >>> Vect(1., 1., 1.) == Vect(1, 1, -1)
          False
        """

        if not isinstance(another, Vect):
            raise Exception("Instances must be of the same type")
        else:
            return all([
                self.x == another.x,
                self.y == another.y,
                self.z == another.z,
                self.epsg() == another.epsg()])

    def __ne__(self, another: 'Vect') -> bool:
        """
        Return False if objects are equal.

        Example:
          >>> Vect(1., 1., 1.) != Vect(0., 0., 0.)
          True
        """

        if not isinstance(another, Vect):
            raise Exception("Instances must be of the same type")
        else:
            return not (self == another)

    @property
    def a(self) -> np.array:
        """
        Return a copy of the object inner array.

        :return: double array of x, y, z values

        Examples:
          >>> np.allclose(Vect(4, 3, 7).a, np.array([ 4.,  3.,  7.]))
          True
        """

        return np.copy(self._a)

    @property
    def x(self) -> numbers.Real:
        """
        Return x value

        Example:
          >>> Vect(1.5, 1, 1).x
          1.5
        """

        return self.a[0]

    @property
    def y(self) -> numbers.Real:
        """
        Return y value

        Example:
          >>> Vect(1.5, 3.0, 1).y
          3.0
        """
        return self.a[1]

    @property
    def z(self) -> numbers.Real:
        """
        Return z value

        Example:
          >>> Vect(1.5, 3.2, 41.).z
          41.0
        """
        return self.a[2]

    @property
    def crs(self) -> Crs:
        """
        Returns the crs of the Vector.

        :return: the crs.
        :rtype: Crs.
        """

        return self._crs

    def epsg(self) -> numbers.Integral:
        """
        Returns the EPSG code of the Vector instance.

        :return: EPSG code.
        :rtype: numbers.Integral.
        """

        return self._crs.epsg()

    def __iter__(self):
        """
        Return the elements of a Point.

        :return:

        """

        return (i for i in [self.x, self.y, self.z, self.epsg()])

    def toXYZ(self) -> Tuple[numbers.Real, numbers.Real, numbers.Real]:
        """
        Returns the spatial components as a tuple of three values.

        :return: the spatial components (x, y, z).
        :rtype: a tuple of three floats.

        Examples:
          >>> Vect(1, 0, 3).toXYZ()
          (1.0, 0.0, 3.0)
        """

        return self.x, self.y, self.z

    def toArray(self) -> np.array:
        """
        Return a double Numpy array representing the point values.

        :return: Numpy array

        Examples:
          >>> np.allclose(Vect(1, 2, 3).toArray(), np.array([ 1., 2., 3.]))
          True
        """

        return self.a

    def pXY(self) -> 'Vect':
        """
        Projection on the x-y plane

        :return: projected object instance

        Examples:
          >>> Vect(2, 3, 4).pXY()
          Vect(2.0000, 3.0000, 0.0000, EPSG: -1)
        """

        return self.__class__(self.x, self.y, 0.0, epsg_cd=self.epsg())

    def pXZ(self) -> 'Vect':
        """
        Projection on the x-z plane

        :return: projected object instance

        Examples:
          >>> Vect(2, 3, 4, epsg_cd=2000).pXZ()
          Vect(2.0000, 0.0000, 4.0000, EPSG: 2000)
        """

        return self.__class__(self.x, 0.0, self.z, epsg_cd=self.epsg())

    def pYZ(self) -> 'Vect':
        """
        Projection on the y-z plane

        :return: projected object instance

        Examples:
          >>> Vect(2, 3, 4).pYZ()
          Vect(0.0000, 3.0000, 4.0000, EPSG: -1)
        """

        return self.__class__(0.0, self.y, self.z, epsg_cd=self.epsg())

    @property
    def len3D(self) -> numbers.Real:
        """
        Spatial distance of the point from the axis origin.

        :return: distance
        :rtype: numbers.Real

        Examples:
          >>> Vect(4.0, 3.0, 0.0).len3D
          5.0
        """

        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    @property
    def len2D(self) -> numbers.Real:
        """
        2D distance of the point from the axis origin.

        Example:
          >>> Vect(3, 4, 0).len2D
          5.0
          >>> Vect(12, 5, 3).len2D
          13.0
        """

        return math.sqrt(self.x * self.x + self.y * self.y)

    def deltaX(self, another: 'Vect') -> Optional[numbers.Real]:
        """
        Delta between x components of two Vect Instances.

        :return: x difference value.
        :rtype: Optional[numbers.Real].

        Examples:
          >>> Vect(1, 2, 3, epsg_cd=2000).deltaX(Vect(4, 7, 1, epsg_cd=2000))
          3.0
        """

        if not isinstance(another, Vect):
            return None

        if self.epsg() != another.epsg():
            return None

        return another.x - self.x

    def deltaY(self, another: 'Vect') -> Optional[numbers.Real]:
        """
        Delta between y components of two Vect Instances.

        :return: y difference value.
        :rtype: Optional[numbers.Real].

        Examples:
          >>> Vect(1, 2, 3, epsg_cd=2000).deltaY(Vect(4, 7, 1, epsg_cd=2000))
          5.0
        """

        if not isinstance(another, Vect):
            return None

        if self.epsg() != another.epsg():
            return None

        return another.y - self.y

    def deltaZ(self, another: 'Vect') -> Optional[numbers.Real]:
        """
        Delta between x components of two Vect Instances.

        :return: z difference value.
        :rtype: Optional[numbers.Real].

        Examples:
          >>> Vect(1, 2, 3, epsg_cd=2000).deltaZ(Vect(4, 7, 1, epsg_cd=2000))
          -2.0
        """

        if not isinstance(another, Vect):
            return None

        if self.epsg() != another.epsg():
            return None

        return another.z - self.z

    def scale(self, scale_factor: numbers.Real) -> Optional['Vect']:
        """
        Create a scaled object.

        Example;
          >>> Vect(1, 0, 1).scale(2.5)
          Vect(2.5000, 0.0000, 2.5000, EPSG: -1)
          >>> Vect(1, 0, 1, 32633).scale(2.5)
          Vect(2.5000, 0.0000, 2.5000, EPSG: 32633)
          >>> Vect(1, 0, 1).scale(np.nan) is None
          True
          >>> Vect(1, 0, 1).scale(np.inf) is None
          True
        """

        if not isinstance(scale_factor, numbers.Real):
            return None

        if not math.isfinite(scale_factor):
            return None

        x, y, z = arrToTuple(self.a * scale_factor)
        return self.__class__(x, y, z, epsg_cd=self.epsg())

    def invert(self) -> 'Vect':
        """
        Create a new object with inverted direction.

        Examples:
          >>> Vect(1, 1, 1, epsg_cd=2000).invert()
          Vect(-1.0000, -1.0000, -1.0000, EPSG: 2000)
          >>> Vect(2, -1, 4).invert()
          Vect(-2.0000, 1.0000, -4.0000, EPSG: -1)
        """

        return self.scale(-1)

    def __repr__(self) -> str:

        return "Vect({:.4f}, {:.4f}, {:.4f}, EPSG: {})".format(self.x, self.y, self.z, self.epsg())

    def __add__(self, another: 'Vect') -> 'Vect':
        """
        Sum of two vectors.

        :param another: the vector to add
        :type another: Vect
        :return: the sum of the two vectors
        :rtype: Vect
        :raise: Exception

        Example:
          >>> Vect(1, 0, 0, epsg_cd=2000) + Vect(0, 1, 1, epsg_cd=2000)
          Vect(1.0000, 1.0000, 1.0000, EPSG: 2000)
          >>> Vect(1, 1, 1, epsg_cd=2000) + Vect(-1, -1, -1, epsg_cd=2000)
          Vect(0.0000, 0.0000, 0.0000, EPSG: 2000)
        """

        check_type(another, "Second vector", Vect)

        check_crs(self, another)

        x, y, z = arrToTuple(self.a + another.a)
        return self.__class__(x, y, z, self.epsg())

    def __sub__(self, another: 'Vect') -> 'Vect':
        """Subtract two vectors.

        :param another: the vector to subtract
        :type another: Vect
        :return: the difference between the two vectors
        :rtype: Vect
        :raise: Exception

        Example:
          >>> Vect(1., 1., 1., epsg_cd=2000) - Vect(1., 1., 1., epsg_cd=2000)
          Vect(0.0000, 0.0000, 0.0000, EPSG: 2000)
          >>> Vect(1., 1., 3., epsg_cd=2000) - Vect(1., 1., 2.2, epsg_cd=2000)
          Vect(0.0000, 0.0000, 0.8000, EPSG: 2000)
        """

        check_type(another, "Second vector", Vect)

        check_crs(self, another)

        x, y, z = arrToTuple(self.a - another.a)
        return self.__class__(x, y, z, self.epsg())

    @property
    def isAlmostZero(self) -> bool:
        """
        Check if the Vect instance length is near zero.

        :return: Boolean

        Example:
          >>> Vect(1, 2, 0).isAlmostZero
          False
          >>> Vect(0.0, 0.0, 0.0).isAlmostZero
          True
        """

        return areClose(self.len3D, 0)

    @property
    def isAlmostUnit(self) -> bool:
        """
        Check if the Vect instance length is near unit.

        :return: Boolean

        Example:
          >>> Vect(1, 2, 0).isAlmostUnit
          False
          >>> Vect(0.0, 1.0, 0.0).isAlmostUnit
          True
        """

        return areClose(self.len3D, 1)

    @property
    def isValid(self) -> bool:
        """
        Check if the Vect instance components are not all valid and the xyz not all zero-valued.

        :return: Boolean

        Example:
          >>> Vect(1, 2, 0).isValid
          True
          >>> Vect(0.0, 0.0, 0.0).isValid
          False
        """

        return not self.isAlmostZero

    def versor(self) -> Optional['Vect']:
        """
        Calculate versor in xyz space.

        Example:
          >>> Vect(5, 0, 0).versor()
          Vect(1.0000, 0.0000, 0.0000, EPSG: -1)
          >>> Vect(0, 0, -1, epsg_cd=32633).versor()
          Vect(0.0000, 0.0000, -1.0000, EPSG: 32633)
          >>> Vect(0, 0, 0).versor() is None
          True
        """

        if not self.isValid:
            return None
        else:
            return self.scale(1.0 / self.len3D)

    def versor2D(self) -> Optional['Vect']:
        """
        Create 2D versor version of the current vector

        :return: unit vector

        Example:
          >>> Vect(7, 0, 10).versor2D()
          Vect(1.0000, 0.0000, 0.0000, EPSG: -1)
          >>> Vect(0, 0, 10).versor2D() is None
          True
        """

        vXY = self.pXY()
        if vXY.isValid:
            return self.pXY().versor()
        else:
            return None

    @property
    def isUpward(self) -> Optional[bool]:
        """
        Check that a vector is upward-directed.

        :return: boolean

        Example:
          >>> Vect(0, 0, 1).isUpward
          True
          >>> Vect(0, 0, -0.5).isUpward
          False
          >>> Vect(1, 3, 0).isUpward
          False
          >>> Vect(0, 0, 0).isUpward is None
          True
        """

        if not self.isValid:
            return None
        else:
            return self.z > 0.0

    @property
    def isDownward(self) -> Optional[bool]:
        """
        Check that a vector is downward-directed.

        :return: boolean

        Example:
          >>> Vect(0, 0, 1).isDownward
          False
          >>> Vect(0, 0, -0.5).isDownward
          True
          >>> Vect(1, 3, 0).isDownward
          False
          >>> Vect(0, 0, 0).isDownward is None
          True
        """

        if not self.isValid:
            return None
        else:
            return self.z < 0.0

    def upward(self) -> Optional['Vect']:
        """
        Calculate a new upward-pointing vector.

        Example:
          >>> Vect(1, 1, 1).upward()
          Vect(1.0000, 1.0000, 1.0000, EPSG: -1)
          >>> Vect(-1, -1, -1).upward()
          Vect(1.0000, 1.0000, 1.0000, EPSG: -1)
          >>> Vect(0, 0, 0).upward() is None
          True
        """

        if not self.isValid:
            return None
        elif self.z < 0.0:
            return self.scale(-1.0)
        else:
            return self.scale(1.0)

    def downward(self) -> Optional['Vect']:
        """
        Calculate a new vector downward-pointing.

        Example:
          >>> Vect(1, 1, 1).downward()
          Vect(-1.0000, -1.0000, -1.0000, EPSG: -1)
          >>> Vect(-1, -1, -1).downward()
          Vect(-1.0000, -1.0000, -1.0000, EPSG: -1)
          >>> Vect(0, 0, 0).downward() is None
          True
        """

        if not self.isValid:
            return None
        elif self.z > 0.0:
            return self.scale(-1.0)
        else:
            return self.scale(1.0)

    def slope_degr(self):
        """
        Slope of a vector expressed as degrees.
        Positive when vector is downward pointing,
        negative when upward pointing.

        Example:
          >>> Vect(1, 0, -1).slope_degr()
          45.0
          >>> Vect(1, 0, 1).slope_degr()
          -45.0
          >>> Vect(0, 1, 0).slope_degr()
          0.0
          >>> Vect(0, 0, 1).slope_degr()
          -90.0
          >>> Vect(0, 0, -1).slope_degr()
          90.0
        """

        hlen = self.len2D
        if hlen == 0.0:
            if self.z > 0.:
                return -90.
            elif self.z < 0.:
                return 90.
            else:
                raise Exception("Zero-valued vector")
        else:
            slope = - math.degrees(math.atan(self.z / self.len2D))
            if abs(slope) > MIN_SCALAR_VALUE:
                return slope
            else:
                return 0.

    def vDot(self, another: 'Vect') -> numbers.Real:
        """
        Vector scalar multiplication.

        Examples:
          >>> Vect(1, 0, 0).vDot(Vect(1, 0, 0))
          1.0
          >>> Vect(1, 0, 0).vDot(Vect(0, 1, 0))
          0.0
          >>> Vect(1, 0, 0).vDot(Vect(-1, 0, 0))
          -1.0
        """

        return self.x * another.x + self.y * another.y + self.z * another.z

    def angleCos(self,
        another: 'Vect'
    ) -> Optional[numbers.Real]:
        """
        Return the cosine of the angle between two vectors.

        Examples:
          >>> Vect(1,0,0).angleCos(Vect(0,0,1))
          0.0
          >>> Vect(1,0,0).angleCos(Vect(-1,0,0))
          -1.0
          >>> Vect(1,0,0).angleCos(Vect(1,0,0))
          1.0
          >>> Vect(0, 0, 0).angleCos(Vect(1,0,0)) is None
          True
          >>> Vect(1, 0, 0).angleCos(Vect(0,0,0)) is None
          True
        """

        if not isinstance(another, Vect):
            return None

        if self.epsg() != another.epsg():
            return None

        if not (self.isValid and another.isValid):
            return None

        val = self.vDot(another) / (self.len3D * another.len3D)
        if val > 1.0:
            return 1.0
        elif val < -1.0:
            return -1.0
        else:
            return val

    def scalarProjection(self,
        another: 'Vect'
    ) -> Optional[numbers.Real]:
        """
        Return the scalar projection of the second vector on the first vector.

        Examples:
          >>> Vect(1,0,0).scalarProjection(Vect(0,0,1))
          0.0
          >>> Vect(2,0,0).scalarProjection(Vect(1,5,0))
          1.0
          >>> Vect(2,0,0).scalarProjection(Vect(-1,5,0))
          -1.0
          >>> Vect(4,0,0).scalarProjection(Vect(7.5, 19.2, -14.72))
          7.5
        """

        check_type(another, "Second vector", Vect)

        check_crs(self, another)

        return self.angleCos(another) * another.len3D

    def fractionalProjection(self,
        another: 'Vect'
    ) -> Optional[numbers.Real]:
        """
        Return the fractional projection of the second vector on the first vector.

        Examples:
          >>> Vect(1,0,0).fractionalProjection(Vect(0,0,1))
          0.0
          >>> Vect(2,0,0).fractionalProjection(Vect(1,5,0))
          0.5
          >>> Vect(2,0,0).fractionalProjection(Vect(-1,5,0))
          -0.5
        """

        check_type(another, "Second vector", Vect)

        check_crs(self, another)

        return self.scalarProjection(another) / self.len3D

    def angle(self, another: 'Vect', unit='d') -> Optional[numbers.Real]:
        """
        Calculate angle between two vectors,
        in 0° - 180° range (as degrees).

        Example:
          >>> Vect(1, 0, 0).angle(Vect(0, 0, 1))
          90.0
          >>> Vect(1, 0, 0).angle(Vect(-1, 0, 0))
          180.0
          >>> Vect(0, 0, 1).angle(Vect(0, 0, -1))
          180.0
          >>> Vect(1, 1, 1).angle(Vect(1, 1,1))
          0.0
          >>> Vect(0, 0, 0).angle(Vect(1,0,0)) is None
          True
          >>> Vect(1, 0, 0).angle(Vect(0,0,0)) is None
          True
        """

        if not isinstance(another, Vect):
            return None

        if self.epsg() != another.epsg():
            return None

        if not (self.isValid and another.isValid):
            return None

        angle_rad = math.acos(self.angleCos(another))
        if unit == 'd':
            return math.degrees(angle_rad)
        elif unit == 'r':
            return angle_rad
        else:
            return None

    def vCross(self, another: 'Vect') -> Optional['Vect']:
        """
        Vector product (cross product).

        Examples:
          >>> Vect(1, 0, 0).vCross(Vect(0, 1, 0))
          Vect(0.0000, 0.0000, 1.0000, EPSG: -1)
          >>> Vect(1, 0, 0).vCross(Vect(1, 0, 0))
          Vect(0.0000, 0.0000, 0.0000, EPSG: -1)
          >>> (Vect(1, 0, 0).vCross(Vect(-1, 0, 0))).isAlmostZero
          True
        """

        if not isinstance(another, Vect):
            raise Exception("Another instance should be Vect but is {}".format(type(another)))

        if self.epsg() != another.epsg():
            raise Exception("Another instance should have {} EPSG code but has {}".format(self.epsg(), another.epsg()))

        x, y, z = arrToTuple(np.cross(self.a[:3], another.a[:3]))
        return Vect(x, y, z, epsg_cd=self.epsg())

    def byMatrix(self, array3x3: np.array) -> 'Vect':
        """
        Matrix multiplication of a vector.

        """

        x, y, z = arrToTuple(array3x3.dot(self.a))
        return Vect(x, y, z)

    @property
    def azimuth(self) -> Optional[numbers.Real]:
        """
        The azimuth between the Y axis and the vector, calculated clockwise.

        :return: angle in degrees.
        :rtype: optional numbers.Real.

        Examples:
          >>> Vect(0, 1, 0).azimuth
          0.0
          >>> Vect(1, 1, 0).azimuth
          45.0
          >>> Vect(1, 0, 0).azimuth
          90.0
          >>> Vect(1, -1, 0).azimuth
          135.0
          >>> Vect(0, -1, 0).azimuth
          180.0
          >>> Vect(-1, -1, 0).azimuth
          225.0
          >>> Vect(-1, 0, 0).azimuth
          270.0
          >>> Vect(-1, 1, 0).azimuth
          315.0
          >>> Vect(0, 0, 1).azimuth is None
          True
          >>> Vect(0, 0, -1).azimuth is None
          True
        """

        y_axis = Vect(0, 1, 0)
        vector_2d = self.versor2D()

        if not vector_2d:
            return None

        angle = vector_2d.angle(y_axis)

        z_comp = y_axis.vCross(vector_2d).z

        if z_comp <= 0.0:
            return angle
        else:
            return 360.0 - angle


if __name__ == "__main__":

    import doctest
    doctest.testmod()
