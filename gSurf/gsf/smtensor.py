# -*- coding: utf-8 -*-

import numpy as np

from .quaternions import Quaternion


class SMTensor(object):
    """
    Class representing the seismic moment tensor
    """

    def __repr__(self):

        return "SMTensor(\n {:.4f}, {:.4f}, {:.4f}\n {:.4f}, {:.4f}, {:.4f}\n {:.4f}, {:.4f}, {:.4f})".format(
            self.t[0, 0], self.t[0, 1], self.t[0, 2],
            self.t[1, 0], self.t[1, 1], self.t[1, 2],
            self.t[2, 0], self.t[2, 1], self.t[2, 2],)

    def __init__(self, m11=float('nan'), m12=float('nan'), m13=float('nan'),
                       m21=float('nan'), m22=float('nan'), m23=float('nan'),
                       m31=float('nan'), m32=float('nan'), m33=float('nan')):
        """
        Constructor from scalar components.

        :param m11: float
        :param m12: float
        :param m13: float
        :param m21: float
        :param m22: float
        :param m23: float
        :param m31: float
        :param m32: float
        :param m33: float

        Example:
          >>> SMTensor(1, 0, 0, 0, 1, 0, 0, 0, 1)
          SMTensor(
           1.0000, 0.0000, 0.0000
           0.0000, 1.0000, 0.0000
           0.0000, 0.0000, 1.0000)
        """

        self.t = np.array([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])

    @classmethod
    def from_quater(cls, quater):
        """
        Class constructor from a Quaternion instance.
        Formula 34 in Kagan, Y., 2008. On geometric complexity of earthquake focal zone and fault zone.
        Cited from Kagan & Jackson, 1994.

        :param quater: a Quaternion instance
        :return: SMTensor instance
        """

        q0, q1, q2, q3 = quater.normalize().components()

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

        m11 = q1q1*q1q1 - 6*q1q2*q1q2  - 2*q1q3*q1q3 + 2*q0q1*q0q1 + 8*q0q1*q2q3 + q2q2*q2q2 \
             + 2*q2q3*q2q3 - 2*q0q2*q0q2 + q3q3*q3q3  - 6*q0q3*q0q3 + q0q0*q0q0

        m12 = 4*(q1q1*q1q2 - q1q2*q2q2 - q0q3*q3q3 + q0q0*q0q3)

        m13 = 2*(q1q1*q1q3 - 3*q0q1*q1q2 -3*q1q2*q2q3 - q1q3*q3q3
                 + 3*q0q0*q1q3 + q0q2*q2q2 + 3*q0q2*q3q3 - q0q0*q0q2)

        m21 = m12

        m22 = - q1q1*q1q1 + 6*q1q2*q1q2 - 2*q1q3*q1q3 + 2*q0q1*q0q1 + 8*q0q1*q2q3 \
               -q2q2*q2q2 + 2*q2q3*q2q3 - 2*q0q2*q0q2 - q3q3*q3q3 + 6*q0q3*q0q3 - q0q0*q0q0

        m23 = 2*(q0q1*q1q1 + 3*q1q1*q2q3 - 3*q0q1*q2q2 + 3*q0q1*q3q3
                 - q0q0*q0q1 - q2q2*q2q3 + q2q3*q3q3 -3*q0q0*q2q3)

        m31 = m13

        m32 = m23

        m33 = 4*(q1q3*q1q3 - q0q1*q0q1 - 4*q0q1*q2q3 - q2q3*q2q3 + q0q2*q0q2)

        obj = cls()
        obj.t = np.array(
            [m11, m12, m13],
            [m21, m22, m23],
            [m31, m32, m33])

        return obj

if __name__ == "__main__":

    import doctest
    doctest.testmod()

