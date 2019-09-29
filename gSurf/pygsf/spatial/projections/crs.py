# -*- coding: utf-8 -*-

import numbers


min_epsg_crs_code = 2000  # checked 2019-06-14 in EPSG database


class Crs(object):
    """
    CRS class.
    Currently it is in a basic form,
    just managing simple comparisons and validity checks.

    """

    def __init__(self, epsg_cd: numbers.Integral = -1):

        self._epsg = int(epsg_cd)

    def epsg(self) -> numbers.Integral:

        return self._epsg

    def valid(self):

        return self.epsg() >= min_epsg_crs_code

    def __repr__(self):

        return "EPSG:{}".format(self.epsg())

    def __eq__(self, another) -> bool:
        """
        Checks for equality between Crs instances.
        Currently it considers equal two Crs instances when they have the
        same EPSG code, even an invalid one (i.e., -1).

        :param another: the Crs instance to compare with.
        :type another: Crs.
        :return: whether the input Crs instance is equal to the current one.
        :rtype: bool.
        :raise: Exception.
        """

        if not (isinstance(another, Crs)):
            raise Exception("Input instance should be Crs but is {}".format(type(another)))

        return self.epsg() == another.epsg()


def check_crs(
    template_element,
    checked_element
) -> None:
    """
    Check whether two spatial elements have the same crs.

    :param template_element: first spatial element.
    :param checked_element: second spatial element.
    :return: whether two spatial elements have the same crs.
    :rtype: None.
    """

    if checked_element.crs != template_element.crs:
        raise Exception("checked {} instance has {} EPSG code but {} expected".format(
            type(checked_element).__name__,
            checked_element.epsg(),
            template_element.epsg()
        )
    )
