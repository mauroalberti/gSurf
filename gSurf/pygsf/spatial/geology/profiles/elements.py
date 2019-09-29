

from pygsf.spatial.vectorial.geometries import *


class Attitude:
    """
    Represent a geological attitude projected onto a vertical profile.
    """

    def __init__(
        self,
        rec_id: int,
        s: numbers.Real,
        z: numbers.Real,
        slope_degr: numbers.Real,
        down_sense: str,
        dist: numbers.Real
    ):
        """
        :param rec_id: the identifier of the observation.
        :type rec_id: int.
        :param s: the signed horizontal distance along the profile.
        :type s: numbers.Real (note: may exceed a profile range, both before, negative values, or after its end.
        :param z: the height of the attitude in the profile.
        :type z: numbers.Real.
        :param slope_degr: the slope of the attitude in the profile. Unit is degrees.
        :type slope_degr: numbers.Real.
        :param down_sense: downward sense, to the right or to the profile left.
        :type down_sense: str.
        :param dist: the distance between the attitude point and the point projection on the profile.
        :type: dist: numbers.Real
        """

        self.id = rec_id
        self.s = s
        self.z = z
        self.slope_degr = slope_degr
        self.down_sense = down_sense
        self.dist = dist

    def __repr__(self) -> str:
        """
        Creates the representation of a ProfileAttitude instance.

        :return: the representation of a ProfileAttitude instance.
        :rtype: str.
        """

        return"ProfileAttitude(id={}, s={}, z={}, slope_degr={}, down_sense={}, dist={})".format(
            self.id,
            self.s,
            self.z,
            self.slope_degr,
            self.down_sense,
            self.dist
        )

    def create_segment_for_plot(
            self,
            profile_length: numbers.Real,
            vertical_exaggeration: numbers.Real = 1,
            segment_scale_factor: numbers.Real = 70.0):
        """

        :param profile_length:
        :param vertical_exaggeration:
        :param segment_scale_factor: the scale factor controlling the attitude segment length in the plot.
        :return:
        """

        ve = float(vertical_exaggeration)

        z0 = self.z

        h_dist = self.s
        slope_rad = radians(self.slope_degr)
        intersection_downward_sense = self.down_sense
        length = profile_length / segment_scale_factor

        s_slope = sin(float(slope_rad))
        c_slope = cos(float(slope_rad))

        if c_slope == 0.0:
            height_corr = length / ve
            structural_segment_s = [h_dist, h_dist]
            structural_segment_z = [z0 + height_corr, z0 - height_corr]
        else:
            t_slope = s_slope / c_slope
            width = length * c_slope

            length_exag = width * sqrt(1 + ve*ve * t_slope*t_slope)

            corr_width = width * length / length_exag
            corr_height = corr_width * t_slope

            structural_segment_s = [h_dist - corr_width, h_dist + corr_width]
            structural_segment_z = [z0 + corr_height, z0 - corr_height]

            if intersection_downward_sense == "left":
                structural_segment_z = [z0 - corr_height, z0 + corr_height]

        return structural_segment_s, structural_segment_z


class LineIntersections:
    """

    """

    def __init__(
        self,
        line_id: int,
        geoms: List[Union[Point, Segment]]
    ):

        check_type(line_id, "Line index", numbers.Integral)
        for geom in geoms:
            check_type(geom, "Intersection geometry", (Point, Segment))

        self._line_id = line_id
        self._geoms = geoms

    @property
    def line_id(self) -> numbers.Integral:
        """
        Return line id.

        :return: the line id
        :rtype: numbers.Integral
        """

        return self._line_id

    @property
    def geoms(self) -> List[Union[Point, Segment]]:
        """
        Returns the intersecting geometries.

        :return: the intersecting geometries.
        :rtype: List[Union[Point, Segment]]
        """

        return self._geoms


class ProfileSubpart:
    """
    A subset of a profile.
    """

    def __init__(self,
                 rec_id: numbers.Integral,
                 parts: List[array]):

        check_type(rec_id, "Input id", numbers.Integral)
        check_type(parts, "List of parts", list)
        for part in parts:
            check_type(part, "Part", array)

        self._id = rec_id
        self._parts = parts

    @property
    def id(self) -> numbers.Integral:
        """
        Return id.

        :return: the id
        :rtype: numbers.Integral
        """

        return self._id

    @property
    def parts(self) -> List[array]:
        """
        Returns the profile parts.
        The arrays represent s values, i.e., distances from profile start.

        :return: the profile parts
        :rtype: List[array]
        """

        return self._parts

