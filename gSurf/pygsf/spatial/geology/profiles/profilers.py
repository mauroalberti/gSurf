
from typing import List, Iterable
from operator import attrgetter


from pygsf.spatial.rasters.geoarray import *

from pygsf.spatial.geology.base import GeorefAttitude

from .sets import *


class LinearProfiler:
    """
    Class storing a linear (straight) profile.
    It intends to represent a vertical profile.
    In a possible future implementations, it would be superseded by a
    plane profiler, not necessarily vertical.
    """

    def __init__(self,
            start_pt: Point,
            end_pt: Point,
            densify_distance: numbers.Real):
        """
        Instantiates a 2D linear profile object.
        It is represented by two 2D points and by a densify distance.

        :param start_pt: the profile start point.
        :type start_pt: Point.
        :param end_pt: the profile end point.
        :type end_pt: Point.
        :param densify_distance: the distance with which to densify the segment profile.
        :type densify_distance: numbers.Real.
        """

        check_type(start_pt, "Input start point", Point)

        check_type(end_pt, "Input end point", Point)

        if start_pt.crs != end_pt.crs:
            raise Exception("Both points must have same CRS")

        if start_pt.dist2DWith(end_pt) == 0.0:
            raise Exception("Input segment length cannot be zero")

        check_type(densify_distance, "Input densify distance", numbers.Real)

        if not isfinite(densify_distance):
            raise Exception("Input densify distance must be finite")

        if densify_distance <= 0.0:
            raise Exception("Input densify distance must be positive")

        epsg_cd = start_pt.epsg()

        self._start_pt = Point(x=start_pt.x, y=start_pt.y, epsg_cd=epsg_cd)
        self._end_pt = Point(x=end_pt.x, y=end_pt.y, epsg_cd=epsg_cd)
        self._crs = Crs(epsg_cd)
        self._densify_dist = float(densify_distance)

    def start_pt(self) -> Point:
        """
        Returns a copy of the segment start 2D point.

        :return: start point copy.
        :rtype: Point.
        """

        return self._start_pt.clone()

    def end_pt(self) -> Point:
        """
        Returns a copy of the segment end 2D point.

        :return: end point copy.
        :rtype: Point.
        """

        return self._end_pt.clone()

    def densify_dist(self) -> numbers.Real:
        """
        Returns the densify distance of the profiler.

        :return: the densify distance of the profiler.
        :rtype: numbers.Real.
        """

        return self._densify_dist

    def __repr__(self):
        """
        Representation of a profile instance.

        :return: the textual representation of the instance.
        :rtype: str.
        """

        return "LinearProfiler(\n\tstart_pt = {},\n\tend_pt = {},\n\tdensify_distance = {})".format(
            self.start_pt(),
            self.end_pt(),
            self.densify_dist()
        )

    @property
    def crs(self) -> Crs:
        """
        Returns the CRS of the profile.

        :return: the CRS of the profile.
        :rtype: Crs.
        """

        return self._crs

    def epsg(self) -> numbers.Integral:
        """
        Returns the EPSG code of the profile.

        :return: the EPSG code of the profile.
        :rtype: numbers.Real.
        """

        return self.crs.epsg()

    def clone(self) -> 'LinearProfiler':
        """
        Returns a deep copy of the current linear profiler.

        :return: a deep copy of the current linear profiler.
        :rtype: LinearProfiler.
        """

        return LinearProfiler(
            start_pt=self.start_pt().clone(),
            end_pt=self.end_pt().clone(),
            densify_distance=self.densify_dist()
        )

    def segment(self) -> Segment:
        """
        Returns the horizontal segment representing the profile.

        :return: segment representing the profile.
        :rtype: Segment.
        """

        return Segment(start_pt=self._start_pt, end_pt=self._end_pt)

    def length(self) -> numbers.Real:
        """
        Returns the length of the profiler section.

        :return: length of the profiler section.
        :rtype: numbers.Real.
        """

        return self.segment().length3D()

    def vector(self) -> Vect:
        """
        Returns the horizontal vector representing the profile.

        :return: vector representing the profile.
        :rtype: Vect.
        """

        return self.segment().vector()

    def versor(self) -> Vect:
        """
        Returns the horizontal versor (unit vector) representing the profile.

        :return: vector representing the profile.
        :rtype: Vect.
        """

        return self.vector().versor()

    def densified_steps(self) -> array:
        """
        Returns an array made up by the incremental steps (2D distances) along the profile.

        :return: array storing incremental steps values.
        :rtype: array.
        """

        return self.segment().densify2d_asSteps(self._densify_dist)

    def num_pts(self) -> numbers.Integral:
        """
        Returns the number of steps making up the profile.

        :return: number of steps making up the profile.
        :rtype: numbers.Integral.
        """

        return len(self.densified_points())

    def densified_points(self) -> List[Point]:
        """
        Returns the list of densified 2D points.

        :return: list of densified points.
        :rtype: List[Point].
        """

        return self.segment().densify2d_asPts(densify_distance=self._densify_dist)

    def vertical_plane(self) -> CPlane:
        """
        Returns the vertical plane of the segment, as a Cartesian plane.

        :return: the vertical plane of the segment, as a Cartesian plane.
        :rtype: CPlane.
        """

        return self.segment().vertical_plane()

    def normal_versor(self) -> Vect:
        """
        Returns the perpendicular (horizontal) versor to the profile (vertical) plane.

        :return: the perpendicular (horizontal) versor to the profile (vertical) plane.
        :rtype: Vect.
        """

        return self.vertical_plane().normVersor()

    def left_norm_vers(self) -> Vect:
        """
        Returns the left horizontal normal versor.

        :return: the left horizontal normal versor.
        :rtype: Vect.
        """

        return Vect(0, 0, 1, epsg_cd=self.epsg()).vCross(self.versor()).versor()

    def right_norm_vers(self) -> Vect:
        """
        Returns the right horizontal normal versor.

        :return: the right horizontal normal versor.
        :rtype: Vect.
        """

        return Vect(0, 0, -1, epsg_cd=self.epsg()).vCross(self.versor()).versor()

    def left_offset(self,
        offset: numbers.Real) -> 'LinearProfiler':
        """
        Returns a copy of the current linear profiler, offset to the left by the provided offset distance.

        :param offset: the lateral offset to apply to create the new LinearProfiler.
        :type: numbers.Real.
        :return: the offset linear profiler.
        :rtype: LinearProfiler
        """

        return LinearProfiler(
            start_pt=self.start_pt().shiftByVect(self.left_norm_vers().scale(offset)),
            end_pt=self.end_pt().shiftByVect(self.left_norm_vers().scale(offset)),
            densify_distance=self.densify_dist()
        )

    def right_offset(self,
        offset: numbers.Real) -> 'LinearProfiler':
        """
        Returns a copy of the current linear profiler, offset to the right by the provided offset distance.

        :param offset: the lateral offset to apply to create the new LinearProfiler.
        :type: numbers.Real.
        :return: the offset linear profiler.
        :rtype: LinearProfiler
        """

        return LinearProfiler(
            start_pt=self.start_pt().shiftByVect(self.right_norm_vers().scale(offset)),
            end_pt=self.end_pt().shiftByVect(self.right_norm_vers().scale(offset)),
            densify_distance=self.densify_dist()
        )

    def point_in_profile(self, pt: Point) -> bool:
        """
        Checks whether a point lies in the profiler plane.

        :param pt: the point to check.
        :type pt: Point.
        :return: whether the point lie in the profiler plane.
        :rtype: bool.
        :raise; Exception.
        """

        return self.vertical_plane().isPointInPlane(pt)

    def point_distance(self, pt: Point) -> numbers.Real:
        """
        Calculates the point distance from the profiler plane.

        :param pt: the point to check.
        :type pt: Point.
        :return: the point distance from the profiler plane.
        :rtype: numbers.Real.
        :raise; Exception.
        """

        return self.vertical_plane().pointDistance(pt)

    def sample_grid(
            self,
            grid: GeoArray) -> array:
        """
        Sample grid values along the profiler points.

        :param grid: the input grid
        :type grid: GeoArray.
        :return: array storing the z values sampled from the grid,
        :rtype: array.
        :raise: Exception
        """

        if not isinstance(grid, GeoArray):
            raise Exception("Input grid must be a GeoArray but is {}".format(type(grid)))

        if self.crs != grid.crs:
            raise Exception("Input grid EPSG code must be {} but is {}".format(self.epsg(), grid.epsg()))

        return array('d', [grid.interpolate_bilinear(pt_2d.x, pt_2d.y) for pt_2d in self.densified_points()])

    def profile_grid(
            self,
            geoarray: GeoArray) -> TopographicProfile:
        """
        Create profile from one geoarray.

        :param geoarray: the source geoarray.
        :type geoarray: GeoArray.
        :return: the profile of the scalar variable stored in the geoarray.
        :rtype: TopographicProfile.
        :raise: Exception.
        """

        check_type(geoarray, "GeoArray", GeoArray)

        return TopographicProfile(
            s_array=self.densified_steps(),
            z_array=self.sample_grid(geoarray))

    def profile_grids(self,
        *grids: Iterable[GeoArray]
    ) -> List[TopographicProfile]:
        """
        Create profiles of one or more grids.

        :param grids: a set of grids, one or more.
        :type grids: Iterable[GeoArray]
        :return:
        :rtype:
        """

        for ndx, grid in enumerate(grids):

            check_type(grid, "{} grid".format(ndx+1), GeoArray)

        for ndx, grid in enumerate(grids):

            check_crs(self, grid)

        topo_profiles = []

        for grid in grids:

            topo_profiles.append(
                TopographicProfile(
                    s_array=self.densified_steps(),
                    z_array=self.sample_grid(grid)
                )
            )

        return topo_profiles

    def intersect_line(self,
       mline: Union[Line, MultiLine],
    ) -> List[Optional[Union[Point, Segment]]]:
        """
        Calculates the intersection with a line/multiline.
        Note: the intersections are intended flat (in a 2D plane, not 3D).

        :param mline: the line/multiline to intersect profile with
        :type mline: Union[Line, MultiLine]
        :return: the possible intersections
        :rtype: List[Optional[Union[Point, Segment]]]
        """

        return mline.intersectSegment(self.segment())

    def intersect_lines(self,
        mlines: Iterable[Union[Line, MultiLine]],
    ) -> List[List[Optional[Union[Point, Segment]]]]:
        """
        Calculates the intersection with a set of lines/multilines.
        Note: the intersections are intended flat (in a 2D plane, not 3D).

        :param mlines: an iterable of Lines or MultiLines to intersect profile with
        :type mlines: Iterable[Union[Line, MultiLine]]
        :return: the possible intersections
        :rtype: List[List[Optional[Point, Segment]]]
        """

        results = [self.intersect_line(mline) for mline in mlines]
        valid_results = [LineIntersections(ndx, res) for ndx, res in enumerate(results) if res]

        return LinesIntersections(valid_results)

    def point_signed_s(
            self,
            pt: Point) -> numbers.Real:
        """
        Calculates the point signed distance from the profiles start.
        The projected point must already lay in the profile vertical plane, otherwise an exception is raised.

        The implementation assumes (and verifies) that the point lies in the profile vertical plane.
        Given that case, it calculates the signed distance from the section start point,
        by using the triangle law of sines.

        :param pt: the point on the section.
        :type pt: Point.
        :return: the signed distance on the profile.
        :rtype: numbers.Real.
        :raise: Exception.
        """

        if not isinstance(pt, Point):
            raise Exception("Projected point should be Point but is {}".format(type(pt)))

        if self.crs != pt.crs:
            raise Exception("Projected point should have {} EPSG but has {}".format(self.epsg(), pt.epsg()))

        if not self.point_in_profile(pt):
            raise Exception("Projected point should lie in the profile plane but there is a distance of {} units".format(self.point_distance(pt)))

        projected_vector = Segment(self.start_pt(), pt).vector()
        cos_alpha = self.vector().angleCos(projected_vector)

        return projected_vector.len3D * cos_alpha

    def segment_signed_s(self,
        segment: Segment
    ) -> Tuple[numbers.Real, numbers.Real]:
        """
        Calculates the segment signed distances from the profiles start.
        The segment must already lay in the profile vertical plane, otherwise an exception is raised.

        :param segment: the analysed segment
        :type segment: Segment
        :return: the segment vertices distances from the profile start
        :rtype: Tuple[numbers.Real, numbers.Real]
        """

        segment_start_distance = self.point_signed_s(segment.start_pt)
        segment_end_distance = self.point_signed_s(segment.end_pt)

        return segment_start_distance, segment_end_distance

    def pt_segm_signed_s(self,
        geom: Union[Point, Segment]
    ) -> array:
        """
        Calculates the point or segment signed distances from the profiles start.

        :param geom: point or segment
        :type: Union[Point, Segment]
        :return: the distance(s) from the profile start
        :rtype: array of double
        """

        if isinstance(geom, Point):
            return array('d', [self.point_signed_s(geom)])
        elif isinstance(geom, Segment):
            return array('d', [*self.segment_signed_s(geom)])
        else:
            return NotImplemented

    def get_intersection_slope(self,
                intersection_vector: Vect) -> Tuple[numbers.Real, str]:
        """
        Calculates the slope (in radians) and the downward sense ('left', 'right' or 'vertical')
        for a profile-laying vector.

        :param intersection_vector: the profile-plane lying vector.
        :type intersection_vector: Vect,
        :return: the slope (in radians) and the downward sense.
        :rtype: Tuple[numbers.Real, str].
        :raise: Exception.
        """

        if not isinstance(intersection_vector, Vect):
            raise Exception("Input argument should be Vect but is {}".format(type(intersection_vector)))

        angle = degrees(acos(self.normal_versor().angleCos(intersection_vector)))
        if abs(90.0 - angle) > 1.0e-4:
            raise Exception("Input argument should lay in the profile plane")

        slope_radians = abs(radians(intersection_vector.slope_degr()))

        scalar_product_for_downward_sense = self.vector().vDot(intersection_vector.downward())
        if scalar_product_for_downward_sense > 0.0:
            intersection_downward_sense = "right"
        elif scalar_product_for_downward_sense == 0.0:
            intersection_downward_sense = "vertical"
        else:
            intersection_downward_sense = "left"

        return slope_radians, intersection_downward_sense

    def calculate_axis_intersection(self,
            map_axis: Axis,
            structural_pt: Point) -> Optional[Point]:
        """
        Calculates the optional intersection point between an axis passing through a point
        and the profiler plane.

        :param map_axis: the projection axis.
        :type map_axis: Axis.
        :param structural_pt: the point through which the axis passes.
        :type structural_pt: Point.
        :return: the optional intersection point.
        :type: Optional[Point].
        :raise: Exception.
        """

        if not isinstance(map_axis, Axis):
            raise Exception("Map axis should be Axis but is {}".format(type(map_axis)))

        if not isinstance(structural_pt, Point):
            raise Exception("Structural point should be Point but is {}".format(type(structural_pt)))

        if self.crs != structural_pt.crs:
            raise Exception("Structural point should have {} EPSG but has {}".format(self.epsg(), structural_pt.epsg()))

        axis_versor = map_axis.asDirect().asVersor()

        l, m, n = axis_versor.x, axis_versor.y, axis_versor.z

        axis_param_line = ParamLine3D(structural_pt, l, m, n)

        return axis_param_line.intersect_cartes_plane(self.vertical_plane())

    def calculate_intersection_versor(
            self,
            attitude_plane: Plane,
            attitude_pt: Point) -> Optional[Vect]:
        """
        Calculate the intersection versor between the plane profiler and
        a geological plane with location defined by a Point.

        :param attitude_plane:
        :type attitude_plane: Plane,
        :param attitude_pt: the attitude point.
        :type attitude_pt: Point.
        :return:
        """

        if not isinstance(attitude_plane, Plane):
            raise Exception("Attitude plane should be Plane but is {}".format(type(attitude_plane)))

        if not isinstance(attitude_pt, Point):
            raise Exception("Attitude point should be Point but is {}".format(type(attitude_pt)))

        if self.crs != attitude_pt.crs:
            raise Exception("Attitude point should has EPSG {} but has {}".format(self.epsg(), attitude_pt.epsg()))

        putative_inters_versor = self.vertical_plane().intersVersor(attitude_plane.toCPlane(attitude_pt))

        if not putative_inters_versor.isValid:
            return None

        return putative_inters_versor

    def nearest_attitude_projection(
            self,
            georef_attitude: GeorefAttitude) -> Point:
        """
        Calculates the nearest projection of a given attitude on a vertical plane.

        :param georef_attitude: geological attitude.
        :type georef_attitude: GeorefAttitude
        :return: the nearest projected point on the vertical section.
        :rtype: pygsf.spatial.vectorial.geometries.Point.
        :raise: Exception.
        """

        if not isinstance(georef_attitude, GeorefAttitude):
            raise Exception("georef_attitude point should be GeorefAttitude but is {}".format(type(georef_attitude)))

        if self.crs != georef_attitude.posit.crs:
            raise Exception("Attitude point should has EPSG {} but has {}".format(self.epsg(), georef_attitude.posit.epsg()))

        attitude_cplane = georef_attitude.attitude.toCPlane(georef_attitude.posit)
        intersection_versor = self.vertical_plane().intersVersor(attitude_cplane)
        dummy_inters_pt = self.vertical_plane().intersPoint(attitude_cplane)
        dummy_structural_vect = Segment(dummy_inters_pt, georef_attitude.posit).vector()
        dummy_distance = dummy_structural_vect.vDot(intersection_versor)
        offset_vector = intersection_versor.scale(dummy_distance)

        projected_pt = Point(
            x=dummy_inters_pt.x + offset_vector.x,
            y=dummy_inters_pt.y + offset_vector.y,
            z=dummy_inters_pt.z + offset_vector.z,
            epsg_cd=self.epsg())

        return projected_pt

    def map_attitude_to_section(
            self,
            georef_attitude: GeorefAttitude,
            map_axis: Optional[Axis] = None) -> Optional[Attitude]:
        """
        Project a georeferenced attitude to the section.

        :param georef_attitude: the georeferenced attitude.
        :type georef_attitude: GeorefAttitude.
        :param map_axis: the map axis.
        :type map_axis: Optional[Axis].
        :return: the optional planar attitude on the profiler vertical plane.
        :rtype: Optional[PlanarAttitude].
        """

        if not isinstance(georef_attitude, GeorefAttitude):
            raise Exception("Georef attitude should be GeorefAttitude but is {}".format(type(georef_attitude)))

        if self.crs != georef_attitude.posit.crs:
            raise Exception("Attitude point should has EPSG {} but has {}".format(self.epsg(), georef_attitude.posit.epsg()))

        if map_axis:
            if not isinstance(map_axis, Axis):
                raise Exception("Map axis should be Axis but is {}".format(type(map_axis)))

        # intersection versor

        intersection_versor = self.calculate_intersection_versor(
            attitude_plane=georef_attitude.attitude,
            attitude_pt=georef_attitude.posit
        )

        # calculate slope of geological plane onto section plane

        slope_radians, intersection_downward_sense = self.get_intersection_slope(intersection_versor)

        # intersection point

        if map_axis is None:
            intersection_point_3d = self.nearest_attitude_projection(
                georef_attitude=georef_attitude)
        else:
            intersection_point_3d = self.calculate_axis_intersection(
                map_axis=map_axis,
                structural_pt=georef_attitude.posit)

        if not intersection_point_3d:
            return None

        # distance along projection vector

        dist = georef_attitude.posit.dist3DWith(intersection_point_3d)

        # horizontal spat_distance between projected structural point and profile start

        signed_distance_from_section_start = self.point_signed_s(intersection_point_3d)

        # solution for current structural point

        return Attitude(
            rec_id=georef_attitude.id,
            s=signed_distance_from_section_start,
            z=intersection_point_3d.z,
            slope_degr=degrees(slope_radians),
            down_sense=intersection_downward_sense,
            dist=dist
        )

    def map_georef_attitudes_to_section(
        self,
        structural_data: List[GeorefAttitude],
        mapping_method: dict,
        height_source: Optional[GeoArray] = None) -> Optional[List[Attitude]]:
        """
        Projects a set of georeferenced _attitudes onto the section profile,
        optionally extracting point heights from a grid.

        defines:
            - 2D x-y position in section
            - plane-plane segment intersection

        :param structural_data: the set of georeferenced _attitudes to plot on the section.
        :type structural_data: List[GeorefAttitude]
        :param mapping_method: the method to map the _attitudes to the section.
        ;type mapping_method; Dict.
        :param height_source: the _attitudes elevation source. Default is None.
        :return: sorted list of ProfileAttitude values.
        :rtype: Optional[List[Attitude]]
        :raise: Exception.
        """

        if height_source:

            if not isinstance(height_source, GeoArray):
                raise Exception("Height source should be GeoArray but is {}".format(type(height_source)))

            attitudes_3d = []
            for georef_attitude in structural_data:
                pt3d = height_source.interpolate_bilinear_point(
                    pt=georef_attitude.posit)
                if pt3d:
                    attitudes_3d.append(GeorefAttitude(
                        georef_attitude.id,
                        pt3d,
                        georef_attitude.attitude))

        else:

            attitudes_3d = structural_data

        if mapping_method['method'] == 'nearest':
            results = [self.map_attitude_to_section(georef_att) for georef_att in attitudes_3d]
        elif mapping_method['method'] == 'common axis':
            map_axis = Axis(mapping_method['trend'], mapping_method['plunge'])
            results = [self.map_attitude_to_section(georef_att, map_axis) for georef_att in attitudes_3d]
        elif mapping_method['method'] == 'individual axes':
            if len(mapping_method['individual_axes_values']) != len(attitudes_3d):
                raise Exception(
                    "Individual axes values are {} but _attitudes are {}".format(
                        len(mapping_method['individual_axes_values']),
                        len(attitudes_3d)
                    )
                )

            results = []
            for georef_att, (trend, plunge) in zip(attitudes_3d, mapping_method['individual_axes_values']):
                try:
                    map_axis = Axis(trend, plunge)
                    result = self.map_attitude_to_section(georef_att, map_axis)
                    if result:
                        results.append(result)
                except Exception:
                    continue
        else:
            return NotImplemented

        if results is None:
            return None

        return Attitudes(sorted(results, key=attrgetter('s')))

    def parse_intersections_for_profile(self,
        lines_intersections: LinesIntersections
    ) -> List:
        """
        Parse the line intersections for incorporation
        as elements in a geoprofile.

        :param lines_intersections: the line intersections
        :type lines_intersections: LinesIntersections
        :return:
        """

        parsed_intersections = ProfileSubpartsSet()

        for line_intersections in lines_intersections:

            line_id = line_intersections.line_id
            inters_geoms = line_intersections.geoms

            intersections_ranges = [self.pt_segm_signed_s(geom) for geom in inters_geoms]

            parsed_intersections.append(ProfileSubpart(line_id, intersections_ranges))

        return parsed_intersections


class ParallelProfilers(list):
    """
    Parallel linear profilers.
    """

    def __init__(self,
        profilers: List[LinearProfiler]):
        """

        :param profilers:
        :return:
        """

        check_type(profilers, "Profilers", List)
        for el in profilers:
            check_type(el, "Profiler", LinearProfiler)

        super(ParallelProfilers, self).__init__(profilers)

    @classmethod
    def fromProfiler(cls,
         base_profiler: LinearProfiler,
         profs_num: numbers.Integral,
         profs_offset: numbers.Real,
         profs_arr: str = "central",  # one of: "left", "central", "right"
         ):
        """
        Initialize the parallel linear profilers.

        :param base_profiler: the base profiler.
        :type base_profiler: LinearProfiler.
        :param profs_num: the number of profilers to create.
        :type profs_num: numbers.Integral.
        :param profs_offset: the lateral offset between profilers.
        :type profs_offset: numbers.Real.
        :param profs_arr: profiles arrangement: one of left", "central", "right".
        :type: str.
        :return: the parallel linear profilers.
        :type: ParallelProfilers.
        :raise: Exception.

        """

        check_type(base_profiler, "Base profiler", LinearProfiler)

        check_type(profs_num, "Profilers number", numbers.Integral)
        if profs_num < 2:
            raise Exception("Profilers number must be >= 2")

        check_type(profs_arr, "Profilers arrangement", str)
        if profs_arr not in ["central", "left", "right"]:
            raise Exception("Profilers arrangement must be 'left', 'central' (default) or 'right'")

        if profs_arr == "central" and profs_num % 2 != 1:
            raise Exception("When profilers arrangement is 'central' profilers number must be odd")

        if profs_arr == "central":

            side_profs_num = profs_num // 2
            num_left_profs = num_right_profs = side_profs_num

        elif profs_arr == "left":

            num_left_profs = profs_num -1
            num_right_profs = 0

        else:

            num_right_profs = profs_num -1
            num_left_profs = 0

        profilers = []

        for i in range(num_left_profs, 0, -1):

            current_offset = profs_offset * i

            profilers.append(base_profiler.left_offset(offset=current_offset))

        profilers.append(base_profiler.clone())

        for i in range(1, num_right_profs + 1):

            current_offset = profs_offset * i

            profilers.append(base_profiler.right_offset(offset=current_offset))

        return cls(profilers)

    def __repr__(self) -> str:
        """
        Represents a parallel linear profilers set.

        :return: the textual representation of the parallel linear profiler set.
        :rtype: str.
        """

        inner_profilers = "\n".join([repr(profiler) for profiler in self])
        return "ParallelProfilers([\n{}]\n)".format(inner_profilers)

    def profile_grid(
            self,
            geoarray: GeoArray) -> TopographicProfileSet:
        """
        Create profile from one geoarray.

        :param geoarray: the source geoarray.
        :type geoarray: GeoArray.
        :return: list of profiles of the scalar variable stored in the geoarray.
        :rtype: TopographicProfileSet.
        :raise: Exception.
        """

        topo_profiles = []

        for profiler in self:

            topo_profiles.append(profiler.profile_grid(geoarray))

        return TopographicProfileSet(topo_profiles)

