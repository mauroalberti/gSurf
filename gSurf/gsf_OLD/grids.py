# -*- coding: utf-8 -*-

from __future__ import division

import numpy
from numpy import *  # general import for compatibility with formula input

from .geometry import *


def formula_to_grid(array_range, array_size, formula):
    """
    Todo: check usages and correctness

    :param array_range:
    :param array_size:
    :param formula:
    :return: three lists of float values
    """

    a_min, a_max, b_max, b_min = array_range  # note: b range reversed for conventional j order in arrays
    array_rows, array_cols = array_size

    a_array = linspace(a_min, a_max, num=array_cols)
    b_array = linspace(b_max, b_min, num=array_rows)  # note: reversed for conventional j order in arrays

    try:
        a_list, b_list = [a for a in a_array for _ in b_array], [b for _ in a_array for b in b_array]
    except:
        raise AnaliticSurfaceCalcException("Error in a-b values")

    try:
        z_list = [eval(formula) for a in a_array for b in b_array]
    except:
        raise AnaliticSurfaceCalcException("Error in applying formula to a and b array values")

    return a_list, b_list, z_list


def ij_transfer_func(i, j, transfer_funcs):
    """
    Return a p_z value as the result of a function (transfer_func_z) applied to a (x, y) point.
    This point is derived from a (i,j) point given two "transfer" functions (transfer_func_y, transfer_func_x).
    All three functions are stored into a tuple (transfer_funcs).

    @param  i:  array i (-p_y) coordinate of a single point.
    @type  i:  float.
    @param  j:  array j (p_x) coordinate of a single point.
    @type  j:  float.
    @param  transfer_funcs:  tuple storing three functions (transfer_func_x, transfer_func_y, transfer_func_z)
                            that derives p_y from i (transfer_func_y), p_x from j (transfer_func_x)
                            and p_z from (p_x,p_y) (transfer_func_z).
    @type  transfer_funcs:  Tuple of Functions.

    @return:  p_z value - float.

    """

    transfer_func_x, transfer_func_y, transfer_func_z = transfer_funcs

    return transfer_func_z(transfer_func_x(j), transfer_func_y(i))

def array_from_function(row_num, col_num, x_transfer_func, y_transfer_func, z_transfer_func):
    """
    Creates an array of p_z values based on functions that map (i,j) indices (to be created)
    into (p_x, p_y) values and then p_z values.

    @param  row_num:  row number of the array to be created.
    @type  row_num:  int.
    @param  col_num:  column number of the array to be created.
    @type  col_num:  int.
    @param  x_transfer_func:  function that derives p_x given a j array index.
    @type  x_transfer_func:  Function.
    @param  y_transfer_func:  function that derives p_y given an i array index.
    @type  y_transfer_func:  Function.
    @param  z_transfer_func:  function that derives p_z given a (p_x,p_y) point.
    @type  z_transfer_func:  Function.

    @return:  array of p_z value - array of float numbers.

    """

    transfer_funcs = (x_transfer_func, y_transfer_func, z_transfer_func)

    return fromfunction(ij_transfer_func, (row_num, col_num), transfer_funcs=transfer_funcs)


class ArrCoord(object):
    """
    2D Array coordinates.
    Manages coordinates in the raster (array) space.

    """

    def __init__(self, ival=0.0, jval=0.0):
        """
        Class constructor.

        @param  ival:  the i (-y) array coordinate of the point.
        @type  ival:  number or string convertible to float.
        @param  jval:  the j (x) array coordinate of the point.
        @type  jval:  number or string convertible to float.

        @return:  self.
        """
        self._i = float(ival)
        self._j = float(jval)

    def g_i(self):
        """
        Get i (row) coordinate value.

        @return:  the i (-y) array coordinate of the point - float.
        """
        return self._i

    def s_i(self, ival):
        """
        Set i (row) coordinate value.

        @param  ival:  the i (-y) array coordinate of the point.
        @type  ival:  number or string convertible to float.

        @return:  self.
        """
        self._i = float(ival)

    # set property for i
    i = property(g_i, s_i)

    def g_j(self):
        """
        Get j (column) coordinate value.

        @return:  the j (x) array coordinate of the point - float.
        """
        return self._j

    def s_j(self, jval):
        """
        Set j (column) coordinate value.

        @param  jval:  the j (x) array coordinate of the point.
        @type  jval:  number or string convertible to float.

        @return:  self.
        """
        self._j = jval

    # set property for j
    j = property(g_j, s_j)

    def grid2geogcoord(self, currGeoGrid):
        currPt_geogr_y = currGeoGrid.domain.g_trcorner().y - self.i * currGeoGrid.cellsize_y()
        currPt_geogr_x = currGeoGrid.domain.g_llcorner().x + self.j * currGeoGrid.cellsize_x()
        return Point(currPt_geogr_x, currPt_geogr_y)


class SpatialDomain(object):
    """
    Rectangular spatial domain class.

    """

    def __init__(self, pt_llc=None, pt_trc=None):
        """
        Class constructor.

        @param  pt_llc:  lower-left corner of the domain.
        @type  pt_llc:  Point.
        @param  pt_trc:  top-right corner of the domain.
        @type  pt_trc:  Point.

        @return:  SpatialDomain instance.
        """
        self._llcorner = pt_llc
        self._trcorner = pt_trc

    def g_llcorner(self):
        """
        Get lower-left corner of the spatial domain.

        @return:  lower-left corner of the spatial domain - Point.
        """
        return self._llcorner

    def g_trcorner(self):
        """
        Get top-right corner of the spatial domain.

        @return:  top-right corner of the spatial domain - Point.
        """
        return self._trcorner

    def g_xrange(self):
        """
        Get x range of spatial domain.

        @return:  x range - float.
        """
        return self._trcorner.x - self._llcorner.x

    def g_yrange(self):
        """
        Get y range of spatial domain.

        @return:  y range - float.
        """
        return self._trcorner.y - self._llcorner.y

    def g_zrange(self):
        """
        Get z range of spatial domain.

        @return:  z range - float.
        """
        return self._trcorner.z - self._llcorner.z

    def g_horiz_area(self):
        """
        Get horizontal area of spatial domain.

        @return:  area - float.
        """
        return self.g_xrange() * self.g_yrange()


class Grid(object):
    """
    Grid class.
    Stores and manages the most of data and processing.

    """

    def __init__(self, source_filename=None, grid_params=None, grid_data=None):
        """
        Grid class constructor.

        @param  source_filename:  name of file from which data and geo-parameters derive.
        @type  source_filename:  string.
        @param  grid_params:  the geo-parameters of the grid.
        @type  grid_params:  class GDALParameters.
        @param  grid_data:  the array storing the data.
        @type  grid_data:  2D np.array.

        @return:  self.
        """
        self._sourcename = source_filename

        if grid_params is not None:
            pt_llc = grid_params.llcorner()
            pt_trc = grid_params.trcorner()
        else:
            pt_llc = None
            pt_trc = None

        self._grid_domain = SpatialDomain(pt_llc, pt_trc)

        if grid_data is not None:
            self._grid_data = grid_data.copy()
        else:
            self._grid_data = None

    def s_domain(self, domain):
        """
        Set spatial domain.

        @param  domain:  Spatial domain to be attributed to the current Grid instance.
        @type  domain:  class SpatialDomain.

        @return: self
        """
        del self._grid_domain
        self._grid_domain = copy.deepcopy(domain)

    def g_domain(self):
        """
        Get spatial domain.

        @return: the spatial domain of the current Grid instance - class SpatialDomain.
        """
        return self._grid_domain

    def d_domain(self):
        """
        Delete current spatial domain of the Grid instance.

        @return: self
        """
        del self._grid_domain

    # set property for spatial domain
    domain = property(g_domain, s_domain, d_domain)

    def s_grid_data(self, data_array):
        """
        Set grid data array.

        @param data_array: numpy.array of data values.
        @param type: 2D numpy.array.

        @return: self.
        """
        if self._grid_data is not None:
            del self._grid_data

        self._grid_data = data_array.copy()

    def g_grid_data(self):
        """
        Get grid data array.

        @return: 2D numpy.array.
        """
        return self._grid_data

    def d_grid_data(self):
        """
        Delete grid data array.

        @return: self.
        """
        del self._grid_data

    data = property(g_grid_data, s_grid_data, d_grid_data)

    def row_num(self):
        """
        Get row number of the grid domain.

        @return: number of rows of data array - int.
        """
        return numpy.shape(self.data)[0]

    def col_num(self):
        """
        Get column number of the grid domain.

        @return: number of columns of data array - int.
        """
        return numpy.shape(self.data)[1]

    def cellsize_x(self):
        """
        Get the cell size of the grid in the x direction.

        @return: cell size in the x (j) direction - float.
        """
        return self.domain.g_xrange() / float(self.col_num())

    def cellsize_y(self):
        """
        Get the cell size of the grid in the y direction.

        @return: cell size in the y (-i) direction - float.
        """
        return self.domain.g_yrange() / float(self.row_num())

    def cellsize_h(self):
        """
        Get the mean horizontal cell size.

        @return: mean horizontal cell size - float.
        """
        return (self.cellsize_x() + self.cellsize_y()) / 2.0

    def geog2array_coord(self, curr_Pt):
        """
        Converts from geographic to raster (array) coordinates.

        @param curr_Pt: point whose geographical coordinates will be converted to raster (array) ones.
        @type curr_Pt: Point.

        @return: point coordinates in raster (array) frame - class ArrCoord.
        """
        currArrCoord_grid_i = (self.domain.g_trcorner().y - curr_Pt.y) / self.cellsize_y()
        currArrCoord_grid_j = (curr_Pt.x - self.domain.g_llcorner().x) / self.cellsize_x()

        return ArrCoord(currArrCoord_grid_i, currArrCoord_grid_j)

    def x(self):
        """
        Creates an array storing the geographical coordinates of the cell centers along the x axis.
        Direction is from left to right.

        @return: numpy.array, shape: 1 x col_num.
        """
        x_values = self.domain.g_llcorner().x + self.cellsize_x() * (0.5 + numpy.arange(self.col_num()))
        return x_values[numpy.newaxis, :]

    def y(self):
        """
        Creates an array storing the geographical coordinates of the cell centers along the y axis.
        Direction is from top to bottom.

        @return: numpy.array, shape: row_num x 1.
        """
        y_values = self.domain.g_trcorner().y - self.cellsize_y() * (0.5 + numpy.arange(self.row_num()))
        return y_values[:, numpy.newaxis]

    def grad_forward_y(self):
        """
        Return an array representing the forward gradient in the y direction (top-wards), with values scaled by cell size.

        @return: numpy.array, same shape as current Grid instance
        """
        gf = numpy.zeros(numpy.shape(self.data)) * numpy.NaN
        gf[1:, :] = self.data[:-1, :] - self.data[1:, :]

        return gf / float(self.cellsize_y())

    def grad_forward_x(self):
        """
        Return an array representing the forward gradient in the x direction (right-wards), with values scaled by cell size.

        @return: numpy.array, same shape as current Grid instance
        """
        gf = numpy.zeros(numpy.shape(self.data), ) * numpy.NaN
        gf[:, :-1] = self.data[:, 1:] - self.data[:, :-1]

        return gf / float(self.cellsize_x())

    def interpolate_bilinear(self, curr_Pt_array_coord):
        """
        Interpolate the z value at a point, given its array coordinates.
        Interpolation method: bilinear.

        @param curr_Pt_array_coord: array coordinates of the point for which the interpolation will be made.
        @type curr_Pt_array_coord: class ArrCoord.

        @return: interpolated z value - float.
        """
        currPt_cellcenter_i = curr_Pt_array_coord.i - 0.5
        currPt_cellcenter_j = curr_Pt_array_coord.j - 0.5

        assert currPt_cellcenter_i > 0, currPt_cellcenter_j > 0

        grid_val_00 = self.data[int(floor(currPt_cellcenter_i)), int(floor(currPt_cellcenter_j))]
        grid_val_01 = self.data[int(floor(currPt_cellcenter_i)), int(ceil(currPt_cellcenter_j))]
        grid_val_10 = self.data[int(ceil(currPt_cellcenter_i)), int(floor(currPt_cellcenter_j))]
        grid_val_11 = self.data[int(ceil(currPt_cellcenter_i)), int(ceil(currPt_cellcenter_j))]

        delta_i = currPt_cellcenter_i - floor(currPt_cellcenter_i)
        delta_j = currPt_cellcenter_j - floor(currPt_cellcenter_j)

        grid_val_y0 = grid_val_00 + (grid_val_10 - grid_val_00) * delta_i
        grid_val_y1 = grid_val_01 + (grid_val_11 - grid_val_01) * delta_i

        grid_val_interp = grid_val_y0 + (grid_val_y1 - grid_val_y0) * delta_j

        return grid_val_interp

    def intersection_with_surface(self, surf_type, srcPt, srcPlaneAttitude):
        """
        Calculates the intersections (as points) between DEM (self) and analytical surface.
        Currently it works only with planes as analytical surface cases.

        @param surf_type: type of considered surface (e.g., plane).
        @type surf_type: String.
        @param srcPt: point, expressed in geographical coordinates, that the plane must contain.
        @type srcPt: Point.
        @param srcPlaneAttitude: orientation of the surface (currently only planes).
        @type srcPlaneAttitude: class StructPlane.

        @return: tuple of six arrays
        """

        if surf_type == 'plane':

            # lambdas to compute the geographic coordinates (in x- and y-) of a cell center
            coord_grid2geog_x = lambda j: self.domain.g_llcorner().x + self.cellsize_x() * (0.5 + j)
            coord_grid2geog_y = lambda i: self.domain.g_trcorner().y - self.cellsize_y() * (0.5 + i)

            # arrays storing the geographical coordinates of the cell centers along the x- and y- axes
            x_values = self.x()
            y_values = self.y()

            ycoords_x, xcoords_y = numpy.broadcast_arrays(x_values, y_values)

            #### x-axis direction intersections

            # 2D array of DEM segment parameters
            x_dem_m = self.grad_forward_x()
            x_dem_q = self.data - x_values * x_dem_m

            # equation for the planar surface that, given (x,y), will be used to derive z
            plane_z = plane_from_geo(srcPt, srcPlaneAttitude)

            # 2D array of plane segment parameters
            x_plane_m = plane_x_coeff(srcPlaneAttitude)
            x_plane_q = array_from_function(self.row_num(), 1, lambda j: 0, coord_grid2geog_y, plane_z)

            # 2D array that defines denominator for intersections between local segments
            x_inters_denomin = numpy.where(x_dem_m != x_plane_m, x_dem_m - x_plane_m, numpy.NaN)
            coincident_x = numpy.where(x_dem_q != x_plane_q, numpy.NaN, ycoords_x)
            xcoords_x = numpy.where(x_dem_m != x_plane_m, (x_plane_q - x_dem_q) / x_inters_denomin, coincident_x)

            xcoords_x = numpy.where(xcoords_x < ycoords_x, numpy.NaN, xcoords_x)
            xcoords_x = numpy.where(xcoords_x >= ycoords_x + self.cellsize_x(), numpy.NaN, xcoords_x)

            #### y-axis direction intersections

            # 2D array of DEM segment parameters
            y_dem_m = self.grad_forward_y()
            y_dem_q = self.data - y_values * y_dem_m

            # 2D array of plane segment parameters
            y_plane_m = plane_y_coeff(srcPlaneAttitude)
            y_plane_q = array_from_function(1, self.col_num(), coord_grid2geog_x, lambda i: 0, plane_z)

            # 2D array that defines denominator for intersections between local segments
            y_inters_denomin = numpy.where(y_dem_m != y_plane_m, y_dem_m - y_plane_m, numpy.NaN)
            coincident_y = numpy.where(y_dem_q != y_plane_q, numpy.NaN, xcoords_y)

            ycoords_y = numpy.where(y_dem_m != y_plane_m, (y_plane_q - y_dem_q) / y_inters_denomin, coincident_y)

            # filter out cases where intersection is outside cell range
            ycoords_y = numpy.where(ycoords_y < xcoords_y, numpy.NaN, ycoords_y)
            ycoords_y = numpy.where(ycoords_y >= xcoords_y + self.cellsize_y(), numpy.NaN, ycoords_y)

            for i in range(xcoords_x.shape[0]):
                for j in range(xcoords_x.shape[1]):
                    if abs(xcoords_x[i, j] - ycoords_x[i, j]) < 1.0e-5 and abs(
                            ycoords_y[i, j] - xcoords_y[i, j]) < 1.0e-5:
                        ycoords_y[i, j] = numpy.NaN

            return xcoords_x, xcoords_y, ycoords_x, ycoords_y


class AnaliticSurfaceCalcException(Exception):
    """
    Exception for Analytical Surface calculation.
    """

    pass


if __name__ == "__main__":

    import doctest
    doctest.testmod()
