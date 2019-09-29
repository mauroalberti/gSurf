# -*- coding: utf-8 -*-

import numpy as np

try:
    from osgeo import ogr, gdal
except ImportError:
    import ogr
    import gdal

from gsf_OLD.grids import *


class GDALParameters(object):
    """
    Manage GDAL parameters from rasters.

    """

    # class constructor
    def __init__(self):
        """
        Class constructor.

        @return:  generic-case GDAL parameters.
        """
        self._nodatavalue = None
        self._topleftX = None
        self._topleftY = None
        self._pixsizeEW = None
        self._pixsizeNS = None
        self._rows = None
        self._cols = None
        self._rotation_GT_2 = 0.0
        self._rotation_GT_4 = 0.0

    def s_noDataValue(self, nodataval):
        """
        Set raster no data value.

        @param  nodataval:  the raster no-data value.
        @type  nodataval:  number or string convertible to float.

        @return:  self.
        """
        self._nodatavalue = float(nodataval)

    def g_noDataValue(self):
        """
        Get raster no-data value.

        @return:  no-data value - float.
        """
        return self._nodatavalue

    # set property for no-data value
    noDataValue = property(g_noDataValue, s_noDataValue)

    def s_topLeftX(self, topleftX):
        """
        Set top-left corner x value of the raster.

        @param  topleftX:  the top-left corner x value, according to GDAL convention.
        @type  topleftX:  number or string convertible to float.

        @return:  self.
        """
        self._topleftX = float(topleftX)

    def g_topLeftX(self):
        """
        Get top-left corner x value of the raster.

        @return:  the top-left corner x value, according to GDAL convention - float.
        """
        return self._topleftX

    # set property for topleftX
    topLeftX = property(g_topLeftX, s_topLeftX)

    def s_topLeftY(self, topleftY):
        """
        Set top-left corner y value of the raster.

        @param  topleftY:  the top-left corner y value, according to GDAL convention.
        @type  topleftY:  number or string convertible to float.

        @return:  self.
        """
        self._topleftY = float(topleftY)

    def g_topLeftY(self):
        """
        Get top-left corner y value of the raster.

        @return:  the top-left corner y value, according to GDAL convention - float.
        """
        return self._topleftY

    # set property for topleftY
    topLeftY = property(g_topLeftY, s_topLeftY)

    def s_pixSizeEW(self, pixsizeEW):
        """
        Set East-West size of the raster cell.

        @param  pixsizeEW:  the top-left y value, according to GDAL convention.
        @type  pixsizeEW:  number or string convertible to float.

        @return:  self.
        """
        self._pixsizeEW = float(pixsizeEW)

    def g_pixSizeEW(self):
        """
        Get East-West size of the raster cell.

        @return:  the East-West size of the raster cell - float.
        """
        return self._pixsizeEW

    # set property for topleftY
    pixSizeEW = property(g_pixSizeEW, s_pixSizeEW)

    # pixsizeNS

    def s_pixSizeNS(self, pixsizeNS):
        """
        Set North-South size of the raster cell.

        @param  pixsizeNS:  the North-South size of the raster cell.
        @type  pixsizeNS:  number or string convertible to float.

        @return:  self.
        """
        self._pixsizeNS = float(pixsizeNS)

    def g_pixSizeNS(self):
        """
        Get North-South size of the raster cell.

        @return:  the North-South size of the raster cell - float.
        """
        return self._pixsizeNS

    # set property for topleftY
    pixSizeNS = property(g_pixSizeNS, s_pixSizeNS)

    def s_rows(self, rows):
        """
        Set row number.

        @param  rows:  the raster row number.
        @type  rows:  number or string convertible to int.

        @return:  self.
        """
        self._rows = int(rows)

    def g_rows(self):
        """
        Get row number.

        @return:  the raster row number - int.
        """
        return self._rows

    # set property for rows
    rows = property(g_rows, s_rows)

    def s_cols(self, cols):
        """
        Set column number.

        @param  cols:  the raster column number.
        @type  cols:  number or string convertible to int.

        @return:  self.
        """
        self._cols = int(cols)

    def g_cols(self):
        """
        Get column number.

        @return:  the raster column number - int.
        """
        return self._cols

    # set property for cols
    cols = property(g_cols, s_cols)

    def s_rotation_GT_2(self, rotation_GT_2):
        """
        Set rotation GT(2) (see GDAL documentation).

        @param  rotation_GT_2:  the raster rotation value GT(2).
        @type  rotation_GT_2:  number or string convertible to float.

        @return:  self.
        """
        self._rotation_GT_2 = float(rotation_GT_2)

    def g_rotation_GT_2(self):
        """
        Get rotation GT(2) (see GDAL documentation).

        @return:  the raster rotation value GT(2). - float.
        """
        return self._rotation_GT_2

    # set property for rotation_GT_2
    rotGT2 = property(g_rotation_GT_2, s_rotation_GT_2)

    def s_rotation_GT_4(self, rotation_GT_4):
        """
        Set rotation GT(4) (see GDAL documentation)

        @param  rotation_GT_4:  the raster rotation value GT(4).
        @type  rotation_GT_4:  number or string convertible to float.

        @return:  self.
        """
        self._rotation_GT_4 = float(rotation_GT_4)

    def g_rotation_GT_4(self):
        """
        Get rotation GT(4) (see GDAL documentation).

        @return:  the raster rotation value GT(4) - float.
        """
        return self._rotation_GT_4

    # set property for rotation_GT_4
    rotGT4 = property(g_rotation_GT_4, s_rotation_GT_4)

    def check_params(self, tolerance=1e-06):
        """
        Check absence of axis rotations or pixel size differences in the raster band.

        @param  tolerance:  the maximum threshold for both pixel N-S and E-W difference, or axis rotations.
        @type  tolerance:  float.

        @return:  None when successful, RasterParametersErrors when pixel differences or axis rotations.

        @raise: RasterParametersErrors - raster geometry incompatible with this module (i.e. different cell sizes or axis rotations).
        """
        # check if pixel size can be considered the same in the two axis directions
        if abs(abs(self._pixsizeEW) - abs(self._pixsizeNS)) / abs(self._pixsizeNS) > tolerance:
            raise RasterParametersErrors('Pixel sizes in x and y directions are different in raster')

        # check for the absence of axis rotations
        if abs(self._rotation_GT_2) > tolerance or abs(self._rotation_GT_4) > tolerance:
            raise RasterParametersErrors('There should be no axis rotation in raster')

        return

    def llcorner(self):
        """
        Creates a point at the lower-left corner of the raster.

        @return:  new Point instance.
        """
        return Point(self.topLeftX, self.topLeftY - abs(self.pixSizeNS) * self.rows)

    def trcorner(self):
        """
        Create a point at the top-right corner of the raster.

        @return:  new Point instance.
        """
        return Point(self.topLeftX + abs(self.pixSizeEW) * self.cols, self.topLeftY)

    def geo_equiv(self, other, tolerance=1.0e-6):
        """
        Checks if two rasters are geographically equivalent.

        @param  other:  a grid to be compared with self.
        @type  other:  Grid instance.
        @param  tolerance:  the maximum threshold for pixel sizes, topLeftX or topLeftY differences.
        @type  tolerance:  float.

        @return:  Boolean.
        """
        if 2 * (self.topLeftX - other.topLeftX) / (self.topLeftX + other.topLeftX) > tolerance or \
                2 * (self.topLeftY - other.topLeftY) / (self.topLeftY + other.topLeftY) > tolerance or \
                2 * (abs(self.pixSizeEW) - abs(other.pixSizeEW)) / (
                abs(self.pixSizeEW) + abs(other.pixSizeEW)) > tolerance or \
                2 * (abs(self.pixSizeNS) - abs(other.pixSizeNS)) / (
                abs(self.pixSizeNS) + abs(other.pixSizeNS)) > tolerance or \
                self.rows != other.rows or self.cols != other.cols or self.projection != other.projection:
            return False
        else:
            return True


def read_dem(in_dem_fn):
    """
    Read input DEM file.

    @param  in_dem_fn: name of file to be read.
    @type  in_dem_fn:  string
    """

    # try reading DEM data
    try:
        dem_params, dem_array = read_raster_band(in_dem_fn)
        dem_params.check_params()
    except (IOError, TypeError, RasterParametersErrors) as e:
        raise IOError("Unable to read data from raster: {}".format(e.message))

    # create current grid
    return Grid(in_dem_fn, dem_params, dem_array)


def read_raster_band(raster_name):
    """
    Read an input raster band, based on GDAL module.

    @param raster_name: name of the raster to be read.
    @type raster_name: QString.

    @return: tuple of a GDALParameters instance and a 2D numpy.array instance.

    @raise IOError: unable to open or read data from raster.
    @raise TypeError: more than one band in raster.
    """

    # GDAL register
    # gdal.AllRegister

    # open raster file and check operation success

    # DEBUG
    print("raster_name is {}".format(raster_name))

    raster_data = gdal.Open(str(raster_name), gdal.GA_ReadOnly)
    if raster_data is None:
        raise IOError('Unable to open raster band')

        # initialize DEM parameters
    raster_params = GDALParameters()

    # get driver type for current raster
    raster_params.driverShortName = raster_data.GetDriver().ShortName

    # get current raster projection
    raster_params.projection = raster_data.GetProjection()

    # get row and column numbers
    raster_params.rows = raster_data.RasterYSize
    raster_params.cols = raster_data.RasterXSize

    # get and check number of raster bands - it must be one
    raster_bands = raster_data.RasterCount
    if raster_bands > 1:
        raise TypeError('More than one raster band in raster')

        # set critical grid values from geotransform array
    raster_params.topLeftX = raster_data.GetGeoTransform()[0]
    raster_params.pixSizeEW = raster_data.GetGeoTransform()[1]
    raster_params.rotGT2 = raster_data.GetGeoTransform()[2]
    raster_params.topLeftY = raster_data.GetGeoTransform()[3]
    raster_params.rotGT4 = raster_data.GetGeoTransform()[4]
    raster_params.pixSizeNS = raster_data.GetGeoTransform()[5]

    # get single band
    band = raster_data.GetRasterBand(1)

    # get no data value for current band
    try:
        raster_params.noDataValue = band.GetNoDataValue()
    except:
        pass
    # read data from band
    grid_values = band.ReadAsArray(0, 0, raster_params.cols, raster_params.rows)
    if grid_values is None:
        raise IOError('Unable to read data from raster')

    # transform data into numpy array
    data = np.asarray(grid_values)

    # if nodatavalue exists, set null values to NaN in numpy array
    if raster_params.noDataValue is not None:
        data = np.where(abs(data - raster_params.noDataValue) > 1e-05, data, np.NaN)

    return raster_params, data


class RasterParametersErrors(Exception):
    """
    Exception for raster parameters.
    """
    pass