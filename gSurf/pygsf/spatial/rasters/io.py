# -*- coding: utf-8 -*-

from typing import Any, Tuple, Dict, Optional, Union
import numbers

import os

from math import isfinite

import numpy as np

import gdal

from pygsf.mathematics.scalars import areClose
from pygsf.spatial.rasters.geoarray import GeoArray
from pygsf.spatial.rasters.geotransform import GeoTransform

from .defaults import GRID_NULL_VALUE


class RasterIOException(Exception):
    """
    Exception for rasters IO.
    """
    pass


def read_raster(file_ref: Any) -> Tuple[gdal.Dataset, Optional[GeoTransform], int, str]:
    """
    Read a raster layer.

    :param file_ref: the reference to the raster
    :type file_ref: Any
    :return: the dataset, its geotransform, the number of bands, the projection.
    :rtype: tuple made up by a gdal.Dataset instance, an optional Geotransform object, and int and a string.
    :raises: RasterIOException

    Examples:
    """

    # open raster file and check operation success

    dataset = gdal.Open(file_ref, gdal.GA_ReadOnly)
    if not dataset:
        raise RasterIOException("No input data open")

    # get raster descriptive infos

    gt = dataset.GetGeoTransform()
    if gt:
        geotransform = GeoTransform.fromGdalGt(gt)
    else:
        geotransform = None

    num_bands = dataset.RasterCount

    projection = dataset.GetProjection()

    return dataset, geotransform, num_bands, projection


def read_band(dataset: gdal.Dataset, bnd_ndx: int = 1) -> Tuple[dict, 'np.array']:
    """
    Read data and metadata of a rasters band based on GDAL.

    :param dataset: the source raster dataset
    :type dataset: gdal.Dataset
    :param bnd_ndx: the index of the band (starts from 1)
    :type bnd_ndx: int
    :return: the band parameters and the data values
    :rtype: dict of data parameters and values as a numpy.array
    :raises: RasterIOException

    Examples:

    """

    band = dataset.GetRasterBand(bnd_ndx)
    data_type = gdal.GetDataTypeName(band.DataType)

    unit_type = band.GetUnitType()

    stats = band.GetStatistics(False, False)
    if stats is None:
        dStats = dict(
            min=None,
            max=None,
            mean=None,
            std_dev=None)
    else:
        dStats = dict(
            min=stats[0],
            max=stats[1],
            mean=stats[2],
            std_dev=stats[3])

    noDataVal = band.GetNoDataValue()

    nOverviews = band.GetOverviewCount()

    colorTable = band.GetRasterColorTable()

    if colorTable:
        nColTableEntries = colorTable.GetCount()
    else:
        nColTableEntries = 0

    # read data from band

    grid_values = band.ReadAsArray()
    if grid_values is None:
        raise RasterIOException("Unable to read data from rasters")

    # if nodatavalue exists, set null values to NaN in numpy array

    if noDataVal is not None and isfinite(noDataVal):
        grid_values = np.where(np.isclose(grid_values, noDataVal), np.NaN, grid_values)

    band_params = dict(
        dataType=data_type,
        unitType=unit_type,
        stats=dStats,
        numOverviews=nOverviews,
        numColorTableEntries=nColTableEntries)

    return band_params, grid_values


def try_read_raster_band(raster_source: str, bnd_ndx: int=1) -> Tuple[bool, Union[str, Tuple[GeoTransform, str, Dict, 'np.array']]]:
    """
    Deprecated. Use "read_raster_band" instead.

    :param raster_source:
    :param bnd_ndx:
    :return:
    """

    # get raster parameters and data
    try:
        dataset, geotransform, num_bands, projection = read_raster(raster_source)
    except (IOError, TypeError, RasterIOException) as err:
        return False, "Exception with reading {}: {}".format(raster_source, err)

    band_params, data = read_band(dataset, bnd_ndx)

    return True, (geotransform, projection, band_params, data)


def read_raster_band(raster_source: str, bnd_ndx: int = 1, epsg_cd: int = -1) -> Optional[GeoArray]:
    """
    Read parameters and values of a raster band.
    Since it is not immediate to get the EPSG code of the input raster,
    the user is advised to provide it directly in the function call.


    :param raster_source: the raster path.
    :param bnd_ndx: the optional band index.
    :param epsg_cd: the EPSG code of the raster.
    :return: the band as a geoarray.
    :rtype: Optional GeoArray.
    """

    try:

        dataset, geotransform, num_bands, projection = read_raster(raster_source)
        band_params, data = read_band(dataset, bnd_ndx)
        ga = GeoArray(
            inGeotransform=geotransform,
            epsg_cd=epsg_cd,
            inLevels=[data]
        )

        return ga

    except:

        return None


def try_write_esrigrid(geoarray: GeoArray, outgrid_flpth: str, esri_nullvalue: numbers.Real=GRID_NULL_VALUE, level_ndx: int=0) -> Tuple[bool, str]:
    """
    Writes ESRI ascii grid.
    
    :param geoarray: 
    :param outgrid_flpth: 
    :param esri_nullvalue: 
    :param level_ndx: index of the level array to write.
    :type level_ndx: int.
    :return: success and descriptive message
    :rtype: tuple made up by a boolean and a string
    """
    
    outgrid_flpth = str(outgrid_flpth)

    out_fldr, out_flnm = os.path.split(outgrid_flpth)
    if not out_flnm.lower().endswith('.asc'):
        out_flnm += '.asc'

    outgrid_flpth = os.path.join(
        out_fldr,
        out_flnm
    )

    # checking existence of output slope grid

    if os.path.exists(outgrid_flpth):
        return False, "Output grid '{}' already exists".format(outgrid_flpth)

    try:
        outputgrid = open(outgrid_flpth, 'w')  # create the output ascii file
    except Exception:
        return False, "Unable to create output grid '{}'".format(outgrid_flpth)

    if outputgrid is None:
        return False, "Unable to create output grid '{}'".format(outgrid_flpth)

    if geoarray.has_rotation:
        return False, "Grid has axes rotations defined"

    cell_size_x = geoarray.src_cellsize_j
    cell_size_y = geoarray.src_cellsize_i

    if not areClose(cell_size_x, cell_size_y):
        return False, "Cell sizes in the x- and y- directions are not similar"

    arr = geoarray.level(level_ndx)
    if arr is None:
        return False, "Array with index {} does not exist".format(level_ndx)

    num_rows, num_cols = arr.shape
    llc_x, llc_y = geoarray.level_llc(level_ndx)

    # writes header of grid ascii file

    outputgrid.write("NCOLS %d\n" % num_cols)
    outputgrid.write("NROWS %d\n" % num_rows)
    outputgrid.write("XLLCORNER %.8f\n" % llc_x)
    outputgrid.write("YLLCORNER %.8f\n" % llc_y)
    outputgrid.write("CELLSIZE %.8f\n" % cell_size_x)
    outputgrid.write("NODATA_VALUE %.8f\n" % esri_nullvalue)

    esrigrid_outvalues = np.where(np.isnan(arr), esri_nullvalue, arr)

    # output of results

    for i in range(0, num_rows):
        for j in range(0, num_cols):
            outputgrid.write("%.8f " % (esrigrid_outvalues[i, j]))
        outputgrid.write("\n")

    outputgrid.close()

    return True, "Data saved in {}".format(outgrid_flpth)


if __name__ == "__main__":

    import doctest
    doctest.testmod()

