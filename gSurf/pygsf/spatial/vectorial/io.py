# -*- coding: utf-8 -*-

from typing import Dict, Tuple, Union, List, Optional

import numbers

import os

from osgeo import ogr, osr

from pygsf.utils.types import *
from pygsf.spatial.vectorial.geometries import Line, MultiLine, Point


class OGRIOException(Exception):
    """
    Exception for OGR IO parameters.
    """
    pass


ogr_simpleline_types = [
    ogr.wkbLineString,
    ogr.wkbLineString25D,
    ogr.wkbLineStringM,
    ogr.wkbLineStringZM
]

ogr_multiline_types = [
    ogr.wkbMultiLineString,
    ogr.wkbMultiLineString25D,
    ogr.wkbMultiLineStringM,
    ogr.wkbMultiLineStringZM
]


def try_open_shapefile(path: str) -> Tuple[bool, Union["OGRLayer", str]]:

    dataSource = ogr.Open(path)

    if dataSource is None:
        return False, "Unable to open shapefile in provided path"

    shapelayer = dataSource.GetLayer()

    return True, shapelayer


def try_read_line_shapefile(
        shp_path: str,
        flds: Optional[List[str]] = None
    ) -> Tuple[bool, Union[str, List[Tuple[list, tuple]]]]:
    """
    Read results geometries from a line shapefile using ogr.
    TODO: it could read also other formats, but it has to be checked.

    :param shp_path: line shapefile path.
    :type shp_path: str.
    :param flds: the fields to extract values from.
    :type flds: Optional[List[str]].
    :return: success status and (error message or results).
    :rtype: Tuple[bool, Union[str, List[Tuple[list, tuple]]]].
    """

    # check input path

    check_type(shp_path, "Shapefile path", str)
    if shp_path == '':
        return False, "Input shapefile path should not be empty"
    if not os.path.exists(shp_path):
        return False, "Input shapefile path does not exist"

    # open input vector layer

    ds = ogr.Open(shp_path, 0)

    if ds is None:
        return False, "Input shapefile path not read"

    # get internal layer

    lyr = ds.GetLayer()

    # get projection

    srs = lyr.GetSpatialRef()
    srs.AutoIdentifyEPSG()
    authority = srs.GetAuthorityName(None)
    if authority.upper() == "EPSG":
        epsg_cd = int(srs.GetAuthorityCode(None))
    else:
        epsg_cd = -1

    # initialize list storing results

    results = []

    # loop in layer features

    for feat in lyr:

        # get attributes

        if flds:
            feat_attributes = tuple(map(lambda fld_nm: feat.GetField(fld_nm), flds))
        else:
            feat_attributes = ()

        # get geometries

        # feat_geometries = []

        curr_geom = feat.GetGeometryRef()

        if curr_geom is None:
            del ds
            return False, "Input shapefile path not read"

        geometry_type = curr_geom.GetGeometryType()
        if geometry_type in ogr_simpleline_types:
            geom_type = "simpleline"
        elif geometry_type in ogr_multiline_types:
            geom_type = "multiline"
        else:
            del ds
            return False, "Geometry type is {}, line expected".format(geom_type)

        if geom_type == "simpleline":

            line = Line(epsg_cd=epsg_cd)

            for i in range(curr_geom.GetPointCount()):
                x, y, z = curr_geom.GetX(i), curr_geom.GetY(i), curr_geom.GetZ(i)

                line.add_pt(Point(x, y, z, epsg_cd=epsg_cd))

            feat_geometries = line

        else:  # multiline case

            multiline = MultiLine(epsg_cd=epsg_cd)

            for line_geom in curr_geom:

                line = Line(epsg_cd=epsg_cd)

                for i in range(line_geom.GetPointCount()):
                    x, y, z = line_geom.GetX(i), line_geom.GetY(i), line_geom.GetZ(i)

                    line.add_pt(Point(x, y, z, epsg_cd=epsg_cd))

                multiline.add_line(line)

            feat_geometries = multiline

        results.append((feat_geometries, feat_attributes))

    del ds

    return True, results


def read_linestring_geometries(line_shp_path: str) -> Optional[MultiLine]:
    """
    Deprecated. Use 'read_lines_geometries'.

    Read linestring geometries from a shapefile using ogr.
    The geometry type of the input shapefile must be LineString (MultiLineString is not currently managed).

    It returns a MultiLine instance.

    :param line_shp_path:  parameter to check.
    :type line_shp_path:  QString or string
    :return: the result of data reading
    :rtype: MultiLine.
    """

    # check input path

    if line_shp_path is None or line_shp_path == '':
        return None

    # open input vector layer

    shape_driver = ogr.GetDriverByName("ESRI Shapefile")

    datasource = shape_driver.Open(str(line_shp_path), 0)

    # layer not read

    if datasource is None:
        return None

    # get internal layer

    layer = datasource.GetLayer()

    # get projection

    srs = layer.GetSpatialRef()
    srs.AutoIdentifyEPSG()
    authority = srs.GetAuthorityName(None)
    if authority == "EPSG":
        epsg_cd = int(srs.GetAuthorityCode(None))
    else:
        epsg_cd = -1

    # initialize list storing vertex coordinates of lines

    lines = []

    # start reading layer features

    feature = layer.GetNextFeature()

    # loop in layer features

    while feature:

        geometry = feature.GetGeometryRef()

        if geometry is None:
            datasource.Destroy()
            return None

        geometry_type = geometry.GetGeometryType()
        if geometry_type not in ogr_simpleline_types:
            datasource.Destroy()
            return None

        line = Line(epsg_cd=epsg_cd)

        for i in range(geometry.GetPointCount()):

            x, y, z = geometry.GetX(i), geometry.GetY(i), geometry.GetZ(i)

            line.add_pt(Point(x, y, z, epsg_cd=epsg_cd))

        feature.Destroy()

        lines.append(line)

        feature = layer.GetNextFeature()

    datasource.Destroy()

    multiline = MultiLine(
        lines=lines,
        epsg_cd=epsg_cd
    )

    return multiline


def parse_ogr_type(ogr_type_str: str) -> 'ogr.OGRFieldType':
    """
    Parse the provided textual field type to return an actual OGRFieldType.

    :param ogr_type_str: the string referring to the ogr field type.
    :type ogr_type_str: str.
    :return: the actural ogr type.
    :rtype: OGRFieldType.
    :raise: Exception.
    """

    if ogr_type_str.endswith("OFTInteger"):
        return ogr.OFTInteger
    elif ogr_type_str.endswith("OFTIntegerList"):
        return ogr.OFTIntegerList
    elif ogr_type_str.endswith("OFTReal"):
        return ogr.OFTReal
    elif ogr_type_str.endswith("OFTRealList"):
        return ogr.OFTRealList
    elif ogr_type_str.endswith("OFTString"):
        return ogr.OFTString
    elif ogr_type_str.endswith("OFTStringList"):
        return ogr.OFTStringList
    elif ogr_type_str.endswith("OFTBinary"):
        return ogr.OFTBinary
    elif ogr_type_str.endswith("OFTDate"):
        return ogr.OFTDate
    elif ogr_type_str.endswith("OFTTime"):
        return ogr.OFTTime
    elif ogr_type_str.endswith("OFTDateTime"):
        return ogr.OFTDateTime
    elif ogr_type_str.endswith("OFTInteger64"):
        return ogr.OFTInteger64
    elif ogr_type_str.endswith("OFTInteger64List"):
        return ogr.OFTInteger64List
    else:
        raise Exception("Debug: not recognized ogr type")


def shapefile_create_def_field(field_def):
    """

    :param field_def:
    :return:
    """

    name = field_def['name']
    ogr_type = parse_ogr_type(field_def['ogr_type'])

    fieldDef = ogr.FieldDefn(name, ogr_type)
    if ogr_type == ogr.OFTString:
        fieldDef.SetWidth(int(field_def['width']))

    return fieldDef


def shapefile_create(path, geom_type, fields_dict_list, crs=None):
    """
    crs_prj4: projection in Proj4 text format
    geom_type = OGRwkbGeometryType: ogr.wkbPoint, ....
    list of:
        field dict: 'name',
                    'type': ogr.OFTString,
                            ogr.wkbLineString,
                            ogr.wkbLinearRing,
                            ogr.wkbPolygon,

                    'width',
    """

    driver = ogr.GetDriverByName("ESRI Shapefile")

    outShapefile = driver.CreateDataSource(str(path))
    if outShapefile is None:
        raise OGRIOException('Unable to save shapefile in provided path')

    if crs is not None:
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromProj4(crs)
        outShapelayer = outShapefile.CreateLayer("layer", spatial_reference, geom_type)
    else:
        outShapelayer = outShapefile.CreateLayer("layer", None, geom_type)

    if not outShapelayer:
        return None, None

    for field_def_params in fields_dict_list:
        field_def = shapefile_create_def_field(field_def_params)
        outShapelayer.CreateField(field_def)

    return outShapefile, outShapelayer


def try_write_pt_shapefile(point_layer, geoms: List[Tuple[numbers.Real, numbers.Real, numbers.Real]], field_names: List[str], attrs: List[Tuple]) -> Tuple[bool, str]:
    """
    Add point records in an existing shapefile, filling attribute values.

    :param point_layer: the existing shapefile layer in which to write.
    :param geoms: the geometric coordinates of the points.
    :type geoms: List of x, y, and z coordinates.
    :param field_names: the field names of the attribute table.
    :type field_names: list of strings.
    :param attrs: the values for each record.
    :type attrs: list of tuple.
    :return: success status and related messages.
    :rtype: tuple of a boolean and a string.
    """

    len_geoms = len(geoms)
    len_attrs = len(attrs)

    if len_geoms != len_attrs:
        return False, "Function error: geometries are {} while attributes are {}".format(len_geoms, len_attrs)

    if len_geoms == 0:
        return True, "No values to be added in shapefile"

    try:

        outshape_featdef = point_layer.GetLayerDefn()

        for ndx_rec in range(len_geoms):

            # pre-processing for new feature in output layer

            curr_Pt_geom = ogr.Geometry(ogr.wkbPoint25D)
            curr_Pt_geom.AddPoint(*geoms[ndx_rec])

            # create a new feature

            curr_pt_shape = ogr.Feature(outshape_featdef)
            curr_pt_shape.SetGeometry(curr_Pt_geom)

            rec_attrs = attrs[ndx_rec]

            for ndx_fld, fld_nm in enumerate(field_names):

                curr_pt_shape.SetField(fld_nm, rec_attrs[ndx_fld])

            # add the feature to the output layer
            point_layer.CreateFeature(curr_pt_shape)

            # destroy no longer used objects
            curr_Pt_geom.Destroy()
            curr_pt_shape.Destroy()

        del outshape_featdef

        return True, ""

    except Exception as e:

        return False, "Exception: {}".format(e)


def try_write_point_shapefile(path: str, field_names: List[str], values: List[Tuple], ndx_x_val: int) -> Tuple[bool, str]:
    """
    Note: candidate for future deprecation.

    Add point records in an existing shapefile, filling attribute values.
    The point coordinates, i.e. x, y, z start at ndx_x_val index (index is zero-based) and are
    assumed to be sequential in order (i.e., 0, 1, 2 or 3, 4, 5).

    :param path: the path of the existing shapefile in which to write.
    :type path: string.
    :param field_names: the field names of the attribute table.
    :type field_names: list of strings.
    :param values: the values for each record.
    :type values: list of tuple.
    :param ndx_x_val: the index of the x coordinate. Y and z should follow.
    :type ndx_x_val: int.
    :return: success status and related messages.
    :rtype: tuple of a boolean and a string.
    """

    success = False
    msg = ""

    try:

        dataSource = ogr.Open(path, 1)

        if dataSource is None:
            return False, "Unable to open shapefile in provided path"

        point_layer = dataSource.GetLayer()

        outshape_featdef = point_layer.GetLayerDefn()

        for pt_vals in values:

            # pre-processing for new feature in output layer
            curr_Pt_geom = ogr.Geometry(ogr.wkbPoint)
            curr_Pt_geom.AddPoint(pt_vals[ndx_x_val], pt_vals[ndx_x_val+1], pt_vals[ndx_x_val+2])

            # create a new feature
            curr_pt_shape = ogr.Feature(outshape_featdef)
            curr_pt_shape.SetGeometry(curr_Pt_geom)

            for ndx, fld_nm in enumerate(field_names):

                curr_pt_shape.SetField(fld_nm, pt_vals[ndx])

            # add the feature to the output layer
            point_layer.CreateFeature(curr_pt_shape)

            # destroy no longer used objects
            curr_Pt_geom.Destroy()
            curr_pt_shape.Destroy()

        del outshape_featdef
        del point_layer
        del dataSource

        success = True

    except Exception as e:

        msg = e

    finally:

        return success, msg


def try_write_line_shapefile(path: str, field_names: List[str], values: Dict) -> Tuple[bool, str]:
    """
    Add point records in an existing shapefile, filling attribute values.


    :param path: the path of the existing shapefile in which to write.
    :type path: string.
    :param field_names: the field names of the attribute table.
    :type field_names: list of strings.
    :param values: the values for each record.
    :type values: dict with values made up by two dictionaries.
    :return: success status and related messages.
    :rtype: tuple of a boolean and a string.
    """

    success = False
    msg = ""

    try:

        dataSource = ogr.Open(path, 1)

        if dataSource is None:
            return False, "Unable to open shapefile in provided path"

        line_layer = dataSource.GetLayer()

        outshape_featdef = line_layer.GetLayerDefn()

        for curr_id in sorted(values.keys()):

            # pre-processing for new feature in output layer
            line_geom = ogr.Geometry(ogr.wkbLineString)

            for id_xyz in values[curr_id]["pts"]:
                x, y, z = id_xyz
                line_geom.AddPoint(x, y, z)

            # create a new feature
            line_shape = ogr.Feature(outshape_featdef)
            line_shape.SetGeometry(line_geom)

            for ndx, fld_nm in enumerate(field_names):

                line_shape.SetField(fld_nm, values[curr_id]["vals"][ndx])

            # add the feature to the output layer
            line_layer.CreateFeature(line_shape)

            # destroy no longer used objects
            line_geom.Destroy()
            line_shape.Destroy()

        del outshape_featdef
        del line_layer
        del dataSource

        success = True

    except Exception as e:

        msg = str(e)

    finally:

        return success, msg


def ogr_get_solution_shapefile(path, fields_dict_list):
    """

    :param path:
    :param fields_dict_list:
    :return:
    """

    driver = ogr.GetDriverByName("ESRI Shapefile")

    dataSource = driver.Open(str(path), 0)

    if dataSource is None:
        raise OGRIOException("Unable to open shapefile in provided path")

    point_shapelayer = dataSource.GetLayer()

    prev_solution_list = []
    in_point = point_shapelayer.GetNextFeature()
    while in_point:
        rec_id = int(in_point.GetField('id'))
        x = in_point.GetField('x')
        y = in_point.GetField('y')
        z = in_point.GetField('z')
        dip_dir = in_point.GetField('dip_dir')
        dip_ang = in_point.GetField('dip_ang')
        descript = in_point.GetField('descript')
        prev_solution_list.append([rec_id, x, y, z, dip_dir, dip_ang, descript])
        in_point.Destroy()
        in_point = point_shapelayer.GetNextFeature()

    dataSource.Destroy()

    if os.path.exists(path):
        driver.DeleteDataSource(str(path))

    outShapefile, outShapelayer = shapefile_create(path, ogr.wkbPoint25D, fields_dict_list, crs=None)
    return outShapefile, outShapelayer, prev_solution_list