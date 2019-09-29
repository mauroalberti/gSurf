
from typing import List, Tuple, Optional

from geopandas import GeoDataFrame

from pygsf.spatial.vectorial.geometries import Point, Plane

from .base import GeorefAttitude


def extract_georeferenced_attitudes(
        geodataframe: GeoDataFrame,
        dip_dir_fldnm: str,
        dip_ang_fldnm: str,
        id_fldnm: Optional[str] = None) -> List[Tuple[Point, Plane]]:
    """
    Extracts the georeferenced _attitudes from a geopandas GeoDataFrame instance representing point records.

    :param geodataframe: the source geodataframe.
    :type geodataframe: GeoDataFrame.
    :param dip_dir_fldnm: the name of the dip direction field in the geodataframe.
    :type dip_dir_fldnm: basestring.
    :param dip_ang_fldnm: the name of the dip angle field in the geodataframe.
    :type dip_ang_fldnm: basestring.
    :return: a collection of Point and Plane values, one for each source record.
    :rtype: List[Tuple[pygsf.spatial.vectorial.geometries.Point, pygsf.spatial.vectorial.geometries.Plane]]
    """

    crs = geodataframe.crs['init']
    if crs.startswith("epsg"):
        epsg_cd = int(crs.split(":")[-1])
    else:
        epsg_cd = -1

    attitudes = []

    for ndx, row in geodataframe.iterrows():

        pt = row['geometry']
        x, y = pt.x, pt.y

        if id_fldnm:
            dip_dir, dip_ang, id = row[dip_dir_fldnm], row[dip_ang_fldnm], row[id_fldnm]
        else:
            dip_dir, dip_ang, id = row[dip_dir_fldnm], row[dip_ang_fldnm], ndx+1

        attitudes.append(GeorefAttitude(id, Point(x, y, epsg_cd=epsg_cd), Plane(dip_dir, dip_ang)))

    return attitudes

