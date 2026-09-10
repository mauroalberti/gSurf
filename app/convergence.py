"""
Grid north against true north.

An attitude is measured from true north and the DEM is on the projection's
grid: the two do not agree, and the gap is neither constant nor negligible.
Every tool that reads a bearing off the field and hands it to a kernel needs
this, which is why it does not live with any one of them.
"""

from __future__ import annotations

import math


class MeridianConvergence:
    """
    The angle between grid north and true north, point by point.

    In a projection the vertical grid lines are not meridians: only on the
    central meridian do the two norths coincide. In the southern Apennines in
    EPSG:25833 the gap runs from +0.41 to +1.04 degrees, which over five
    kilometres of trace is up to 91 metres -- twenty times the DEM cell.

    The value is measured rather than read off a formula: take a hundred-metre
    step along true north and see what azimuth that step has on the grid. Eight
    microseconds, and it holds for any projection, including the ones that do
    not let themselves be written in PROJ.

    The sign, checked on three points to four decimals:

        grid_azimuth = true_azimuth - convergence
    """

    STEP_M = 100.0

    def __init__(self, crs):
        self.available = False
        self._to_geographic = None

        if crs is None:
            return

        import pyproj

        try:
            self._to_geographic = pyproj.Transformer.from_crs(crs, 4326, always_xy=True)
            self._to_projected = pyproj.Transformer.from_crs(4326, crs, always_xy=True)
            self._geod = pyproj.CRS.from_user_input(crs).get_geod()
        except Exception:
            return

        self.available = self._geod is not None

    def at(self, x, y):
        """Convergence in degrees at the point, positive east of the central meridian."""

        if not self.available:
            return 0.0

        lon, lat = self._to_geographic.transform(x, y)
        lon_n, lat_n, _ = self._geod.fwd(lon, lat, 0.0, self.STEP_M)
        x_n, y_n = self._to_projected.transform(lon_n, lat_n)

        return -math.degrees(math.atan2(x_n - x, y_n - y))

    def to_grid(self, true_azimuth, x, y):
        return (true_azimuth - self.at(x, y)) % 360.0

    def geographic(self, x, y):
        """
        Longitude and latitude of the point, or None without a usable CRS.

        The transformation already exists because convergence needs it on every
        frame: here it is only exposed, because a point written in projected
        coordinates alone is unusable outside its EPSG -- in a notebook, in a
        GPS, in a paper.

        Outside the projection's domain pyproj returns infinity rather than
        raising: the finiteness check is what tells an out-of-range point from
        a good coordinate.
        """

        if self._to_geographic is None:
            return None

        lon, lat = self._to_geographic.transform(x, y)

        if not (math.isfinite(lon) and math.isfinite(lat)):
            return None

        return lon, lat
