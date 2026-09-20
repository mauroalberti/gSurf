"""
The DEM, opened rather than loaded, and the crops the kernels run on.

A mosaic is read one window at a time: the cost of a frame goes with the cells
scanned, not with the size of the file, and holding a 314 Mpx raster in memory
would be 2.5 GB in float64 for no gain.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows
from rasterio.windows import Window


def hillshade(z, dx, dy, azimuth=315.0, altitude=45.0):
    """Hillshade in the ESRI convention, with rows running south."""

    d_row, d_col = np.gradient(z, dy, dx)
    dz_dx, dz_dy = d_col, -d_row

    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(dz_dy, -dz_dx)

    zenith = np.radians(90.0 - altitude)
    az = np.radians(360.0 - azimuth + 90.0)

    shaded = np.cos(zenith) * np.cos(slope) + np.sin(zenith) * np.sin(slope) * np.cos(az - aspect)

    return np.clip(shaded, 0.0, 1.0)

class ComputeWindow:
    """
    The full-resolution crop the kernel runs on.

    It exists apart from the DEM because the cost of a frame goes with the
    number of cells scanned, not with the size of the file: on a 314 Mpx mosaic
    the kernel would take seconds, on a 1000x1000 window it fits in 22 ms.
    """

    def __init__(self, data, geotransform, bounds, offset):
        self.data = data
        self.geotransform = geotransform
        self.bounds = bounds
        self.offset = offset

    @property
    def shape(self):
        return self.data.shape

    def covers(self, x, y):
        left, bottom, right, top = self.bounds

        return left <= x <= right and bottom <= y <= top

    def contains(self, box):
        """Whether a whole rectangle is inside this window."""

        left, bottom, right, top = self.bounds
        x0, y0, x1, y1 = box

        return left <= x0 and y0 >= bottom and x1 <= right and y1 <= top

    def rectangle_xy(self):
        left, bottom, right, top = self.bounds

        return (left, bottom), right - left, top - bottom


class Dem:
    """
    The DEM opened without loading it: an overview for the background,
    full-resolution windows on demand.

    A 314 Mpx mosaic would be 2.5 GB in float64, so holding it all in memory is
    not an option and is not needed either: the kernel reads one window at a
    time, and that read costs 4.9 ms on 1000x1000.
    """

    def __init__(self, path, display_max=1600):
        self.path = Path(path)
        self._src = rasterio.open(path)

        self.crs = self._src.crs
        self.nodata = self._src.nodata
        self.bounds = self._src.bounds
        self.width = self._src.width
        self.height = self._src.height
        self.res_x = abs(self._src.transform.a)
        self.res_y = abs(self._src.transform.e)

        self.extent = [
            self.bounds.left,
            self.bounds.right,
            self.bounds.bottom,
            self.bounds.top,
        ]

        # The background does not need full resolution: past a couple of
        # thousand pixels it would not show anyway, and hillshading a whole
        # mosaic would cost minutes.
        self.decimation = max(1, math.ceil(max(self.width, self.height) / display_max))
        shape = (self.height // self.decimation, self.width // self.decimation)
        overview = self._src.read(1, out_shape=shape).astype(float)

        if self.nodata is not None:
            overview[overview == self.nodata] = np.nan

        self.hillshade = hillshade(
            overview,
            self.res_x * self.decimation,
            self.res_y * self.decimation,
        )
        self.z_median = float(np.nanmedian(overview))

        # Off the overview, not the full raster: a section's vertical axis has
        # to be settled before the first profile is drawn, and it has to stay
        # settled while the trace is dragged -- an axis that rescaled under a
        # moving profile would make every frame a different picture. The
        # decimated minimum can miss the bottom of a gorge by a few metres,
        # which is why what uses this pads it rather than trusting it.
        self.z_range = (float(np.nanmin(overview)), float(np.nanmax(overview)))

    def close(self):
        self._src.close()

    def center(self):
        return (
            (self.bounds.left + self.bounds.right) / 2.0,
            (self.bounds.bottom + self.bounds.top) / 2.0,
        )

    def elevation_at(self, x, y):
        """Elevation at the map coordinate, or None off-grid / on nodata."""

        row, col = self._src.index(x, y)
        row, col = int(row), int(col)

        if not (0 <= row < self.height and 0 <= col < self.width):
            return None

        z = float(self._src.read(1, window=Window(col, row, 1, 1))[0, 0])

        return None if self.nodata is not None and z == self.nodata else z

    def shade_for(self, xmin, xmax, ymin, ymax, max_px=1200):
        """
        Hillshade of the current view alone, at the resolution it needs.

        The initial background is decimated over the whole DEM: on a large
        mosaic that means cells tens of metres across, and zooming in leaves
        mush exactly as the trace becomes detailed. Here the framed portion is
        re-read at the decimation right for that scale.

        Returns None if the view is entirely off the DEM.
        """

        left = max(xmin, self.bounds.left)
        right = min(xmax, self.bounds.right)
        bottom = max(ymin, self.bounds.bottom)
        top = min(ymax, self.bounds.top)

        if right <= left or top <= bottom:
            return None

        window = rasterio.windows.from_bounds(
            left, bottom, right, top, self._src.transform
        ).round_offsets().round_lengths()

        window = window.intersection(Window(0, 0, self.width, self.height))

        if window.width < 2 or window.height < 2:
            return None

        step = max(1, math.ceil(max(window.width, window.height) / max_px))
        shape = (max(2, int(window.height) // step), max(2, int(window.width) // step))

        band = self._src.read(1, window=window, out_shape=shape).astype(float)

        if self.nodata is not None:
            band[band == self.nodata] = np.nan

        shade = hillshade(band, self.res_x * step, self.res_y * step)
        left, bottom, right, top = rasterio.windows.bounds(window, self._src.transform)

        return shade, [left, right, bottom, top], step

    def window_over(self, box, margin=0.0, nodata_as_nan=True):
        """
        A full-resolution window covering a rectangle, clipped to the DEM.

        `window_at` is for a kernel that wants a fixed cost per frame; this is
        for whatever has to cover an area it was given. A section trace is the
        case: it is as long as it is drawn, and a square window sized for the
        longest one would read tens of megabytes to sample a line.

        The margin is there so a trace nudged a few metres does not fall off
        the edge and force a reread on the next frame.
        """

        left, bottom, right, top = box
        left, bottom = left - margin, bottom - margin
        right, top = right + margin, top + margin

        row0, col0 = self._src.index(left, top)
        row1, col1 = self._src.index(right, bottom)

        col_off = max(0, min(int(col0), int(col1)))
        row_off = max(0, min(int(row0), int(row1)))
        col_end = min(self.width, max(int(col0), int(col1)) + 1)
        row_end = min(self.height, max(int(row0), int(row1)) + 1)

        if col_end <= col_off or row_end <= row_off:
            return None

        window = Window(col_off, row_off, col_end - col_off, row_end - row_off)
        band = self._src.read(1, window=window).astype(np.float64)

        if nodata_as_nan and self.nodata is not None:
            band[band == self.nodata] = np.nan

        return ComputeWindow(
            np.ascontiguousarray(band),
            list(rasterio.windows.transform(window, self._src.transform).to_gdal()),
            rasterio.windows.bounds(window, self._src.transform),
            (col_off, row_off),
        )

    def window_at(self, x, y, side):
        """A `side`-cell window centred on (x, y), clipped to the DEM."""

        row, col = self._src.index(x, y)
        col_off = int(col) - side // 2
        row_off = int(row) - side // 2

        # At the edges the window shifts rather than shrinking, so the per-frame
        # cost stays the advertised one wherever you take it.
        col_off = max(0, min(col_off, self.width - side))
        row_off = max(0, min(row_off, self.height - side))

        width = min(side, self.width)
        height = min(side, self.height)

        window = Window(col_off, row_off, width, height)
        band = self._src.read(1, window=window)
        transform = rasterio.windows.transform(window, self._src.transform)
        bounds = rasterio.windows.bounds(window, self._src.transform)

        # misah wants contiguous f64. Real DEMs are often f32: the conversion is
        # paid for here, not inside the loop.
        return ComputeWindow(
            np.ascontiguousarray(band.astype(np.float64)),
            list(transform.to_gdal()),
            bounds,
            (col_off, row_off),
        )
