"""
Real-time geological plane / DEM intersection.

The misah kernel intersects an unbounded plane with the grid and returns the
marching-squares chords; around it is the minimum needed to steer it by hand
and watch the answer move. The status bar reports kernel, drawing and frame
rate on every frame, so the cost stays visible while you work.

Usage:
    python realtime_intersection.py
    python realtime_intersection.py <dem.tif> [--polygons PATH[:LAYER]]
                                    [--lines PATH[:LAYER]] [--points PATH[:LAYER]]
                                    [--categories FIELD] [--x E] [--y N] [--z Z]
                                    [--window N] [--settings <file.json>]

With no arguments a dialog asks for the same things. The only one required is
the DEM: the three vector slots -- polygons, lines, points -- answer where the
plane is being laid down, which is a different question from the calculation.
The layers offered in each slot are filtered on geometry read from the
metadata, so faults never appear among the polygons.

The DEM can be as large as you like: it is never loaded into memory. The
background is a decimated overview, while the kernel runs on a full-resolution
window centred on the source point, whose side --window sets and which decides
how smooth the ride is (1000 px sits around 38 fps, 500 px around 100).

The source point is fixed component by component, and what you leave out the
DEM decides: no --x and --y puts it at the centre, no --z takes the ground
elevation. An elevation you give stays given even as you move the point --
that is how a plane is laid on a horizon passing above or below today's
topography. The "elevation from DEM" checkbox makes and breaks the tie at any
time.

Dip direction is read and written in **true azimuth**, as it is measured in
the field. The DEM, though, is on the projection's grid, and the two norths do
not agree: meridian convergence is subtracted before the kernel is called, and
shown under the attitude. In the southern Apennines it runs between +0.41 and
+1.04 degrees, which over five kilometres of trace is up to 91 metres.

In the window:
    - dial and slider for dip direction and dip angle;
    - scroll to zoom around the cursor, navigation bar for pan, rubber-band
      zoom and back to the full view;
    - click on the map to move the source point, or drag the point itself (with
      pan and zoom off, or the two gestures are the same one);
    - screenshot to the clipboard or to a file, current attitude as JSON,
      computed trace as a shapefile. The source point goes into both in both
      forms, projected and geographic.

Polygonal outcrops are coloured per unit -- the field is set by --categories,
`code` by default. The colours come from the layer's complete list of values
and not from whichever units are in view, so a formation keeps its colour as
you pan.
"""

# The vector slots are three and generic, but this tool grew up around a single
# geopackage: `--geology` survives as the shortcut for it, putting `carbonates`
# among the polygons and `faults` among the lines.

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np
import rasterio
from rasterio.windows import Window

# PyQt6 has to be imported before the backend: matplotlib picks the binding by
# looking at what is already in sys.modules.
import PyQt6.QtCore  # noqa: F401
from PyQt6 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from misah.kernels import intersect_plane_grid


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


class VectorSource:
    """
    One backdrop vector layer, in the role it was given.

    The roles are three -- polygons, lines, points -- and they are not a matter
    of style: they decide what it makes sense to ask of the layer. A field of
    polygons coloured per unit says which two formations a contact separates;
    the same twenty-three tints spread over four hundred faults cannot be read.
    So categorisation starts on for polygons and off for the rest, and it is
    the user who decides in the end.

    The layers are static: they are drawn once and end up in the background
    that blitting recaptures, so per frame they cost nothing. Each carries its
    own CRS -- in geology.gpkg the carbonates are in UTM 32N and the faults in
    geographic -- and is reprojected on its own onto the DEM's, never the file
    as a block.
    """

    ROLES = ("polygons", "lines", "points")

    # The OGR type suffix: 'Polygon' and 'MultiPolygon' both end in 'Polygon',
    # and so for the other two pairs. A layer with no geometry --
    # `fault_attitudes` in geology.gpkg is a pure table -- has no suffix and
    # stays out of all three roles, which is where it belongs.
    GEOMETRY_SUFFIX = {
        "polygons": "Polygon",
        "lines": "LineString",
        "points": "Point",
    }

    FLAT_STYLE = {
        "polygons": dict(facecolor="#4daf7c", edgecolor="#2f7a52", alpha=0.25, linewidth=0.5),
        "lines": dict(color="#1f4fd8", linewidth=1.0),
        "points": dict(color="#d95f02", markersize=26, marker="^", edgecolor="#4a2200"),
    }

    CATEGORY_STYLE = {
        "polygons": dict(edgecolor="#333333", linewidth=0.4, alpha=0.38),
        "lines": dict(linewidth=1.3),
        "points": dict(markersize=30, marker="^", edgecolor="#222222"),
    }

    # Points over lines, lines over polygons: the order in which a map is read.
    # With the single zorder of before, an outcrop drawn later covered the
    # faults you were using to find your way.
    ZORDER = {"polygons": 2, "lines": 3, "points": 4}

    # The field that plugs the holes in the chosen one: in geology.gpkg five
    # polygons have no `code`, and with no fallback they would end up in a
    # single "n/a" category mixing three different ones.
    CATEGORY_FALLBACK = "name"

    # Past a dozen entries the legend eats the map it is supposed to explain;
    # the categories in excess stay coloured, they are just not listed.
    MAX_LEGEND_ENTRIES = 12

    def __init__(self, path, role, crs, bounds, layer=None, category_field=None):
        import geopandas as gpd  # heavy to import: only when actually needed
        from shapely.geometry import box

        self.path = Path(path)
        self.role = role
        self.layer = layer
        self.category_field = category_field
        self.colors = {}
        self.labels = {}
        self.frame = None
        self.problem = None

        try:
            complete = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
        except Exception as err:
            self.problem = str(err).split("\n")[0]
            return

        if complete.crs is None:
            self.problem = "no CRS"
            return

        window = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        visible = complete.to_crs(crs)
        visible = visible[visible.intersects(window)]

        if visible.empty:
            self.problem = "no feature on the DEM"
            return

        self.frame = self._categorize(complete, visible)

    # -- reading the container, without loading the data ------------------

    @staticmethod
    def candidate_layers(path, role):
        """
        The layers in the file whose geometry fits the role.

        Read from the metadata alone, so listing the layers of a half-gigabyte
        geopackage costs what listing an empty one costs -- and that is what
        lets the dialog filter while the user chooses.
        """

        import pyogrio

        suffix = VectorSource.GEOMETRY_SUFFIX[role]

        return [
            str(name)
            for name, geometry in pyogrio.list_layers(path)
            if geometry is not None and str(geometry).endswith(suffix)
        ]

    @staticmethod
    def text_fields(path, layer=None):
        """The layer's text fields: the only ones worth categorising on."""

        import pyogrio

        info = pyogrio.read_info(path, layer=layer) if layer else pyogrio.read_info(path)

        return [
            str(field)
            for field, dtype in zip(info["fields"], info["dtypes"])
            if str(dtype) == "object"
        ]

    # -- categories --------------------------------------------------------

    @property
    def is_loaded(self):
        return self.frame is not None

    def _values(self, frame):
        """The column to tell things apart by, with the holes plugged by the name."""

        if not self.category_field or self.category_field not in frame.columns:
            return None

        values = frame[self.category_field].astype("string")

        if self.CATEGORY_FALLBACK in frame.columns:
            values = values.fillna(frame[self.CATEGORY_FALLBACK].astype("string"))

        return values.fillna("n/a").astype(str)

    def _categorize(self, complete, visible):
        """
        Assigns one colour per category, decided on the layer's complete list.

        On the complete list and not on the visible one on purpose: if the
        colours came from whichever categories happen to fall in the window,
        the same formation would change colour as you pan or change DEM, and
        that is the one thing a legend cannot afford.
        """

        values = self._values(complete)

        if values is None:
            return visible

        from matplotlib import colormaps

        # Twenty plus twenty: the units mapped in geology.gpkg are twenty-three,
        # and with tab20 alone two of them would come out identical.
        wheel = list(colormaps["tab20"].colors) + list(colormaps["tab20b"].colors)
        order = sorted(values.unique())

        self.colors = {value: wheel[i % len(wheel)] for i, value in enumerate(order)}

        if self.CATEGORY_FALLBACK in complete.columns and self.category_field != self.CATEGORY_FALLBACK:
            named = complete[self.CATEGORY_FALLBACK].astype("string")
            self.labels = {
                value: (group.dropna().iloc[0] if len(group.dropna()) else "")
                for value, group in named.groupby(values)
            }

        return visible.assign(_gsurf_category=self._values(visible))

    # -- drawing -----------------------------------------------------------

    def draw(self, axes):
        table = self.CATEGORY_STYLE if self.colors else self.FLAT_STYLE
        style = dict(table[self.role])

        if self.colors:
            style["color"] = [self.colors[v] for v in self.frame["_gsurf_category"]]
        else:
            style.setdefault("label", self.layer or self.path.stem)

        self.frame.plot(ax=axes, zorder=self.ZORDER[self.role], **style)

    def _legend_label(self, value, width=28):
        name = self.labels.get(value, "")
        text = f"{value} - {name}" if name and name != value else str(value)

        return text if len(text) <= width else text[: width - 1] + "…"

    def _handle(self, label, color=None):
        """
        The dummy artist standing in for one legend entry.

        Needed because geopandas draws with collections matplotlib cannot
        represent on its own: without these the polygons would drop out of the
        legend silently. The shape follows the role, so across three
        categorised layers you can still tell whose entry is whose.
        """

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        style = dict((self.CATEGORY_STYLE if self.colors else self.FLAT_STYLE)[self.role])
        style.pop("markersize", None)
        style.pop("color", None)

        if self.role == "polygons":
            return Patch(facecolor=color or style.pop("facecolor", "#4daf7c"), label=label, **style)

        if self.role == "lines":
            return Line2D([], [], color=color or "#1f4fd8", label=label, **style)

        marker = style.pop("marker", "^")
        edge = style.pop("edgecolor", "#222222")

        return Line2D(
            [], [],
            linestyle="none",
            marker=marker,
            markerfacecolor=color or "#d95f02",
            markeredgecolor=edge,
            markersize=7,
            label=label,
        )

    def legend_handles(self):
        if not self.colors:
            return [self._handle(self.layer or self.path.stem)]

        # In the legend only the categories actually on show, and in order of
        # weight: alphabetically, the cut at twelve would throw out Qt, PL and
        # Op -- which are half the map -- to make room for AV, which is a
        # single polygon. Weight is area for polygons, length for lines, count
        # for points.
        frame = self.frame

        if self.role == "polygons":
            weight = frame.area
        elif self.role == "lines":
            weight = frame.length
        else:
            weight = 1.0

        present = list(
            frame.assign(_gsurf_weight=weight)
            .groupby("_gsurf_category")["_gsurf_weight"]
            .sum()
            .sort_values(ascending=False)
            .index
        )

        handles = [
            self._handle(self._legend_label(value), self.colors[value])
            for value in present[: self.MAX_LEGEND_ENTRIES]
        ]

        if len(present) > self.MAX_LEGEND_ENTRIES:
            from matplotlib.patches import Patch

            handles.append(
                Patch(
                    facecolor="none",
                    edgecolor="none",
                    label=f"+{len(present) - self.MAX_LEGEND_ENTRIES} more in {self.role}",
                )
            )

        return handles

    def summary(self):
        where = self.layer or self.path.name

        if not self.is_loaded:
            return f"{self.role}: {where} skipped ({self.problem})"

        if self.colors:
            distinct = len(set(self.frame["_gsurf_category"]))

            return (
                f"{self.role}: {where}, {len(self.frame)} in {distinct} "
                f"categories ({self.category_field})"
            )

        return f"{self.role}: {where}, {len(self.frame)}"


class Overlay:
    """
    The backdrop vector layers held together, in the order they are read in.

    None of them is necessary: the DEM alone is enough to intersect a plane.
    They answer where that plane is being laid down, which is a different
    question from the calculation and one the DEM does not answer.
    """

    def __init__(self, sources=()):
        sources = list(sources)

        self.sources = [s for s in sources if s.is_loaded]
        self.rejected = [s for s in sources if not s.is_loaded]

    def __bool__(self):
        return bool(self.sources)

    @property
    def is_categorized(self):
        return any(source.colors for source in self.sources)

    def draw(self, axes):
        """Draws on the axes, without letting geopandas rescale the view."""

        limits = axes.get_xlim(), axes.get_ylim()

        for source in sorted(self.sources, key=lambda s: VectorSource.ZORDER[s.role]):
            source.draw(axes)

        axes.set_xlim(limits[0])
        axes.set_ylim(limits[1])

    def legend_handles(self):
        handles = []

        for source in sorted(self.sources, key=lambda s: VectorSource.ZORDER[s.role]):
            handles.extend(source.legend_handles())

        return handles

    def summary(self):
        lines = [s.summary() for s in self.sources] + [s.summary() for s in self.rejected]

        return "; ".join(lines) if lines else "no vector layer"


def merged_traces(points, segments):
    """
    The marching-squares chords welded into polylines, elevation and all.

    The kernel returns thousands of two-vertex segments: written out that way
    they are unusable in a GIS. `linemerge` stitches them back into the
    continuous traces they really are, and keeps the Z.
    """

    from shapely.geometry import LineString
    from shapely.ops import linemerge

    if not len(segments):
        return []

    chords = [LineString(points[pair]) for pair in segments]
    merged = linemerge(chords)

    return list(merged.geoms) if hasattr(merged, "geoms") else [merged]


class Toolbar(NavigationToolbar2QT):
    """
    The navigation bar, with saving diverted.

    The toolbar's save button calls `savefig` on its own, and that path knows
    nothing of the animated artists: the file would come out with the map and
    without the trace on top. Here it ends up in the same place as the "Save
    screenshot" button, so the two cannot drift apart.
    """

    def __init__(self, canvas, parent, save_handler, view_changed):
        super().__init__(canvas, parent)
        self._save_handler = save_handler
        self._view_changed = view_changed

    def save_figure(self, *args):
        self._save_handler()

    # Every way the bar can change the framing has to report it, or the
    # background stays at the previous resolution.

    def release_pan(self, event):
        super().release_pan(event)
        self._view_changed()

    def release_zoom(self, event):
        super().release_zoom(event)
        self._view_changed()

    def home(self, *args):
        super().home(*args)
        self._view_changed()

    def back(self, *args):
        super().back(*args)
        self._view_changed()

    def forward(self, *args):
        super().forward(*args)
        self._view_changed()


VECTOR_FILTER = (
    "Vector (*.gpkg *.shp *.geojson *.json *.gml *.kml *.sqlite *.fgb);;"
    "All files (*)"
)

RASTER_FILTER = "Raster (*.tif *.tiff *.vrt *.asc *.img *.dt2 *.hgt);;All files (*)"


def as_number(text):
    """The text as a number, or None if it is empty or is not one.

    A comma counts as a point: an Italian keyboard puts the comma on the numeric
    keypad, and rejecting '1187,4' would be a small cruelty."""

    text = (text or "").strip().replace(",", ".")

    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        return None


class VectorPicker(QtWidgets.QGroupBox):
    """
    Choosing one layer for one role: file, layer within the file, categories.

    The layers offered are filtered on the role's geometry, read from the
    metadata alone: in the polygon slot the faults simply never appear, and a
    table without geometry appears nowhere. One less error to diagnose
    downstream, at the cost of a read that never touches the data.
    """

    # The names a category column usually goes by. The Italian ones are kept
    # alongside the English: the geological maps this is used on are surveyed
    # in Italy, and their attribute tables say `sigla` and `unita`. On polygons
    # categorisation starts on because it is almost always what you want; on
    # lines and points it starts off, since twenty tints over four hundred
    # faults cannot be read.
    PREFERRED_FIELDS = ("code", "sigla", "unit", "unita", "type", "tipo", "name", "nome")

    def __init__(self, role, parent=None):
        super().__init__(role.capitalize(), parent)

        self.role = role
        self._path = None

        self.path_label = QtWidgets.QLineEdit()
        self.path_label.setReadOnly(True)
        self.path_label.setPlaceholderText("none (optional)")

        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(self._browse)

        self.clear_button = QtWidgets.QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear)
        self.clear_button.setEnabled(False)

        self.layer_combo = QtWidgets.QComboBox()
        self.layer_combo.setEnabled(False)
        self.layer_combo.currentTextChanged.connect(self._on_layer_changed)

        self.category_combo = QtWidgets.QComboBox()
        self.category_combo.setEnabled(False)

        grid = QtWidgets.QGridLayout(self)
        grid.addWidget(self.path_label, 0, 0, 1, 2)
        grid.addWidget(browse, 0, 2)
        grid.addWidget(self.clear_button, 0, 3)
        grid.addWidget(QtWidgets.QLabel("layer"), 1, 0)
        grid.addWidget(self.layer_combo, 1, 1, 1, 3)
        grid.addWidget(QtWidgets.QLabel("categories"), 2, 0)
        grid.addWidget(self.category_combo, 2, 1, 1, 3)
        grid.setColumnStretch(1, 1)

    def _browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Choose the file: {self.role}", "", VECTOR_FILTER
        )

        if path:
            self.set_path(path)

    def set_path(self, path, layer=None, category_field=None):
        """Loads the list of layers fit for the role. Returns False if there are none."""

        try:
            candidates = VectorSource.candidate_layers(path, self.role)
        except Exception as err:
            QtWidgets.QMessageBox.warning(
                self, "Unreadable file", f"{Path(path).name}\n\n{str(err).splitlines()[0]}"
            )
            return False

        if not candidates:
            QtWidgets.QMessageBox.information(
                self,
                "No suitable layer",
                f"{Path(path).name} holds no {self.role} layer.",
            )
            return False

        self._path = Path(path)
        self.path_label.setText(str(path))
        self.path_label.setToolTip(str(path))
        self.clear_button.setEnabled(True)

        with QtCore.QSignalBlocker(self.layer_combo):
            self.layer_combo.clear()
            self.layer_combo.addItems(candidates)

            if layer and layer in candidates:
                self.layer_combo.setCurrentText(layer)

        self.layer_combo.setEnabled(True)
        self._on_layer_changed(self.layer_combo.currentText(), preferred=category_field)

        return True

    def _on_layer_changed(self, layer, preferred=None):
        if not self._path or not layer:
            return

        try:
            fields = VectorSource.text_fields(self._path, layer)
        except Exception:
            fields = []

        with QtCore.QSignalBlocker(self.category_combo):
            self.category_combo.clear()
            self.category_combo.addItem("(none)")
            self.category_combo.addItems(fields)

            chosen = None

            if preferred and preferred in fields:
                chosen = preferred
            elif self.role == "polygons":
                chosen = next((f for f in self.PREFERRED_FIELDS if f in fields), None)

            self.category_combo.setCurrentText(chosen or "(none)")

        self.category_combo.setEnabled(bool(fields))

    def clear(self):
        self._path = None
        self.path_label.clear()
        self.path_label.setToolTip("")
        self.clear_button.setEnabled(False)
        self.layer_combo.clear()
        self.layer_combo.setEnabled(False)
        self.category_combo.clear()
        self.category_combo.setEnabled(False)

    def value(self):
        """The chosen role as a dictionary, or None if the slot is empty."""

        if self._path is None:
            return None

        field = self.category_combo.currentText()

        return dict(
            path=str(self._path),
            role=self.role,
            layer=self.layer_combo.currentText() or None,
            category_field=None if field in ("", "(none)") else field,
        )


class SourcesDialog(QtWidgets.QDialog):
    """
    What to open, asked before the working window opens.

    The DEM is the only one required, because it is the only one the kernel
    needs: the other three answer where the plane is being laid down, which is
    a different question from the calculation.

    The source point can be fixed here too, component by component: leave it
    empty and you go to the centre of the DEM at ground elevation, and an
    elevation typed by hand outranks the ground -- that is how a plane is laid
    on a horizon passing above today's topography.
    """

    def __init__(self, parent=None, dem=None, vectors=(), point=(None, None, None)):
        super().__init__(parent)

        self.setWindowTitle("gSurf - sources")
        self.setMinimumWidth(560)

        self.dem_label = QtWidgets.QLineEdit()
        self.dem_label.setReadOnly(True)
        self.dem_label.setPlaceholderText("required")

        dem_browse = QtWidgets.QPushButton("Browse...")
        dem_browse.clicked.connect(self._browse_dem)

        self.dem_info = QtWidgets.QLabel()
        self.dem_info.setStyleSheet("color: gray; font-size: 10px;")

        dem_box = QtWidgets.QGroupBox("DEM")
        dem_grid = QtWidgets.QGridLayout(dem_box)
        dem_grid.addWidget(self.dem_label, 0, 0)
        dem_grid.addWidget(dem_browse, 0, 1)
        dem_grid.addWidget(self.dem_info, 1, 0, 1, 2)
        dem_grid.setColumnStretch(0, 1)

        self.pickers = {role: VectorPicker(role) for role in VectorSource.ROLES}

        # The boxes stay line edits and not spin boxes: a spin box cannot be
        # empty, and "empty" is exactly the value that here means "you decide".
        numeric = QtGui.QDoubleValidator()
        numeric.setLocale(QtCore.QLocale.c())

        self.easting_edit = QtWidgets.QLineEdit()
        self.northing_edit = QtWidgets.QLineEdit()
        self.elevation_edit = QtWidgets.QLineEdit()

        for edit in (self.easting_edit, self.northing_edit, self.elevation_edit):
            edit.setValidator(numeric)

        self.easting_edit.setPlaceholderText("DEM centre")
        self.northing_edit.setPlaceholderText("DEM centre")
        self.elevation_edit.setPlaceholderText("DEM elevation")

        point_box = QtWidgets.QGroupBox("Source point (optional)")
        point_grid = QtWidgets.QGridLayout(point_box)
        for column, (caption, edit) in enumerate(
            (
                ("E", self.easting_edit),
                ("N", self.northing_edit),
                ("Z", self.elevation_edit),
            )
        ):
            point_grid.addWidget(QtWidgets.QLabel(caption), 0, column * 2)
            point_grid.addWidget(edit, 0, column * 2 + 1)
            point_grid.setColumnStretch(column * 2 + 1, 1)

        note = QtWidgets.QLabel(
            "An elevation typed here does not follow the DEM: the plane rests on it."
        )
        note.setStyleSheet("color: gray; font-size: 10px;")
        point_grid.addWidget(note, 1, 0, 1, 6)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Open
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        # Qt's standard buttons already read "Open" and "Cancel" untranslated,
        # so nothing has to be written over them here.

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(dem_box)
        for role in VectorSource.ROLES:
            layout.addWidget(self.pickers[role])
        layout.addWidget(point_box)
        layout.addWidget(self.buttons)

        self._dem_path = None
        self._set_dem(dem)

        for spec in vectors or ():
            picker = self.pickers.get(spec.get("role"))

            if picker is not None:
                picker.set_path(spec["path"], spec.get("layer"), spec.get("category_field"))

        for edit, value in zip(
            (self.easting_edit, self.northing_edit, self.elevation_edit), point
        ):
            if value is not None:
                edit.setText(f"{float(value):g}")

    def _browse_dem(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose the DEM", "", RASTER_FILTER
        )

        if path:
            self._set_dem(path)

    def _set_dem(self, path):
        """
        Opens the DEM for its metadata alone, and says what it is from those.

        It also catches a file that is not a raster right away, rather than
        after the dialog has closed -- and writes the real centre coordinates
        into the placeholders, which is the value you would get by leaving them
        empty.
        """

        self._refresh_ok()

        if not path:
            return

        try:
            with rasterio.open(path) as src:
                epsg = src.crs.to_epsg() if src.crs else None
                centre_x = (src.bounds.left + src.bounds.right) / 2.0
                centre_y = (src.bounds.bottom + src.bounds.top) / 2.0
                info = (
                    f"{src.width}x{src.height} "
                    f"({src.width * src.height / 1e6:.1f} Mpx), "
                    f"EPSG:{epsg or '?'}, cell {abs(src.transform.a):g} m"
                )
        except Exception as err:
            QtWidgets.QMessageBox.warning(
                self, "Unreadable DEM", f"{Path(path).name}\n\n{str(err).splitlines()[0]}"
            )
            return

        self._dem_path = str(path)
        self.dem_label.setText(str(path))
        self.dem_label.setToolTip(str(path))
        self.dem_info.setText(info)

        self.easting_edit.setPlaceholderText(f"centre: {centre_x:.0f}")
        self.northing_edit.setPlaceholderText(f"centre: {centre_y:.0f}")

        self._refresh_ok()

    def _refresh_ok(self):
        self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Open).setEnabled(
            self._dem_path is not None
        )

    def choices(self):
        """DEM, vector layers and point, in the shape `main` knows how to use."""

        vectors = [p.value() for p in self.pickers.values()]

        return dict(
            dem=self._dem_path,
            vectors=[v for v in vectors if v],
            point=(
                as_number(self.easting_edit.text()),
                as_number(self.northing_edit.text()),
                as_number(self.elevation_edit.text()),
            ),
        )


class RealtimeWindow(QtWidgets.QMainWindow):

    PICK_RADIUS_PX = 12
    ZOOM_STEP = 1.3

    # QDial puts its minimum at six o'clock, not at twelve: measured by
    # grabbing the widget and hunting for the needle, value 0 points 181 degrees
    # from twelve o'clock and value 180 points to 360. It runs clockwise, like
    # an azimuth, so between the widget's scale and geological dip direction
    # there is only half a turn of offset.
    DIAL_NORTH_OFFSET = 180

    # Where the legend goes. Categorised it runs to some thirty entries, and
    # inside the map it covers the corner you were most likely looking at:
    # beside it the map stays whole and the legend grows in its own column.
    # Hidden is for when the tints are already known and the map just has to be
    # read -- or for a figure whose caption lists them elsewhere.
    LEGEND_PLACEMENTS = (
        ("beside the map", "beside"),
        ("inside the map", "inside"),
        ("hidden", "hidden"),
    )

    def __init__(
        self,
        dem,
        overlay=None,
        side=1000,
        attitude=(90.0, 30.0),
        source=None,
        z_follows_dem=None,
        legend="beside",
    ):
        super().__init__()

        self.dem = dem
        self.overlay = overlay
        self.background = None
        self.legend = None
        self.dragging = False
        self.frame_times = deque(maxlen=20)
        self.convergence = MeridianConvergence(dem.crs)
        self.last_result = ([], [])

        # The three components are independent: you can fix the elevation alone
        # and let the point sit at the centre, or the other way round.
        x, y, z = (tuple(source) + (None, None, None))[:3] if source else (None, None, None)
        centre_x, centre_y = dem.center()

        x = centre_x if x is None else float(x)
        y = centre_y if y is None else float(y)

        # An elevation given explicitly wants to stay that one: it is the case
        # of a projected horizon, or of a measurement taken above or below the
        # ground. Giving it is therefore also the way of saying it must not
        # follow the DEM, unless somebody asks for that explicitly.
        self.z_follows_dem = (z is None) if z_follows_dem is None else bool(z_follows_dem)

        if z is None:
            surface = dem.elevation_at(x, y)
            z = dem.z_median if surface is None else surface

        self.source_point = [x, y, float(z)]

        self.side = min(side, dem.width, dem.height)
        self.window = dem.window_at(self.source_point[0], self.source_point[1], self.side)

        self.setWindowTitle(f"gSurf - real-time intersection - {dem.path.name}")
        self._build_ui(attitude, legend)
        self._draw_base_map()

        self.update_intersection()

    # -- construction -----------------------------------------------------

    def _build_ui(self, attitude, legend):
        self.figure = Figure(figsize=(8, 8), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)

        self.toolbar = Toolbar(self.canvas, self, self.save_screenshot, self.schedule_shade_refresh)

        # Reloading the hillshade is paid for in tens of milliseconds: too much
        # for every notch of the wheel, about right once the hand stops. Hence
        # the delay.
        self.shade_timer = QtCore.QTimer(self)
        self.shade_timer.setSingleShot(True)
        self.shade_timer.timeout.connect(self._refresh_shade)

        self.canvas.mpl_connect("draw_event", self._on_draw)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        # The dial for the hand, the box for the number. The dial alone steps by
        # one degree, and meridian convergence here is 0.8: without the tenth of
        # a degree the correction would be smaller than the control meant to
        # apply it, and therefore useless. The box is the authoritative source,
        # the dial follows it.
        self.dip_dir_dial = QtWidgets.QDial()
        self.dip_dir_dial.setRange(0, 359)
        self.dip_dir_dial.setWrapping(True)
        self.dip_dir_dial.setNotchesVisible(True)
        self.dip_dir_dial.setMinimumSize(140, 140)

        self.dip_dir_spin = QtWidgets.QDoubleSpinBox()
        self.dip_dir_spin.setRange(0.0, 359.9)
        self.dip_dir_spin.setDecimals(1)
        self.dip_dir_spin.setSingleStep(0.1)
        self.dip_dir_spin.setWrapping(True)
        self.dip_dir_spin.setSuffix("°  dip dir")

        self.dip_angle_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
        self.dip_angle_slider.setRange(0, 90)
        self.dip_angle_slider.setTickInterval(10)
        self.dip_angle_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksRight)

        self.dip_angle_spin = QtWidgets.QDoubleSpinBox()
        self.dip_angle_spin.setRange(0.0, 90.0)
        self.dip_angle_spin.setDecimals(1)
        self.dip_angle_spin.setSingleStep(0.1)
        self.dip_angle_spin.setSuffix("°  dip")

        self.attitude_label = QtWidgets.QLabel()
        self.attitude_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.convergence_label = QtWidgets.QLabel()
        self.convergence_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.convergence_label.setStyleSheet("color: gray; font-size: 10px;")

        self.dip_dir_spin.setValue(float(attitude[0]) % 360.0)
        self.dip_angle_spin.setValue(float(attitude[1]))
        self._sync_dial_from_spin()
        self._sync_slider_from_spin()

        # The point of it: recomputed while you drag, not on a button.
        self.dip_dir_dial.valueChanged.connect(self._on_dial_moved)
        self.dip_dir_spin.valueChanged.connect(self._on_dip_dir_typed)
        self.dip_angle_slider.valueChanged.connect(self._on_slider_moved)
        self.dip_angle_spin.valueChanged.connect(self._on_dip_angle_typed)

        self.side_spin = QtWidgets.QSpinBox()
        self.side_spin.setRange(100, min(4000, max(self.dem.width, self.dem.height)))
        self.side_spin.setSingleStep(100)
        self.side_spin.setValue(self.side)
        self.side_spin.setSuffix(" px")
        self.side_spin.valueChanged.connect(self._on_side_changed)

        self.side_label = QtWidgets.QLabel()
        self.side_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # The box is the authoritative source here too: `_apply_legend` takes no
        # argument, it reads where the legend goes off this widget.
        self.legend_combo = QtWidgets.QComboBox()
        for text, mode in self.LEGEND_PLACEMENTS:
            self.legend_combo.addItem(text, mode)

        modes = [mode for _, mode in self.LEGEND_PLACEMENTS]
        self.legend_combo.setCurrentIndex(modes.index(legend) if legend in modes else 0)
        self.legend_combo.currentIndexChanged.connect(lambda _: self._apply_legend())

        # The source point, typeable as well as draggable: in the field a
        # station has coordinates, and re-entering them by hunting with the
        # mouse is a way of losing them.
        # The range runs past the DEM by one of its widths on each side rather
        # than stopping at the edge: the plane is unbounded and the point
        # holding it up need not sit on it. Stopping at the edge would mean an
        # --x outside the DEM was silently pulled back inside, and the box would
        # say something different from the point being computed.
        span_x = self.dem.bounds.right - self.dem.bounds.left
        span_y = self.dem.bounds.top - self.dem.bounds.bottom

        self.easting_spin = QtWidgets.QDoubleSpinBox()
        self.easting_spin.setDecimals(1)
        self.easting_spin.setSingleStep(50.0)
        self.easting_spin.setRange(self.dem.bounds.left - span_x, self.dem.bounds.right + span_x)
        self.easting_spin.setPrefix("E ")

        self.northing_spin = QtWidgets.QDoubleSpinBox()
        self.northing_spin.setDecimals(1)
        self.northing_spin.setSingleStep(50.0)
        self.northing_spin.setRange(self.dem.bounds.bottom - span_y, self.dem.bounds.top + span_y)
        self.northing_spin.setPrefix("N ")

        # The elevation goes below sea level -- the foredeep reaches it around
        # here -- and above the highest summit, because a plane can rest on a
        # horizon standing in mid-air over today's topography.
        self.elevation_spin = QtWidgets.QDoubleSpinBox()
        self.elevation_spin.setDecimals(1)
        self.elevation_spin.setSingleStep(10.0)
        self.elevation_spin.setRange(-6000.0, 9000.0)
        self.elevation_spin.setPrefix("Z ")
        self.elevation_spin.setSuffix(" m")

        # On, the point crawls along the topography. Off, the elevation stays
        # the one typed and the plane comes off the ground: which is what a
        # projected horizon needs, or a measurement taken on a cliff face.
        self.follow_dem_check = QtWidgets.QCheckBox("elevation from DEM")
        self.follow_dem_check.setChecked(self.z_follows_dem)

        self.elevation_label = QtWidgets.QLabel()
        self.elevation_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.elevation_label.setStyleSheet("color: gray; font-size: 10px;")

        self._sync_point_boxes()

        self.easting_spin.valueChanged.connect(self._on_point_typed)
        self.northing_spin.valueChanged.connect(self._on_point_typed)
        self.elevation_spin.valueChanged.connect(self._on_elevation_typed)
        self.follow_dem_check.toggled.connect(self._on_follow_dem_toggled)

        controls = QtWidgets.QWidget()
        controls.setMaximumWidth(200)
        layout = QtWidgets.QVBoxLayout(controls)
        layout.addWidget(QtWidgets.QLabel("Dip direction"))
        layout.addWidget(self.dip_dir_dial)
        layout.addWidget(self.dip_dir_spin)
        layout.addWidget(QtWidgets.QLabel("Dip angle"))
        layout.addWidget(self.dip_angle_slider, stretch=1)
        layout.addWidget(self.dip_angle_spin)
        layout.addWidget(self.attitude_label)
        layout.addWidget(self.convergence_label)

        layout.addSpacing(8)
        layout.addWidget(QtWidgets.QLabel("Source point"))
        layout.addWidget(self.easting_spin)
        layout.addWidget(self.northing_spin)
        layout.addWidget(self.elevation_spin)
        layout.addWidget(self.follow_dem_check)
        layout.addWidget(self.elevation_label)

        layout.addSpacing(8)
        layout.addWidget(QtWidgets.QLabel("Compute window"))
        layout.addWidget(self.side_spin)
        layout.addWidget(self.side_label)

        layout.addSpacing(8)
        layout.addWidget(QtWidgets.QLabel("Legend"))
        layout.addWidget(self.legend_combo)

        layout.addSpacing(8)
        for text, slot in (
            ("Copy screenshot", self.copy_screenshot),
            ("Save screenshot...", self.save_screenshot),
            ("Save settings...", self.save_settings),
            ("Export trace...", self.export_traces),
        ):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            layout.addWidget(button)

        map_side = QtWidgets.QWidget()
        map_layout = QtWidgets.QVBoxLayout(map_side)
        map_layout.setContentsMargins(0, 0, 0, 0)
        map_layout.addWidget(self.toolbar)
        map_layout.addWidget(self.canvas, stretch=1)

        # The panel has five groups and no longer fits on a short screen: inside
        # a scroll area it shortens instead of cutting the buttons off.
        panel = QtWidgets.QScrollArea()
        panel.setWidget(controls)
        panel.setWidgetResizable(True)
        panel.setMaximumWidth(224)
        panel.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.addWidget(map_side, stretch=1)
        main_layout.addWidget(panel)
        self.setCentralWidget(central)

        self.statusBar().showMessage(
            "scroll to zoom; drag the yellow point, or click elsewhere to move it"
        )

    def _draw_base_map(self):
        self.shade_image = self.axes.imshow(
            self.dem.hillshade,
            cmap="gray",
            extent=self.dem.extent,
            origin="upper",
            interpolation="bilinear",
        )
        self.shade_step = self.dem.decimation
        epsg = self.dem.crs.to_epsg() if self.dem.crs else "?"
        self.axes.set_xlabel(f"E (m, EPSG:{epsg})")
        self.axes.set_ylabel("N (m)")
        self.axes.set_aspect("equal")

        if self.overlay is not None:
            self.overlay.draw(self.axes)

        # animated=True keeps these artists out of the normal draw: only
        # blitting redraws them, which is what keeps the loop inside the frame.
        #
        # A single Line2D with NaN separators, not a LineCollection: the
        # marching-squares chords are thousands of loose segments, and for
        # matplotlib one broken path costs 4-5 times less than as many separate
        # paths (measured: 2.0 ms against 9.1 on 1000x1000).
        (self.intersections,) = self.axes.plot(
            [], [], "-", color="red", linewidth=1.2, animated=True
        )

        (self.source_marker,) = self.axes.plot(
            [self.source_point[0]],
            [self.source_point[1]],
            marker="o",
            color="yellow",
            markeredgecolor="black",
            markersize=8,
            animated=True,
        )

        corner, width, height = self.window.rectangle_xy()
        self.window_patch = Rectangle(
            corner,
            width,
            height,
            fill=False,
            edgecolor="orange",
            linestyle="--",
            linewidth=1.0,
            animated=True,
        )
        self.axes.add_patch(self.window_patch)

        self._apply_legend()

        # The full view has to go at the bottom of the bar's stack, or "home"
        # takes you back to the first framing the bar happened to see, which is
        # some arbitrary point of the zoom and not the extent of the DEM.
        self.toolbar.update()
        self.toolbar.push_current()

    def _legend_handles(self):
        """
        The legend has to be built by hand: animated artists do not show up in
        the normal draw, and geopandas polygons carry no handler.
        """

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        handles = [
            Line2D([], [], color="red", linewidth=1.2, label="intersection"),
            Patch(facecolor="none", edgecolor="orange", linestyle="--", label="compute window"),
        ]

        if self.overlay is not None:
            handles.extend(self.overlay.legend_handles())

        return handles

    def _apply_legend(self):
        """
        Rebuilds the legend where the box says it goes, the map included.

        Rebuilt and not hidden: a legend made invisible beside the map would
        still hold its column, and the map would stay narrow to explain
        nothing. Outside the axes it is `figure.legend` and not `axes.legend`,
        because only that way does the layout reserve the column for it instead
        of letting it spill over; and it lives in the figure, not in a Qt widget
        alongside, or it would drop out of both outputs -- the saved screenshot
        and the copied one.
        """

        if self.legend is not None:
            self.legend.remove()
            self.legend = None

        mode = self.legend_combo.currentData()

        if mode != "hidden":
            # With distinct units the entries are a dozen instead of three, and
            # at normal body size they would not fit in height.
            categorized = self.overlay is not None and self.overlay.is_categorized
            style = dict(
                handles=self._legend_handles(),
                fontsize="x-small" if categorized else "small",
                framealpha=0.85,
            )

            self.legend = (
                self.axes.legend(loc="upper right", **style)
                if mode == "inside"
                else self.figure.legend(loc="outside right upper", **style)
            )

        # The axes box has just moved: the blitting background cut on the
        # previous one would be worth nothing now. The draw_event fired from
        # here recaptures it.
        self.canvas.draw()

    # -- interaction ------------------------------------------------------

    def _on_draw(self, event):
        """The background only changes on resize or zoom: here it is recaptured."""

        self.background = self.canvas.copy_from_bbox(self.axes.bbox)
        self._draw_animated()

    def _draw_animated(self):
        self.axes.draw_artist(self.window_patch)
        self.axes.draw_artist(self.intersections)
        self.axes.draw_artist(self.source_marker)

    def _near_source(self, event):
        """Nearness measured in screen pixels, not in metres: the threshold has
        to stay the same at every zoom scale."""

        px, py = self.axes.transData.transform(self.source_point[:2])

        return math.hypot(event.x - px, event.y - py) <= self.PICK_RADIUS_PX

    def _move_source(self, x, y):
        surface = self.dem.elevation_at(x, y)

        # With the box unticked the elevation is left alone: whoever typed it
        # wants it where it is, and a drag across the map is a gesture in the
        # horizontal plane. On nodata the previous one is kept rather than
        # refusing the move: breaking off a drag halfway is worse than resting
        # the plane on an elevation a few pixels old.
        if self.z_follows_dem and surface is not None:
            z = surface
        else:
            z = self.source_point[2]

        self.source_point = [x, y, z]
        self.source_marker.set_data([x], [y])
        self._sync_point_boxes()

        return surface is not None

    def _sync_point_boxes(self):
        """Puts the three boxes back on the point, without letting them answer."""

        for box, value in (
            (self.easting_spin, self.source_point[0]),
            (self.northing_spin, self.source_point[1]),
            (self.elevation_spin, self.source_point[2]),
        ):
            with QtCore.QSignalBlocker(box):
                box.setValue(value)

        self.elevation_spin.setEnabled(not self.z_follows_dem)
        self._report_elevation()

    def _report_elevation(self):
        """The gap from the ground, the one thing the elevation alone does not say."""

        surface = self.dem.elevation_at(self.source_point[0], self.source_point[1])

        if surface is None:
            self.elevation_label.setText("outside DEM")
            return

        if self.z_follows_dem:
            self.elevation_label.setText(f"ground {surface:.0f} m")
            return

        gap = self.source_point[2] - surface
        self.elevation_label.setText(f"ground {surface:.0f} m\n{gap:+.0f} m from ground")

    def _on_point_typed(self, value):
        """Coordinates typed by hand: the point goes where they say, the window follows."""

        x = float(self.easting_spin.value())
        y = float(self.northing_spin.value())

        self._move_source(x, y)
        self._recenter_window()
        self.update_intersection()

    def _on_elevation_typed(self, value):
        self.source_point[2] = float(value)
        self._report_elevation()
        self.update_intersection()

    def _on_follow_dem_toggled(self, checked):
        """
        Hooking the elevation back onto the DEM puts it on the ground at once.

        Not the other way round: unticking leaves the elevation where it was,
        which is the natural starting point for moving it a little.
        """

        self.z_follows_dem = bool(checked)

        if checked:
            surface = self.dem.elevation_at(self.source_point[0], self.source_point[1])

            if surface is not None:
                self.source_point[2] = surface

        self._sync_point_boxes()
        self.update_intersection()

    def _recenter_window(self):
        """Re-reads the window if the point has left it. True if it changed."""

        fresh = self.dem.window_at(self.source_point[0], self.source_point[1], self.side)

        if fresh.offset == self.window.offset:
            return False

        self.window = fresh
        corner, width, height = fresh.rectangle_xy()
        self.window_patch.set_xy(corner)
        self.window_patch.set_width(width)
        self.window_patch.set_height(height)

        return True

    def dip_direction(self):
        """
        Dip direction as it is measured in the field: azimuth from true north.

        This is the number the dial shows and the one that goes into the
        exports, because it is what a compass reads once declination has been
        corrected. The kernel, instead, works on the grid, and wants
        `grid_dip_direction`.
        """

        return float(self.dip_dir_spin.value())

    def set_dip_direction(self, azimuth):
        self.dip_dir_spin.setValue(float(azimuth) % 360.0)

    def _sync_dial_from_spin(self):
        with QtCore.QSignalBlocker(self.dip_dir_dial):
            self.dip_dir_dial.setValue(
                int(round(self.dip_dir_spin.value() - self.DIAL_NORTH_OFFSET)) % 360
            )

    def _sync_slider_from_spin(self):
        with QtCore.QSignalBlocker(self.dip_angle_slider):
            self.dip_angle_slider.setValue(int(round(self.dip_angle_spin.value())))

    def _on_dial_moved(self, value):
        """The dial moves the box, and the box is what commands."""

        with QtCore.QSignalBlocker(self.dip_dir_spin):
            self.dip_dir_spin.setValue(float((value + self.DIAL_NORTH_OFFSET) % 360))

        self.update_intersection()

    def _on_dip_dir_typed(self, value):
        self._sync_dial_from_spin()
        self.update_intersection()

    def _on_slider_moved(self, value):
        with QtCore.QSignalBlocker(self.dip_angle_spin):
            self.dip_angle_spin.setValue(float(value))

        self.update_intersection()

    def _on_dip_angle_typed(self, value):
        self._sync_slider_from_spin()
        self.update_intersection()

    def grid_dip_direction(self):
        """Dip direction turned onto grid north, which is what the DEM has."""

        return self.convergence.to_grid(
            self.dip_direction(), self.source_point[0], self.source_point[1]
        )

    def convergence_here(self):
        return self.convergence.at(self.source_point[0], self.source_point[1])

    def source_geographic(self):
        """The source point in longitude and latitude, or None."""

        return self.convergence.geographic(self.source_point[0], self.source_point[1])

    def dip_angle(self):
        return float(self.dip_angle_spin.value())

    def _navigating(self):
        """True while pan or rubber-band zoom are active in the bar.

        Without this check a pan would drag the source point along too, because
        the two gestures are the same one: left button held down and moved."""

        return bool(self.toolbar.mode)

    def _on_press(self, event):
        if self._navigating() or event.inaxes is not self.axes or event.xdata is None:
            return

        if self._near_source(event):
            self.dragging = True
            return

        self._move_source(event.xdata, event.ydata)
        self._recenter_window()
        self.update_intersection()

    def _on_scroll(self, event):
        """Zoom around the cursor, which stays put on the point it was on."""

        if event.inaxes is not self.axes or event.xdata is None:
            return

        factor = 1.0 / self.ZOOM_STEP if event.button == "up" else self.ZOOM_STEP

        for axis, limits, anchor in (
            (self.axes.set_xlim, self.axes.get_xlim(), event.xdata),
            (self.axes.set_ylim, self.axes.get_ylim(), event.ydata),
        ):
            low, high = limits
            axis((anchor + (low - anchor) * factor, anchor + (high - anchor) * factor))

        # Every notch goes on the stack, so the bar's back/forward arrows
        # retrace the zooms made with the wheel as well.
        self.toolbar.push_current()

        # The background has changed: a full draw is needed, and the draw_event
        # recaptures it for the blitting of the frames that follow.
        self.canvas.draw()
        self.schedule_shade_refresh()

    def schedule_shade_refresh(self, delay_ms=180):
        self.shade_timer.start(delay_ms)

    def _refresh_shade(self):
        """Re-reads the hillshade for the current view, if anything changes."""

        xmin, xmax = self.axes.get_xlim()
        ymin, ymax = self.axes.get_ylim()

        result = self.dem.shade_for(xmin, xmax, ymin, ymax)
        if result is None:
            return

        shade, extent, step = result
        if step == self.shade_step and extent == list(self.shade_image.get_extent()):
            return

        started = perf_counter()

        # set_extent rescales the axes if you let it, and the view would jump on
        # every reload: the limits have to be put back the way they were.
        limits = self.axes.get_xlim(), self.axes.get_ylim()
        self.shade_image.set_data(shade)
        self.shade_image.set_extent(extent)
        self.axes.set_xlim(limits[0])
        self.axes.set_ylim(limits[1])
        self.shade_step = step

        self.canvas.draw()

        metres = self.dem.res_x * step
        self.statusBar().showMessage(
            f"background redrawn at {metres:.0f} m/cell in {(perf_counter() - started) * 1000:.0f} ms"
        )

    def _on_motion(self, event):
        if not self.dragging or event.inaxes is not self.axes or event.xdata is None:
            return

        # During a drag the window stays put: re-reading it at every step would
        # cost 4.9 ms on 1000x1000, and the trace inside the window is right
        # anyway, because the plane is unbounded and the source point need not
        # sit inside it. It recentres on release.
        self._move_source(event.xdata, event.ydata)
        self.update_intersection()

    def _on_release(self, event):
        if not self.dragging:
            return

        self.dragging = False

        if self._recenter_window():
            self.update_intersection()

    def _on_side_changed(self, value):
        self.side = value
        self.window = self.dem.window_at(self.source_point[0], self.source_point[1], self.side)

        corner, width, height = self.window.rectangle_xy()
        self.window_patch.set_xy(corner)
        self.window_patch.set_width(width)
        self.window_patch.set_height(height)

        self.update_intersection()

    # -- loop -------------------------------------------------------------

    def update_intersection(self):
        dip_angle = self.dip_angle()

        # The dial is in true azimuth, the DEM is on the grid: convergence sits
        # between the two and has to come off before the kernel is called.
        convergence = self.convergence_here()
        grid_dip_dir = (self.dip_direction() - convergence) % 360.0

        start = perf_counter()
        points, segments = intersect_plane_grid(
            self.window.data,
            self.window.geotransform,
            self.source_point,
            grid_dip_dir,
            dip_angle,
            self.dem.nodata,
        )
        kernel_done = perf_counter()

        self.last_result = (points, segments)

        # points is (N, 3) in map coordinates, segments (M, 2) of indices. The
        # chords become a single path: end, end, NaN, and the NaN breaks the
        # line between one chord and the next.
        if len(segments):
            chords = points[segments][:, :, :2]
            path = np.full((len(chords) * 3, 2), np.nan)
            path[0::3] = chords[:, 0]
            path[1::3] = chords[:, 1]
            self.intersections.set_data(path[:, 0], path[:, 1])
        else:
            self.intersections.set_data([], [])

        if self.background is None:
            self.canvas.draw()
        else:
            self.canvas.restore_region(self.background)
            self._draw_animated()
            self.canvas.blit(self.axes.bbox)

        self.canvas.flush_events()
        drawn = perf_counter()

        self.frame_times.append(drawn - start)
        self._report(
            grid_dip_dir,
            convergence,
            dip_angle,
            len(points),
            kernel_done - start,
            drawn - kernel_done,
        )

    def _report(self, grid_dip_dir, convergence, dip_angle, n_points, kernel_s, draw_s):
        # On the label the true number, the one you measure and write in the
        # notebook; below it, spelled out, what the grid makes of it.
        self.attitude_label.setText(f"{self.dip_direction():03.0f} / {dip_angle:02.0f}")
        self.convergence_label.setText(
            f"true north\nconvergence {convergence:+.2f}°\ngrid {grid_dip_dir:05.1f}°"
            if self.convergence.available
            else "grid north\n(convergence not\ncomputable)"
        )

        rows, cols = self.window.shape
        km = cols * self.dem.res_x / 1000.0
        self.side_label.setText(f"{cols}x{rows} = {km:.1f} km")

        mean_frame = sum(self.frame_times) / len(self.frame_times)
        fps = 1.0 / mean_frame if mean_frame else 0.0

        self.statusBar().showMessage(
            f"{n_points} points   "
            f"kernel {kernel_s * 1000:5.1f} ms   "
            f"draw {draw_s * 1000:5.1f} ms   "
            f"total {(kernel_s + draw_s) * 1000:5.1f} ms   "
            f"{fps:4.1f} fps"
        )

    # -- outputs ----------------------------------------------------------

    def _rendered_figure(self, path, dpi=150):
        """
        Saves the figure with the animated artists in it too.

        An animated artist takes no part in the normal draw, so savefig on its
        own would give back the map without the trace. Here they are turned
        off, it is saved, they are turned back on, and the final draw remakes
        the blitting background.
        """

        animated = [self.intersections, self.source_marker, self.window_patch]

        for artist in animated:
            artist.set_animated(False)

        try:
            self.figure.savefig(path, dpi=dpi)
        finally:
            for artist in animated:
                artist.set_animated(True)
            self.canvas.draw()

    def copy_screenshot(self):
        QtWidgets.QApplication.clipboard().setPixmap(self.canvas.grab())
        self.statusBar().showMessage("screenshot copied to the clipboard")

    def save_screenshot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save screenshot", str(self._suggested_name(".png")), "PNG (*.png)"
        )
        if not path:
            return

        self._rendered_figure(path)
        self.statusBar().showMessage(f"screenshot saved to {path}")

    def save_settings(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save settings", str(self._suggested_name(".json")), "JSON (*.json)"
        )
        if not path:
            return

        # The reference has to be written out in full: a dip direction without
        # the north it rests on is ambiguous by almost a degree around here.
        #
        # And the point goes in both forms. The projected ones are what the
        # kernel runs on, but without the geographic ones the file cannot be
        # read outside its EPSG -- and the EPSG is the very line that goes
        # missing first.
        lon_lat = self.source_geographic()

        settings = {
            "dem": str(self.dem.path),
            "dip_dir": self.dip_direction(),
            "dip_dir_reference": "true north",
            "dip_dir_grid": self.grid_dip_direction(),
            "meridian_convergence": self.convergence_here(),
            "dip_angle": self.dip_angle(),
            "source_point": [float(v) for v in self.source_point],
            "source_point_reference": (
                f"EPSG:{self.dem.crs.to_epsg()}" if self.dem.crs else "unknown"
            ),
            "source_lon": lon_lat[0] if lon_lat else None,
            "source_lat": lon_lat[1] if lon_lat else None,
            "source_lon_lat_reference": "EPSG:4326",
            # Without these two lines an elevation lifted off the ground reads
            # back as a DEM lookup gone wrong, instead of as the choice it was:
            # it has to say the lift is deliberate, and by how much.
            "source_z_from_dem": bool(self.z_follows_dem),
            "dem_elevation": self.dem.elevation_at(self.source_point[0], self.source_point[1]),
            "window_px": int(self.side),
            "epsg": self.dem.crs.to_epsg() if self.dem.crs else None,
        }

        Path(path).write_text(json.dumps(settings, indent=2), encoding="utf-8")
        self.statusBar().showMessage(f"settings saved to {path}")

    def export_traces(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export trace", str(self._suggested_name(".shp")), "Shapefile (*.shp)"
        )
        if not path:
            return

        import geopandas as gpd

        points, segments = self.last_result
        traces = merged_traces(points, segments)

        if not traces:
            self.statusBar().showMessage("no intersection to export")
            return

        dip_dir = self.dip_direction()
        dip_angle = self.dip_angle()

        # Names within the ten characters a shapefile allows, and both azimuths
        # present: whoever reopens the file should not have to guess which
        # north. Same reason for the two pairs of coordinates: the .prj says
        # which EPSG src_x and src_y are in, but the .prj is the file that goes
        # missing. `src_z_dem` is the ground elevation under the point: if it
        # differs from `src_z` the plane was lifted on purpose, and without the
        # comparison that would look like a mistake.
        lon_lat = self.source_geographic()
        surface = self.dem.elevation_at(self.source_point[0], self.source_point[1])

        frame = gpd.GeoDataFrame(
            {
                "dip_dir": [dip_dir] * len(traces),
                "dipdir_grd": [self.grid_dip_direction()] * len(traces),
                "converg": [self.convergence_here()] * len(traces),
                "dip": [dip_angle] * len(traces),
                "src_x": [self.source_point[0]] * len(traces),
                "src_y": [self.source_point[1]] * len(traces),
                "src_z": [self.source_point[2]] * len(traces),
                "src_lon": [lon_lat[0] if lon_lat else None] * len(traces),
                "src_lat": [lon_lat[1] if lon_lat else None] * len(traces),
                "src_z_dem": [surface] * len(traces),
                "z_from_dem": [bool(self.z_follows_dem)] * len(traces),
            },
            geometry=traces,
            crs=self.dem.crs,
        )
        frame.to_file(path, driver="ESRI Shapefile")

        self.statusBar().showMessage(f"{len(traces)} traces exported to {path}")

    def _suggested_name(self, suffix):
        stem = self.dem.path.stem
        attitude = f"{int(self.dip_direction()):03d}-{int(self.dip_angle()):02d}"

        return self.dem.path.with_name(f"{stem}_{attitude}{suffix}")


def split_layer(spec, role):
    """
    `path` or `path:layer`, resolved without guessing.

    A path can hold a colon of its own, so the rule is to look at the disk
    rather than read the string: if the whole thing is an existing file, it is
    a path; otherwise the last piece is peeled off and tried.
    """

    if spec is None:
        return None

    if Path(spec).exists():
        return dict(path=spec, role=role, layer=None)

    head, _, tail = spec.rpartition(":")

    if head and Path(head).exists():
        return dict(path=head, role=role, layer=tail)

    return dict(path=spec, role=role, layer=None)


def fit_to_screen(window, width, height):
    """
    Shows the window at the wanted size, or maximised if it does not fit.

    1180x880 is the right measure for the map plus the panel, but on a 1366x768
    screen the bottom of the window -- the buttons and the status bar -- ends up
    under the edge, and unlike the buttons the status bar cannot be reached by
    scrolling the panel.

    Maximised rather than resized to the available area, because the window
    frame is not ours to measure: right after show() the title bar does not
    exist yet, and it only appears once the window manager has reparented the
    window, some hundred milliseconds later -- a delay there is no honest way to
    wait for. Maximising hands the arithmetic to the window manager, which knows
    how thick its own decorations are. The one case this leaves rough is a
    screen tall enough for the client area but not for the title bar too, where
    the window sticks out by the height of that bar.
    """

    available = window.screen().availableGeometry()

    if width > available.width() or height > available.height():
        window.showMaximized()
        return

    window.resize(width, height)
    window.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dem",
        nargs="?",
        help="DEM in any format rasterio can read; a dialog asks for it if missing",
    )
    for role, explanation in (
        ("polygons", "outcrops, units, any fill"),
        ("lines", "faults, contacts, traces"),
        ("points", "stations, measurements, samples"),
    ):
        parser.add_argument(
            f"--{role}",
            metavar="PATH[:LAYER]",
            help=f"{role} layer to put under the plane ({explanation})",
        )
    parser.add_argument(
        "--geology",
        metavar="GPKG",
        help="shortcut: carbonates as polygons and faults as lines from the same geopackage",
    )
    parser.add_argument(
        "--categories",
        metavar="FIELD",
        default="code",
        help=(
            "field to colour the polygons on (default 'code'); "
            "'none' for a single colour"
        ),
    )
    parser.add_argument("--x", type=float, help="easting of the source point (default: DEM centre)")
    parser.add_argument("--y", type=float, help="northing of the source point (default: DEM centre)")
    parser.add_argument(
        "--z",
        type=float,
        help=(
            "elevation of the source point (default: the DEM's); "
            "if given, the plane stays at this elevation as you move it"
        ),
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1000,
        metavar="N",
        help="side in cells of the compute window (default 1000)",
    )
    parser.add_argument(
        "--settings",
        metavar="JSON",
        help="saved attitude to start from",
    )
    parser.add_argument(
        "--legend",
        choices=[mode for _, mode in RealtimeWindow.LEGEND_PLACEMENTS],
        default="beside",
        help="where to put the legend at startup (default 'beside'); changeable from the panel",
    )
    args = parser.parse_args()

    attitude = (90.0, 30.0)
    point = (args.x, args.y, args.z)
    z_follows_dem = None
    side = args.window

    if args.settings:
        saved = json.loads(Path(args.settings).read_text(encoding="utf-8"))
        attitude = (saved["dip_dir"], saved["dip_angle"])

        # The Italian keys are still read: files written before the interface
        # changed language are the same attitudes, and refusing them would be a
        # rename breaking data it had no business touching.
        side = saved.get("window_px", saved.get("finestra_px", side))
        z_follows_dem = saved.get("source_z_from_dem", saved.get("source_z_dal_dem"))

        # Explicit arguments beat the file: passing --z together with a saved
        # attitude means the command line is the one just written.
        stored = saved.get("source_point") or (None, None, None)
        point = tuple(
            given if given is not None else was for given, was in zip(point, stored)
        )

        if args.z is not None:
            z_follows_dem = False

        print(f"settings: {attitude[0]:.0f}/{attitude[1]:.0f}, window {side} px")

    vectors = [
        spec
        for spec in (
            split_layer(args.polygons, "polygons"),
            split_layer(args.lines, "lines"),
            split_layer(args.points, "points"),
        )
        if spec
    ]

    # The historical shortcut, kept because it is how this tool has always been
    # launched: the two layers of geology.gpkg in their natural roles.
    if args.geology:
        vectors.append(dict(path=args.geology, role="polygons", layer="carbonates"))
        vectors.append(dict(path=args.geology, role="lines", layer="faults"))

    for spec in vectors:
        spec.setdefault(
            "category_field",
            None if args.categories == "none" or spec["role"] != "polygons" else args.categories,
        )

    # The QApplication before any dialog, or Qt exits without saying why.
    app = QtWidgets.QApplication(sys.argv)

    if not args.dem:
        dialog = SourcesDialog(dem=args.dem, vectors=vectors, point=point)

        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        chosen = dialog.choices()
        args.dem = chosen["dem"]
        vectors = chosen["vectors"]
        point = chosen["point"]

        # An elevation typed into the dialog is a choice like the one from the
        # command line, and counts the same way.
        if point[2] is not None:
            z_follows_dem = False

    dem = Dem(args.dem)
    print(
        f"DEM {dem.width}x{dem.height} ({dem.width * dem.height / 1e6:.1f} Mpx), "
        f"EPSG:{dem.crs.to_epsg()}, nodata={dem.nodata}, "
        f"background decimated 1:{dem.decimation}"
    )

    overlay = Overlay(
        VectorSource(
            spec["path"],
            spec["role"],
            dem.crs,
            dem.bounds,
            layer=spec.get("layer"),
            category_field=spec.get("category_field"),
        )
        for spec in vectors
    )

    if vectors:
        print(f"vectors: {overlay.summary()}")

    window = RealtimeWindow(
        dem,
        overlay,
        side=side,
        attitude=attitude,
        source=point,
        z_follows_dem=z_follows_dem,
        legend=args.legend,
    )
    fit_to_screen(window, 1180, 880)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
