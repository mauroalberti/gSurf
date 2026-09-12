"""
Real-time geological plane / DEM intersection.

The misah kernel intersects an unbounded plane with the grid and returns the
marching-squares chords; around it is the minimum needed to steer it by hand
and watch the answer move. The status bar reports kernel, drawing and frame
rate on every frame, so the cost stays visible while you work.

Usage:
    python -m gsurf                     and pick it from the launcher
    python -m gsurf.tools.intersection
    python -m gsurf.tools.intersection <dem.tif> [--polygons PATH[:LAYER]]
                                       [--lines PATH[:LAYER]] [--points PATH[:LAYER]]
                                       [--categories FIELD] [--x E] [--y N] [--z Z]
                                       [--window N] [--settings <file.json>]

With no arguments a dialog asks for the files. The DEM is the one it marks as
required: the three vector slots -- polygons, lines, points -- answer where the
plane is being laid down, which is a different question from the calculation.
The layers offered in each slot are filtered on geometry read from the
metadata, so faults never appear among the polygons. The source point is not
asked for, because it is the one thing here that is set by pointing at it.

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

# PyQt6 has to be imported before the backend: matplotlib picks the binding by
# looking at what is already in sys.modules.
import PyQt6.QtCore  # noqa: F401
from PyQt6 import QtCore, QtWidgets

from matplotlib.patches import Rectangle

from misah.kernels import intersect_plane_grid

from gsurf.mapview import LegendControls, MapView, fit_to_screen
from gsurf.sources import SourcesDialog, open_session
from gsurf.vectors import split_layer

# What this tool can be opened on. The DEM is the surface being intersected, so
# there is no version of this without one; the three vector slots answer where
# the plane is being laid down, which is a different question from the
# calculation and one it runs perfectly well without.
WANTS = dict(dem="required", polygons="optional", lines="optional", points="optional")


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


class RealtimeWindow(QtWidgets.QMainWindow):
    """
    The plane on the DEM: the controls that aim it, and the loop that redraws it.

    Everything that is not about this one calculation -- the map underneath,
    the blitting, the navigation, the legend -- lives in `MapView`, so that the
    next tool inherits it instead of copying it.
    """

    PICK_RADIUS_PX = 12

    # QDial puts its minimum at six o'clock, not at twelve: measured by
    # grabbing the widget and hunting for the needle, value 0 points 181 degrees
    # from twelve o'clock and value 180 points to 360. It runs clockwise, like
    # an azimuth, so between the widget's scale and geological dip direction
    # there is only half a turn of offset.
    DIAL_NORTH_OFFSET = 180

    def __init__(
        self,
        session,
        side=1000,
        attitude=(90.0, 30.0),
        source=None,
        z_follows_dem=None,
        legend="beside",
    ):
        super().__init__()

        if session.dem is None:
            raise ValueError("the plane / DEM intersection needs a DEM")

        self.session = session
        self.dem = session.dem
        self.dragging = False
        self.frame_times = deque(maxlen=20)
        self.convergence = session.convergence
        self.last_result = ([], [])

        dem = self.dem

        # The three components are independent: you can fix the elevation alone
        # and let the point sit at the centre, or the other way round.
        x, y, z = (tuple(source) + (None, None, None))[:3] if source else (None, None, None)
        centre_x, centre_y = session.center()

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

        self.setWindowTitle(f"gSurf - real-time intersection - {session.label}")
        self._build_ui(attitude, legend)
        self._draw_base_map()

        self.update_intersection()

        # After the first frame and not before it: update_intersection ends by
        # reporting what that frame cost, so a message set during construction
        # was overwritten before it could ever be read.
        self.statusBar().showMessage(self._opening_hint)

    # -- construction -----------------------------------------------------

    def _build_ui(self, attitude, legend):
        self.map_view = MapView(self.session, legend=legend)

        # The legend needs the entries for the artists this tool draws; the
        # backdrop's the map adds by itself.
        self.map_view.legend_handles_provider = self._legend_handles

        self.map_view.pressed.connect(self._on_map_pressed)
        self.map_view.dragged.connect(self._on_map_dragged)
        self.map_view.released.connect(self._on_map_released)
        self.map_view.save_requested.connect(self.save_screenshot)
        self.map_view.status.connect(self.statusBar().showMessage)

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

        # Where the legend goes is the map's business; the box that says so is
        # a control, and belongs in the panel with the others.
        self.legend_controls = LegendControls(self.map_view, legend)
        self.legend_combo = self.legend_controls.combo

        # The source point, typeable as well as draggable: in the field a
        # station has coordinates, and re-entering them by hunting with the
        # mouse is a way of losing them.
        # The range runs past the DEM by one of its widths on each side rather
        # than stopping at the edge: the plane is unbounded and the point
        # holding it up need not sit on it. Stopping at the edge would mean an
        # --x outside the DEM was silently pulled back inside, and the box would
        # say something different from the point being computed.
        left, bottom, right, top = self.session.bounds
        span_x = right - left
        span_y = top - bottom

        self.easting_spin = QtWidgets.QDoubleSpinBox()
        self.easting_spin.setDecimals(1)
        self.easting_spin.setSingleStep(50.0)
        self.easting_spin.setRange(left - span_x, right + span_x)
        self.easting_spin.setPrefix("E ")

        self.northing_spin = QtWidgets.QDoubleSpinBox()
        self.northing_spin.setDecimals(1)
        self.northing_spin.setSingleStep(50.0)
        self.northing_spin.setRange(bottom - span_y, top + span_y)
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
        layout.addWidget(self.legend_controls)

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
        main_layout.addWidget(self.map_view, stretch=1)
        main_layout.addWidget(panel)
        self.setCentralWidget(central)

        hint = "scroll to zoom; drag the yellow point, or click elsewhere to move it"

        # Said only when there is something to click. The two entries this tool
        # draws are what the hand is steering and switch nothing, so it takes a
        # backdrop -- categorised or not, a layer is switched by its own entry.
        if self.session.overlay:
            hint += " - click a legend entry to take what it names off the map"

        self._opening_hint = hint

    def _draw_base_map(self):
        axes = self.map_view.axes

        self.map_view.draw_base_map()

        # Registered with the map as animated: they stay out of the normal draw
        # and only blitting redraws them, which is what keeps the loop inside
        # the frame.
        #
        # A single Line2D with NaN separators, not a LineCollection: the
        # marching-squares chords are thousands of loose segments, and for
        # matplotlib one broken path costs 4-5 times less than as many separate
        # paths (measured: 2.0 ms against 9.1 on 1000x1000).
        (line,) = axes.plot([], [], "-", color="red", linewidth=1.2)
        self.intersections = self.map_view.add_animated(line)

        (marker,) = axes.plot(
            [self.source_point[0]],
            [self.source_point[1]],
            marker="o",
            color="yellow",
            markeredgecolor="black",
            markersize=8,
        )
        self.source_marker = self.map_view.add_animated(marker)

        corner, width, height = self.window.rectangle_xy()
        self.window_patch = self.map_view.add_animated(
            Rectangle(
                corner,
                width,
                height,
                fill=False,
                edgecolor="orange",
                linestyle="--",
                linewidth=1.0,
            )
        )
        axes.add_patch(self.window_patch)

        self.map_view.refresh_legend()
        self.map_view.anchor_home()

    def _legend_handles(self):
        """
        This tool's own legend entries. Built by hand because animated artists
        do not show up in the normal draw.
        """

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        return [
            Line2D([], [], color="red", linewidth=1.2, label="intersection"),
            Patch(facecolor="none", edgecolor="orange", linestyle="--", label="compute window"),
        ]

    # -- interaction ------------------------------------------------------

    def _near_source(self, x, y):
        """Nearness measured in screen pixels, not in metres: the threshold has
        to stay the same at every zoom scale."""

        px, py = self.map_view.display_xy(*self.source_point[:2])
        ex, ey = self.map_view.display_xy(x, y)

        return math.hypot(ex - px, ey - py) <= self.PICK_RADIUS_PX

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

    def _on_map_pressed(self, x, y):
        """A click on the map: on the point it starts a drag, elsewhere it
        moves the point there. Pan and zoom never get this far."""

        if self._near_source(x, y):
            self.dragging = True
            return

        self._move_source(x, y)
        self._recenter_window()
        self.update_intersection()

    def _on_map_dragged(self, x, y):
        if not self.dragging:
            return

        # During a drag the window stays put: re-reading it at every step would
        # cost 4.9 ms on 1000x1000, and the trace inside the window is right
        # anyway, because the plane is unbounded and the source point need not
        # sit inside it. It recentres on release.
        self._move_source(x, y)
        self.update_intersection()

    def _on_map_released(self):
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

        self.map_view.blit()
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

    def copy_screenshot(self):
        self.map_view.copy_screenshot()

    def save_screenshot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save screenshot", str(self._suggested_name(".png")), "PNG (*.png)"
        )
        if not path:
            return

        self.map_view.rendered_figure(path)
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
                f"EPSG:{self.session.epsg}" if self.session.epsg else "unknown"
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
            "epsg": self.session.epsg,
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
            crs=self.session.crs,
        )
        frame.to_file(path, driver="ESRI Shapefile")

        self.statusBar().showMessage(f"{len(traces)} traces exported to {path}")

    def _suggested_name(self, suffix):
        attitude = f"{int(self.dip_direction()):03d}-{int(self.dip_angle()):02d}"

        return self.session.suggested_name(suffix, tag=attitude)


def build(session, chosen, legend="beside"):
    """
    The window, on a session somebody else has already opened.

    This is what the launcher calls, and it takes nothing out of `chosen` that
    the session does not already hold: the source point starts at the centre of
    the map and is moved by clicking on it, the window side and the attitude
    have defaults, and all three are changed from inside the window anyway. The
    argument is taken all the same, so that every tool is built the same way.
    """

    window = RealtimeWindow(session, legend=legend)
    fit_to_screen(window, 1180, 880)

    return window


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
        choices=[mode for _, mode in MapView.LEGEND_PLACEMENTS],
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

    chosen = dict(dem=args.dem)

    for spec in (
        split_layer(args.polygons, "polygons"),
        split_layer(args.lines, "lines"),
        split_layer(args.points, "points"),
    ):
        if spec:
            chosen[spec["role"]] = spec

    # The historical shortcut, kept because it is how this tool has always been
    # launched: the two layers of geology.gpkg in their natural roles. It fills
    # the slots left empty rather than adding to them -- naming a layer outright
    # is the more particular statement of the two, and wins.
    if args.geology:
        chosen.setdefault("polygons", dict(path=args.geology, role="polygons", layer="carbonates"))
        chosen.setdefault("lines", dict(path=args.geology, role="lines", layer="faults"))

    for role in ("polygons", "lines", "points"):
        if chosen.get(role):
            chosen[role].setdefault(
                "category_field",
                None if args.categories == "none" or role != "polygons" else args.categories,
            )

    # The QApplication before any dialog, or Qt exits without saying why.
    app = QtWidgets.QApplication(sys.argv)

    if not args.dem:
        dialog = SourcesDialog(
            wants=WANTS,
            chosen=chosen,
            title="gSurf - plane on a DEM",
        )

        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        chosen = dialog.choices()

    session = open_session(chosen)
    print(f"session: {session.summary()}, nodata={session.dem.nodata}")

    if session.overlay:
        print(f"vectors: {session.overlay.summary()}")

    window = RealtimeWindow(
        session,
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
