"""
Fold axes under a moving window.

Drag a circle across the map and watch the bedding poles inside it fall onto a
stereonet, with the best-fit girdle and the axis they turn about recomputed on
every frame. The question a fold axis answers is local -- a plunge measured
over a whole map sheet is the average of every structure on it -- so the tool
is a window you move rather than a button you press.

Usage:
    python -m gsurf               and pick it from the launcher
    python -m gsurf.tools.fold_axes
    python -m gsurf.tools.fold_axes <attitudes.gpkg[:layer]> --dip-dir FIELD --dip FIELD
                        [--dem DEM] [--polygons PATH[:LAYER]] [--lines PATH[:LAYER]]
                        [--radius M] [--step M] [--strike-rhr]
                        [--min-points N] [--max-k K]

With no arguments a dialog asks for the files, and asks for the two angle
fields by offering the layer's numeric columns rather than making you remember
what they are called. The attitude layer is the one it marks as required, and
it is what the session is built on: with no DEM the projection and the extent
come from the layer itself. A DEM, if given, is backdrop and nothing else --
this calculation never reads an elevation.

What the window reports is not one number but four, and the fourth is the one
that decides whether the other three mean anything:

    axis      the minimum eigenvector of the orientation tensor of the poles
    K, C      Woodcock's shape and strength
    n         how many attitudes it was computed from
    verdict   whether the poles form a girdle at all

Above K = 1 the poles cluster instead of spreading, which is a homocline, and
the minimum eigenvector of a cluster is the least determined direction in the
data rather than a fold axis. On the Potenza-Irsina sheet three windows in four
fail that test. The axis is still drawn when it fails, in grey rather than
hidden: watching it turn colour as the window crosses a hinge is the point of a
live net, and a blank stereonet would say the same thing as no data at all.

The same question can be asked everywhere at once: `Compute grid` puts a window
at every node of a square grid `--step` apart and draws a tick where one passes,
coloured by plunge. Moving a threshold afterwards re-decides the whole field
without recomputing a tensor -- K, C and the count are what the gate reads and
they are already there -- which is the only practical way to see how much of a
map depends on where the threshold was put.

Attitudes are read in true azimuth, as they are measured. The map is on the
projection's grid, so the axis is turned onto the grid before it is drawn --
meridian convergence, taken at the window's centre. Both bearings go into the
export.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

# PyQt6 has to be imported before the backend: matplotlib picks the binding by
# looking at what is already in sys.modules.
import PyQt6.QtCore  # noqa: F401
from PyQt6 import QtCore, QtWidgets

from matplotlib.patches import Circle

from gsurf.attitudes import AttitudeSource
from gsurf.folds import (
    Gate,
    describe_sampling,
    field_cost,
    fold_axis,
    fold_axis_field,
    grid_centres,
)
from gsurf.mapview import LegendControls, MapView, fit_to_screen
from gsurf.sources import SourcesDialog, open_session
from gsurf.stereonet import StereonetView
from gsurf.vectors import split_layer

# What this tool can be opened on. The attitudes are the data itself, so there
# is nothing to compute without them; the DEM is backdrop here, which is why it
# sits in the optional half of a tool that is otherwise about the same map as
# the one that cannot run without it.
WANTS = dict(attitudes="required", dem="optional", polygons="optional", lines="optional")


class FoldAxesWindow(QtWidgets.QMainWindow):
    """
    The map with a window on it, the stereonet of what is inside, and the
    verdict on whether that is a fold.
    """

    PICK_RADIUS_PX = 12

    def __init__(self, session, attitudes, radius=2000.0, step=None, gate=None, legend="beside"):
        super().__init__()

        self.session = session
        self.attitudes = attitudes
        self.gate = gate or Gate()
        self.radius = float(radius)
        self.step = float(step) if step else max(250.0, self.radius / 2.0)
        self.dragging = False
        self.result = None
        self.indices = np.empty(0, dtype=int)
        self.field = None

        # The net is drawn only while its window is open, so these two say
        # whether what is on it is the window we are in, and whether it has
        # been put where it goes yet.
        self._net_is_current = False
        self._net_placed = False

        self.centre = list(session.center())

        self.setWindowTitle(f"gSurf - fold axes - {session.label}")
        self._build_ui(legend)
        self._draw_base_map()

        self.update_window()

        # After the first window and not before it: update_window ends by
        # reporting the frame it drew, so a message set during construction was
        # overwritten before it could ever be read.
        self.statusBar().showMessage(self._opening_hint)

    # -- construction -----------------------------------------------------

    def _build_ui(self, legend):
        self.map_view = MapView(self.session, legend=legend)
        self.map_view.legend_handles_provider = self._legend_handles

        self.map_view.pressed.connect(self._on_map_pressed)
        self.map_view.dragged.connect(self._on_map_dragged)
        self.map_view.released.connect(self._on_map_released)
        self.map_view.save_requested.connect(self.save_screenshot)
        self.map_view.status.connect(self.statusBar().showMessage)

        # The net in a window of its own, floating over the map.
        #
        # It is the half of the answer you watch while the other hand drags the
        # window, and in the panel it was as wide as the panel let it be. A
        # floating dock is a Qt::Tool window: it stays above the map without
        # standing over the rest of the desktop, closes by its own X, and docks
        # back into the side if it is dragged there. The button that brings it
        # back is that same action, so the two cannot fall out of step -- not
        # even when the window is closed from its own corner.
        self.stereonet = StereonetView()

        self.stereonet_dock = QtWidgets.QDockWidget("Stereonet", self)
        self.stereonet_dock.setObjectName("stereonet")
        self.stereonet_dock.setWidget(self.stereonet)
        self.stereonet_dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.stereonet_dock)
        self.stereonet_dock.visibilityChanged.connect(self._on_stereonet_shown)

        self.stereonet_button = QtWidgets.QToolButton()
        self.stereonet_button.setDefaultAction(self.stereonet_dock.toggleViewAction())
        self.stereonet_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly
        )
        self.stereonet_button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed
        )

        # The radius is the one parameter with no right value: too small and
        # the tensor is noise, too large and the fold is averaged away with its
        # neighbours. It is a control and not a setting for that reason.
        self.radius_spin = QtWidgets.QSpinBox()
        self.radius_spin.setRange(50, 100000)
        self.radius_spin.setSingleStep(250)
        self.radius_spin.setValue(int(self.radius))
        self.radius_spin.setSuffix(" m")
        self.radius_spin.valueChanged.connect(self._on_radius_changed)

        self.min_points_spin = QtWidgets.QSpinBox()
        self.min_points_spin.setRange(3, 500)
        self.min_points_spin.setValue(self.gate.min_points)
        self.min_points_spin.setPrefix("n ≥ ")
        self.min_points_spin.valueChanged.connect(self._on_gate_changed)

        self.max_k_spin = QtWidgets.QDoubleSpinBox()
        self.max_k_spin.setRange(0.1, 10.0)
        self.max_k_spin.setDecimals(2)
        self.max_k_spin.setSingleStep(0.1)
        self.max_k_spin.setValue(self.gate.max_k)
        self.max_k_spin.setPrefix("K ≤ ")
        self.max_k_spin.valueChanged.connect(self._on_gate_changed)

        self.easting_spin = QtWidgets.QDoubleSpinBox()
        self.northing_spin = QtWidgets.QDoubleSpinBox()

        left, bottom, right, top = self.session.bounds
        span_x, span_y = right - left, top - bottom

        for box, (low, high), prefix in (
            (self.easting_spin, (left - span_x, right + span_x), "E "),
            (self.northing_spin, (bottom - span_y, top + span_y), "N "),
        ):
            box.setDecimals(1)
            box.setSingleStep(100.0)
            box.setRange(low, high)
            box.setPrefix(prefix)
            box.valueChanged.connect(self._on_centre_typed)

        self._sync_centre_boxes()

        # The readout. Big for the axis, small for what qualifies it -- but the
        # qualification is on screen at all times, not behind a tooltip.
        self.axis_label = QtWidgets.QLabel()
        self.axis_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.axis_label.setStyleSheet("font-size: 20px; font-weight: bold;")

        self.verdict_label = QtWidgets.QLabel()
        self.verdict_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.verdict_label.setWordWrap(True)

        self.shape_label = QtWidgets.QLabel()
        self.shape_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.shape_label.setStyleSheet("color: gray; font-size: 10px;")

        # The grid asks the same question everywhere at once. Its step is the
        # one number that decides whether that takes half a second or a minute,
        # so the cost of the step currently typed is on screen beside it,
        # before the button is pressed rather than after.
        self.step_spin = QtWidgets.QSpinBox()
        self.step_spin.setRange(25, 20000)
        self.step_spin.setSingleStep(100)
        self.step_spin.setValue(int(self.step))
        self.step_spin.setSuffix(" m  step")
        self.step_spin.valueChanged.connect(self._refresh_cost)

        # Two different kinds of thing, so two labels. The cost says how long
        # you will wait; the sampling says what the answer is worth, and a
        # reader who takes a field of a thousand axes for a thousand
        # observations has misread it by a factor of fifty. Sharing one grey
        # block would have given them the same weight, and the second one sits
        # closest to the control that sets it.
        self.sampling_label = QtWidgets.QLabel()
        self.sampling_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.sampling_label.setWordWrap(True)
        self.sampling_label.setStyleSheet("color: #7a5c00; font-size: 10px;")

        self.cost_label = QtWidgets.QLabel()
        self.cost_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.cost_label.setStyleSheet("color: gray; font-size: 10px;")

        self.show_refused_check = QtWidgets.QCheckBox("show cells that failed")
        self.show_refused_check.toggled.connect(lambda _: self._draw_field())

        # This tool had no legend control at all: the three placements existed
        # but only on the command line, where they cannot be changed once the
        # map is open.
        self.legend_controls = LegendControls(self.map_view, legend)
        self.legend_combo = self.legend_controls.combo

        controls = QtWidgets.QWidget()
        controls.setMaximumWidth(300)
        layout = QtWidgets.QVBoxLayout(controls)

        layout.addWidget(self.axis_label)
        layout.addWidget(self.verdict_label)
        layout.addWidget(self.shape_label)
        layout.addWidget(self.stereonet_button)

        layout.addSpacing(8)
        layout.addWidget(QtWidgets.QLabel("Window radius"))
        layout.addWidget(self.radius_spin)

        layout.addSpacing(8)
        layout.addWidget(QtWidgets.QLabel("Centre"))
        layout.addWidget(self.easting_spin)
        layout.addWidget(self.northing_spin)

        layout.addSpacing(8)
        layout.addWidget(QtWidgets.QLabel("A girdle needs"))
        layout.addWidget(self.min_points_spin)
        layout.addWidget(self.max_k_spin)

        layout.addSpacing(8)
        layout.addWidget(QtWidgets.QLabel("Grid"))
        layout.addWidget(self.step_spin)
        layout.addWidget(self.sampling_label)
        layout.addWidget(self.cost_label)
        layout.addWidget(self.show_refused_check)

        for text, slot in (
            ("Compute grid", self.compute_field),
            ("Clear grid", self.clear_field),
            ("Export grid...", self.export_field),
        ):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            layout.addWidget(button)

        layout.addSpacing(8)
        layout.addWidget(self.legend_controls)

        layout.addSpacing(8)
        for text, slot in (
            ("Copy screenshot", self.copy_screenshot),
            ("Save screenshot...", self.save_screenshot),
            ("Save this window...", self.save_result),
        ):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            layout.addWidget(button)

        layout.addStretch(1)

        panel = QtWidgets.QScrollArea()
        panel.setWidget(controls)
        panel.setWidgetResizable(True)
        panel.setMaximumWidth(324)
        panel.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.addWidget(self.map_view, stretch=1)
        main_layout.addWidget(panel)
        self.setCentralWidget(central)

        # There is always something to switch here -- the attitudes themselves,
        # if nothing else -- so the hint is not conditional the way the
        # intersection tool's is.
        self._opening_hint = (
            "drag the circle, or click elsewhere to move it; the net follows"
            " - click a legend entry to take what it names off the map"
        )

    def _draw_base_map(self):
        axes = self.map_view.axes

        self.map_view.draw_base_map()

        # The stations are static: they belong in the background, which
        # blitting recaptures, and cost nothing per frame there. One Line2D and
        # not a scatter, for the same reason as everywhere else.
        #
        # Kept, rather than drawn and forgotten, because its legend entry
        # switches it: on a dense survey the dots are what the computed field
        # has to be read through.
        (stations,) = axes.plot(
            self.attitudes.xy[:, 0], self.attitudes.xy[:, 1],
            linestyle="none", marker=".", markersize=3.0,
            color="#555555", zorder=5,
        )
        self.station_dots = stations

        # What is inside the window, drawn over them. This is the half of the
        # highlight the stereonet cannot show: which measurements on the ground
        # the poles on the net are.
        (selected,) = axes.plot(
            [], [], linestyle="none", marker="o", markersize=5.0,
            markerfacecolor="#d62728", markeredgecolor="black", markeredgewidth=0.4,
            zorder=6,
        )
        self.selected_marker = self.map_view.add_animated(selected)

        self.window_patch = self.map_view.add_animated(
            Circle(
                tuple(self.centre), self.radius,
                fill=False, edgecolor="orange", linestyle="--", linewidth=1.2,
            )
        )
        axes.add_patch(self.window_patch)

        (centre_marker,) = axes.plot(
            [self.centre[0]], [self.centre[1]],
            marker="o", color="yellow", markeredgecolor="black", markersize=8,
        )
        self.centre_marker = self.map_view.add_animated(centre_marker)

        # The axis as a tick through the centre, turned onto the grid: an axis
        # drawn in true azimuth on a projected map points slightly wrong, by
        # the same degree the intersection tool corrects for.
        (tick,) = axes.plot([], [], "-", color="#d62728", linewidth=2.5)
        self.axis_tick = self.map_view.add_animated(tick)

        # The grid's own artists are not animated: a field is computed once and
        # then sits there, so it belongs in the background that blitting
        # recaptures rather than being redrawn on every frame of a drag.
        self.field_collection = None
        self.field_colorbar = None

        # Faint on purpose, and off by default. These are the cells whose
        # answer was refused, and they exist to tell a blank patch that has no
        # data from one whose data said no -- a distinction worth a whisper.
        # Drawn at any real weight they outnumber the axes three to one and end
        # up more prominent than the result, which is the confidence ordering
        # backwards.
        (refused,) = axes.plot(
            [], [], linestyle="none", marker=".", markersize=1.5,
            color="#c8c8c8", alpha=0.6, zorder=4,
        )
        self.refused_marker = refused

        self.map_view.refresh_legend()
        self.map_view.anchor_home()
        self._refresh_cost()

    def _legend_handles(self):
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        # The measurements can be taken off the map from their entries, the way
        # the backdrop's units can. The axis and the circle cannot: those are
        # what the hand is steering, and one dragged invisible is worse than one
        # in the way.
        switchable = self.map_view.switchable

        handles = [
            switchable(
                Line2D([], [], linestyle="none", marker=".", color="#555555",
                       label="attitude"),
                self.station_dots,
            ),
            switchable(
                Line2D(
                    [], [], linestyle="none", marker="o", markerfacecolor="#d62728",
                    markeredgecolor="black", color="none", label="in the window",
                ),
                self.selected_marker,
            ),
            Line2D([], [], color="#d62728", linewidth=2.5, label="fold axis"),
            Patch(facecolor="none", edgecolor="orange", linestyle="--", label="window"),
        ]

        # No entry for the grid's own ticks: they are coloured by plunge, and a
        # single swatch beside the word "grid axis" would say they are one
        # colour. The colourbar is their legend.
        if self.field is not None and self.show_refused_check.isChecked():
            handles.append(
                Line2D([], [], linestyle="none", marker=".", color="#c8c8c8",
                       label="cell that failed")
            )

        return handles

    # -- the net's window -------------------------------------------------

    def showEvent(self, event):
        """
        Floats the net the first time the window appears, beside it if there is
        room on the screen and tucked into its corner if there is not.

        Not in the constructor: before show() the main window has no geometry to
        be placed against, and fit_to_screen may yet maximise it. The flag is
        because showEvent fires again on every unminimise, and a net the user
        had docked or moved would jump back here each time.
        """

        super().showEvent(event)

        if self._net_placed:
            return

        self._net_placed = True
        self.stereonet_dock.setFloating(True)

        frame = self.frameGeometry()
        available = self.screen().availableGeometry()
        width, height = 420, 460

        left = frame.right() + 12
        if left + width > available.right():
            left = max(available.left(), frame.right() - width - 24)

        self.stereonet_dock.setGeometry(left, frame.top() + 48, width, height)

        # visibilityChanged carries the first drawing on most platforms, but not
        # dependably: the net is one gesture behind until something moves, and
        # that is the one frame nobody would think to look for.
        self._refresh_stereonet()

    def _refresh_stereonet(self, admitted=None):
        """
        Puts the current window on the net, unless the net is closed.

        Closed it is not drawn at all. A canvas nobody can see still costs its
        milliseconds, and the draw time this window reports would then be
        measuring something that is not on screen. What that leaves is a net one
        window behind, which is what _net_is_current remembers and why reopening
        it comes back through here.
        """

        if not self.stereonet_dock.isVisible():
            self._net_is_current = False
            return

        if admitted is None:
            admitted = self.gate.admits(self.result)

        self.stereonet.show_window(
            self.attitudes.dip_directions()[self.indices],
            self.attitudes.dips[self.indices],
            self.result,
            admitted,
        )
        self._net_is_current = True

    def _on_stereonet_shown(self, visible):
        if visible and not self._net_is_current:
            self._refresh_stereonet()

    # -- interaction ------------------------------------------------------

    def _near_centre(self, x, y):
        px, py = self.map_view.display_xy(*self.centre)
        ex, ey = self.map_view.display_xy(x, y)

        return np.hypot(ex - px, ey - py) <= self.PICK_RADIUS_PX

    def _on_map_pressed(self, x, y):
        if self._near_centre(x, y):
            self.dragging = True
            return

        self._move_centre(x, y)
        self.update_window()

    def _on_map_dragged(self, x, y):
        if not self.dragging:
            return

        self._move_centre(x, y)
        self.update_window()

    def _on_map_released(self):
        self.dragging = False

    def _move_centre(self, x, y):
        self.centre = [float(x), float(y)]
        self.centre_marker.set_data([x], [y])
        self.window_patch.set_center((x, y))
        self._sync_centre_boxes()

    def _sync_centre_boxes(self):
        for box, value in ((self.easting_spin, self.centre[0]), (self.northing_spin, self.centre[1])):
            with QtCore.QSignalBlocker(box):
                box.setValue(value)

    def _on_centre_typed(self, value):
        self._move_centre(self.easting_spin.value(), self.northing_spin.value())
        self.update_window()

    def _on_radius_changed(self, value):
        self.radius = float(value)
        self.window_patch.set_radius(self.radius)
        self._refresh_cost()
        self.update_window()

    def _on_gate_changed(self, value):
        self.gate = Gate(
            min_points=int(self.min_points_spin.value()),
            max_k=float(self.max_k_spin.value()),
            min_c=self.gate.min_c,
        )

        # A grid already computed answers the new gate without recomputing
        # anything: K, C and the count are what the gate reads and they are
        # already in the field. Moving a threshold and watching how much of the
        # map survives is how you find out whether the answer depends on it.
        if self.field is not None:
            self.field.regate(self.gate)
            self._draw_field()
            self.statusBar().showMessage(self.field.summary())

        self._refresh_cost()
        self.update_window()

    # -- loop -------------------------------------------------------------

    def update_window(self):
        start = perf_counter()

        self.indices = self.attitudes.within(self.centre[0], self.centre[1], self.radius)
        found = perf_counter()

        self.result = fold_axis(self.attitudes.poles_at(self.indices))
        computed = perf_counter()

        admitted = self.gate.admits(self.result)

        inside = self.attitudes.xy[self.indices]
        self.selected_marker.set_data(inside[:, 0], inside[:, 1])
        self._draw_axis_tick(admitted)
        self.map_view.blit()

        self._refresh_stereonet(admitted)
        drawn = perf_counter()

        self._report(admitted, found - start, computed - found, drawn - computed)

    def _draw_axis_tick(self, admitted):
        """
        The axis as a bar through the centre of the window, on the grid.

        Length is a fixed share of the radius rather than of the plunge: a bar
        foreshortened by its own plunge would read as a shallower axis in a
        smaller window, which is a claim the data does not make.
        """

        if self.result is None or not admitted:
            self.axis_tick.set_data([], [])
            return

        trend, _ = self.result.axis
        grid_trend = self.session.convergence.to_grid(trend, *self.centre)

        half = self.radius * 0.9
        dx = half * np.sin(np.radians(grid_trend))
        dy = half * np.cos(np.radians(grid_trend))

        self.axis_tick.set_data(
            [self.centre[0] - dx, self.centre[0] + dx],
            [self.centre[1] - dy, self.centre[1] + dy],
        )

    def _report(self, admitted, search_s, kernel_s, draw_s):
        refusal = self.gate.refusal(self.result)

        if self.result is None:
            self.axis_label.setText("--")
            self.verdict_label.setText("no attitude in the window")
            self.shape_label.setText("")
        else:
            trend, plunge = self.result.axis
            self.axis_label.setText(f"{trend:03.0f} / {plunge:02.0f}")

            if admitted:
                self.verdict_label.setText(f"fold axis, from {self.result.n} attitudes")
                self.verdict_label.setStyleSheet("color: #1a7f37;")
            else:
                self.verdict_label.setText(f"not a fold axis - {refusal}")
                self.verdict_label.setStyleSheet("color: #a03000;")

            s1, s2, s3 = self.result.eigenvalues
            convergence = self.session.convergence.at(*self.centre)
            self.shape_label.setText(
                f"K = {self.result.k:.2f}   C = {self.result.c:.2f}   "
                f"{self.result.description}\n"
                f"S = {s1:.3f} / {s2:.3f} / {s3:.3f}\n"
                f"true north; convergence {convergence:+.2f}°"
            )

        self.statusBar().showMessage(
            f"{len(self.indices)} attitudes   "
            f"search {search_s * 1000:5.2f} ms   "
            f"tensor {kernel_s * 1000:5.2f} ms   "
            f"draw {draw_s * 1000:5.1f} ms   "
            f"total {(search_s + kernel_s + draw_s) * 1000:5.1f} ms"
        )

    # -- the grid ---------------------------------------------------------

    def _field_bounds(self):
        """
        The area to grid: where the attitudes are, not where the map is.

        With a DEM in the session the map can be far larger than the data, and
        gridding the difference would be computing thousands of empty cells to
        draw nothing in them.
        """

        left, bottom = self.attitudes.xy.min(axis=0)
        right, top = self.attitudes.xy.max(axis=0)

        return float(left), float(bottom), float(right), float(top)

    def _refresh_cost(self):
        step = float(self.step_spin.value())
        centres = grid_centres(self._field_bounds(), step)
        cost = field_cost(self.attitudes, centres, self.radius, self.gate)

        seconds = cost["seconds"]
        spelled = f"{seconds:.1f} s" if seconds >= 1.0 else f"{seconds * 1000:.0f} ms"

        self.sampling_label.setText(
            describe_sampling(self._field_bounds(), self.radius, float(self.step_spin.value()))
        )
        self.cost_label.setText(
            f"{cost['cells']} cells, about {cost['occupied']} with data - roughly {spelled}"
        )

    def compute_field(self):
        centres = grid_centres(self._field_bounds(), float(self.step_spin.value()))

        dialog = QtWidgets.QProgressDialog(
            f"{len(centres)} windows...", "Stop", 0, len(centres), self
        )
        dialog.setWindowTitle("Computing the grid")
        dialog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(400)

        def progress(done, total):
            dialog.setValue(done)
            QtWidgets.QApplication.processEvents()

            return not dialog.wasCanceled()

        started = perf_counter()
        self.field = fold_axis_field(
            self.attitudes, centres, self.radius, self.gate, progress=progress
        )
        elapsed = perf_counter() - started
        dialog.setValue(len(centres))

        self._draw_field()

        # The time first: the summary ends with the sampling clause, and an
        # elapsed time tacked after it read as though tiling the area were what
        # took 1.8 seconds.
        self.statusBar().showMessage(f"computed in {elapsed:.1f} s - {self.field.summary()}")

    def clear_field(self):
        self.field = None
        self._draw_field()
        self.statusBar().showMessage("grid cleared")

    def _draw_field(self):
        """
        Draws the field, or takes it off the map.

        The ticks are a LineCollection and not the single NaN-separated Line2D
        that everything drawn per frame here is. That rule is about the cost of
        a frame, and this is not paid per frame: a field is drawn once and then
        captured into the blitting background, which buys the room to colour
        every tick by its own plunge -- information the length of a bar cannot
        carry, since a bar shortened by its plunge would read as a shallower
        axis in a smaller window.
        """

        from matplotlib.collections import LineCollection

        if self.field_colorbar is not None:
            self.field_colorbar.remove()
            self.field_colorbar = None

        if self.field_collection is not None:
            self.field_collection.remove()
            self.field_collection = None

        self.refused_marker.set_data([], [])

        if self.field is not None:
            taken = self.field.admitted
            half = self.field.step * 0.45

            if taken.any():
                centres = self.field.centres[taken]
                trends = np.array([
                    self.session.convergence.to_grid(t, x, y)
                    for t, (x, y) in zip(self.field.trends[taken], centres)
                ])
                dx = half * np.sin(np.radians(trends))
                dy = half * np.cos(np.radians(trends))

                segments = np.stack(
                    [
                        np.c_[centres[:, 0] - dx, centres[:, 1] - dy],
                        np.c_[centres[:, 0] + dx, centres[:, 1] + dy],
                    ],
                    axis=1,
                )

                self.field_collection = LineCollection(
                    segments, linewidths=1.6, cmap="viridis", zorder=7
                )
                self.field_collection.set_array(self.field.plunges[taken])
                self.field_collection.set_clim(0.0, 90.0)
                self.map_view.axes.add_collection(self.field_collection)

                # A colour scale nobody can read is a decoration. The plunge is
                # half of what an axis is, and a bar cannot carry it: a tick
                # shortened by its own plunge would read as a shallower axis in
                # a smaller window, which is a claim the data does not make.
                self.field_colorbar = self.map_view.figure.colorbar(
                    self.field_collection,
                    ax=self.map_view.axes,
                    fraction=0.035,
                    pad=0.02,
                    label="axis plunge (°)",
                )

            if self.show_refused_check.isChecked():
                failed = self.field.occupied & ~taken
                self.refused_marker.set_data(
                    self.field.centres[failed, 0], self.field.centres[failed, 1]
                )

        # A full draw, not a blit: the field belongs to the background, and the
        # draw_event this fires is what recaptures it with the field in it.
        self.map_view.refresh_legend()

    def export_field(self):
        if self.field is None:
            self.statusBar().showMessage("no grid to export - compute one first")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export grid", str(self._suggested_name("_grid.gpkg")),
            "GeoPackage (*.gpkg);;Shapefile (*.shp)",
        )
        if not path:
            return

        import geopandas as gpd
        from shapely.geometry import Point

        # Occupied cells only: an empty cell says nothing that the absence of a
        # point does not say more compactly.
        rows = np.flatnonzero(self.field.occupied)
        centres = self.field.centres[rows]

        grid_trends = [
            self.session.convergence.to_grid(t, x, y) if np.isfinite(t) else None
            for t, (x, y) in zip(self.field.trends[rows], centres)
        ]

        # Names within the ten characters a shapefile allows, and the gate
        # repeated on every row: the thresholds are a choice, and a field whose
        # verdicts cannot be checked against the rule that produced them is a
        # picture rather than a measurement.
        frame = gpd.GeoDataFrame(
            {
                "trend": self.field.trends[rows],
                "plunge": self.field.plunges[rows],
                "trend_grd": grid_trends,
                "k": self.field.k[rows],
                "c": self.field.c[rows],
                "n": self.field.counts[rows],
                "is_axis": self.field.admitted[rows],
                "radius_m": self.field.radius,
                "step_m": self.field.step,
                "min_pts": self.field.gate.min_points,
                "max_k": self.field.gate.max_k,
                "min_c": self.field.gate.min_c,
            },
            geometry=[Point(x, y) for x, y in centres],
            crs=self.session.crs,
        )
        frame.to_file(path)

        self.statusBar().showMessage(
            f"{len(frame)} cells exported to {path} "
            f"({int(self.field.admitted[rows].sum())} of them fold axes)"
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

    def save_result(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save this window", str(self._suggested_name(".json")), "JSON (*.json)"
        )
        if not path:
            return

        Path(path).write_text(json.dumps(self.as_record(), indent=2), encoding="utf-8")
        self.statusBar().showMessage(f"window saved to {path}")

    def as_record(self):
        """
        The window as it stands, with everything needed to read it later.

        The verdict is written down beside the axis, not left to be recomputed:
        the thresholds are a choice, and a file that records the answer without
        the choice that produced it cannot be checked.
        """

        admitted = self.gate.admits(self.result)
        lon_lat = self.session.convergence.geographic(*self.centre)

        record = {
            "centre": [float(v) for v in self.centre],
            "centre_reference": f"EPSG:{self.session.epsg}" if self.session.epsg else "unknown",
            "centre_lon": lon_lat[0] if lon_lat else None,
            "centre_lat": lon_lat[1] if lon_lat else None,
            "radius_m": self.radius,
            "attitudes": int(len(self.indices)),
            "source": str(self.attitudes.path),
            "source_layer": self.attitudes.layer,
            "is_fold_axis": bool(admitted),
            "verdict": self.gate.refusal(self.result) or "girdle",
            "gate": {
                "min_points": self.gate.min_points,
                "max_k": self.gate.max_k,
                "min_c": self.gate.min_c,
            },
        }

        if self.result is not None:
            trend, plunge = self.result.axis
            record.update(
                axis_trend=trend,
                axis_plunge=plunge,
                axis_trend_reference="true north",
                axis_trend_grid=self.session.convergence.to_grid(trend, *self.centre),
                meridian_convergence=self.session.convergence.at(*self.centre),
                girdle_dip_direction=self.result.girdle[0],
                girdle_dip_angle=self.result.girdle[1],
                woodcock_k=self.result.k,
                woodcock_c=self.result.c,
                woodcock_description=self.result.description,
                eigenvalues=list(self.result.eigenvalues),
                principal_axes=[list(axis) for axis in self.result.principal],
            )

        return record

    def _suggested_name(self, suffix):
        tag = f"fold_{int(self.centre[0])}_{int(self.centre[1])}_r{int(self.radius)}"

        return self.session.suggested_name(suffix, tag=tag)


def read_attitudes(session, spec, parent=None):
    """
    The attitude layer as a source, or None once the refusal has been shown.

    The two ways in -- the command line and the dialog -- fail the same way and
    in the same place, which is why this is not written twice.
    """

    attitudes = AttitudeSource(
        spec["path"],
        session.crs,
        layer=spec.get("layer"),
        dip_dir_field=spec.get("dip_dir_field"),
        dip_field=spec.get("dip_field"),
        is_rhr_strike=spec.get("is_rhr_strike", False),
        bounds=session.bounds,
    )
    print(f"attitudes: {attitudes.summary()}")

    if attitudes.problem:
        QtWidgets.QMessageBox.critical(
            parent, "Unusable attitudes", f"{spec['path']}\n\n{attitudes.problem}"
        )
        return None

    return attitudes


def refuse_geographic(session, parent=None):
    """
    True once a session in degrees has been refused, with the reason shown.

    A window radius is in metres, and the search that uses it is a plain
    distance between coordinates. On a geographic CRS those coordinates are
    degrees, and the comparison would be quietly meaningless rather than wrong
    in any way that shows: refused here, where it can still be said.
    """

    if session.crs is None or not session.crs.is_geographic:
        return False

    QtWidgets.QMessageBox.critical(
        parent,
        "Geographic CRS",
        "The window radius is in metres and this session is in degrees "
        f"(EPSG:{session.epsg}).\n\nReproject the attitudes, or give a "
        "projected DEM, before looking for fold axes.",
    )

    return True


def build(session, chosen, legend="beside"):
    """
    The window, on a session somebody else has already opened.

    What the launcher calls. Radius, step and gate keep their defaults and are
    all moved from inside the window, so the only thing taken out of `chosen`
    is the attitude layer -- the one thing this tool cannot do without, and the
    same layer that framed the session.
    """

    if refuse_geographic(session):
        return None

    source = read_attitudes(session, chosen["attitudes"])

    if source is None:
        return None

    window = FoldAxesWindow(session, source, legend=legend)
    fit_to_screen(window, 1280, 900)

    return window


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("attitudes", nargs="?",
                        help="point layer of attitudes, PATH[:LAYER]; a dialog asks if missing")
    parser.add_argument("--dip-dir", metavar="FIELD",
                        help="field holding the dip direction (or the strike, with --strike-rhr)")
    parser.add_argument("--dip", metavar="FIELD",
                        help="field holding the dip angle")
    parser.add_argument("--strike-rhr", action="store_true",
                        help="read the azimuth field as a right-hand-rule strike")
    parser.add_argument("--dem", metavar="PATH", help="DEM for the backdrop; not read otherwise")
    parser.add_argument("--polygons", metavar="PATH[:LAYER]", help="backdrop polygons")
    parser.add_argument("--lines", metavar="PATH[:LAYER]", help="backdrop lines")
    parser.add_argument("--radius", type=float, default=2000.0, metavar="M",
                        help="window radius in metres (default 2000)")
    parser.add_argument("--step", type=int, metavar="M",
                        help="grid step in metres (default: half the radius)")
    parser.add_argument("--min-points", type=int, default=Gate.min_points, metavar="N",
                        help=f"attitudes a girdle needs (default {Gate.min_points})")
    parser.add_argument("--max-k", type=float, default=Gate.max_k, metavar="K",
                        help=f"largest Woodcock K still read as a girdle (default {Gate.max_k})")
    parser.add_argument("--legend", default="beside",
                        choices=[mode for _, mode in MapView.LEGEND_PLACEMENTS])
    args = parser.parse_args()

    chosen = dict(dem=args.dem)

    attitude_spec = split_layer(args.attitudes, "points")

    if attitude_spec:
        attitude_spec.update(
            dip_dir_field=args.dip_dir,
            dip_field=args.dip,
            is_rhr_strike=args.strike_rhr,
        )
        chosen["attitudes"] = attitude_spec

    for backdrop in (split_layer(args.polygons, "polygons"), split_layer(args.lines, "lines")):
        if backdrop:
            chosen[backdrop["role"]] = backdrop

    # The QApplication before anything that could raise into a dialog.
    app = QtWidgets.QApplication(sys.argv)

    # Naming the layer is not the same as saying what its columns mean, and
    # either can be left out: whatever is missing is asked for, with the layer
    # and the fields already filled in from what was given.
    if not attitude_spec or not args.dip_dir or not args.dip:
        dialog = SourcesDialog(wants=WANTS, chosen=chosen, title="gSurf - fold axes")

        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        chosen = dialog.choices()

    session = open_session(chosen)
    print(f"session: {session.summary()}")

    if refuse_geographic(session):
        return

    attitudes = read_attitudes(session, chosen["attitudes"])

    if attitudes is None:
        return

    window = FoldAxesWindow(
        session,
        attitudes,
        radius=args.radius,
        step=args.step,
        gate=Gate(min_points=args.min_points, max_k=args.max_k),
        legend=args.legend,
    )
    fit_to_screen(window, 1280, 900)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
