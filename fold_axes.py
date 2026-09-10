"""
Fold axes under a moving window.

Drag a circle across the map and watch the bedding poles inside it fall onto a
stereonet, with the best-fit girdle and the axis they turn about recomputed on
every frame. The question a fold axis answers is local -- a plunge measured
over a whole map sheet is the average of every structure on it -- so the tool
is a window you move rather than a button you press.

Usage:
    python fold_axes.py <attitudes.gpkg[:layer]> --dip-dir FIELD --dip FIELD
                        [--dem DEM] [--polygons PATH[:LAYER]] [--lines PATH[:LAYER]]
                        [--radius M] [--strike-rhr] [--min-points N] [--max-k K]

The attitude layer is the only thing required, and it is what the session is
built on: with no DEM the projection and the extent come from the layer itself.
A DEM, if given, is backdrop and nothing else -- this calculation never reads
an elevation.

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

from app.attitudes import AttitudeSource
from app.folds import Gate, fold_axis
from app.mapview import MapView, fit_to_screen
from app.session import Session
from app.stereonet import StereonetView
from app.vectors import split_layer


class FoldAxesWindow(QtWidgets.QMainWindow):
    """
    The map with a window on it, the stereonet of what is inside, and the
    verdict on whether that is a fold.
    """

    PICK_RADIUS_PX = 12

    def __init__(self, session, attitudes, radius=2000.0, gate=None, legend="beside"):
        super().__init__()

        self.session = session
        self.attitudes = attitudes
        self.gate = gate or Gate()
        self.radius = float(radius)
        self.dragging = False
        self.result = None
        self.indices = np.empty(0, dtype=int)

        self.centre = list(session.center())

        self.setWindowTitle(f"gSurf - fold axes - {session.label}")
        self._build_ui(legend)
        self._draw_base_map()

        self.update_window()

    # -- construction -----------------------------------------------------

    def _build_ui(self, legend):
        self.map_view = MapView(self.session, legend=legend)
        self.map_view.legend_handles_provider = self._legend_handles

        self.map_view.pressed.connect(self._on_map_pressed)
        self.map_view.dragged.connect(self._on_map_dragged)
        self.map_view.released.connect(self._on_map_released)
        self.map_view.save_requested.connect(self.save_screenshot)
        self.map_view.status.connect(self.statusBar().showMessage)

        self.stereonet = StereonetView()

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

        controls = QtWidgets.QWidget()
        controls.setMaximumWidth(300)
        layout = QtWidgets.QVBoxLayout(controls)

        layout.addWidget(self.stereonet)
        layout.addWidget(self.axis_label)
        layout.addWidget(self.verdict_label)
        layout.addWidget(self.shape_label)

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

        self.statusBar().showMessage(
            "drag the circle, or click elsewhere to move it; the net follows"
        )

    def _draw_base_map(self):
        axes = self.map_view.axes

        self.map_view.draw_base_map()

        # The stations are static: they belong in the background, which
        # blitting recaptures, and cost nothing per frame there. One Line2D and
        # not a scatter, for the same reason as everywhere else.
        axes.plot(
            self.attitudes.xy[:, 0], self.attitudes.xy[:, 1],
            linestyle="none", marker=".", markersize=3.0,
            color="#555555", zorder=5,
        )

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

        self.map_view.refresh_legend()
        self.map_view.anchor_home()

    def _legend_handles(self):
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        return [
            Line2D([], [], linestyle="none", marker=".", color="#555555", label="attitude"),
            Line2D(
                [], [], linestyle="none", marker="o", markerfacecolor="#d62728",
                markeredgecolor="black", color="none", label="in the window",
            ),
            Line2D([], [], color="#d62728", linewidth=2.5, label="fold axis"),
            Patch(facecolor="none", edgecolor="orange", linestyle="--", label="window"),
        ]

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
        self.update_window()

    def _on_gate_changed(self, value):
        self.gate = Gate(
            min_points=int(self.min_points_spin.value()),
            max_k=float(self.max_k_spin.value()),
            min_c=self.gate.min_c,
        )
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

        dip_dirs = self.attitudes.dip_directions()[self.indices]
        self.stereonet.show_window(
            dip_dirs, self.attitudes.dips[self.indices], self.result, admitted
        )
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("attitudes", help="point layer of attitudes, PATH[:LAYER]")
    parser.add_argument("--dip-dir", required=True, metavar="FIELD",
                        help="field holding the dip direction (or the strike, with --strike-rhr)")
    parser.add_argument("--dip", required=True, metavar="FIELD",
                        help="field holding the dip angle")
    parser.add_argument("--strike-rhr", action="store_true",
                        help="read the azimuth field as a right-hand-rule strike")
    parser.add_argument("--dem", metavar="PATH", help="DEM for the backdrop; not read otherwise")
    parser.add_argument("--polygons", metavar="PATH[:LAYER]", help="backdrop polygons")
    parser.add_argument("--lines", metavar="PATH[:LAYER]", help="backdrop lines")
    parser.add_argument("--radius", type=float, default=2000.0, metavar="M",
                        help="window radius in metres (default 2000)")
    parser.add_argument("--min-points", type=int, default=Gate.min_points, metavar="N",
                        help=f"attitudes a girdle needs (default {Gate.min_points})")
    parser.add_argument("--max-k", type=float, default=Gate.max_k, metavar="K",
                        help=f"largest Woodcock K still read as a girdle (default {Gate.max_k})")
    parser.add_argument("--legend", default="beside",
                        choices=[mode for _, mode in MapView.LEGEND_PLACEMENTS])
    args = parser.parse_args()

    attitude_spec = split_layer(args.attitudes, "points")

    backdrop = [
        spec
        for spec in (split_layer(args.polygons, "polygons"), split_layer(args.lines, "lines"))
        if spec
    ]

    # The QApplication before anything that could raise into a dialog.
    app = QtWidgets.QApplication(sys.argv)

    # The attitude layer frames the session without being drawn by it: with no
    # DEM it is what says where we are, but the stations are this tool's own
    # data and it draws them itself.
    session = Session.open(dem_path=args.dem, vectors=backdrop, frame_layers=[attitude_spec])
    print(f"session: {session.summary()}")

    # A window radius is in metres, and the search that uses it is a plain
    # distance between coordinates. On a geographic CRS those coordinates are
    # degrees, and the comparison would be quietly meaningless rather than
    # wrong in any way that shows: refused here, where it can still be said.
    if session.crs is not None and session.crs.is_geographic:
        QtWidgets.QMessageBox.critical(
            None,
            "Geographic CRS",
            "The window radius is in metres and this session is in degrees "
            f"(EPSG:{session.epsg}).\n\nReproject the attitudes, or give a "
            "projected DEM, before looking for fold axes.",
        )
        return

    attitudes = AttitudeSource(
        attitude_spec["path"],
        session.crs,
        layer=attitude_spec.get("layer"),
        dip_dir_field=args.dip_dir,
        dip_field=args.dip,
        is_rhr_strike=args.strike_rhr,
        bounds=session.bounds,
    )
    print(f"attitudes: {attitudes.summary()}")

    if attitudes.problem:
        QtWidgets.QMessageBox.critical(
            None, "Unusable attitudes", f"{attitude_spec['path']}\n\n{attitudes.problem}"
        )
        return

    window = FoldAxesWindow(
        session,
        attitudes,
        radius=args.radius,
        gate=Gate(min_points=args.min_points, max_k=args.max_k),
        legend=args.legend,
    )
    fit_to_screen(window, 1280, 900)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
