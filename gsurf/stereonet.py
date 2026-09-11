"""
An equal-area stereonet that redraws while you move.

The same discipline as the map, for the same reason: the net, its grid and the
primitive circle never change, so they are drawn once and captured; the poles,
the girdle and the axis change on every frame and are blitted over them.

Two rules decide whether that works. The artists are created once and updated
with `set_data`, never recreated -- `ax.pole` and `ax.line` make a new Line2D
per call, and a scatter makes a new PathCollection, so drawing a window that
way would leave a hundred dead artists on the axes within a second of dragging.
And the poles are one Line2D with `linestyle="none"`, not one per measurement,
which is the same choice as the marching-squares chords on the map.
"""

from __future__ import annotations

import numpy as np

import PyQt6.QtCore  # noqa: F401
from PyQt6 import QtWidgets

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# Imported for its side effect as much as for its functions: importing
# mplstereonet is what registers `equal_area_stereonet` with matplotlib, so an
# import deferred into the drawing code leaves `add_subplot` unable to find the
# projection. Not lazy, therefore, unlike geopandas elsewhere -- there is no
# version of this widget that works without it.
import mplstereonet


class StereonetView(QtWidgets.QWidget):
    """
    Lower hemisphere, equal area, with the poles of one window on it.

    Equal area and not equal angle: the question being asked is whether a
    population spreads on a girdle or clusters, and only an equal-area net lets
    density be read off the picture without correcting for where on it you are
    looking.
    """

    ADMITTED = dict(axis="#d62728", girdle="#1f4fd8")
    REFUSED = dict(axis="#9a9a9a", girdle="#9a9a9a")

    def __init__(self, parent=None):
        super().__init__(parent)

        self.figure = Figure(figsize=(3.6, 3.6), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111, projection="equal_area_stereonet")

        self.axes.grid(True, color="#cccccc", linewidth=0.4)
        self.axes.set_azimuth_ticks(range(0, 360, 90), labels=["N", "E", "S", "W"])

        self.background = None

        # Created once and only ever updated. The order is the order they are
        # drawn in: poles under the girdle, the girdle under the axis.
        (self.poles,) = self.axes.plot(
            [], [],
            linestyle="none", marker="o", markersize=3.0,
            markerfacecolor="#333333", markeredgecolor="none",
            animated=True, zorder=3,
        )
        (self.girdle,) = self.axes.plot(
            [], [], linestyle="--", linewidth=1.3, animated=True, zorder=4
        )
        (self.axis,) = self.axes.plot(
            [], [],
            linestyle="none", marker="D", markersize=8.0,
            markeredgecolor="black", markeredgewidth=0.6,
            animated=True, zorder=5,
        )

        self._animated = (self.poles, self.girdle, self.axis)

        self.canvas.mpl_connect("draw_event", self._on_draw)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    # -- the frame --------------------------------------------------------

    def _on_draw(self, event):
        self.background = self.canvas.copy_from_bbox(self.axes.bbox)
        self._draw_animated()

    def _draw_animated(self):
        for artist in self._animated:
            self.axes.draw_artist(artist)

    def blit(self):
        if self.background is None:
            self.canvas.draw()
        else:
            self.canvas.restore_region(self.background)
            self._draw_animated()
            self.canvas.blit(self.axes.bbox)

        self.canvas.flush_events()

    # -- what is on it ----------------------------------------------------

    def show_window(self, dip_directions, dips, result=None, admitted=False):
        """
        Puts one window on the net: its poles, and its axis if it has one.

        A refused axis is greyed rather than hidden. Hiding it would answer the
        question "is this a fold?" by showing nothing, which reads the same as a
        window with no data in it; greyed, you watch it turn colour as the
        window moves onto the hinge, which is the reason for a live net at all.
        """

        if len(dips):
            # mplstereonet takes a plane by its right-hand-rule strike.
            strikes = (np.asarray(dip_directions) - 90.0) % 360.0
            self.poles.set_data(*mplstereonet.pole(strikes, np.asarray(dips)))
        else:
            self.poles.set_data([], [])

        if result is None:
            self.girdle.set_data([], [])
            self.axis.set_data([], [])
            self.blit()
            return

        colors = self.ADMITTED if admitted else self.REFUSED

        girdle_dip_dir, girdle_dip = result.girdle
        lons, lats = mplstereonet.plane((girdle_dip_dir - 90.0) % 360.0, girdle_dip)
        self.girdle.set_data(np.ravel(lons), np.ravel(lats))
        self.girdle.set_color(colors["girdle"])

        trend, plunge = result.axis
        self.axis.set_data(*mplstereonet.line(plunge, trend))
        self.axis.set_markerfacecolor(colors["axis"])

        self.blit()

    def clear(self):
        self.show_window([], [])
