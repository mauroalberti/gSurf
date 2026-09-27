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
    Lower hemisphere, equal area, with either a population or one measurement.

    Equal area and not equal angle: the question being asked is whether a
    population spreads on a girdle or clusters, and only an equal-area net lets
    density be read off the picture without correcting for where on it you are
    looking.

    **Two uses, and they are not the same picture.** `show_window` puts a
    population on it -- poles, and the girdle and axis fitted to them -- which is
    the fold tool's question. `show_attitude` puts a single plane on it as a
    great circle with whatever lineations were read on that plane, which is the
    editor's: at one measurement per station there is no density to look at, and
    what there is to look at is where the striae sit within the plane. A net
    showing one pole says less than the two numbers written out; a net showing
    the plane and the line on it says something neither number does.

    Each method clears the other's artists, so a widget handed to both cannot
    show a fitted axis over a plane it was not fitted to.
    """

    ADMITTED = dict(axis="#d62728", girdle="#1f4fd8")
    REFUSED = dict(axis="#9a9a9a", girdle="#9a9a9a")

    # Where a single measurement draws with nothing said about it. Overridden per
    # call, because the tool that asks for one has a palette in which the colour
    # already means something.
    PLANE_COLOR = "#333333"

    # Square, and asked for in inches because that is what a Figure takes. This
    # is also the width the widget asks a layout for, so a tool that puts the net
    # beside something else is choosing how much of that something else it costs:
    # in a dock, 3.6 in is 366 px off the map, and a single plane does not need
    # 366 px to be read.
    FIGSIZE = 3.6

    def __init__(self, parent=None, figsize=None):
        super().__init__(parent)

        side = figsize or self.FIGSIZE
        self.figure = Figure(figsize=(side, side), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111, projection="equal_area_stereonet")

        self.axes.grid(True, color="#cccccc", linewidth=0.4)
        self.axes.set_azimuth_ticks(range(0, 360, 90), labels=["N", "E", "S", "W"])

        # And this is what makes those four letters appear, which until now they
        # never did -- on this net or on the fold tool's. mplstereonet keeps the
        # azimuth labels on a second, hidden polar axes underneath, positioned
        # just outside the primitive circle. In a figure with room to spare that
        # is outside the stereonet axes too and they are drawn; in one sized to a
        # widget, the layout inflates the axes until its own opaque background
        # covers them, and they are painted over by the thing they label. A net
        # with no north on it is a circle of lines, so: no background.
        self.axes.patch.set_alpha(0.0)

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

        # One measurement's plane, and the lineations read on it. Solid where the
        # girdle is dashed, because one is a plane somebody put a compass on and
        # the other is a surface fitted to a scatter, and a picture that draws
        # them the same way invites the two to be read as the same kind of claim.
        (self.great_circle,) = self.axes.plot(
            [], [], linestyle="-", linewidth=1.6, animated=True, zorder=4
        )

        # A marker and not an arrow. An arrow on a slickenline is a statement
        # about which block went which way, and a trend read off a striated
        # surface is a line and not a vector: the same striae are consistent with
        # a rake and with that rake turned through 180 degrees. Where the sense
        # is genuinely known it is a separate fact and would need drawing as one.
        (self.lineation,) = self.axes.plot(
            [], [],
            linestyle="none", marker="s", markersize=6.5,
            markeredgecolor="black", markeredgewidth=0.6,
            animated=True, zorder=6,
        )

        self._animated = (
            self.poles, self.girdle, self.axis, self.great_circle, self.lineation
        )

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

        self.great_circle.set_data([], [])
        self.lineation.set_data([], [])

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

    def show_attitude(self, dip_dir, dip, lineations=(), color=None):
        """
        Puts one measured plane on the net, with the lines read on it.

        The plane as a great circle and as its pole, which are the same fact
        drawn twice on purpose: the pole is where it would sit in a population
        and is how this net is compared with a fold tool's, the great circle is
        what a lineation has to lie on for the pair to be believable.

        `lineations` are `(trend, plunge)`, plural because one striated surface
        can carry more than one set -- two generations of movement on the same
        plane is a thing the field notes in this project record. Passing none is
        the ordinary case, not an empty result: the plane is the measurement and
        the striae are a second one that mostly was not made.

        `dip_dir` of `None` empties the net, so a caller with nothing selected
        does not need a second method to say so.
        """

        self.poles.set_data([], [])
        self.girdle.set_data([], [])
        self.axis.set_data([], [])

        if dip_dir is None or dip is None:
            self.great_circle.set_data([], [])
            self.lineation.set_data([], [])
            self.blit()
            return

        strike = (dip_dir - 90.0) % 360.0
        lons, lats = mplstereonet.plane(strike, dip)
        self.great_circle.set_data(np.ravel(lons), np.ravel(lats))
        self.great_circle.set_color(color or self.PLANE_COLOR)

        self.poles.set_data(*mplstereonet.pole(strike, dip))
        self.poles.set_markerfacecolor(color or self.PLANE_COLOR)

        if lineations:
            # One artist with N points, as the poles are: the same reason, which
            # is that a second generation of striae must not cost a second Line2D
            # on a widget that blits.
            trends = [trend for trend, _ in lineations]
            plunges = [plunge for _, plunge in lineations]
            self.lineation.set_data(*mplstereonet.line(plunges, trends))
            self.lineation.set_markerfacecolor(color or self.PLANE_COLOR)
        else:
            self.lineation.set_data([], [])

        self.blit()

    def clear(self):
        self.show_window([], [])
