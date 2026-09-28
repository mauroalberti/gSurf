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
    the fold tool's question. `show_planes` puts the planes of one structure on
    it as great circles, with whatever lineations were read on them, which is the
    editor's: a fault carries one to four planes here and there is no density to
    look at, so what there is to look at is how they agree and where the striae
    sit within them. A net showing four poles says less than the four pairs of
    numbers written out; a net showing four great circles shows whether they are
    the same surface, which no column of numbers does. So the poles are
    `show_window`'s artist and only its.

    Each method clears the other's artists, so a widget handed to both cannot
    show a fitted axis over planes it was not fitted to.
    """

    ADMITTED = dict(axis="#d62728", girdle="#1f4fd8")
    REFUSED = dict(axis="#9a9a9a", girdle="#9a9a9a")

    # Where a plane draws with nothing said about it. Overridden per call, because
    # the tool that asks for one has a palette in which the colour already means
    # something.
    PLANE_COLOR = "#333333"

    # What a plane keeps of itself while another one is being pointed at. Dimming
    # the rest and not recolouring the one, because the colour on this net says
    # what kind of claim a circle is and the hover must not spend it: a plane
    # pointed at is still a compass reading or still a fit, and after the cursor
    # leaves it has to be the same circle it was before.
    DIMMED = 0.3

    # Square, and asked for in inches because that is what a Figure takes. Where
    # the net starts and not where it stays -- the primitive circle follows the
    # widget, measured at 268 px across in a 276 px one and 412 in a 420. What
    # the figure size does decide is the width the widget asks a layout for, so
    # a tool that docks the net beside something else is choosing how much of
    # that something else it costs: 3.6 in is 366 px off a map.
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

        # The measured planes currently on it, kept so that `mark` can draw one of
        # them again without being handed them a second time. The artist holds
        # coordinates on the net and there is no way back from those to a dip
        # direction, which is the same reason the editor keeps its station records
        # beside the dots it drew them as.
        self._measured = []

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

        # The planes of one structure. Two artists and not one per plane, and not
        # one per structure either: `nan` between the great circles breaks the
        # polyline, which is how the map draws two rejected stretches without
        # joining them across the good ground between, and it leaves the
        # create-once rule above intact for any number of planes.
        #
        # Two of them because the difference between the two kinds is drawn and not
        # written. Solid for a plane somebody put a compass on, dashed for a
        # surface fitted to a scatter: a picture that drew them the same way would
        # invite the two to be read as the same kind of claim, and on a fault that
        # carries both -- six of the forty-five in `merid_faults` -- reading them
        # as the same kind is the one mistake this net exists to prevent.
        (self.measured,) = self.axes.plot(
            [], [], linestyle="-", linewidth=1.6, animated=True, zorder=4
        )
        (self.fitted,) = self.axes.plot(
            [], [], linestyle="--", linewidth=1.4, animated=True, zorder=4
        )

        # And the one being pointed at, drawn again over the rest. A third artist
        # and not a restyling of the first, because a `nan`-joined polyline has one
        # colour and one width for every circle in it: the only way to make one of
        # them answer is to draw that one twice. Which is what the map does with
        # `traces` and `highlight`, for the same reason.
        (self.marked,) = self.axes.plot(
            [], [], linestyle="-", linewidth=3.2, animated=True, zorder=5
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
            self.poles, self.girdle, self.axis,
            self.measured, self.fitted, self.marked, self.lineation,
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

        self.measured.set_data([], [])
        self.fitted.set_data([], [])
        self.marked.set_data([], [])
        self.lineation.set_data([], [])

        # And not just the artist: a `mark` arriving after this must not be able to
        # draw a plane from the structure the net was showing before.
        self._measured = []

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

    @staticmethod
    def _arcs(planes):
        """
        Great circles as one polyline, with a break between them.

        `mplstereonet.plane` is asked once per plane rather than once with a
        vector of strikes, which it would take: vectorised it returns one row per
        plane and joining the rows is the same concatenation done less legibly,
        and at four planes there is nothing to win. The break is what makes one
        artist hold several circles -- without it the last point of one would be
        joined to the first point of the next by a chord across the net.
        """

        xs, ys = [], []

        for dip_dir, dip in planes:
            # mplstereonet takes a plane by its right-hand-rule strike.
            lons, lats = mplstereonet.plane((dip_dir - 90.0) % 360.0, dip)
            xs.extend(np.ravel(lons).tolist())
            ys.extend(np.ravel(lats).tolist())
            xs.append(np.nan)
            ys.append(np.nan)

        return xs, ys

    def show_planes(
        self, measured=(), fitted=(), lineations=(),
        marked=None, measured_color=None, fitted_color=None,
    ):
        """
        Puts the planes of one structure on the net, with the lines read on them.

        `measured` and `fitted` are `(dip_dir, dip)`, and they are two arguments
        rather than one list with a kind on each entry because the net draws them
        as two artists: sorting a mixed list back into the two would be work done
        here to undo work done by the caller, who has them apart already.

        The planes as great circles, and as nothing else. Poles were drawn here
        too at first, on the argument that a pole is how this net would be
        compared with a fold tool's -- but that is a reason to draw poles where
        there is a population to see the shape of, and one fault's one-to-four
        planes are not one. What this picture is read for is whether the circles
        are the same surface and where the striae sit within them; the markers
        inside the primitive circle are the striae, and a pole is a mark inside
        that circle which is not one.

        `lineations` are `(trend, plunge)` and are not split by which plane they
        were read on. Plural because one striated surface can carry more than one
        set -- two generations of movement on the same plane is a thing the field
        notes in this project record -- and pooled because a stria drawn on the
        net lies in the plane it was read on, so which circle it belongs to is
        already visible: it is the one it sits on.

        `marked` is an index into `measured` and draws that one again on top,
        with everything else dimmed. Into `measured` and not into both because
        what points at a circle is the cursor resting on a station dot, and a fit
        has no dot: it holds over a stretch of the trace and not at a point on it.

        Called with nothing, it empties the net -- so a caller with nothing
        selected does not need a second method to say so.
        """

        self.poles.set_data([], [])
        self.girdle.set_data([], [])
        self.axis.set_data([], [])

        self._measured = list(measured)

        self.measured.set_data(*self._arcs(self._measured))
        self.measured.set_color(measured_color or self.PLANE_COLOR)
        self.fitted.set_data(*self._arcs(fitted))
        self.fitted.set_color(fitted_color or self.PLANE_COLOR)

        self._apply_mark(marked)

        if lineations:
            # One artist with N points, as the poles are: the same reason, which
            # is that a second generation of striae must not cost a second Line2D
            # on a widget that blits.
            trends = [trend for trend, _ in lineations]
            plunges = [plunge for _, plunge in lineations]
            self.lineation.set_data(*mplstereonet.line(plunges, trends))
            self.lineation.set_markerfacecolor(measured_color or self.PLANE_COLOR)
        else:
            self.lineation.set_data([], [])

        self.blit()

    def _apply_mark(self, marked):
        """Which of the planes on the net is being pointed at, without drawing."""

        if marked is not None and 0 <= marked < len(self._measured):
            self.marked.set_data(*self._arcs([self._measured[marked]]))
            self.marked.set_color(self.measured.get_color())
            alpha = self.DIMMED
        else:
            self.marked.set_data([], [])
            alpha = 1.0

        # The dimmed copy of the marked plane stays under its own full-strength
        # one, which is 3.2 points wide over 1.6: covered, and cheaper than
        # arranging for it to be left out.
        self.measured.set_alpha(alpha)
        self.fitted.set_alpha(alpha)

    def mark(self, marked=None):
        """
        Points at one of the planes already on the net, or at none of them.

        Separate from `show_planes` because it is called on a different gesture at
        a different rate: the planes change when a structure is selected, and this
        changes when a cursor crosses onto a station dot. Going back through
        `show_planes` would work and would rebuild every circle on the net to move
        a highlight between two of them -- the arcs are what this exists not to
        recompute.

        Out of range or `None` is not an error but the ordinary way to say "none of
        them": a cursor leaving the dots says it on every frame it is off them.
        """

        self._apply_mark(marked)
        self.blit()

    def clear(self):
        self.show_window([], [])
