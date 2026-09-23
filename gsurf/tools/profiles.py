"""
Sections: a trace dragged over the map, and the geology under it redrawn as it
moves.

The section is the oldest thing a structural geologist draws and the slowest
thing to iterate on, because moving it by two hundred metres has always meant
redoing it. Here it moves with the mouse, which turns the section from a result
into an instrument: you find where the fault is by watching where it goes.

**One while it moves, the bundle when it stops.** A single profile recomputes
in about forty milliseconds, a bundle of thirteen in a quarter of a second --
the difference between a line you drag and a line that lurches after you. So
the drag redraws one profile, and the parallel bundle is recomputed on release,
the same bargain `MapView.schedule_shade_refresh` already strikes with the
hillshade. Both panels are blitted: the scaffolding of a `ProfilesView` is
built once and only the data artists are redrawn.

**The trace drawn, never the trace sampled.** An intersection costs (profile
segments x trace vertices), and a section line stored densified to DEM step --
1276 vertices over 6.4 km, which is how they come out of QGIS -- costs 1275
times what the two-point trace costs and finds exactly the same crossings. The
densification is for sampling the topography and the sampler does its own. What
goes to the profiler here is the two ends and nothing between them.

**How far a measurement reaches.** A plane fitted to a trace owns that trace.
A compass reading taken at one outcrop owns a point, and how much of the fault
it speaks for is a judgement about that fault -- fifty metres where it is a
local break, the whole kilometre where the surface has been walked. That
judgement is made here, against the section, and not in whatever wrote the
layer: `TraceRecord` keeps the anchor apart from the span for exactly this, and
the reach box is the control that sets it.

**Three windows, not one window with docks.** The map, the section and the
trace records are top-level windows in their own right, so the section can be
given a screen and the size a section wants rather than the strip a dock leaves
it. They are parented to the map all the same, which is what has Qt destroy
them with the tool and what keeps closing one from taking the application down
while the launcher waits hidden underneath. Where they were left is remembered.

**What is remembered, and what a remembered number means.** A section is
arrived at rather than specified -- you drag until it crosses the thing you
are after -- and closing the window used to throw that away and come up west
to east through the middle again. So the trace, the framing, the bundle and
the reach are written on the way out, and so is whether the legend is up. They
divide, though, into habits and places: how many profiles at what spacing is a
way of working and carries to whatever opens next, while a trace is metres in a
projection and means somewhere else under another one. Places come back only
over the source they were written on. See `applicable`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.lines import Line2D
from PyQt6 import QtCore, QtGui, QtWidgets

from gsurf.attitudes import DEFAULT_HALF_SPAN, TraceAttitudeSource
from gsurf.mapview import LegendControls, MapView, fit_to_screen
from gsurf.vectors import VectorSource, single_parts

# The DEM is the section: without a topographic surface there is no profile to
# draw anything under, which is what puts it in the required half where the
# fold-axis tool has its attitudes. Everything else is what gets drawn on it,
# and a section of bare topography is a legitimate thing to want.
WANTS = dict(
    dem="required",
    traces="optional",
    polygons="optional",
    lines="optional",
    attitudes="optional",
)

BUNDLE_DEFAULT = 5
OFFSET_DEFAULT = 500.0
MAX_DEFAULT_LENGTH = 10000.0

# What the controls can hold, in one place because a remembered state is
# checked against the same bounds: a number the spin box would clamp and a
# section computed from the number before clamping are two different things,
# and the gap between them is a box that disagrees with the map. The count is
# odd throughout -- a central bundle has a middle, and `Profilers` raises
# without one.
BUNDLE_RANGE = (1, 41)
OFFSET_RANGE = (10.0, 20000.0)
REACH_RANGE = (0.0, 50000.0)

PANEL_MIN_PX = 130
PANEL_WIDTH_PX = 430

# What the legend beside the panels takes, and how many names it spells out in
# a group before it starts counting instead. Narrow: it holds unit names, and a
# name that does not fit is cut rather than given room the section wants.
SECTION_LEGEND_PX = 210
SECTION_LEGEND_NOTES = 6

# What the satellites come up as, the first time and nothing being remembered.
# The section is wider and taller than the dock it replaces because it no
# longer has to leave room for a map underneath it: 520 px is three panels of
# a bundle before the scroll area has anything to do. The legend is added to
# that width rather than taken out of it, so the panels keep the 900 they had.
SECTION_WINDOW_PX = (900 + SECTION_LEGEND_PX, 520)
TRACES_WINDOW_PX = (PANEL_WIDTH_PX, 620)

# Below this there is no arrangement of three windows that does not cover the
# map, and the window manager's own placement is a better guess than ours.
ROOM_TO_PLACE_PX = 1600

# Room above and below what the DEM holds. The axis is settled when the view is
# built and must not move afterwards, so it is given the whole elevation range
# of the DEM plus a margin: a profile dragged onto the highest ground in the
# map still has air over it, and the decimated minimum still has floor under it.
Z_MARGIN = 200.0

# The backdrop roles that become section geometry; points are drawn on the map
# but there is nothing to cross a section with. The single-part shapely type
# each one converts from is `VectorSource.GEOMETRY_SUFFIX`, which says the same
# thing for the layer declaration: 'Polygon' is both what 'MultiPolygon' ends
# with and what a MultiPolygon is made of.
BACKDROP_ROLES = ("polygons", "lines")


class SectionCanvas(QtWidgets.QWidget):
    """
    A `ProfilesView` in a widget, redrawn by blitting.

    The view builds its own figure and hands out the artists that carry data;
    everything this adds is the canvas, the captured background and the two
    lines of restore-and-blit. The background is recaptured on every full draw,
    which is what a resize is.

    The data artists are held out of that full draw. A background captured with
    the profile in it is a background with a section printed on it: the next
    frame restores it and draws the new curve on top, and the panel shows two
    sections -- the live one, and one from wherever the trace was standing when
    the background was taken. `MapView.add_animated` strikes the same bargain
    once and for all; here the flag has to go back on after every `update`,
    because the view makes new artists each time and hands the old ones to the
    garbage collector.
    """

    def __init__(self, view, parent=None):
        super().__init__(parent)

        self.view = view
        self.canvas = FigureCanvasQTAgg(view.figure)
        self.background = None

        # `ProfilesView` draws its first frame while building, so there are
        # artists to keep out of the draw before anything has been redrawn.
        self._mark_animated()

        # A panel needs a height to be a section rather than a band, and a
        # bundle of twenty-five of them needs more height than any dock has.
        # Asking for the height outright and letting the scroll area deal with
        # it is what keeps the map from being squeezed to nothing by a dock
        # whose size hint came from the figure.
        self.canvas.setMinimumHeight(PANEL_MIN_PX * max(1, len(view.axes)))

        self.canvas.mpl_connect("draw_event", self._on_draw)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def _mark_animated(self):
        """Takes the artists that carry data out of the normal draw."""

        for artist in self.view.data_artists():
            artist.set_animated(True)

    def _draw_data(self):
        for artist in self.view.data_artists():
            artist.axes.draw_artist(artist)

    def _on_draw(self, event):
        # Captured without the data, then the data put back on top of it: the
        # panel has to come out of a full draw with its section still on it.
        self.background = self.canvas.copy_from_bbox(self.view.figure.bbox)
        self._draw_data()

    def redraw(self, geoprofiles):
        """The new data onto panels that have not moved."""

        self.view.update(geoprofiles)
        self._mark_animated()

        if self.background is None:
            self.canvas.draw()
            return

        self.canvas.restore_region(self.background)
        self._draw_data()

        self.canvas.blit(self.view.figure.bbox)
        self.canvas.flush_events()


class SectionLegend(QtWidgets.QScrollArea):
    """
    What the colours in the panels stand for, for the section now drawn.

    Not the map's legend moved across. That one is the *window's* -- every unit
    in it, in the project's order, each entry a click that takes what it names
    off the map -- and it answers "what is around here". A section asks the
    narrower question, "what does this line go through", so this one lists what
    the profiles actually cross and nothing else. It switches nothing: a panel
    is redrawn from the intersections, and the library takes the palette as
    given.

    The library offers a legend of its own, and the reason gSurf leaves it off
    (`polygon_intersections_legend=False`) is what decides the shape of this
    class. A figure's legend is settled when the figure is built, and the
    bundle's figure is rebuilt only when the panel count or the section's
    length changes -- so dragged across a sheet it would go on naming the units
    of a section that is no longer on screen. This one is offered every frame
    and rebuilt only when what it would say has changed, which makes a drag
    along one unit a tuple comparison and crossing into a new one a column of
    labels.
    """

    SWATCH_PX = (20, 12)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._entries = None

        body = QtWidgets.QWidget()

        self._rows = QtWidgets.QVBoxLayout(body)
        self._rows.setContentsMargins(8, 8, 8, 8)
        self._rows.setSpacing(2)
        self._rows.addStretch(1)

        self.setWidget(body)
        self.setWidgetResizable(True)
        self.setFixedWidth(SECTION_LEGEND_PX)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        # A name too long for the column is cut where it is built, so there is
        # nothing to scroll sideways to; a bundle through a sheet's worth of
        # units does need the vertical one.
        self.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

    def set_entries(self, entries):
        """Rebuilds the column, unless this is what it is already showing."""

        entries = tuple(entries)

        if entries == self._entries:
            return

        self._entries = entries

        # Unparented *and* deleted, in that order, and the first half is what
        # matters. Taking the item out of the layout leaves the widget a child
        # of the body, drawn where it was, and `deleteLater` only gets to it
        # when the event loop next runs -- which during a drag is after several
        # more frames. Without the unparenting the old column stays on screen
        # under the new one, two legends deep and both legible.
        while self._rows.count():
            item = self._rows.takeAt(0)
            widget = item.widget()

            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        for kind, label, color in entries:
            self._rows.addWidget(self._row(kind, label, color))

        # Last, so a short legend sits at the top of the column rather than
        # spreading itself down the height of the window.
        self._rows.addStretch(1)

    def _row(self, kind, label, color):
        if kind == "heading":
            heading = QtWidgets.QLabel(label)

            font = heading.font()
            font.setBold(True)
            heading.setFont(font)
            heading.setContentsMargins(0, 6, 0, 0)

            return heading

        row = QtWidgets.QWidget()

        line = QtWidgets.QHBoxLayout(row)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(6)

        # Kept even when there is nothing to put in it: an entry with no colour
        # of its own -- a count, or a name in a group that shares one mark --
        # then lines up under the ones that have, indented by the space a
        # swatch would have taken.
        swatch = QtWidgets.QLabel()
        swatch.setFixedSize(*self.SWATCH_PX)

        if color is not None:
            swatch.setPixmap(self._swatch(kind, color))

        # Cut with an ellipsis rather than left to run off the edge of the
        # column, which is what a plain label does: `Membro di Ganca di Campo
        # Longo` clipped mid-word reads as the name of something else. The
        # whole of it is a hover away.
        text = QtWidgets.QLabel()
        text.setToolTip(label)
        text.setText(
            text.fontMetrics().elidedText(
                label, QtCore.Qt.TextElideMode.ElideRight, self._room()
            )
        )

        line.addWidget(swatch)
        line.addWidget(text, stretch=1)

        return row

    def _room(self):
        """The width a label has, once the swatch and the margins are out."""

        margins = self._rows.contentsMargins()

        return (
            SECTION_LEGEND_PX
            - self.SWATCH_PX[0]
            - self._rows.spacing()
            - margins.left()
            - margins.right()
            # The vertical scroll bar, whether or not it is up: a legend that
            # reflowed the moment it grew past the window would be worse than
            # one that is a few pixels shy of the frame.
            - self.style().pixelMetric(
                QtWidgets.QStyle.PixelMetric.PM_ScrollBarExtent
            )
        )

    def _swatch(self, kind, color):
        """The mark the panel draws this with, at the size of a line of text."""

        from matplotlib.colors import to_rgba

        width, height = self.SWATCH_PX

        pixmap = QtGui.QPixmap(width, height)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)

        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Alpha and all: an attitude tick is drawn at half opacity and a
        # swatch that showed it solid would be a swatch of another colour.
        shade = QtGui.QColor.fromRgbF(*to_rgba(color))

        if kind == "patch":
            painter.fillRect(0, 3, width, height - 6, shade)
        elif kind == "dot":
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(shade)
            painter.drawEllipse(QtCore.QPointF(width / 2.0, height / 2.0), 3.0, 3.0)
        else:
            pen = QtGui.QPen(shade)
            pen.setWidthF(2.5)

            painter.setPen(pen)
            painter.drawLine(
                QtCore.QPointF(2.0, height - 2.0), QtCore.QPointF(width - 2.0, 2.0)
            )

        painter.end()

        return pixmap


class TracePanel(QtWidgets.QWidget):
    """
    Every plane the traces hold, with what it is allowed to say editable.

    The three things worth arguing with are here and nowhere else: whether a
    record speaks at all, what attitude it carries, and how far along its trace
    it reaches. None of them is written back to the layer. A source file is a
    record of what was surveyed, and a section is an argument about it; the
    argument is saved as its own assertion -- `Write curation` -- so that
    re-reading the layer tomorrow gives the survey back, not yesterday's
    opinion of it.

    The `crosses` column is the reason the panel is next to the section rather
    than in a dialog: fifty-six planes on a nine-kilometre line produce three
    crossings, and which three changes as the trace moves.
    """

    COLUMNS = ("category", "attitude", "source", "anchor", "reach", "crosses")

    changed = QtCore.pyqtSignal()

    def __init__(self, source, parent=None):
        super().__init__(parent)

        self.source = source
        self._filling = False

        self.table = QtWidgets.QTableWidget(len(source.traces), len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.itemChanged.connect(self._on_item_changed)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)

        self.save_button = QtWidgets.QPushButton("Write curation...")
        self.save_button.setToolTip(
            "Save the edits as a gstruct fragment: one assertion per record "
            "changed, over a source layer left as it was found."
        )
        self.save_button.clicked.connect(self.write_curation)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.table, stretch=1)
        layout.addWidget(self.save_button)

        self.fill()

    # -- the table --------------------------------------------------------

    def fill(self):
        """Every record as a row, with only the editable cells editable."""

        self._filling = True

        editable = (
            QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsSelectable
            | QtCore.Qt.ItemFlag.ItemIsEditable
        )
        fixed = QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable

        for row, record in enumerate(self.source.traces):
            name = QtWidgets.QTableWidgetItem(record.category)
            name.setFlags(fixed | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            name.setCheckState(
                QtCore.Qt.CheckState.Checked if record.enabled
                else QtCore.Qt.CheckState.Unchecked
            )
            self.table.setItem(row, 0, name)

            attitude = QtWidgets.QTableWidgetItem(
                f"{record.plane.dipazim:.0f}/{record.plane.dipang:.0f}"
            )
            attitude.setFlags(editable)
            self.table.setItem(row, 1, attitude)

            source = QtWidgets.QTableWidgetItem(str(record.attrs.get("src", "")))
            source.setFlags(fixed)
            self.table.setItem(row, 2, source)

            anchor = QtWidgets.QTableWidgetItem(
                "-" if record.anchor is None else f"{record.anchor:.0f} m"
            )
            anchor.setFlags(fixed)
            self.table.setItem(row, 3, anchor)

            # Editable only where there is a point to reach out from. A record
            # with no anchor is its trace, and there is no number to set.
            reach = QtWidgets.QTableWidgetItem(self._reach_text(record))
            reach.setFlags(editable if record.anchor is not None else fixed)
            self.table.setItem(row, 4, reach)

            crosses = QtWidgets.QTableWidgetItem("")
            crosses.setFlags(fixed)
            self.table.setItem(row, 5, crosses)

        self._filling = False

    def _reach_text(self, record):
        if record.anchor is None:
            return "whole trace"

        s0, s1 = record.extent(self.source.half_span)

        return f"{(s1 - s0) / 2.0:.0f}"

    def _on_item_changed(self, item):
        if self._filling:
            return

        record = self.source.traces[item.row()]
        column = item.column()

        if column == 0:
            self.source.set_enabled(
                record, item.checkState() == QtCore.Qt.CheckState.Checked
            )
        elif column == 1:
            if not self._apply_attitude(record, item.text()):
                return
        elif column == 4:
            if not self._apply_reach(record, item.text()):
                return
        else:
            return

        self.changed.emit()

    def _apply_attitude(self, record, text):
        """`dip direction / dip`, or the old value back."""

        try:
            azimuth, dip = (float(part) for part in text.replace(",", "/").split("/"))
        except ValueError:
            self.refresh()
            return False

        if not 0.0 <= dip <= 90.0:
            self.refresh()
            return False

        self.source.set_plane(record, azimuth, dip)

        return True

    def _apply_reach(self, record, text):
        """Half-span in metres; empty hands the record back to the default."""

        stripped = text.strip()

        if not stripped:
            self.source.set_span(record, None, None)
            return True

        try:
            half = float(stripped)
        except ValueError:
            self.refresh()
            return False

        self.source.set_span(record, record.anchor - half, record.anchor + half)

        return True

    def refresh(self):
        """The table back in step with the records, after an edit or a reach."""

        self._filling = True

        for row, record in enumerate(self.source.traces):
            self.table.item(row, 1).setText(
                f"{record.plane.dipazim:.0f}/{record.plane.dipang:.0f}"
            )
            self.table.item(row, 4).setText(self._reach_text(record))

        self._filling = False

    def show_crossings(self, geoprofiles):
        """Which records the section actually meets, and on how many profiles."""

        per_profile = geoprofiles.lines_with_attitudes_intersections or []

        met = {}

        for profile in per_profile:
            for category, traces in profile.items():
                for trace in traces:
                    key = (category, round(trace.src_dip_dir), round(trace.src_dip_ang))
                    met[key] = met.get(key, 0) + 1

        self._filling = True

        for row, record in enumerate(self.source.traces):
            key = (
                record.category,
                round(record.plane.dipazim),
                round(record.plane.dipang),
            )
            count = met.get(key, 0)
            self.table.item(row, 5).setText(str(count) if count else "")

        self._filling = False

    # -- saying it somewhere that lasts -----------------------------------

    def write_curation(self):
        """
        The edits as a gstruct fragment: one assertion per record changed.

        `reach` is a proposed axis, not one the format already defines. It
        behaves like the axes that are there -- an interval on a trace, last
        one covering a progressive wins -- so `value_at` would read it without
        being taught anything; but the name is a suggestion and should be
        settled before anything is built on it.
        """

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Write curation", "curation_gsurf.gstruct", "gstruct (*.gstruct)"
        )

        if not path:
            return

        text, written = self.curation_text()

        Path(path).write_text(text, encoding="utf8")

        QtWidgets.QMessageBox.information(
            self,
            "Curation written",
            f"{written} structure(s) carry an assertion.\n\n{path}",
        )

    def curation_text(self):
        """The fragment and how many structures it speaks about."""

        lines = [
            "gstruct 0.1",
            f"crs EPSG:{self.source_epsg()}",
            'project "Curatela da gSurf: portata e giaciture decise in sezione"',
            'note "Si applica sopra il dataset sorgente. Ogni riga e\' '
            'un\'asserzione umana, non un derivato."',
            "",
        ]

        written = 0

        for record in self.source.traces:
            assertions = []

            if not record.enabled:
                assertions.append("  span use * * excluded src=gsurf")

            # Only a reach somebody set. A record still living on the tool's
            # default has had no decision made about it, and writing the
            # default out as an assertion would put words in the geologist's
            # mouth -- and freeze a number that is meant to be moved.
            ends = (
                record.reach_endpoints(self.source.half_span)
                if record.span is not None else None
            )

            if ends is not None:
                (x0, y0), (x1, y1) = ends
                s0, s1 = record.extent(self.source.half_span)
                assertions.append(
                    f"  span reach @{x0:.2f},{y0:.2f} @{x1:.2f},{y1:.2f} "
                    f"{(s1 - s0) / 2.0:.0f} src=gsurf plane={record.plane.dipazim:.0f}/"
                    f"{record.plane.dipang:.0f}"
                )

            if not assertions:
                continue

            lines.append(f'structure "{record.category}"')
            lines.extend(assertions)
            lines.append("")
            written += 1

        return "\n".join(lines), written

    def source_epsg(self):
        crs = getattr(self.source, "crs", None)

        return crs.to_epsg() if crs is not None else 0


def remembered():
    """
    Where the windows were left last time, if there is a desktop they were left on.

    Off-screen there is no window manager, no screen to be placed against and
    no session to continue -- and a layout restored there is one real run's
    arrangement pushed into a run that measures pixels. `checks/run.py` works
    in exactly that mode, and a section window last left as a strip would make
    its ink counts meaningless. So off-screen the windows come up at their own
    size and nothing is written back.
    """

    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        return None

    return QtCore.QSettings("gSurf", "sections")


# The section itself, in the same store as the layout. One JSON string rather
# than a key per number, for the reason `recent.py` gives about its own lists:
# QSettings has no faithful round trip for a nested list, and reads a
# one-element one back as a scalar.
STATE_KEY = "state/last"


def read_state(settings):
    """What the last run wrote about the section, or nothing."""

    if settings is None:
        return {}

    raw = settings.value(STATE_KEY)

    if not isinstance(raw, str):
        return {}

    try:
        state = json.loads(raw)
    except ValueError:
        return {}

    return state if isinstance(state, dict) else {}


def applicable(session, state):
    """
    The part of a remembered state that belongs to the session being opened.

    Two kinds of thing are in there and they travel differently.

    How many profiles at what spacing, and how far a point measurement
    reaches, are ways of working. They are not tied to anywhere and they come
    back whatever is open -- somebody who works in bundles of thirteen at
    250 m works that way in the next area too.

    Where the trace was and how the map was framed are *places*: metres in a
    projection. The same pair of numbers is somewhere else under a different
    projection and nowhere at all under a DEM of another region, so those come
    back only over the source they were written on, and are dropped rather
    than guessed at -- a section restored off the DEM would come up empty with
    nothing to say why, which is worse than coming up in the middle.
    """

    kept = {}
    count, offset = state.get("profiles"), state.get("offset")

    # Checked against what the controls hold rather than taken as written. This
    # file is one a hand can reach, and a window is built on it before there is
    # a running application to put an error in front of: an even count would
    # come out of `Profilers` as an exception during construction, and an
    # out-of-range spacing as a spin box quietly saying something the section
    # was not computed from. Anything that fails falls back to the default,
    # which is the one number known to work.
    if isinstance(count, int) and count % 2 == 1 and _within(count, BUNDLE_RANGE):
        kept["profiles"] = int(count)

    if isinstance(offset, (int, float)) and _within(offset, OFFSET_RANGE):
        kept["offset"] = float(offset)

    # None is a value here and the one the box calls "whole trace", so it is
    # kept as it comes; a zero is not, the box reading that as the whole trace
    # too and `_on_reach_changed` never writing one.
    reach = state.get("reach")

    if reach is None and "reach" in state:
        kept["reach"] = None
    elif isinstance(reach, (int, float)) and REACH_RANGE[0] < reach <= REACH_RANGE[1]:
        kept["reach"] = float(reach)

    # Whether the legend is up is a habit like the rest of them: somebody
    # reading sections wants the names, somebody preparing a figure does not,
    # and neither has anything to do with which DEM is open.
    if isinstance(state.get("legend"), bool):
        kept["legend"] = state["legend"]

    if state.get("source") != _source_key(session) or state.get("epsg") != session.epsg:
        return kept

    trace = _as_trace(state.get("trace"), session.bounds)
    if trace is not None:
        kept["trace"] = trace

    extent = _as_extent(state.get("extent"))
    if extent is not None:
        kept["extent"] = extent

    return kept


def _within(value, limits):
    return limits[0] <= value <= limits[1]


def _source_key(session):
    """What the coordinates in a state were measured on."""

    return str(session.base_path) if session.base_path is not None else ""


def _as_trace(value, bounds):
    """Two ends on this DEM, or None for anything else."""

    try:
        (x0, y0), (x1, y1) = ((float(x), float(y)) for x, y in value)
    except (TypeError, ValueError):
        return None

    left, bottom, right, top = bounds

    for x, y in ((x0, y0), (x1, y1)):
        if not (left <= x <= right and bottom <= y <= top):
            return None

    # A section with no length is what the tool refuses to compute anyway, and
    # a press with no drag after it leaves exactly that behind.
    if np.hypot(x1 - x0, y1 - y0) < 1.0:
        return None

    return [(x0, y0), (x1, y1)]


def _as_extent(value):
    """A framing that can be set on an axis, or None."""

    try:
        left, right, bottom, top = (float(v) for v in value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite([left, right, bottom, top]).all():
        return None

    if not (left < right and bottom < top):
        return None

    return [left, right, bottom, top]


class SatelliteWindow(QtWidgets.QWidget):
    """
    A panel that used to be a dock, given a window of its own.

    Parented to the map and flagged `Window`. Parented for two reasons that do
    not show: Qt destroys it along with the tool, so there is no lifetime to
    keep track of; and a window that has a parent does not count towards
    `quitOnLastWindowClosed` -- which matters, because while a tool runs the
    launcher is hidden underneath it, and without the parent closing this one
    would be the last window closed and take the application with it.

    Flagged a window rather than left the `Qt::Tool` a floating dock becomes: a
    tool window on X11 stays over its parent and out of the taskbar, and the
    point of taking the section out of the dock was to be able to put it on the
    other screen and leave it there.

    Closing hides. What is inside is expensive -- the bundle's panels are
    rebuilt only when the section's length moves enough to matter -- and a
    window shut by accident should not cost that. The way back is the Windows
    menu, and the menu follows the window rather than the other way round, so a
    close from the title bar unticks its own box.
    """

    visibility_changed = QtCore.pyqtSignal(bool)

    def __init__(self, title, content, size, parent=None):
        super().__init__(parent)

        self.setWindowFlag(QtCore.Qt.WindowType.Window, True)
        self.setWindowTitle(title)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(content)

        self.resize(*size)

    def closeEvent(self, event):
        event.ignore()
        self.hide()

    def hideEvent(self, event):
        super().hideEvent(event)
        self.visibility_changed.emit(False)

    def showEvent(self, event):
        super().showEvent(event)
        self.visibility_changed.emit(True)


class OddSpinBox(QtWidgets.QSpinBox):
    """
    A spin box that will not hold an even number.

    The bundle is central: the trace you drag is the middle profile and the
    rest are laid off it in pairs, which is what `reverse` relies on and what
    makes the line on the map the section rather than its edge. A count with
    no middle is not a bundle this tool can draw, and `Profilers` will not
    pick a side on anyone's behalf -- it raises.

    A step of two keeps the arrows on odd numbers, but a box can be typed
    into, and the keyboard reached past the step to the one thing underneath
    that cannot cope. What made it worth a class rather than a guard further
    down is where the raise lands: `_on_count_changed` is a Qt slot, and an
    exception out of a slot under PyQt6 is not an error message but qFatal --
    the process aborts, with the section and everything unsaved in it.

    So the refusal is the box's own, and it is a refusal rather than a
    correction made behind the typing. An even number is `Intermediate`: it
    may stand in the line edit, because 4 is on the way to 41, but it is not
    handed on as a value -- nothing recomputes, and the count the map was
    drawn from still holds. What settles it is `fixup`, when the edit ends,
    and it goes up: four profiles asked for, five given, because five is the
    nearest bundle with a middle and the alternative is to quietly draw one
    fewer than was typed.

    Both ends of `BUNDLE_RANGE` are odd, which is what makes going up safe --
    clamping to a bound can then never land back on an even number.
    """

    def validate(self, text, pos):
        verdict = QtGui.QValidator.State
        state, text, pos = super().validate(text, pos)

        # Read back through the box's own conversion rather than `int`, so
        # that what is being tested for evenness is the number the box would
        # arrive at from this text and not a second opinion about it.
        if state == verdict.Acceptable and self.valueFromText(text) % 2 == 0:
            return verdict.Intermediate, text, pos

        return state, text, pos

    def fixup(self, text):
        fixed = super().fixup(text)
        value = self.valueFromText(fixed)

        if value % 2 == 0:
            return self.textFromValue(min(value + 1, self.maximum()))

        return fixed

    def setValue(self, value):
        # The other door into the value, and it does not pass the validator:
        # Qt only clamps `setValue` to the range. A caller handing over an
        # even number would set one and emit `valueChanged` carrying it, which
        # is the same abort reached without anybody typing -- so the rounding
        # is repeated here rather than the promise being made only about the
        # keyboard.
        value = int(value)

        super().setValue(value + 1 if value % 2 == 0 else value)


class ProfilesWindow(QtWidgets.QMainWindow):
    """The map with a section trace on it; the section and the records beside it."""

    HANDLE_RADIUS_PX = 12

    def __init__(
        self,
        session,
        traces=None,
        num_profiles=BUNDLE_DEFAULT,
        offset=OFFSET_DEFAULT,
        legend="beside",
        state=None,
    ):
        super().__init__()

        self.session = session
        self.traces = traces
        self.num_profiles = int(num_profiles)
        self.offset = float(offset)

        self.dragging = None          # "start", "end" or None
        self.window = None            # the DEM crop the sampling reads
        self.geoprofiles = None
        self.single = None
        self.bundle = None
        self._bundle_reach = None
        self._satellites_up = False

        # On, because a section whose colours are not explained anywhere is
        # what the legend was added for; off is for the run where the section
        # window is a figure being got ready rather than something being read.
        self.legend_beside_section = True

        left, bottom, right, top = session.bounds
        cx, cy = session.center()

        # West to east through the middle, which is where a first look goes.
        # Capped, because the session can be a whole mosaic: a third of the
        # short side of this one is a 24 km section, and the first frame would
        # sample five thousand points to draw something nobody asked for. Ten
        # kilometres is a section; past that it is a transect, and one drag
        # makes it.
        span = min(min(right - left, top - bottom) / 6.0, MAX_DEFAULT_LENGTH / 2.0)

        self.trace = [(cx - span, cy), (cx + span, cy)]

        # Where the last run left off, over the defaults and before anything is
        # built from them: the spin boxes take their values from these
        # attributes, and the first bundle is computed once, on the trace that
        # is being continued rather than on the one nobody asked for.
        if state is None:
            state = read_state(remembered())

        state = applicable(session, state)
        self._apply_state(state)

        self.polygons, self.lines, self.overlay_dropped = self._overlay_geometry()

        self.setWindowTitle(f"gSurf - sections - {session.label}")
        self._build_ui(legend)
        self._draw_base_map()

        # After the base map, which is what settles the axis limits in the
        # first place: set before it, the hillshade's own extent would overrule
        # them.
        if "extent" in state:
            self.map_view.restore_framing(state["extent"])

        self.update_bundle()

        hint = "drag an end to move the section; the bundle follows on release"

        if self.overlay_dropped:
            detail = ", ".join(
                f"{count} {reason}" for reason, count in self.overlay_dropped.items()
            )
            hint += f" (backdrop: {detail}, left out)"

        self.statusBar().showMessage(hint)

    # -- construction -----------------------------------------------------

    def _build_ui(self, legend):
        self.map_view = MapView(self.session, legend=legend)
        self.map_view.legend_handles_provider = self._legend_handles

        self.map_view.pressed.connect(self._on_map_pressed)
        self.map_view.dragged.connect(self._on_map_dragged)
        self.map_view.released.connect(self._on_map_released)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.map_view, stretch=1)
        layout.addWidget(LegendControls(self.map_view, placement=legend))

        self.setCentralWidget(central)

        # The stack stays what it was and so does its scroll area: a bundle of
        # twenty-five panels is taller than any window, screen or not, and the
        # section asks for its height outright in `SectionCanvas`.
        self.stack = QtWidgets.QStackedWidget()

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.stack)

        self.section_legend = SectionLegend()
        self.section_legend.setVisible(self.legend_beside_section)

        # Outside the scroll area, not in it: the panels scroll and the legend
        # is the same for all of them, being the bundle's and not one profile's.
        framed = QtWidgets.QWidget()

        beside = QtWidgets.QHBoxLayout(framed)
        beside.setContentsMargins(0, 0, 0, 0)
        beside.setSpacing(0)
        beside.addWidget(scroll, stretch=1)
        beside.addWidget(self.section_legend)

        self.section_window = SatelliteWindow(
            "gSurf - section", framed, SECTION_WINDOW_PX, parent=self
        )

        self.panel = None
        self.traces_window = None

        if self.traces is not None:
            self.panel = TracePanel(self.traces)
            self.panel.changed.connect(self.update_bundle)

            self.traces_window = SatelliteWindow(
                "gSurf - traces", self.panel, TRACES_WINDOW_PX, parent=self
            )

        # Keyed, because geometry is saved and restored under these names and a
        # tool opened without traces has two windows rather than three. Not
        # `windows`, a letter away from `self.window` -- which is the DEM crop
        # the sampling reads, and nothing to do with any of this.
        self.window_group = {"map": self, "section": self.section_window}

        if self.traces_window is not None:
            self.window_group["traces"] = self.traces_window

        self._build_controls()
        self._build_menu()

    def _build_controls(self):
        bar = self.addToolBar("section")
        bar.setMovable(False)

        # Said outright because the bar holds one action and it has no icon. A
        # QToolButton does fall back to its text when the icon is null, but
        # that is a fallback and not a promise, and an empty 20-pixel button
        # would be a hard thing to guess at.
        bar.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)

        self.count_spin = OddSpinBox()
        self.count_spin.setRange(*BUNDLE_RANGE)
        self.count_spin.setSingleStep(2)
        self.count_spin.setValue(self.num_profiles)

        # Read back rather than assumed. `num_profiles` arrives as an argument
        # and nothing on that path has been through the box, so an even one
        # would be rounded up in the box and left as it came here -- and this
        # is the attribute the section is computed from. Taken before the
        # signal is connected, so it is this line that has to do it: a
        # `setValue` above would have nothing to tell.
        self.num_profiles = self.count_spin.value()

        self.count_spin.setToolTip(
            "How many parallel profiles the bundle holds, the trace being the "
            "middle one -- so the count is odd, and an even one typed in is "
            "rounded up. Recomputed on release, not while dragging."
        )
        self.count_spin.valueChanged.connect(self._on_count_changed)

        self.offset_spin = QtWidgets.QDoubleSpinBox()
        self.offset_spin.setRange(*OFFSET_RANGE)
        self.offset_spin.setSingleStep(100.0)
        self.offset_spin.setDecimals(0)
        self.offset_spin.setSuffix(" m")
        self.offset_spin.setValue(self.offset)
        self.offset_spin.setToolTip("Spacing between the parallel profiles.")
        self.offset_spin.valueChanged.connect(self._on_offset_changed)

        # Next to the trace's own numbers rather than in a menu: turning a
        # section round is something you do while looking at it, often twice in
        # a row to see which way reads better.
        self.reverse_action = QtGui.QAction("&Reverse", self)
        self.reverse_action.setShortcut("Ctrl+R")
        self.reverse_action.setToolTip(
            "Turn the section round (Ctrl+R). The profile is mirrored and the "
            "panels of the bundle arrive in the opposite order; the lines on "
            "the map do not move."
        )
        self.reverse_action.triggered.connect(self.reverse)
        self.addAction(self.reverse_action)

        bar.addWidget(QtWidgets.QLabel("  profiles "))
        bar.addWidget(self.count_spin)
        bar.addWidget(QtWidgets.QLabel("  spacing "))
        bar.addWidget(self.offset_spin)

        if self.traces is not None:
            self.reach_spin = QtWidgets.QDoubleSpinBox()
            self.reach_spin.setRange(*REACH_RANGE)
            self.reach_spin.setSingleStep(50.0)
            self.reach_spin.setDecimals(0)
            self.reach_spin.setSuffix(" m")
            self.reach_spin.setSpecialValueText("whole trace")
            self.reach_spin.setValue(self.traces.half_span or 0.0)
            self.reach_spin.setToolTip(
                "How far a measurement taken at one point reaches along the "
                "trace it was taken on. Traces that carry no anchor are not "
                "affected: the trace is their evidence and they keep all of it."
            )
            self.reach_spin.valueChanged.connect(self._on_reach_changed)

            bar.addWidget(QtWidgets.QLabel("  reach "))
            bar.addWidget(self.reach_spin)

        bar.addSeparator()
        bar.addAction(self.reverse_action)

    # -- the window group --------------------------------------------------

    def _build_menu(self):
        """The way back to a window that was closed, and to one that is buried."""

        menu = self.menuBar().addMenu("&Windows")

        for label, window in (
            ("&Section", self.section_window),
            ("&Traces", self.traces_window),
        ):
            if window is None:
                continue

            action = QtGui.QAction(label, self)
            action.setCheckable(True)
            action.setChecked(True)
            action.triggered.connect(
                lambda shown, w=window: self._show_satellite(w, shown)
            )

            # The window is the authority on whether it is up: closed from its
            # own title bar it has to untick this box itself, or the menu would
            # claim a window that is not there.
            window.visibility_changed.connect(action.setChecked)

            menu.addAction(action)

        menu.addSeparator()

        # Here rather than on the section's own frame, which has no bar of its
        # own to put it on, and next to the windows because that is what it is:
        # a piece of the section window being shown or not.
        self.legend_action = QtGui.QAction("Section &legend", self)
        self.legend_action.setCheckable(True)
        self.legend_action.setChecked(self.legend_beside_section)
        self.legend_action.setToolTip(
            "The units, lines and attitudes the section crosses, named beside "
            "the panels."
        )
        self.legend_action.toggled.connect(self._show_section_legend)
        menu.addAction(self.legend_action)

        menu.addSeparator()

        front = QtGui.QAction("Bring all to &front", self)
        front.triggered.connect(self._raise_group)
        menu.addAction(front)

    def _show_section_legend(self, shown):
        self.legend_beside_section = bool(shown)
        self.section_legend.setVisible(self.legend_beside_section)

        # What it would have said while it was hidden was never worked out, so
        # it is caught up here rather than at the next release of the trace.
        if self.legend_beside_section and self.geoprofiles is not None:
            self._refresh_section_legend(self.geoprofiles)

    def _show_satellite(self, window, shown):
        window.setVisible(shown)

        if shown:
            # Shown is not the same as seen: a window put back from the menu
            # can come up behind the map it was asked for from.
            window.raise_()
            window.activateWindow()

    def _raise_group(self):
        for window in self.window_group.values():
            if window.isVisible():
                window.raise_()

    def showEvent(self, event):
        """
        The satellites come up with the map -- the first time, and only then.

        After that their being on screen is the user's business: a section
        window closed on purpose must not come back because the map happened to
        be un-minimised. That this hangs off `showEvent` rather than off `build`
        is what lets a `ProfilesWindow` constructed directly -- which is how
        `checks/check_sections.py` builds one -- get its whole group by calling
        `show()`, with nothing to remember to do.
        """

        super().showEvent(event)

        if self._satellites_up:
            return

        self._satellites_up = True

        for window in self.window_group.values():
            if window is not self:
                window.show()

    def restore_geometry(self):
        """
        Puts the group back where it was left, and says whether the map was.

        The map's own answer is the one the caller needs: `fit_to_screen` sizes
        a window that has nothing remembered about it, and calling it over a
        restored geometry would undo the restoring. The satellites have no such
        competition and are simply put back.
        """

        settings = remembered()

        if settings is None:
            self._place_unremembered()
            return False

        restored = set()

        for name, window in self.window_group.items():
            saved = settings.value(f"geometry/{name}")

            if isinstance(saved, QtCore.QByteArray) and window.restoreGeometry(saved):
                restored.add(name)

        # Whatever was not remembered -- a first run, a tool opened with traces
        # for the first time -- still has to be put somewhere.
        if not restored.issuperset(set(self.window_group) - {"map"}):
            self._place_unremembered(skip=restored)

        return "map" in restored

    def _place_unremembered(self, skip=()):
        """
        Where a satellite goes with nothing remembered about it.

        Down the right edge of the screen, and only if the screen has the room
        for it: on one 1920-wide desktop with the map maximised there is no
        arrangement that does not cover it, and the window manager's own
        placement is a better guess than ours. This runs once in the life of an
        installation -- after it, there is something remembered.
        """

        available = self.screen().availableGeometry()

        if available.width() < ROOM_TO_PLACE_PX:
            return

        top = available.top() + 40

        for name, window in self.window_group.items():
            if name == "map" or name in skip:
                continue

            window.move(available.right() - window.width() - 20, top)
            top += window.height() + 40

    def save_geometry(self):
        settings = remembered()

        if settings is None:
            return

        for name, window in self.window_group.items():
            settings.setValue(f"geometry/{name}", window.saveGeometry())

    # -- the section, from one run to the next -----------------------------

    def _apply_state(self, state):
        """
        Takes on whatever of a remembered state has survived `applicable`.

        Called before the controls exist, so it writes the attributes and not
        the widgets: `_build_controls` reads these to set the spin boxes, and
        doing it the other way round would fire their signals and recompute a
        bundle per restored number.
        """

        if "profiles" in state:
            self.num_profiles = int(state["profiles"])

        if "offset" in state:
            self.offset = float(state["offset"])

        if "trace" in state:
            self.trace = list(state["trace"])

        if "reach" in state and self.traces is not None:
            self.traces.set_half_span(state["reach"])

        if "legend" in state:
            self.legend_beside_section = bool(state["legend"])

    def current_state(self):
        """The section as it stands, in the shape `read_state` hands back."""

        return {
            "source": _source_key(self.session),
            "epsg": self.session.epsg,
            "trace": [[float(x), float(y)] for x, y in self.trace],
            "extent": self.map_view.framing,
            "profiles": int(self.num_profiles),
            "offset": float(self.offset),
            "reach": self.traces.half_span if self.traces is not None else None,
            "legend": bool(self.legend_beside_section),
        }

    def save_state(self):
        settings = remembered()

        if settings is None:
            return

        settings.setValue(STATE_KEY, json.dumps(self.current_state()))

    def closeEvent(self, event):
        # Saved on the way out rather than as each window moves: the arrangement
        # worth keeping is the one the work ended on, and a window dragged
        # across a screen would otherwise write settings on every frame of it.
        # The same argument covers the trace, which moves a great deal more.
        self.save_geometry()
        self.save_state()

        super().closeEvent(event)

    def _draw_base_map(self):
        self.map_view.draw_base_map()

        axes = self.map_view.axes
        (x0, y0), (x1, y1) = self.trace

        self.trace_line = self.map_view.add_animated(
            axes.add_line(
                Line2D([x0, x1], [y0, y1], color="crimson", linewidth=1.6, zorder=6)
            )
        )
        self.handles = self.map_view.add_animated(
            axes.add_line(
                Line2D(
                    [x0, x1],
                    [y0, y1],
                    linestyle="None",
                    marker="o",
                    markersize=7,
                    markerfacecolor="white",
                    markeredgecolor="crimson",
                    zorder=7,
                )
            )
        )
        self.bundle_lines = self.map_view.add_animated(
            axes.add_line(
                Line2D([], [], color="crimson", linewidth=0.7, alpha=0.5, zorder=5)
            )
        )

        # The reach drawn where it is decided. A trace is on the map as a thin
        # backdrop line if it was opened as one, but the part of it a plane is
        # held to speak for is not in any layer -- it is the number in the
        # panel, and without this it would be a number with nothing under it.
        self.reach_lines = self.map_view.add_animated(
            axes.add_line(
                Line2D([], [], color="#1f77b4", linewidth=2.2, alpha=0.85, zorder=4)
            )
        )

        # Which end the section starts at, drawn over the handle that grabs it.
        # Without it the trace is a line with two identical ends and reversing
        # it changes nothing anyone can see on the map -- the section panel
        # mirrors, and the map stays exactly as it was.
        self.start_marker = self.map_view.add_animated(
            axes.add_line(
                Line2D(
                    [x0],
                    [y0],
                    linestyle="None",
                    marker="o",
                    markersize=4,
                    markerfacecolor="crimson",
                    markeredgecolor="crimson",
                    zorder=8,
                )
            )
        )

        self._draw_reach()

        self.map_view.refresh_legend()

        # Home is the whole area, not whichever framing the bar first saw --
        # which matters more here than in the other tools, since a run that
        # comes up on a remembered zoom would otherwise have no way back out.
        self.map_view.anchor_home()

    def _legend_handles(self):
        # The section itself is not switchable: it is what the hand is
        # steering, and one dragged invisible is worse than one in the way. The
        # bundle is, because at twenty-five profiles it is a hatch over the map
        # and the middle line is the one being aimed.
        handles = [
            Line2D([], [], color="crimson", linewidth=1.6, label="section"),
            self.map_view.switchable(
                Line2D([], [], color="crimson", linewidth=0.7, alpha=0.5, label="bundle"),
                self.bundle_lines,
            ),
        ]

        if self.traces is not None:
            handles.append(
                self.map_view.switchable(
                    Line2D([], [], color="#1f77b4", linewidth=2.2, label="reach"),
                    self.reach_lines,
                )
            )

        return handles

    # -- interaction ------------------------------------------------------

    def _near(self, x, y, point):
        px, py = self.map_view.display_xy(*point)
        ex, ey = self.map_view.display_xy(x, y)

        return np.hypot(ex - px, ey - py) <= self.HANDLE_RADIUS_PX

    def _on_map_pressed(self, x, y):
        """An end grabbed, or a new trace started where the press landed."""

        for name, ndx in (("start", 0), ("end", 1)):
            if self._near(x, y, self.trace[ndx]):
                self.dragging = name
                return

        # Not on a handle: this is a new section, rubber-banded from here.
        self.trace = [(float(x), float(y)), (float(x), float(y))]
        self.dragging = "end"
        self._redraw_trace()

    def _on_map_dragged(self, x, y):
        if self.dragging is None:
            return

        ndx = 0 if self.dragging == "start" else 1
        self.trace[ndx] = (float(x), float(y))

        self._redraw_trace()
        self.update_single()

    def _on_map_released(self):
        if self.dragging is None:
            return

        self.dragging = None
        self.update_bundle()

    def _redraw_trace(self):
        (x0, y0), (x1, y1) = self.trace

        self.trace_line.set_data([x0, x1], [y0, y1])
        self.handles.set_data([x0, x1], [y0, y1])
        self.start_marker.set_data([x0], [y0])

    def reverse(self):
        """
        Turns the section round: the end it starts from becomes the end it ends on.

        Not cosmetic, and not a move either -- nothing on the ground changes.
        A profile is read from its own start, so the section comes out mirrored:
        the fault that was on the left of the panel is on the right of it. And
        the bundle is laid out from the trace's direction, `Profilers` putting
        half of it to the left of the line and half to the right, so the same
        set of lines comes back indexed the other way and the panels arrive in
        the opposite order down the window. Which is the point of the button: a
        section read against the way the ground is usually drawn reads as a
        structure dipping the wrong way, and the fix is not to redraw it.

        The two ends only swap, so the length cannot change: the fitted axis
        still fits, and this costs one bundle and no rebuild of the panels.
        """

        self.trace = [self.trace[1], self.trace[0]]

        self._redraw_trace()
        self.update_bundle()

    def _on_count_changed(self, value):
        self.num_profiles = int(value)
        self._bundle_reach = None       # the panel count changed: rebuild
        self.update_bundle()

    def _on_offset_changed(self, value):
        self.offset = float(value)
        self.update_bundle()

    def _on_reach_changed(self, value):
        self.traces.set_half_span(None if value <= 0.0 else float(value))

        # The default moved, so every record that was living on it moved too:
        # the panel shows numbers that are now stale until it is told.
        if self.panel is not None:
            self.panel.refresh()

        self.update_bundle()

    # -- the section ------------------------------------------------------

    def _length(self):
        (x0, y0), (x1, y1) = self.trace

        return float(np.hypot(x1 - x0, y1 - y0))

    def _profilers(self, count):
        """The bundle, from the two ends and nothing between them."""

        from geogst.core.geology.profiles.profilers import Profilers
        from geogst.core.geometries.shapes.lines import Ln

        return Profilers.make_parallel_from_line(
            src_trace=Ln(np.asarray(self.trace, dtype=float)),
            src_crs=self.session.crs.to_wkt(),
            num_profiles=count,
            offset=self.offset,
        )

    def _grid(self, profilers):
        """
        The DEM crop the sampling reads, reread only when the bundle leaves it.

        Sampling takes its elevations from a grid held in memory, and the
        profiles move; rereading per frame would put a disk read inside the
        loop for a window that almost always still fits.
        """

        from geogst.core.geometries.grids.rasters import Grid
        from geogst.core.geometries.projections.geotransform import GeoTransform

        xs, ys = [], []

        for line in profilers.lines:
            coords = line.coords
            xs.extend([coords[:, 0].min(), coords[:, 0].max()])
            ys.extend([coords[:, 1].min(), coords[:, 1].max()])

        box = (min(xs), min(ys), max(xs), max(ys))

        if self.window is None or not self.window.contains(box):
            margin = max(self.offset, 0.1 * self._length(), 200.0)
            self.window = self.session.dem.window_over(box, margin=margin)

            if self.window is None:
                return None

            self._grid_cache = Grid(
                self.window.data,
                GeoTransform.from_gdal_geotransform(self.window.geotransform),
                self.session.crs.to_wkt(),
            )

        return self._grid_cache

    def _overlay_geometry(self):
        """
        The backdrop layers in the shapes the profiler takes, converted once.

        A section of bare topography with three dip ticks on it is not a
        geological section: the units the line crosses are what the ticks mean
        something against. The backdrop is already reprojected and clipped by
        the session, and it does not move while the trace does, so converting
        it here rather than per frame is the whole of the trick.

        What a layer of one role holds of another is counted and left behind,
        rather than converted into something it is not: a 4 m dangle put in
        with the lines would be drawn across the section as a mapped contact.
        A feature with no geometry at all is counted too -- 5 of the 236
        carbonate units carry none, and the Calabrian CASMEZ sheet 299 -- since
        a row that draws nothing is a thing to know about the layer, not an
        absence to pass over.
        """

        from collections import defaultdict
        from itertools import repeat

        from geogst.core.geometries.shapes.lines import Ln
        from geogst.core.geometries.shapes.polygons import Polygon

        polygons, lines = defaultdict(list), defaultdict(list)
        dropped = defaultdict(int)

        for source in self.session.overlay.sources:
            if source.role not in BACKDROP_ROLES:
                continue

            wanted = VectorSource.GEOMETRY_SUFFIX[source.role]

            # A layer categorised by nothing has no such column: `_categorize`
            # adds it only when there is a field to categorise by. That is not
            # an edge: no field is guessed for a line backdrop, so a fault
            # layer taken as it comes has none, and this used to end the tool
            # here with a KeyError the launcher could only repeat verbatim.
            # Uncategorised, the layer is one category named after itself --
            # which is what its single legend entry has always said.
            if "_gsurf_category" in source.frame.columns:
                categories = source.frame["_gsurf_category"]
            else:
                categories = repeat(source.layer or source.path.stem)

            for category, geometry in zip(categories, source.frame.geometry):
                if geometry is None or geometry.is_empty:
                    dropped["with no geometry"] += 1
                    continue

                parts, skipped = single_parts(geometry, wanted)

                if skipped:
                    dropped[f"not a {wanted.lower()}"] += skipped

                name = str(category)

                for part in parts:
                    if source.role == "lines":
                        lines[name].append(Ln(np.asarray(part.coords)[:, :2]))
                    else:
                        polygons[name].append(
                            Polygon(
                                outer=Ln(np.asarray(part.exterior.coords)[:, :2]),
                                inner=[
                                    Ln(np.asarray(hole.coords)[:, :2])
                                    for hole in part.interiors
                                ],
                            )
                        )

        return dict(polygons), dict(lines), dict(dropped)

    def _compute(self, count):
        """A GeoProfiles with the topography and whatever was opened on it."""

        from geogst.core.geology.profiles.geoprofiles import GeoProfiles

        if self._length() < 1.0:
            return None, "the section has no length yet"

        profilers = self._profilers(count)
        grid = self._grid(profilers)

        if grid is None:
            return None, "the section is off the DEM"

        wkt = self.session.crs.to_wkt()
        geoprofiles = GeoProfiles(profilers=list(profilers), crs=wkt)

        err = geoprofiles.sample_grid(grid)
        if err:
            return None, str(err)

        if self.traces is not None and self.traces.records:
            err = geoprofiles.intersect_lines_with_attitudes(lines=self.traces.records)
            if err:
                return None, str(err)

        if self.polygons:
            err = geoprofiles.intersect_polygons(polygons=self.polygons, polygons_crs=wkt)
            if err:
                return None, str(err)

        if self.lines:
            err = geoprofiles.intersect_lines(lines=self.lines, lines_crs=wkt)
            if err:
                return None, str(err)

        return geoprofiles, None

    def _reach(self):
        """The longest section this map can hold: its diagonal."""

        left, bottom, right, top = self.session.bounds

        return float(np.hypot(right - left, top - bottom))

    def _axis_params(self, s_max):
        """
        The window the panels keep, said outright rather than autoscaled.

        Both axes, because the first section drawn is rarely the longest and
        `ProfilesView` settles its window on the first frame: elevation from
        the DEM's own range, distance from whatever the caller can promise.
        """

        from geogst.plots.parameters import AxisPlotParams

        low, high = self.session.dem.z_range

        return AxisPlotParams(
            z_min=float(low) - Z_MARGIN,
            z_max=float(high) + Z_MARGIN,
            s_min=0.0,
            s_max=s_max,
            vertical_exaggeration=1.0,
        )

    def _polygon_colors(self):
        """
        The colour of each unit, so that the section is drawn in the map's.

        Handed over, the library uses them; withheld, it spreads a hue ramp of
        its own over whatever it was given -- and the map above was meanwhile
        colouring the same units from a different wheel entirely. Two palettes,
        neither wrong on its own, disagreeing about the one thing a section and
        the map over it have in common. This is the whole of the fix, and it
        applies whether or not the colours came from a QGIS project.

        Every unit in the window, not the ones the current trace happens to
        cross: the library paints an unknown category red, and a palette that
        changed as the trace moved would repaint the section under the hand
        moving it. All of them, however many there are -- a sheet's worth is
        scores, and what that costs is a dict of tuples.

        Which is why the legend goes with it. The library reads a palette this
        wide as a legend that wide, and settles it at build: the section panel
        would carry the names of units the trace has since been dragged away
        from. The map above is doing the dragging and has the same colours in
        its own legend, cut and switchable, so the section says the colours and
        the map says what they mean.
        """

        colors = {}

        for source in self.session.overlay.sources:
            if source.role != "polygons":
                continue

            for value, colour in source.colors.items():
                colors[str(value)] = colour

        return colors

    def _view_for(self, geoprofiles, s_max):
        from geogst.plots.profiles import ProfilesView

        return ProfilesView(
            geoprofiles,
            axis_params=self._axis_params(s_max),
            height=1.6,
            line_attitudes_intersections=_dock_style(),
            line_intersections=_crossing_style(),
            polygon_intersections=self._polygon_colors(),
            polygon_intersections_legend=False,
        )

    # -- what the panels are showing, said beside them --------------------

    def _refresh_section_legend(self, geoprofiles):
        """
        The legend brought up to what has just been drawn, if it is on show.

        Hidden it is not computed either: the walk over the intersections is
        cheap but it is per frame, and a frame during a drag has 26 ms to
        spend. `_show_section_legend` is what catches it up on the way back.

        What it costs when it is up, on the Monte Alpi backdrop and a bundle of
        five: 0.14 ms to work out what to say, 0.001 ms to find that it is
        already saying it, and 1.8 ms on the frame where it actually changes --
        which is the frame the section crossed into a new unit on.
        """

        if not self.legend_beside_section:
            return

        self.section_legend.set_entries(self._section_legend_entries(geoprofiles))

    def _section_legend_entries(self, geoprofiles):
        """
        What to say beside the panels about the section that was just drawn.

        Read off the intersections and not off the layers, because the layers
        are the window and the window is a sheet: a bundle over Monte Alpi
        crosses twelve of the thirty carbonate units, and a legend naming all
        thirty would be the map's legend, which is already open next to it.

        The colour each unit gets is the one the panel draws it in, which is
        not always the one the map used. The library paints a category its
        palette does not name in a default red, and an uncategorised polygon
        layer hands over no palette at all -- so `red` here is not a guess, it
        is what is on the screen. That the map has the same unit in green is a
        disagreement to see rather than to hide.

        The lines go the other way round. They are crossed by category and the
        categories are worth reading -- thirty-one crossings of `n/a` and nine
        of `?transcurrent` is the fault picture of a section -- but the panel
        draws every one of them in the same colour, so the colour is claimed
        once and the categories are listed under it as names, heaviest first.
        See `_crossing_style`.
        """

        from matplotlib.colors import to_rgba

        entries = []
        palette = self._polygon_colors()
        crossed = _crossings(geoprofiles.polygons_intersections)

        for source in self.session.overlay.sources:
            if source.role != "polygons":
                continue

            # A layer categorised by nothing is one category named after
            # itself, which is the name `_overlay_geometry` gave its geometry
            # and so the one the intersections come back under.
            heading = source.layer or source.path.stem
            order = source.category_order or [heading]

            listed = [value for value in order if crossed.get(str(value))]

            if not listed:
                continue

            entries.append(("heading", heading, None))

            shown = listed[: VectorSource.MAX_LEGEND_ENTRIES]
            rest = listed[VectorSource.MAX_LEGEND_ENTRIES :]

            for value in shown:
                colour = palette.get(str(value), "red")

                # `_legend_label` on purpose, private as it is: a unit spelled
                # one way beside the map and another beside the section would
                # read as two units.
                entries.append(
                    ("patch", source._legend_label(value), _as_color(colour))
                )

            if rest:
                entries.append(("note", f"+{len(rest)} more", None))

        met = sorted(
            ((name, count) for name, count in
             _crossings(geoprofiles.lines_intersections).items() if count),
            key=lambda pair: (-pair[1], pair[0]),
        )

        if met:
            style = _crossing_style()

            entries.append(("heading", self._role_heading("lines"), None))
            entries.append(
                (
                    "dot",
                    f"crossings ({sum(count for _, count in met)})",
                    _as_color(to_rgba(style.color, style.alpha)),
                )
            )

            for name, count in met[:SECTION_LEGEND_NOTES]:
                # An empty category is a row with no value in the field it was
                # categorised by, and it is usually the commonest one there is.
                entries.append(("note", f"{name or '(unnamed)'} ({count})", None))

            if len(met) > SECTION_LEGEND_NOTES:
                entries.append(
                    ("note", f"+{len(met) - SECTION_LEGEND_NOTES} more", None)
                )

        crossings = _attitude_crossings(geoprofiles)

        if crossings:
            style = _dock_style()

            entries.append(
                ("heading", self.traces.layer or self.traces.path.stem, None)
            )
            entries.append(
                (
                    "tick",
                    f"attitudes ({crossings})",
                    _as_color(to_rgba(style.color, style.alpha)),
                )
            )

        return entries

    def _role_heading(self, role):
        """
        What to call a group whose crossings arrive merged.

        `_overlay_geometry` keys its geometry by category and not by layer, so
        two fault layers crossing one section come back as one set of
        categories with no way to say which file each came from. Both names,
        then, rather than one of them chosen silently.
        """

        names = [
            source.layer or source.path.stem
            for source in self.session.overlay.sources
            if source.role == role
        ]

        return ", ".join(names) if names else role

    def update_single(self):
        """One profile, while the trace is moving."""

        start = perf_counter()

        geoprofiles, problem = self._compute(1)

        if geoprofiles is None:
            self.statusBar().showMessage(problem)
            return

        computed = perf_counter()

        if self.single is None:
            # The whole diagonal, once: while the hand is moving, a panel that
            # rescaled under the line would make every frame a different
            # picture and throw away the blitting background with it.
            self.single = SectionCanvas(self._view_for(geoprofiles, self._reach()))
            self.stack.addWidget(self.single)

        self.stack.setCurrentWidget(self.single)
        self.single.redraw(geoprofiles)
        self.map_view.blit()

        # Of the one profile on screen, not of the bundle it will become: the
        # legend names what is being looked at, and during a drag that is this.
        self._refresh_section_legend(geoprofiles)

        self._report(geoprofiles, 1, computed - start, perf_counter() - computed)

    def update_bundle(self):
        """The parallel bundle, once the hand has stopped."""

        start = perf_counter()

        geoprofiles, problem = self._compute(self.num_profiles)

        if geoprofiles is None:
            self.statusBar().showMessage(problem)
            return

        computed = perf_counter()

        self._draw_bundle_outline(geoprofiles)
        self._draw_reach()

        # The hand has stopped, so this one can be fitted to the section that
        # was actually drawn instead of to everywhere it might have gone: a
        # seven-kilometre section on a seventeen-kilometre axis is readable but
        # it is not the figure anybody wants to keep. Rebuilt only when the
        # length has moved enough to matter, since a rebuild is the panels.
        length = self._length()
        fitted = self.bundle is not None and self._bundle_reach is not None and (
            0.55 * self._bundle_reach <= length <= self._bundle_reach
        )

        if not fitted:
            if self.bundle is not None:
                self.stack.removeWidget(self.bundle)
                self.bundle.deleteLater()

            self._bundle_reach = length * 1.02
            self.bundle = SectionCanvas(self._view_for(geoprofiles, self._bundle_reach))
            self.stack.addWidget(self.bundle)

        self.stack.setCurrentWidget(self.bundle)
        self.bundle.redraw(geoprofiles)
        self.map_view.blit()

        self.geoprofiles = geoprofiles

        self._refresh_section_legend(geoprofiles)

        if self.panel is not None:
            self.panel.show_crossings(geoprofiles)

        self._report(geoprofiles, self.num_profiles, computed - start, perf_counter() - computed)

    def _draw_reach(self):
        """
        The stretch of each trace its plane is held to speak for.

        One path with NaN between the pieces, because there is one artist and
        a couple of hundred vertices: this is the same trick the bundle
        outline uses, and it is what keeps the map to one Line2D per thing
        rather than one per record.
        """

        if self.traces is None:
            return

        pieces = [
            line.coords[:, :2]
            for records in self.traces.records.values()
            for _, lines in records
            for line in lines
        ]

        if not pieces:
            self.reach_lines.set_data([], [])
            return

        gap = np.full((1, 2), np.nan)
        path = np.vstack([row for piece in pieces for row in (piece, gap)])

        self.reach_lines.set_data(path[:, 0], path[:, 1])

    def _draw_bundle_outline(self, geoprofiles):
        """The other profiles on the map, as one path broken by NaN."""

        lines = geoprofiles.profilers.lines

        if len(lines) < 2:
            self.bundle_lines.set_data([], [])
            return

        path = np.full((len(lines) * 3, 2), np.nan)

        for ndx, line in enumerate(lines):
            coords = line.coords
            path[ndx * 3] = coords[0, :2]
            path[ndx * 3 + 1] = coords[-1, :2]

        self.bundle_lines.set_data(path[:, 0], path[:, 1])

    def _report(self, geoprofiles, count, compute_s, draw_s):
        crossings = 0
        attitudes = geoprofiles.lines_with_attitudes_intersections

        if attitudes:
            crossings = sum(
                len(traces) for profile in attitudes for traces in profile.values()
            )

        total = compute_s + draw_s

        self.statusBar().showMessage(
            f"{count} profile{'s' if count != 1 else ''}, "
            f"{self._length() / 1000.0:.2f} km   "
            f"{crossings} attitude crossings   "
            f"compute {compute_s * 1000:6.1f} ms   draw {draw_s * 1000:5.1f} ms   "
            f"total {total * 1000:6.1f} ms   {1.0 / total if total else 0.0:4.1f} fps"
        )


def _crossings(intersections):
    """
    How many pieces each category contributes, whatever the nesting.

    Walked rather than indexed: how deep the lists go is the library's
    business, and the leaves are what carry an id.

    Counted and not collected, which is the whole reason this exists. A
    category can be in there with nothing in it -- the profiler hands back an
    entry per category it was given, crossed or not, and on Monte Alpi that is
    thirty entries for the twelve units a bundle actually goes through. The
    library's own `polygon_intersections_categories` reads the ids and not the
    lengths, so it cannot be used as the list of what is on the panel.
    """

    counts = {}

    def walk(node):
        if node is None:
            return

        if isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

            return

        name, pieces = getattr(node, "id", None), getattr(node, "arrays", None)

        if name is None or pieces is None:
            return

        counts[str(name)] = counts.get(str(name), 0) + len(list(pieces))

    walk(intersections)

    return counts


def _as_color(value):
    """
    A colour in a shape that can be compared for equality.

    The legend decides whether to rebuild by comparing what it is about to
    show with what it is showing, and a palette read from a QGIS project comes
    back as lists: two equal colours in two lists are equal, but a list cannot
    sit in the tuple the comparison is made on without the whole thing becoming
    unhashable further down the line. Names pass through as they are.
    """

    return tuple(value) if isinstance(value, (list, tuple)) else value


def _attitude_crossings(geoprofiles):
    """How many times the traces' planes meet the profiles."""

    per_profile = geoprofiles.lines_with_attitudes_intersections or []

    return sum(len(traces) for profile in per_profile for traces in profile.values())


def _crossing_style():
    """
    Where a mapped line meets the section, in one colour for all of them.

    The library's default, said out loud rather than left to be inherited: the
    legend beside the panels has to name the colour the panel actually draws,
    and the two now read it from the same place.

    One colour for every line layer and every category in them is the
    library's shape and not a choice made here -- `plot_line_intersections`
    gathers the lot into a single `plot` call. Which is why the legend claims
    the colour once and lists the categories under it as names: a swatch each
    would be a key to a code that does not exist.
    """

    from geogst.plots.parameters import PointPlotParams

    return PointPlotParams()


def _dock_style():
    """
    Fault attitudes at a size a dock can show, in a colour the section shows through.

    The library's defaults were chosen for a figure the width of a page: a
    linewidth of 10 draws a dip tick as a band rather than a line.

    Black at 0.9 had the tick as the most solid thing in the panel, which is
    the wrong way round. A tick is a measurement laid *over* a section, and the
    section -- the topography, and the units the profile is crossing in the
    colours the project gives them -- is what it is being read against; an
    opaque black one covers the very thing that makes it mean something. Yellow
    at half opacity puts that back underneath, and the extra half point of
    width is what keeps the tick from thinning away as the opacity comes off.

    The marker went up with it, from 4 to 6, for the same reason and one more.
    The library shares one colour and one alpha between the marker and the
    segment on purpose, the two being one datum, so the dot lost half its
    weight along with the tick -- and the dot is the part that says *where*,
    the tick only saying at what angle. Size is the only way to give that back
    here; a black edge, which is what `fold_axes` puts round its own yellow, is
    not on offer for the same shared-style reason.

    What remains is that yellow is the faintest hue there is against white, so
    off the profile line a tick reads as a pale mark rather than a firm one;
    and that a unit whose own colour is near yellow will take a tick crossing
    it and give very little back.

    `segment_scale_factor` is a **divisor** -- `create_segment_for_plot` takes
    `profile_length / factor` -- so the number goes up to make the tick
    shorter, not down. The default of 3 puts a tick a third of the section long
    across it; 20 makes it a twentieth, which on a seven-kilometre section is
    360 m: long enough to read an angle off, short enough to sit at the place
    it is reporting.
    """

    from geogst.plots.parameters import LineAttitudePlotParams

    return LineAttitudePlotParams(
        color="yellow",
        width=2.0,
        alpha=0.5,
        markersize=6,
        segment_scale_factor=20.0,
    )


def read_traces(session, spec, parent=None):
    """The trace layer as a source, or None once the refusal has been shown."""

    traces = TraceAttitudeSource(
        spec["path"],
        session.crs,
        layer=spec.get("layer"),
        category_field=spec.get("category_field"),
        dip_dir_field=spec.get("dip_dir_field"),
        dip_field=spec.get("dip_field"),
        is_rhr_strike=spec.get("is_rhr_strike", False),
        bounds=session.bounds,
        half_span=DEFAULT_HALF_SPAN,
    )
    print(f"traces: {traces.summary()}")

    if traces.problem:
        QtWidgets.QMessageBox.critical(
            parent, "Unusable traces", f"{spec['path']}\n\n{traces.problem}"
        )
        return None

    return traces


def build(session, chosen, legend="beside"):
    """The window, on a session somebody else has already opened."""

    traces = None

    if chosen.get("traces"):
        traces = read_traces(session, chosen["traces"])

        if traces is None:
            return None

    window = ProfilesWindow(session, traces=traces, legend=legend)

    # The satellites follow from the map's own `showEvent`, so whichever of
    # these two shows it brings the group up with it.
    if window.restore_geometry():
        window.show()
    else:
        fit_to_screen(window, 1400, 950)

    return window
