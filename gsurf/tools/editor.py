"""
The trace editor: a file in this format, read against the map and written in
its own words.

The other two ways into gstruct are readers. A `.gstruct` opens in the traces
slot like a layer, and the section panel writes a curation *over* one -- a file
of differences, laid on a source it never touches. This is the third thing, and
the one the format was for: the file itself, open, with what it says drawn along
the ground it says it about.

**What a text editor cannot show you.** Precedence in this format is a
computation over several lines at once -- a refusal beats a measurement, a
measurement within reach beats a fit, a fit beats a measurement further off --
so which line is winning at a given metre, and where along the fault the winner
changes, is not readable by looking at them. `attitude_at` answers that at one
place; the band at the top of the panel asks it everywhere. On F0055 of
`merid_faults` it shows a fit holding for 3177 m and a compass reading taking
over for the last 354, with the dip stepping from 31 to 35 where they meet. That
is one line of the file beating another, and neither line says so.

**Three windows.** The map is the tool; the panel and the net are windows beside
it. The panel used to be one frame split down the middle with the map, and a
splitter cannot be dragged across a screen boundary, so the map could not be
made bigger without making the table smaller and neither could be moved -- while
the arrangement this tool is for is a monitor of map with the file open next to
it. The net was a dock and is a window for the same reason read off a different
measurement: a dock is as wide as the map can spare, which was 276 px, and 276
px of widget is a stereonet 268 px across. All three are parented to the map and
the group is `gsurf.windows`, the section tool's; what that costs here is two
shortcuts and a status bar, and `_build_shortcuts` and `say` are those.

**Finding the one to open.** 45 of the 393 faults of `merid_faults` carry a
plane; the other 348 are mapped contacts nobody has read one off yet. So the
first question this tool has to answer is which forty-five, and it answers it
twice over: the table sorts on what is written, and the map draws a trace
carrying something firmly and one carrying nothing faintly. Both go through
`carries`, so the two can never disagree. A row picked in the table then brings
its trace into view -- the one thing a highlight cannot do is say where to look,
and on an AOI-wide framing a 1 km fault is three pixels of orange somewhere.

**What a dot cannot say about itself.** A station dot is drawn on the trace, at
the progressive the anchor gives; the reading was taken wherever somebody stood,
and `off` is how far apart those two are. Over the 23 measurements of
`merid_faults` it runs from 0.0 to 69.7 m -- inside the width of the line at
1:25000 and a visible lie at 1:5000, with nothing to tell the two apart by
looking. Nor does the dot show that the curation has *refused* it: a rejected
measurement stays drawn, because somebody did stand there, and until now the band
in the panel was the only thing that said it does not hold. Resting on the dot
says both, which is why the hover exists and why what it says is text: `off` is a
number and a refusal is a word, and neither is a picture.

**And the one thing text cannot say.** This argument was first made as a reason
not to draw a net at all -- one measurement per station, no population, and a
single pole says less than `145/35` written out. The first half is right and the
conclusion did not follow. A pole is a population's way of drawing a plane; a
*great circle* is a plane, and a striation drawn on it sits somewhere along it,
and where along it is the difference between a fault that moved down its dip and
one that moved along its strike. No pair of numbers shows that, and the net shows
it without arithmetic. So the window beside the map holds both: the plane the
cursor is resting on, and the lineations read on it. Only those -- the pole was
drawn there at first and is not, because with one plane on the net the markers
inside the circle are the striae, and a pole is a mark inside it that is not one.

There are none. `merid_faults` has zero `lineation` records in it, and three of
its 23 attitudes mention striae in an Italian note -- `lineazione N080°` at S20,
`lineazione N075°` at S19, and at S4 two pitches whose values are on a paper
sheet. A trend alone would be enough, a striation lying in the plane it was read
on, so the plunge follows: 18.3 deg and 30.7 deg for those two. Turning prose
into a record is curation, and it is not something to do from inside a drawing
routine -- so the net asks `Structure.lineations` and currently draws nothing,
and the day somebody writes one it appears.

**What it does not rewrite.** Saving replaces the lines of the structure that
was edited and leaves every other byte alone. Not fastidiousness -- measured:
`curation.gstruct` through a load and a dump comes back without the ten lines of
comment in it, because comments are not in the model and a writer can only write
what it has, and those ten lines are the argument for why five thrusts are
`exposed`. A Save that deletes the geologist's reasoning is not a Save. See
`curation.Document`, which is where the splice lives and why.

**Two coordinate systems, and one place each crosses.** The model stays in the
file's own projection, because the model is the file: the progressives
`attitude_at` reasons over are metres on the ruler the file was written with.
The map is in the session's. So the paths are transformed on the way to being
drawn, and a picked anchor is transformed back on the way to being written, and
those are the only two crossings there are.

**Editing is textual, which is the format's own shape.** A correction here is a
line added, never a line changed: the last span covering a progressive wins, so
the way to say "not on this stretch after all" is to say it, under what was said
before. A panel of widgets would have had to invent an order of operations the
format already has. What the box is given is the block exactly as it stands in
the file, path and all -- median six vertices over the 393 faults, longest 35,
so there is nothing worth hiding and nothing hidden.
"""

from __future__ import annotations

import textwrap
from collections import Counter

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PyQt6 import QtCore, QtGui, QtWidgets

from gsurf.curation import (
    DEFAULT_MAX_GAP,
    PROVENANCE,
    SUFFIX,
    UNCONSTRAINED,
    Document,
    degrees_not_metres,
    is_gstruct,
    nearest_structure,
    place_on,
    point_on,
    provenance_of,
    stretch,
)
from gsurf.mapview import LegendControls, MapView, fit_to_screen
from gsurf.stereonet import StereonetView
from gsurf.windows import SatelliteWindow, WindowGroup

# The file is the subject, so it is the one thing this cannot run without. The
# DEM is optional and the tool is usable without it -- the traces are geometry
# and draw at any scale -- but a fault with no topography under it is a line on
# a white field, and where it runs is half of why it was drawn there.
WANTS = dict(
    traces="required",
    dem="optional",
    polygons="optional",
    lines="optional",
)

# And the slot takes one kind of file, which no other tool's does. A mapped
# layer in `traces` is a perfectly good answer to the sections and no answer at
# all here: there is nothing in a layer to edit, and every part of this window
# reads text. Said to the dialog rather than only at the door, so the choice is
# never offered -- `build` still refuses, for the ways in that do not pass
# through a dialog.
ONLY = dict(traces=SUFFIX)

# The five answers `attitude_at` can give, as colours. Measured and computed are
# different hues rather than different shades, because the distinction between
# them is the one the format exists to keep; far-off measurement is the pale
# version of measurement, because that is what it is. Refused is the only red on
# the panel, and absent is the colour of nothing.
PROVENANCE_TINT = {
    "misurata": "#1b7837",
    "misurata-lontana": "#a6dba0",
    "fit": "#2166ac",
    "rifiutata": "#b2182b",
    "assente": "#e4e4e4",
}

# How the `use` axis paints, it being the one axis that decides rather than
# describes. The others are grey: they are notes about the contact, and colouring
# them would put them in competition with the band above for the eye.
USE_TINT = {"rejected": "#b2182b", "accepted": "#1b7837", "unknown": "#9a9a9a"}
SPAN_TINT = "#6a7f95"

# What the provenance band is sampled at. The band's edges are therefore good to
# one four-hundredth of the trace -- nine metres on a 3.5 km fault -- which is a
# picture of where the answer changes and not a measurement of it. The number
# where it changes exactly is in the file, as the anchor somebody wrote.
SAMPLES = 400

# And what the table's one-word summary of the same computation is sampled at.
# Far coarser, because it is 393 sweeps and not one: at this rate the whole
# column is 39 ms, against 250 at the band's own rate. What it buys is a
# fraction good to a percent and a half, which is the precision of a word.
HOLDS_SAMPLES = 64

# How a trace is drawn, by whether anything has been read off it. The difference
# is weight and darkness rather than hue, because every hue on this map already
# means something -- orange is the selection, red a refusal, green a measurement
# -- and a trace carrying only a compass reading is not a fit and must not be
# painted as one. It also puts the 348 that carry nothing where they belong:
# still on the map, no longer competing for the eye.
CARRYING_STYLE = dict(colors="#1a1a1a", linewidths=1.5)
BARE_STYLE = dict(colors="#8a8a8a", linewidths=0.7)

# Room around a trace a made framing leaves, and the smallest window it will
# make. Both are in the docstring of `framing_for`, which is where the numbers
# they are measured against are.
FRAME_MARGIN = 0.25
FRAME_MIN_SPAN_M = 1200.0

# How long the framing waits before it moves. Arrow-keying down the table is one
# selection per keystroke, and each framing is a full redraw plus a hillshade
# reread -- tens to hundreds of milliseconds, paid for a view nobody looked at.
# Deferred like this a run of them costs one move, which is the same bargain
# `MapView.schedule_shade_refresh` strikes and for the same reason.
FRAME_DELAY_MS = 140

# How near the cursor has to come, in screen pixels, for a click to have been
# aimed at a trace. In pixels and not in metres so that it means the same thing
# at every zoom.
PICK_RADIUS_PX = 14

# And for the cursor to be resting on a station dot rather than near one. Tied to
# the marker, which is 5 points across: a threshold smaller than what is drawn
# would ask for an aim the dot does not reward, and a much larger one would claim
# ground where there is visibly nothing. Separate from `PICK_RADIUS_PX` because
# the two aim at different shapes -- a click is aimed at a line, which is long,
# and this at a dot, which is neither long nor wide.
HOVER_RADIUS_PX = 10

# Where a note in a tooltip is folded. Notes here run to 65 characters and a
# tooltip does not wrap plain text by itself, so a file with a paragraph in one
# would open a tooltip wider than the map it is covering.
TIP_WRAP = 64

# How near, along the trace, an unlabelled lineation has to be anchored to count
# as read at the same place as a plane. Anchors in these files are written to a
# hundredth of a metre, so this is not a tolerance for rounding: it is the width
# of one outcrop as somebody standing on it would place two readings. A lineation
# that carries `station=` is matched by that instead, and is not subject to this.
SAME_OUTCROP_M = 2.0

# What the net's window is called, and what it says with nothing on it. A title
# and not a label inside the figure, because an empty equal-area net is a circle
# with a grid in it and reads as a widget that has not loaded rather than as one
# waiting to be pointed at.
NET_TITLE = "gSurf - stereonet"
NET_EMPTY_TITLE = f"{NET_TITLE} - rest on a green dot"

# How big the net opens. Its own window, so the map is no longer paying for it,
# and that is the whole of the difference: as a dock it had to stop at 276 px,
# because past that it came off the map's axes, and 276 px of widget is a
# primitive circle 268 px across. 420 px is 412. The circle follows the widget
# from there -- the figure size says where it starts, the window says where it
# stays, and unlike a floating dock's, that is remembered.
NET_FIGSIZE_IN = 3.6
NET_WINDOW_PX = (420, 440)

MAX_GAP_RANGE = (0.0, 20000.0)

# The three lines a curation is made of, with `*` where an anchor goes. They are
# not a form and do not save any typing worth counting: what they do is put the
# cursor somewhere known. An anchor is picked on the map and written at the
# cursor, so without a place to put it the first shift-click of a session lands
# wherever the box was last left -- which is the top of the block, where it
# reads as a word glued to `structure`.
#
# `use` and not one of the other two axes, because it is the one that decides:
# `certainty` and `exposure` describe the contact, and this is the tool for
# saying what holds.
TEMPLATES = (
    ("+ span", '  span use * * rejected reason=""'),
    ("+ attitude", "  attitude * plane 000/00 station= src=field"),
    ("+ fit", "  fit plane * * 000/00 from="),
)

PANEL_WIDTH_PX = 520

# What the panel comes up as, the first time and nothing being remembered. Taller
# than the 880 the whole tool used to ask for, because it is no longer sharing a
# frame with the map: the table wants rows and the band and the box want height,
# and both of those used to be cut to whatever was left beside a canvas.
PANEL_WINDOW_PX = (PANEL_WIDTH_PX, 940)

# What the map comes up as where the two do not fit side by side, and the least
# map worth putting a panel next to. Below that floor the map would be narrower
# than the panel, which is the wrong way round for a tool whose subject is where
# a fault runs, so the map takes the screen and the panel comes up over it.
MAP_WINDOW_PX = (1080, 880)
MAP_FLOOR_PX = 700

# What is left between them, and what is left for the map window's own frame.
# The gap is a gap; the allowance is a guess and has to be -- right after
# `setGeometry` the title bar does not exist yet, the window manager not having
# reparented the window, so its thickness cannot be measured before the window
# has to be placed. 40 px is the same allowance `WindowGroup.place_unremembered`
# already makes, and erring high costs a strip of desktop rather than the bottom
# of a window.
TILE_GAP_PX = 6
FRAME_ALLOWANCE_PX = 40

# What this tool's windows are kept under. `sections` is the other one.
SETTINGS_NAME = "editor"

# How much of the panel the table opens with, and the least it can be dragged
# to. The opening size is about eight rows, which is enough of a list to sort and
# read; the floor is four, which is enough to see that sorting did something.
TABLE_OPENING_PX = 240
TABLE_FLOOR_PX = 140

# Above this fraction of the trace, a bar has room for its own word in it.
LABEL_FRACTION = 0.14


def carries(structure):
    """
    Whether anything has been read off this trace: a measurement, or a fit.

    The one predicate the table's filter, the table's sorting and the map's two
    weights all go through, so that what the list calls carrying and what the map
    draws firmly are the same forty-five faults. A span is not counted: every one
    of the 393 has one, most of them saying `certainty` or `exposure`, and none
    of that is a plane.
    """

    return bool(structure.attitudes or structure.fits)


def _as_number(written):
    """
    A length an attribute claims to be, or nothing.

    Attributes are text and the format does not type them, so a file is free to
    put a word, an empty string or a range where a distance goes -- and a tooltip
    is the last place that should be the thing which raises.
    """

    try:
        return float(written)
    except (TypeError, ValueError):
        return None


def holds_along(structure, max_gap=DEFAULT_MAX_GAP, samples=HOLDS_SAMPLES):
    """
    The provenance covering most of the trace, as `(kind, fraction of it)`.

    `provenance_of` reduced to one word, which is what fits in a cell. It is a
    summary and reads as one: a fault measured over its first three hundred
    metres and fitted over the remaining three thousand says `fit`, because that
    is what most of it is. Where the answer changes is the band, and this column
    is how you find the trace worth opening the band on.

    `(None, 0.0)` where there is no ruler to sample along -- a structure with one
    vertex, or none.
    """

    sampled = provenance_of(structure, samples=samples, max_gap=max_gap)

    if not sampled:
        return None, 0.0

    kind, count = Counter(kind for _, _, _, kind in sampled).most_common(1)[0]

    return kind, count / len(sampled)


def framing_for(path, margin=FRAME_MARGIN, floor=FRAME_MIN_SPAN_M):
    """
    A view with one trace across the middle of it, as `extent` is ordered.

    The margin comes off the longer side and not off each one: a fault running
    due north has no width to take a fraction of, and a fraction of nothing is
    nothing. That is the reasoning `sections.framing_for` is built on; this is
    the same thing for a polyline rather than a two-point trace, and the floor is
    what the polyline adds to it.

    The floor is there because these lengths run over three orders of magnitude.
    The shortest fault in `merid_faults` is 12 m against a median of 1048 and a
    longest of 29641, and a window fitted to that short one is a screenful of
    hillshade with nothing in it to say where you are standing.
    """

    if not path:
        return None

    xs = [x for x, _ in path]
    ys = [y for _, y in path]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)

    span = max(x1 - x0, y1 - y0)
    room = max(margin * span, (floor - span) / 2.0, 0.0)

    return [x0 - room, x1 + room, y0 - room, y1 + room]


def runs_of(sampled):
    """
    The sampled provenance as contiguous `(s0, s1, kind, said)` stretches.

    One rectangle per stretch rather than one per sample: four hundred patches
    would draw the same picture and take four hundred times as long to, and the
    run is also the honest unit -- what is being shown is where the answer
    changes, and between two changes there is one answer.
    """

    runs = []

    for s, _, said, kind in sampled:
        if runs and runs[-1][2] == kind:
            runs[-1][1] = s
            continue

        if runs:
            runs[-1][1] = s

        runs.append([s, s, kind, said])

    return [tuple(run) for run in runs]


class StructureTable(QtWidgets.QTableWidget):
    """
    Every structure in the file as a row: what is written on it, and what holds.

    This replaces a combo box that could be typed into, and that was all it could
    do. What was missing is the two things a list is for. Sorting: the faults
    carrying a plane are 45 of 393, and the way to them is a click on the `att`
    header, not a scroll. And a selection the map shares -- picking a row here and
    clicking a trace there are now one gesture, reported twice.

    Rows are never removed, only hidden, which is why the filter can no longer
    take away the structure being worked on. The combo was rebuilt on every
    filter change and had to notice when what was open had dropped out of it; a
    hidden row is still the open row, and there is nothing to notice.

    The sorting is what makes a row's position worth nothing, so the structure's
    index in the document travels in the ident cell rather than being the row
    number. `index_of` and `_row_of` are the two directions of that, and every
    write goes through `_hold_open` because a sort moves the current row out from
    under whatever the panel has on screen.
    """

    COLUMNS = ("ident", "m", "att", "fit", "span", "holds")

    chosen = QtCore.pyqtSignal(int)
    framing_asked = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(0, len(self.COLUMNS), parent)

        self._filling = False
        self._open = None

        self.setHorizontalHeaderLabels(self.COLUMNS)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.setSortingEnabled(True)
        self.setToolTip(
            "Click a header to sort: `att` and `fit` are what has been read off "
            "the trace, `span` what has been said about it, `holds` the class "
            "covering most of it. Double-click a row to bring its trace into "
            "view."
        )

        # `Interactive` and widths fitted once per fill, rather than
        # `ResizeToContents` -- which is what the section panel's table uses and
        # what this one used first. Measured on the 393x6 cells of `merid_faults`,
        # inside this window rather than on a table by itself: rewriting them
        # costs 7.9 ms with this mode and 25.7 *seconds* with the header fitting
        # to contents, because that mode reflows every column on every cell
        # written. A table on its own does not show it -- with no laid-out
        # viewport the reflow never runs -- which is why the number above is from
        # the real window.
        header = self.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)

        # Said rather than left to Qt, which starts a header's indicator on
        # section 0 *descending* -- measured, not assumed. Enabling the sorting
        # then sorts by that, and a table filled in file order came out reversed:
        # F0395 on the first row, F0000 on the last, and nothing on screen saying
        # so. Ident ascending is the order the file is written in for anything the
        # import produced, which is what a reader opening it expects to see.
        header.setSortIndicator(0, QtCore.Qt.SortOrder.AscendingOrder)

        self.currentCellChanged.connect(self._on_current)
        self.doubleClicked.connect(
            lambda index: self.framing_asked.emit(self.index_of(index.row()))
        )

    # -- rows and indices --------------------------------------------------

    def index_of(self, row):
        """Which structure the row stands for, or None off the end."""

        item = self.item(row, 0)

        return None if item is None else item.data(QtCore.Qt.ItemDataRole.UserRole)

    def _row_of(self, index):
        """
        Which row a structure is on now, or -1.

        A scan and not a dictionary: 393 rows is microseconds, and a dictionary
        would have to be rebuilt on every sort by somebody who remembered to.
        """

        for row in range(self.rowCount()):
            if self.index_of(row) == index:
                return row

        return -1

    def point_at(self, index):
        """
        Puts the current row on a structure without saying anybody asked.

        The counterpart of the map holding its highlight: when the panel refuses
        to leave a block that was typed and not applied, this is what puts the
        table back on the one still on screen.
        """

        self._open = index
        row = self._row_of(index)

        if row < 0 or row == self.currentRow():
            return

        self._filling = True
        self.setCurrentCell(row, 0)
        self._filling = False

        self.scrollToItem(
            self.item(row, 0), QtWidgets.QAbstractItemView.ScrollHint.EnsureVisible
        )

    def _on_current(self, row, _column, previous, _previous_column):
        if self._filling or row < 0 or row == previous:
            return

        index = self.index_of(row)

        if index is not None:
            self.chosen.emit(index)

    # -- filling -----------------------------------------------------------

    def fill(self, structures, max_gap=DEFAULT_MAX_GAP):
        """Every structure as a row, in the order the file has them."""

        with self._hold_open():
            self.setRowCount(len(structures))

            for index, structure in enumerate(structures):
                self._write_row(index, index, structure, max_gap)

        # Once, here, and not on the incremental writers: a column that resized
        # every time a row was rewritten would shift under the hand, and the
        # widths a whole file fits in are the right ones for one row of it.
        self.resizeColumnsToContents()

    def update_row(self, index, structure, max_gap=DEFAULT_MAX_GAP):
        """One row again, for a block that has just been applied."""

        row = self._row_of(index)

        if row < 0:
            return

        with self._hold_open():
            self._write_row(row, index, structure, max_gap)

    def refresh_holds(self, structures, max_gap=DEFAULT_MAX_GAP):
        """
        The `holds` column again, for a reach that has just changed.

        Only that column: the reach decides what holds and says nothing about
        what is written, so the tallies beside it are not in question.
        """

        with self._hold_open():
            for row in range(self.rowCount()):
                index = self.index_of(row)

                if index is not None:
                    self._holds(row, structures[index], max_gap)

    def set_filter(self, wanted, carrying_only, structures):
        """Hides the rows that do not match, and says how many are left."""

        wanted = (wanted or "").strip().lower()
        shown = 0

        for row in range(self.rowCount()):
            index = self.index_of(row)
            structure = None if index is None else structures[index]

            hidden = structure is None or (
                (carrying_only and not carries(structure))
                or bool(wanted and wanted not in (structure.ident or "").lower())
            )

            self.setRowHidden(row, hidden)
            shown += not hidden

        return shown

    # -- one row -----------------------------------------------------------

    def _hold_open(self):
        """
        Writes cells with the sorting off, and puts the current row back after.

        Both halves are needed. Writing into a sorted table reorders it as you
        write, so a fill would be laying rows down on ground that moves; and
        turning the sorting back on sorts, which moves the current row out from
        under whatever the panel has on screen -- the table would end up pointing
        at a different structure from the one whose text is in the box.
        """

        table = self

        class Held:
            def __enter__(self):
                table._filling = True
                table.setSortingEnabled(False)

            def __exit__(self, *_):
                table.setSortingEnabled(True)
                table._filling = False

                if table._open is not None:
                    table.point_at(table._open)

        return Held()

    def _write_row(self, row, index, structure, max_gap):
        ident = QtWidgets.QTableWidgetItem(structure.ident or "")
        ident.setData(QtCore.Qt.ItemDataRole.UserRole, index)
        self.setItem(row, 0, ident)

        self._number(row, 1, round(structure.length))
        self._number(row, 2, len(structure.attitudes))
        self._number(row, 3, len(structure.fits))
        self._number(row, 4, len(structure.spans))
        self._holds(row, structure, max_gap)

    def _number(self, row, column, value):
        """
        A count or a length, sorted as the number it is.

        Set as text it would sort as text, and a column reading 9, 84, 1048 would
        come out 1048, 84, 9 -- which is the order that hides the long faults at
        the top of the very sort meant to find them.
        """

        item = QtWidgets.QTableWidgetItem()
        item.setData(QtCore.Qt.ItemDataRole.DisplayRole, int(value))
        item.setTextAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )

        self.setItem(row, column, item)

    def _holds(self, row, structure, max_gap):
        """
        What holds over most of the trace, tinted as the band tints it.

        The same colours and the same white-on-dark rule as `ProvenanceView._band`,
        so that the column and the picture under it are read as the one thing they
        are, and a row of `assente` grey is a trace with nothing on it at a glance.
        """

        kind, fraction = holds_along(structure, max_gap)
        item = QtWidgets.QTableWidgetItem(
            "" if kind is None else f"{kind} {fraction:.0%}"
        )

        if kind is not None:
            item.setBackground(QtGui.QColor(PROVENANCE_TINT.get(kind, "#cccccc")))
            item.setForeground(
                QtGui.QColor("#6a6a6a" if kind == "assente" else "white")
            )

        self.setItem(row, 5, item)


class ProvenanceView(QtWidgets.QWidget):
    """
    One structure along its own trace: what holds, what says so, and what it is.

    Three rows over a shared axis of metres. The band is `attitude_at` swept end
    to end. The lanes under it are the lines of the file that were competing to
    answer -- the measurements as ticks where they were taken, the fits as the
    intervals they were computed over, the axes as the stretches they cover. The
    bottom is the plane that won, which is what makes a precedence boundary
    visible as the thing it is: a step in the dip.

    The gap between the band and the lanes is where the reading is. A fit drawn
    in the lanes with `assente` over it in the band is a fit its own verdict
    threw out -- a straight trace does not constrain a dip -- and that is a
    common and confusing state to be in with nothing but the text in front of
    you.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.figure = Figure(figsize=(6, 3.8), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(230)

        # In a label and not in the figure: five entries would take a row of
        # the plot to explain what a row of coloured text explains beside it,
        # and the figure has three axes to fit already.
        key = QtWidgets.QLabel(
            " ".join(
                f'<span style="background:{PROVENANCE_TINT[kind]}; color:{PROVENANCE_TINT[kind]}">'
                f"&nbsp;&nbsp;&nbsp;</span>&nbsp;{kind}"
                for kind in PROVENANCE
            )
        )
        key.setTextFormat(QtCore.Qt.TextFormat.RichText)
        key.setStyleSheet("font-size: 10px;")
        key.setWordWrap(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(key)

    def show_structure(self, structure, max_gap=DEFAULT_MAX_GAP):
        """Redraws for one structure, or says why there is nothing to draw."""

        self.figure.clear()

        if structure is None or len(structure.path) < 2 or structure.length <= 0.0:
            self._nothing(
                "no path, and so no ruler to lay anything along"
                if structure is not None else "nothing selected"
            )
            return

        sampled = provenance_of(structure, samples=SAMPLES, max_gap=max_gap)

        if not sampled:
            self._nothing("nothing to sample along")
            return

        band, lanes, angles = self.figure.subplots(
            3, 1, sharex=True, height_ratios=[0.5, 2.0, 1.5]
        )

        length = structure.length

        self._band(band, runs_of(sampled), length)
        self._lanes(lanes, structure, length)
        self._angles(angles, sampled)

        angles.set_xlim(0.0, length)
        angles.set_xlabel("m along the trace")

        self.canvas.draw_idle()

    def _nothing(self, why):
        axes = self.figure.subplots()
        axes.text(0.5, 0.5, why, ha="center", va="center", color="#8a8a8a", fontsize=9)
        axes.set_axis_off()

        self.canvas.draw_idle()

    def _band(self, axes, runs, length):
        for s0, s1, kind, said in runs:
            axes.axvspan(s0, s1, color=PROVENANCE_TINT.get(kind, "#cccccc"), lw=0)

            if s1 - s0 > LABEL_FRACTION * length:
                # The detail and not the class: the class is the colour, and
                # `misurata:S26@245m` is the sentence -- which station, and how
                # far away it was when it won.
                axes.text(
                    (s0 + s1) / 2.0,
                    0.5,
                    said,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if kind != "assente" else "#6a6a6a",
                )

        axes.set_yticks([])
        axes.set_ylabel("holds", fontsize=8)

    def _lanes(self, axes, structure, length):
        axes.set_ylabel("written", fontsize=8)

        # The axes the file actually uses on this structure, in the order they
        # were written. Showing the three the vocabulary defines whether or not
        # they are there would be three empty rows saying nothing.
        axis_rows = []

        for span in structure.spans:
            if span.axis not in axis_rows:
                axis_rows.append(span.axis)

        rows = ["attitudes", "fits"] + axis_rows
        at = {name: -n for n, name in enumerate(rows)}

        for attitude in structure.attitudes:
            if attitude.s is None:
                continue

            axes.vlines(
                attitude.s, at["attitudes"] - 0.3, at["attitudes"] + 0.3,
                color="#1b7837", lw=1.8,
            )
            axes.text(
                attitude.s, at["attitudes"] + 0.36,
                " ".join(
                    part for part in (
                        attitude.attrs.get("station", ""),
                        "" if attitude.plane is None else str(attitude.plane),
                    ) if part
                ),
                ha="center", va="bottom", fontsize=7, color="#1b7837",
            )

        for fit in structure.fits:
            s0 = 0.0 if fit.s0 is None else fit.s0
            s1 = length if fit.s1 is None else fit.s1

            # The same test `attitude_at` makes, and the reason a fit can be
            # drawn here under an `assente` band: a plane read off a straight
            # trace is a number the arithmetic produced, not a measurement.
            counted = (
                fit.plane is not None
                and fit.attrs.get("verdict") != UNCONSTRAINED
            )

            axes.barh(
                at["fits"], max(s1 - s0, length * 0.002), left=s0, height=0.6,
                color=PROVENANCE_TINT["fit"] if counted else "#d6d6d6",
                edgecolor="#8a8a8a" if not counted else "none", lw=0.6,
            )

            said = fit.attrs.get("verdict") or fit.attrs.get("from") or "fit"

            if s1 - s0 > LABEL_FRACTION * length:
                axes.text(
                    (s0 + s1) / 2.0, at["fits"],
                    f"{fit.plane} {said}" if fit.plane is not None else said,
                    ha="center", va="center", fontsize=7,
                    color="white" if counted else "#4a4a4a",
                )

        stacked = {}

        for span in structure.spans:
            if span.s0 is None or span.axis not in at:
                continue

            # Later spans are drawn narrower and in front of earlier ones,
            # because that is what the rule does: the last span covering a
            # progressive is the one in force, and a correction in this format
            # is written by adding a line over the one being corrected. Drawn
            # flat they would be one bar with two words printed on top of each
            # other -- which is how this read before, on the two `exposure`
            # lines of F0055.
            layer = stacked.get(span.axis, 0)
            stacked[span.axis] = layer + 1

            middle = (span.s0 + span.s1) / 2.0
            tint = USE_TINT.get(span.value, SPAN_TINT) if span.axis == "use" else SPAN_TINT

            # Whether this is the line in force where it is widest. A proxy, and
            # said as one: a span can hold over part of itself and be shadowed
            # over the rest, and the picture of that is the stacking rather than
            # the tint. What the tint is for is the common case -- two lines
            # over one stretch, one of them answering.
            answering = structure.span_at(span.axis, middle) is span

            axes.barh(
                at[span.axis], max(span.s1 - span.s0, length * 0.002),
                left=span.s0, height=max(0.6 - 0.17 * layer, 0.16),
                color=tint, alpha=1.0 if answering else 0.35, zorder=3 + layer,
            )

            # Only the one that is winning. The shadowed line stays drawn,
            # because it is still in the file and taking it off would hide why
            # the winner is the winner -- but it is not what holds, and putting
            # its word on the chart would say that it was.
            if answering and span.s1 - span.s0 > LABEL_FRACTION * length:
                axes.text(
                    middle, at[span.axis], span.value, ha="center", va="center",
                    fontsize=7, color="white", zorder=12,
                )

        axes.set_ylim(-len(rows) + 0.4, 0.75)
        axes.set_yticks([at[name] for name in rows])
        axes.set_yticklabels(rows, fontsize=8)
        axes.tick_params(axis="y", length=0)

        for side in ("top", "right", "left"):
            axes.spines[side].set_visible(False)

    def _angles(self, axes, sampled):
        progressives = [s for s, _, _, _ in sampled]
        dips = [np.nan if plane is None else plane.dip for _, plane, _, _ in sampled]
        azimuths = [
            np.nan if plane is None else plane.dip_dir for _, plane, _, _ in sampled
        ]

        axes.step(progressives, dips, where="post", color="#333333", lw=1.4)
        axes.set_ylim(0, 90)
        axes.set_yticks([0, 30, 60, 90])
        axes.set_ylabel("dip", fontsize=8, color="#333333")

        # Two scales because they are two different measurements of one plane,
        # and drawing a dip direction on an axis that stops at 90 would fold
        # three quarters of the compass onto the top of the frame.
        twin = axes.twinx()
        twin.step(progressives, azimuths, where="post", color="#d95f02", lw=1.1, ls="--")
        twin.set_ylim(0, 360)
        twin.set_yticks([0, 90, 180, 270, 360])
        twin.set_ylabel("dip dir", fontsize=8, color="#d95f02")

        for one in (axes, twin):
            one.tick_params(labelsize=7)


class EditorPanel(QtWidgets.QWidget):
    """
    The structure being worked on: which one, what holds along it, and its lines.

    The buttons divide the way the file does. `Apply` settles a block -- it goes
    through the parser, so a block that would not read is refused with the
    parser's own words and the text stays on screen to be fixed. `Save` puts the
    document on disk. Nothing is written to disk until it is asked for, and
    nothing is applied to the model until it parses.
    """

    selected = QtCore.pyqtSignal(int)
    applied = QtCore.pyqtSignal(int)

    # Asked of the window, which is the only thing here that knows what a view
    # is. Separate from `selected` because the two are not the same question: a
    # click on the map selects without moving the view, and it would be a poor
    # map that jumped to the thing you had just pointed at.
    framing_asked = QtCore.pyqtSignal(int)

    def __init__(self, document, parent=None):
        super().__init__(parent)

        self.document = document
        self.index = None

        self.table = StructureTable()
        self.table.setMinimumHeight(TABLE_FLOOR_PX)
        self.table.chosen.connect(self._chosen)
        self.table.framing_asked.connect(self._asked_framing)

        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText("find an ident")
        self.filter.setClearButtonEnabled(True)
        self.filter.setToolTip(
            "Narrows the table to the idents containing what is typed. It hides "
            "rows rather than removing them, so the structure being worked on "
            "stays the one being worked on."
        )
        self.filter.textChanged.connect(lambda _: self._apply_filter())

        self.carrying = QtWidgets.QCheckBox("only the ones carrying a plane")
        self.carrying.setToolTip(
            "45 of the 393 faults of merid_faults carry an attitude or a fit. "
            "The rest are mapped contacts nobody has read a plane off yet, and "
            "they are still editable -- this only shortens the list."
        )
        self.carrying.toggled.connect(lambda _: self._apply_filter())

        self.shown = QtWidgets.QLabel()
        self.shown.setStyleSheet("color: #6a6a6a; font-size: 11px;")

        self.frame_wanted = QtWidgets.QCheckBox("frame it")
        self.frame_wanted.setChecked(True)
        self.frame_wanted.setToolTip(
            "Bring the trace into view when it is picked from the table. A click "
            "on the map never moves the view -- you are already looking at what "
            "you clicked. A double-click on a row frames it whatever this says."
        )

        self.gap_spin = QtWidgets.QDoubleSpinBox()
        self.gap_spin.setRange(*MAX_GAP_RANGE)
        self.gap_spin.setSingleStep(50.0)
        self.gap_spin.setValue(DEFAULT_MAX_GAP)
        self.gap_spin.setSuffix(" m")
        self.gap_spin.setToolTip(
            "How far from where it was taken a measurement still answers. The "
            "same judgement the section panel's reach makes, from the other "
            "end: past this the fit under it takes over."
        )
        self.gap_spin.valueChanged.connect(lambda _: self._reach_changed())

        self.view = ProvenanceView()

        self.text = QtWidgets.QPlainTextEdit()
        self.text.setFont(QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.SystemFont.FixedFont
        ))
        self.text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.text.setTabStopDistance(28)
        self.text.setMinimumHeight(120)

        self.problem = QtWidgets.QLabel()
        self.problem.setWordWrap(True)
        self.problem.setStyleSheet("color: #b2182b; font-size: 11px;")
        self.problem.setVisible(False)

        self.apply_button = QtWidgets.QPushButton("Apply")

        # No shortcut of its own: the window owns both of this tool's shortcuts
        # and hands them to every window of the group, which is what makes them
        # work from the map as well. A second Ctrl+Return here would be an
        # ambiguous one -- see `EditorWindow._build_shortcuts`.
        self.apply_button.setToolTip(
            "Read the block back through the parser and put it in the document "
            "(Ctrl+Return). Nothing reaches the file until Save."
        )
        self.apply_button.clicked.connect(self.apply_block)

        self.revert_button = QtWidgets.QPushButton("Revert")
        self.revert_button.setToolTip("Put back the block as the document has it.")
        self.revert_button.clicked.connect(self._redraw)

        buttons = QtWidgets.QHBoxLayout()

        for label, template in TEMPLATES:
            adder = QtWidgets.QPushButton(label)
            adder.setToolTip(
                f"Write `{template.strip()}` above the path, with the first "
                f"anchor selected: shift-click the map to fill it in, and the "
                f"next one is selected in turn."
            )
            adder.clicked.connect(lambda _, line=template: self.add_line(line))
            buttons.addWidget(adder)

        buttons.addStretch(1)
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.revert_button)

        heading = QtWidgets.QHBoxLayout()
        heading.addWidget(self.filter, stretch=1)
        heading.addWidget(self.carrying)
        heading.addWidget(self.frame_wanted)

        gap = QtWidgets.QHBoxLayout()
        gap.addWidget(QtWidgets.QLabel("a measurement answers for"))
        gap.addWidget(self.gap_spin)
        gap.addStretch(1)
        gap.addWidget(self.shown)

        finding = QtWidgets.QWidget()
        finding_layout = QtWidgets.QVBoxLayout(finding)
        finding_layout.setContentsMargins(0, 0, 0, 0)
        finding_layout.setSpacing(4)
        finding_layout.addLayout(heading)
        finding_layout.addWidget(self.table, stretch=1)

        working = QtWidgets.QWidget()
        working_layout = QtWidgets.QVBoxLayout(working)
        working_layout.setContentsMargins(0, 0, 0, 0)
        working_layout.addLayout(gap)
        working_layout.addWidget(self.view, stretch=3)
        working_layout.addWidget(self.text, stretch=2)
        working_layout.addWidget(self.problem)
        working_layout.addLayout(buttons)

        # In a splitter because the two halves are wanted at different times and
        # the panel is not tall enough for both at their best: hunting for the
        # trace worth opening wants rows, and working on the one that is open
        # wants the band and the box. Neither is allowed to collapse to nothing --
        # a table dragged shut has no way back that looks like one.
        #
        # Opened at sizes rather than at stretch factors, which is what this had
        # first: a stretch factor divides what is left over after the size hints
        # are met, and a table's hint asks for almost nothing -- the panel came up
        # with 70 pixels of table in it, which is one row, which is a list you
        # cannot read to find anything.
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        split.addWidget(finding)
        split.addWidget(working)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        split.setChildrenCollapsible(False)
        split.setSizes([TABLE_OPENING_PX, 3 * TABLE_OPENING_PX])

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(split)

        self.table.fill(self.document.dataset.structures, self.max_gap)
        self._apply_filter()

        if self.document.dataset.structures:
            # Pointed at rather than chosen: the window emits nothing yet -- it
            # has not connected to this panel -- and it opens the first structure
            # itself once it has a map to draw it on.
            self.show_index(0)

    # -- the list ----------------------------------------------------------

    def _apply_filter(self):
        """
        Narrows the table, and says how much of the file is on show.

        The count is the point of the label: `45 of 393` is the sentence the
        `carrying` box exists to produce, and without it the box looks like it
        has thrown most of the file away.
        """

        structures = self.document.dataset.structures
        shown = self.table.set_filter(
            self.filter.text(), self.carrying.isChecked(), structures
        )

        self.shown.setText(f"{shown} of {len(structures)}")

    def _chosen(self, index):
        """A row picked in the table, which the panel may refuse to leave."""

        if not self.show_index(index):
            self.table.point_at(self.index)
            return

        self.selected.emit(index)

        if self.frame_wanted.isChecked():
            self.framing_asked.emit(index)

    def _asked_framing(self, index):
        """
        A double-click, which frames whatever the box says.

        Only on the row that is already open, which after a double-click is
        always this one: the first of the two clicks has been through `_chosen`
        already. A refusal there leaves `index` where it was, and this then does
        nothing -- which is right, because the trace worth looking at is still
        the one whose text is in the box.
        """

        if index == self.index:
            self.framing_asked.emit(index)

    def _reach_changed(self):
        """
        The dial: the picture, and the one column that reads from the same number.

        Not the tallies beside it. How far a measurement answers for decides what
        holds and says nothing about what is written, and a dial that rewrote the
        counts would be claiming otherwise.
        """

        self._redraw_view()
        self.table.refresh_holds(self.document.dataset.structures, self.max_gap)

    # -- one structure -----------------------------------------------------

    def show_index(self, index):
        """
        Puts a structure on screen, whoever asked. False if the old one was kept.

        Selecting is the one gesture that would throw away work without being
        asked to: a block typed and not applied lives only in the box, and a
        click on another trace would have redrawn over it. So leaving is a
        question, and the caller has to be able to hear no -- the map holds its
        highlight where it was, and the table goes back to the row it was on.
        """

        if index == self.index:
            return True

        if not self._may_leave():
            return False

        self.index = index

        self.table.point_at(index)
        self._redraw()

        return True

    def _may_leave(self):
        """Asks before a block that was typed and never applied is redrawn over."""

        if self.index is None or self.text.toPlainText() == self.document.text_of(
            self.index
        ):
            return True

        answer = QtWidgets.QMessageBox.question(
            self,
            "Not applied",
            f"{self.document.dataset.structures[self.index].ident} has been "
            f"edited in the box and not applied.\n\nApply reads it into the "
            f"document; moving on without it leaves it behind.",
            QtWidgets.QMessageBox.StandardButton.Apply
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel,
        )

        if answer == QtWidgets.QMessageBox.StandardButton.Cancel:
            return False

        if answer == QtWidgets.QMessageBox.StandardButton.Apply:
            # A block that will not parse is a block that cannot be left on
            # these terms either: the error is now on screen under the text it
            # is about, which is where it can be fixed.
            return self.apply_block()

        return True

    def _redraw(self):
        """The block put back as the document has it, and the picture with it."""

        if self.index is None:
            return

        self.problem.setVisible(False)
        self.text.setPlainText(self.document.text_of(self.index))
        self._park_cursor()
        self._redraw_view()

    def _redraw_view(self):
        """
        The picture alone, for the things that change it and not the file.

        The reach is one of those: how far a measurement answers for decides
        what holds along the trace and says nothing about what is written on it,
        so turning that dial must not put the block back and take a half-written
        line away with it.
        """

        if self.index is None:
            return

        self.view.show_structure(
            self.document.dataset.structures[self.index], self.max_gap
        )

    @property
    def max_gap(self):
        return float(self.gap_spin.value())

    def apply_block(self):
        """Parses the box into the document, or shows why it will not go."""

        if self.index is None:
            return False

        try:
            self.document.replace(self.index, self.text.toPlainText())
        except ValueError as err:
            self.problem.setText(str(err))
            self.problem.setVisible(True)
            return False

        self.problem.setVisible(False)

        # Re-read from the document rather than left as typed: what is on screen
        # is now what the file holds, and the two being the same object is what
        # keeps `Revert` meaning something.
        self.text.setPlainText(self.document.text_of(self.index))
        self._park_cursor()
        self._redraw_view()

        # The row as well: an applied block can have added the first attitude to
        # a trace, which changes its tally, what holds along it, and whether the
        # `carrying` filter keeps it -- and the filter is re-run for that last
        # one, or a trace would stay hidden from a list it now belongs in.
        self.table.update_row(
            self.index, self.document.dataset.structures[self.index], self.max_gap
        )
        self._apply_filter()

        self.applied.emit(self.index)

        return True

    # -- writing a line ----------------------------------------------------

    def _path_line(self, lines):
        """Which line the geometry starts on, or the end of the block."""

        return next(
            (n for n, line in enumerate(lines) if line.strip().partition(" ")[0] == "path"),
            len(lines),
        )

    def add_line(self, template):
        """
        Writes a template line above the path, and selects its first anchor.

        Above the path because that is where an assertion goes: `dumps` puts the
        geometry last, so a line written after it would sit among the vertices
        and read as one of them.
        """

        if self.index is None:
            return

        lines = self.text.toPlainText().splitlines()
        at = self._path_line(lines)

        lines.insert(at, template)

        self.text.setPlainText("\n".join(lines))
        self._aim_at_anchor(sum(len(line) + 1 for line in lines[:at]))

    def _park_cursor(self):
        """Puts the cursor where a new line goes, rather than at the top."""

        lines = self.text.toPlainText().splitlines()
        at = self._path_line(lines)

        cursor = self.text.textCursor()
        cursor.setPosition(max(sum(len(line) + 1 for line in lines[:at]) - 1, 0))

        self.text.setTextCursor(cursor)

    def _aim_at_anchor(self, start, same_line=False):
        """Selects the next `*`, so that a picked anchor replaces it."""

        text = self.text.toPlainText()
        stop = len(text)

        if same_line:
            ends = text.find("\n", start)
            stop = len(text) if ends < 0 else ends

        at = text.find("*", start)

        if at < 0 or at >= stop:
            return False

        cursor = self.text.textCursor()
        cursor.setPosition(at)
        cursor.setPosition(at + 1, QtGui.QTextCursor.MoveMode.KeepAnchor)

        self.text.setTextCursor(cursor)

        return True

    def insert_anchor(self, x, y):
        """
        Writes a picked anchor over the selected `*`, or at the cursor.

        A space goes in front of it where one is needed, an anchor being a token
        and not a suffix: without it a shift-click at the end of a line glues
        `@x,y` onto whatever that line ended with, and the parser reads the pair
        as one word it cannot make sense of.
        """

        cursor = self.text.textCursor()
        written = f"@{x:.2f},{y:.2f}"
        at = cursor.selectionStart() if cursor.hasSelection() else cursor.position()

        if not cursor.hasSelection():
            before = self.text.toPlainText()[:at]

            if before and not before[-1].isspace():
                written = " " + written

        cursor.insertText(written)

        # The next one on the same line, so that `span use * *` takes two clicks
        # and stops there rather than running on into the line below.
        self._aim_at_anchor(at + len(written), same_line=True)

        self.text.setFocus()


class EditorWindow(QtWidgets.QMainWindow):
    """The map with the file's traces on it, and the file beside them."""

    def __init__(self, session, document, legend="beside"):
        super().__init__()

        self.session = session
        self.document = document
        self.index = None

        self._forward, self._back = self._transformers()

        # The paths in map coordinates, worked out once. The model is in the
        # file's projection and stays there; this is the other side of that.
        self._drawn = [self.on_map(st.path) for st in document.dataset.structures]

        # Whether map and panel were laid side by side for want of anything
        # remembered about them, which is what `build` reads to know that the
        # map has already been given a size.
        self.tiled = False

        # Which trace the framing is on its way to, and the timer it waits on.
        self._framing = None
        self._frame_timer = QtCore.QTimer(self)
        self._frame_timer.setSingleShot(True)
        self._frame_timer.timeout.connect(lambda: self.frame_now())

        # The station dots that are drawn, as `(where on the map, the record)`,
        # and which of them the tooltip is currently about. The artist holds
        # coordinates and nothing else, so the records they were made from have
        # to be kept alongside or there is no way back from a dot to a station.
        self._marked = []
        self._tipped = None

        # And which of them the net is showing, which is not the same question:
        # the tooltip follows the cursor and the net stays where it was put. See
        # `_on_map_hover`.
        self._netted = None

        self._build_ui(legend)
        self._draw_base_map()

        self.select(0)

        self._retitle()
        self.say(self._opening())

    # -- the two projections ----------------------------------------------

    def _transformers(self):
        """
        File to map and back, or `(None, None)` where they are the same.

        A file with no CRS line is read as the session's rather than refused:
        the coordinates are then taken at face value, which is the only thing
        left to do with them, and it is a thing worth being able to fix from
        inside the editor.
        """

        from pyproj import CRS, Transformer

        declared = self.document.dataset.crs
        theirs = CRS.from_user_input(declared) if declared else None
        ours = self.session.crs

        if theirs is None or ours is None or theirs.equals(ours):
            return None, None

        return (
            Transformer.from_crs(theirs, ours, always_xy=True),
            Transformer.from_crs(ours, theirs, always_xy=True),
        )

    def on_map(self, path):
        """A path in the file's projection, as the map's coordinates."""

        if not path:
            return []

        if self._forward is None:
            return [(float(x), float(y)) for x, y in path]

        xs, ys = self._forward.transform(*zip(*path))

        return [(float(x), float(y)) for x, y in zip(xs, ys)]

    def in_file(self, x, y):
        """A point off the map, in the projection the file is written in."""

        if self._back is None:
            return float(x), float(y)

        moved = self._back.transform(float(x), float(y))

        return float(moved[0]), float(moved[1])

    # -- construction ------------------------------------------------------

    def _build_ui(self, legend):
        self.map_view = MapView(self.session, legend=legend)
        self.map_view.legend_handles_provider = self._legend_handles
        self.map_view.pressed.connect(self._on_map_pressed)
        self.map_view.hovered.connect(self._on_map_hover)
        self.map_view.hover_off.connect(lambda: self._tip_on(None))
        self.map_view.status.connect(self.say)

        # A third window, and not the dock it was first built as. The fold tool's
        # argument for a dock -- small, looked at beside the map, draggable back
        # into the side -- holds there and came apart here on two counts, and
        # both of them are about how big the picture is allowed to be. A dock is
        # as wide as the map can spare, which was 276 px and a circle 268 px
        # across; and the one way out of that, floating it, is a Qt::Tool window
        # whose geometry this tool does not save, so a net dragged out and made
        # readable would have to be made readable again every run. In the group
        # it is saved with the map and the panel, it goes on the second monitor
        # the rest of this tool is built around, and the circle is as big as the
        # window -- 412 px at the size it opens.
        self.net = StereonetView(figsize=NET_FIGSIZE_IN)
        self.net_window = SatelliteWindow(
            NET_EMPTY_TITLE, self.net, NET_WINDOW_PX, parent=self
        )

        self.panel = EditorPanel(self.document)
        self.panel.selected.connect(self.select)
        self.panel.applied.connect(self._on_applied)
        self.panel.framing_asked.connect(self.frame_on)

        self.save_button = QtWidgets.QPushButton("Save")
        self.save_button.setToolTip(
            "Write the file, replacing the lines of the structures that were "
            "edited and leaving every other byte of it alone (Ctrl+S)."
        )
        self.save_button.clicked.connect(self.save)

        self.save_as_button = QtWidgets.QPushButton("Save as...")
        self.save_as_button.clicked.connect(self.save_as)

        writing = QtWidgets.QHBoxLayout()
        writing.addWidget(self.save_button)
        writing.addWidget(self.save_as_button)
        writing.addStretch(1)

        # The status bar's words again, beside the panel. The windows can be on
        # two screens, and which of them the news belongs on depends on the
        # news: the map reports what a click found, the panel what Apply and Save
        # did, and each is read from the other window often enough to matter.
        # `say` writes both -- one call, no state between them, so there is
        # nothing in the echo that can drift out of step with the bar.
        self.echo = QtWidgets.QLabel()
        self.echo.setStyleSheet("color: #555555; font-size: 11px;")
        self.echo.setMinimumHeight(16)

        side = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.addWidget(self.panel, stretch=1)
        side_layout.addLayout(writing)
        side_layout.addWidget(self.echo)
        side.setMinimumWidth(PANEL_WIDTH_PX)

        # The map is the window and the panel is beside it, which is not the same
        # arrangement as before: the two were one frame split down the middle, so
        # the map could not be made bigger without making the table smaller, and
        # neither could go on the other screen. What the split cost is not room
        # but reach -- a second monitor is the map at the size the DEM deserves
        # and the file open next to it, and a splitter cannot be dragged across a
        # screen boundary.
        self.panel_window = SatelliteWindow(
            self._panel_title(), side, PANEL_WINDOW_PX, parent=self
        )

        self.group = WindowGroup(
            self,
            SETTINGS_NAME,
            {"panel": self.panel_window, "net": self.net_window},
            placer=self._place_unremembered,
        )

        central = QtWidgets.QWidget()
        map_layout = QtWidgets.QHBoxLayout(central)
        map_layout.setContentsMargins(0, 0, 0, 0)
        map_layout.addWidget(self.map_view, stretch=1)
        map_layout.addWidget(LegendControls(self.map_view, placement=legend))

        self.setCentralWidget(central)

        self._build_menu()
        self._build_shortcuts()

    def _build_menu(self):
        """The way back to the panel and the net, once they have been closed."""

        menu = self.menuBar().addMenu("&Windows")

        self.window_actions = self.group.actions_into(
            menu, {"panel": "&Structures", "net": "Stereo&net"}
        )

        menu.addSeparator()

        front = QtGui.QAction("Bring all to &front", self)
        front.triggered.connect(self.group.raise_all)
        menu.addAction(front)

    def _build_shortcuts(self):
        """
        Save and Apply from either window, which takes saying so.

        Both were shortcuts on their buttons, and a button's shortcut reaches
        only the window the button is in -- which used to be the only window
        there was. Left alone, Ctrl+S would have worked over the panel and done
        nothing over the map, where half the work is: clicking anchors along a
        trace and then writing the file is one motion, and it goes through the
        map.

        Given to every window of the group, rather than made
        `ApplicationShortcut` -- which reaches every window of the application,
        including another tool's. The section tool binds Ctrl+S to saving a
        section, and two tools open at once would have had one ambiguous
        shortcut between them and Qt firing neither.

        Every window, including the one nothing is typed into: the net is a
        canvas and clicking it gives it the keyboard, so leaving it out would
        make Ctrl+S depend on which window was last clicked -- which is worse
        than no shortcut, because it fails silently and only sometimes.
        """

        for label, shortcut, slot in (
            ("Save", "Ctrl+S", self.save),
            ("Apply", "Ctrl+Return", self.panel.apply_block),
        ):
            action = QtGui.QAction(label, self)
            action.setShortcut(shortcut)
            action.triggered.connect(slot)

            self.addAction(action)

            for satellite in self.group.satellites.values():
                satellite.addAction(action)

    def _panel_title(self):
        mark = "*" if self.document.dirty else ""

        return f"gSurf - structures - {self.document.path.name}{mark}"

    def say(self, message):
        """One line of news, on the map's bar and in the panel's echo."""

        self.statusBar().showMessage(message)

        # Set as the tooltip too, because the label is one line in a 520-pixel
        # panel and the longer messages here run past that. Clipped text with
        # nothing behind it would be a sentence the panel silently shortens.
        self.echo.setText(message)
        self.echo.setToolTip(message)

    def showEvent(self, event):
        """The panel comes up with the map, the first time and only then."""

        super().showEvent(event)

        self.group.show_satellites()

    def _place_unremembered(self, skip=()):
        """
        Map and panel side by side, filling the screen, the first time ever.

        The group's own answer -- satellites down the right edge, and only on a
        desktop 1600 wide -- is for a tool whose windows do not fit, where the
        window manager's guess is as good as any. Two of these three do fit
        where that rule says they do not: on the 1366x741 of usable area this is
        written on, 840 of map beside 520 of panel is the splitter's own
        arrangement made out of two windows. Which is the one thing this change
        must not be worse than -- taking a pane out of a frame should buy a
        second monitor, not cost a first one.

        It places the map as well, which is why `build` asks whether this ran
        before falling back to `fit_to_screen`: a map maximised over a panel
        placed beside it would be the arrangement this exists to avoid.

        Once. The first drag onto another screen is remembered, and after that
        there is something remembered and this never runs again.
        """

        available = self.main_screen_area()

        # The net goes first and goes over the map, there being no third
        # rectangle once the other two have filled the screen. The corner is a
        # choice about what it covers: the map is re-framed on the trace being
        # edited every time a row is picked, so what the cursor is hunting for
        # is in the middle of it. Lifted clear of the status bar, which runs the
        # width of the map window and is where `say` writes -- a net resting on
        # the bottom edge would cover the left end of every message, which is
        # where they start.
        #
        # Placed on any screen, narrow or not, because it is small enough to
        # land somewhere on all of them, and because a window covering a corner
        # of the map is undone by one drag -- which is then remembered.
        if "net" not in skip:
            net_width, net_height = NET_WINDOW_PX

            self.net_window.setGeometry(
                available.left() + TILE_GAP_PX,
                available.bottom()
                - net_height
                - FRAME_ALLOWANCE_PX
                - self.statusBar().sizeHint().height(),
                net_width,
                net_height,
            )

        width = available.width() - PANEL_WIDTH_PX - TILE_GAP_PX

        if width < MAP_FLOOR_PX:
            self.tiled = False
            return

        height = available.height() - FRAME_ALLOWANCE_PX

        if "panel" not in skip:
            self.panel_window.setGeometry(
                available.right() - PANEL_WIDTH_PX + 1,
                available.top(),
                PANEL_WIDTH_PX,
                height,
            )

        if "map" not in skip:
            self.setGeometry(available.left(), available.top(), width, height)

        self.tiled = True

    def main_screen_area(self):
        """
        The desktop the windows are placed against.

        A method rather than the one call it is, so that a check can stand a
        narrow screen in front of it: the placement has two branches and a run
        can only ever be on one screen, which off-screen is 800x800 -- narrow
        enough that the branch a real laptop takes would never be exercised.
        """

        return self.screen().availableGeometry()

    def _draw_base_map(self):
        self.map_view.draw_base_map()

        axes = self.map_view.axes

        # Two collections and not 393 lines: they change only when a block is
        # applied, so they belong in the background the blitting is cut from, and
        # one artist is one draw.
        #
        # Two rather than one because which forty-five carry a plane is the
        # question the window opens on, and a map that cannot answer it leaves the
        # table to answer it alone -- you would be able to list them and not to
        # aim at one. The carrying ones go on top, so that where the two cross it
        # is the one with something written on it that stays whole.
        self.traces = {
            carrying: axes.add_collection(
                LineCollection(
                    self._segments(carrying),
                    zorder=3.5 if carrying else 3.0,
                    **(CARRYING_STYLE if carrying else BARE_STYLE),
                )
            )
            for carrying in (False, True)
        }

        self.highlight = self.map_view.add_animated(
            axes.add_line(Line2D([], [], color="#ff7f0e", lw=2.6, zorder=6))
        )
        self.refused = self.map_view.add_animated(
            axes.add_line(
                Line2D([], [], color=PROVENANCE_TINT["rifiutata"], lw=3.4, zorder=7)
            )
        )
        self.marks = self.map_view.add_animated(
            axes.add_line(
                Line2D(
                    [], [], color=PROVENANCE_TINT["misurata"], marker="o",
                    markersize=5, linestyle="none", zorder=8,
                )
            )
        )
        self.picked = self.map_view.add_animated(
            axes.add_line(
                Line2D(
                    [], [], color="#000000", marker="+", markersize=11,
                    markeredgewidth=1.6, linestyle="none", zorder=9,
                )
            )
        )

        # Built here and not left to the placement combo, which is what used to
        # happen: the other three tools ask for the legend once the map is drawn
        # and this one never did, so its four entries were made on every rebuild
        # and there was no rebuild -- `legend_handles` answered and nothing
        # called it. The window opened with no legend at all until somebody moved
        # the combo, and a colour nobody can look up is a colour that says nothing.
        self.map_view.refresh_legend()
        self.map_view.anchor_home()

    def _segments(self, carrying):
        """The drawn paths of the traces on one side of `carries`."""

        return [
            path
            for path, structure in zip(self._drawn, self.document.dataset.structures)
            if len(path) > 1 and carries(structure) is carrying
        ]

    def _reset_traces(self):
        """
        Hands both collections their geometry again.

        Both, and not the one that was edited: applying a block can add the first
        attitude to a trace, and the trace then belongs to the other collection.
        A pale line that has just been given a plane has to stop being pale, or
        the map is answering last minute's question.
        """

        for carrying, collection in self.traces.items():
            collection.set_segments(self._segments(carrying))

    def _legend_handles(self):
        # The two weights are switchable, and taking the pale one off is the
        # third way to the same place the table's filter and its sorting lead:
        # 348 contacts off the map, and what is left is the forty-five with
        # something written on them. The steering artists below are not -- a
        # highlight switched off would leave a selection with nothing to show it.
        return [
            self.map_view.switchable(
                Line2D(
                    [], [],
                    color=CARRYING_STYLE["colors"],
                    lw=CARRYING_STYLE["linewidths"],
                    label="carrying a plane",
                ),
                self.traces[True],
            ),
            self.map_view.switchable(
                Line2D(
                    [], [],
                    color=BARE_STYLE["colors"],
                    lw=BARE_STYLE["linewidths"],
                    label="mapped, nothing read",
                ),
                self.traces[False],
            ),
            Line2D([], [], color="#ff7f0e", lw=2.6, label="selected"),
            Line2D(
                [], [], color=PROVENANCE_TINT["rifiutata"], lw=3.4,
                label="rejected stretch",
            ),
            Line2D(
                [], [], color=PROVENANCE_TINT["misurata"], marker="o",
                markersize=5, linestyle="none", label="measured",
            ),
        ]

    def _opening(self):
        dataset = self.document.dataset
        said = (
            f"{len(dataset.structures)} structure(s), "
            f"{sum(1 for s in dataset.structures if carries(s))} carrying a plane: "
            f"{sum(len(s.attitudes) for s in dataset.structures)} attitude(s), "
            f"{sum(len(s.fits) for s in dataset.structures)} fit(s)"
        )

        if dataset.observations:
            # Preserved and not editable: an observation attaches to no path, so
            # there is no trace to select it on and no ruler to draw it along.
            # It stays in the file untouched, and is counted here so that it is
            # a thing left alone rather than a thing gone missing.
            said += f", {len(dataset.observations)} loose observation(s), kept as they are"

        if not dataset.crs:
            said += "; no CRS declared, read as the session's"

        return (
            f"{said}. Click a trace to select it, shift-click to pick an anchor; "
            f"or pick a row in the table, which brings its trace into view."
        )

    # -- selection ---------------------------------------------------------

    def select(self, index):
        """
        Puts one structure under the hand, on the map and in the panel.

        The panel is asked first and can say no, which is what keeps a click on
        the map from redrawing over a block that was typed and not applied. When
        it does, nothing here moves: the highlight stays on the structure whose
        text is still in the box, so the two never disagree about which one is
        being worked on.
        """

        if not self.panel.show_index(index):
            return

        self.index = index

        structure = self.document.dataset.structures[index]
        drawn = self._drawn[index]

        # `[y for _, y in ...]`, and it read `[y for y, _ in ...]` from the day
        # this was written: the name bound is the first of the pair whatever it
        # is called, so the highlight was drawn at (easting, easting) -- off the
        # map, on a diagonal no extent here ever covers. Selecting a trace worked
        # and showed nothing, which is not a thing the eye reads as a bug in the
        # drawing. `check_editor` now asserts the highlight is on the trace.
        self.highlight.set_data([x for x, _ in drawn], [y for _, y in drawn])

        self._mark_refusals(structure)
        self._mark_attitudes(structure)
        self.picked.set_data([], [])

        self.map_view.blit()

    def _mark_refusals(self, structure):
        """The stretches somebody has rejected, drawn where they are."""

        xs, ys = [], []

        for span in structure.spans:
            if span.axis != "use" or span.value != "rejected" or span.s0 is None:
                continue

            for x, y in self.on_map(stretch(structure.path, span.s0, span.s1)):
                xs.append(x)
                ys.append(y)

            # A break, so two rejected stretches do not join across the good
            # ground between them.
            xs.append(np.nan)
            ys.append(np.nan)

        self.refused.set_data(xs, ys)

    def _mark_attitudes(self, structure):
        marked = [
            attitude for attitude in structure.attitudes if attitude.s is not None
        ]
        drawn = self.on_map([point_on(structure.path, a.s) for a in marked])

        # The same reversed pair as in `select`, and the same consequence: the
        # green dot marking where a plane was measured was never on the fault.
        self.marks.set_data([x for x, _ in drawn], [y for _, y in drawn])

        # Kept for the hover, and dropped first: the tooltip is held by index
        # into this list, and index 0 of the trace just selected is not index 0
        # of the one before it. Without the reset a cursor that had not moved
        # would go on showing the station of a dot that is no longer there.
        self._tip_on(None)

        # And the net with it, this being the change it is cleared by. A net is
        # meant to outlast the cursor leaving its dot, which is the whole reason
        # `hover_off` does not touch it -- but it must not outlast the trace it
        # belongs to, or the window would show one fault with another's plane
        # beside it and nothing on screen saying so.
        self._clear_net()
        self._marked = list(zip(drawn, marked))

    def _on_applied(self, index):
        """A block that parsed: the map has to agree with it again."""

        self._drawn[index] = self.on_map(self.document.dataset.structures[index].path)

        # The static collections hold a copy of the geometry, so a path edited
        # in the box has to be handed over again -- and then a full draw, which
        # is what recaptures the background the rest is blitted over.
        self._reset_traces()
        self.map_view.canvas.draw()

        self.select(index)
        self._retitle()
        self.say(
            f"{self.document.dataset.structures[index].ident} applied; "
            f"the file is written by Save"
        )

    # -- where the view is -------------------------------------------------

    def frame_on(self, index):
        """
        Asks for the view to move onto one trace, in a moment.

        Deferred rather than done, because a run of selections is one gesture:
        holding the down arrow in the table is a row a keystroke, and each of
        these is a full redraw and a hillshade reread for a view nobody stopped
        to look at.
        """

        self._framing = index
        self._frame_timer.start(FRAME_DELAY_MS)

    def frame_now(self, index=None):
        """Moves the view onto a trace, and says where it went."""

        self._frame_timer.stop()

        if index is None:
            index, self._framing = self._framing, None

        if index is None:
            return False

        structure = self.document.dataset.structures[index]
        extent = framing_for(self._drawn[index])

        if extent is None:
            self.say(f"{structure.ident} has no path to frame on")
            return False

        self.map_view.restore_framing(extent)

        # The way back said out loud, because this is the one thing in the window
        # that moves the map without the hand having moved it.
        self.say(
            f"framed on {structure.ident}, {structure.length:.0f} m "
            f"-- the back arrow returns to where you were"
        )

        return True

    # -- what the cursor is resting on --------------------------------------

    def _on_map_hover(self, x, y):
        """
        A cursor resting on a station dot, or on nothing.

        Two answers with two lifetimes, which is why they are two calls. The
        tooltip is about where the cursor is and goes away with it -- that is
        what `hover_off` is connected to. The net is a figure, and reading a
        figure means looking away from the dot that asked for it: one that
        emptied as the cursor left would only ever be seen out of the corner of
        an eye. So the net is filled by a hover and cleared by a change of
        trace, and in between it keeps saying whose it is in its own title.
        """

        which = self._station_near(x, y)

        self._tip_on(which)

        if which is not None:
            self._net_on(which)

    def _station_near(self, x, y):
        """
        Which drawn station dot the cursor is on, as an index, or nothing.

        In display pixels, through the axes' own transform, because the dots are
        five points across whatever the view is showing: a reach in metres would
        be a target a kilometre wide framed on the whole AOI and unhittable
        framed on one fault.

        The nearest of the ones in reach and not the first. The two anchors on
        F0273 are 26 m apart -- one dot at most scales, two that overlap just
        before they separate -- and which of them is meant is the nearer one.
        """

        if not self._marked:
            return None

        transform = self.map_view.axes.transData
        here = transform.transform((x, y))
        offsets = transform.transform([point for point, _ in self._marked]) - here
        squared = (offsets * offsets).sum(axis=1)
        nearest = int(np.argmin(squared))

        if squared[nearest] > HOVER_RADIUS_PX**2:
            return None

        return nearest

    def _tip_on(self, which):
        """
        Puts the tooltip on one station dot, or takes it off.

        Guarded on which dot and not on where the cursor is, because this runs on
        every motion event the canvas sees: crossing a dot is two changes and the
        hundreds of pixels either side of it are none, so the text is built when
        the answer changes rather than when the mouse moves. Which makes a motion
        event 0.032 ms when the answer stands and 0.063 when it does not -- the
        guard is not what makes this affordable, it is what keeps a tooltip from
        being rebuilt sixty times a second while the cursor sits still.

        Qt's own tooltip, so it appears after the delay everything else on this
        desktop appears after and goes away without being told to. What that
        costs is that a tooltip already on screen does not re-read its text, so
        moving between two dots close enough to be under one tooltip can show the
        first one's -- which is why they are 10 pixels apart at most and why this
        is the place it would be noticed.
        """

        if which == self._tipped:
            return

        self._tipped = which
        self.map_view.canvas.setToolTip(
            "" if which is None else self._station_tip(self._marked[which][1])
        )

    def _net_on(self, which):
        """
        Puts one station's plane on the net, with the lines read on it.

        The great circle is what the two numbers in the tooltip already say and
        the net says differently: `145/35` is a plane you have to picture, and a
        picture is what this is for. What it is really for is the second fact --
        where a striation sits within the plane -- because a trend and a dip
        direction written side by side do not show whether the movement was down
        the dip or along the strike, and the net does, at a glance, without
        arithmetic.

        Guarded like the tooltip, and for a stronger reason: filling this is a
        `set_data` on every artist the net has and a blit, which is a thousand
        times a motion event's own cost. `None` is not a value this takes --
        clearing is `_clear_net`, called when the trace changes and not when the
        cursor moves.

        Drawn whether the window is open or not, which is the opposite of what
        the fold tool does with the same widget. The difference is what makes it
        redraw: there it is every frame of a drag, and a hidden canvas costs
        frame budget and makes the frame cost that tool reports a measurement of
        something nobody can see. Here it is a cursor crossing onto a different
        dot. Skipping it would save 2.3 ms on an occasional event and cost the
        thing it is there for -- a net put back would show whichever station it
        happened to be closed on.
        """

        if which == self._netted:
            return

        self._netted = which
        attitude = self._marked[which][1]
        plane = attitude.plane

        if plane is None:
            self._clear_net()
            return

        station = attitude.attrs.get("station") or "no station code"
        lineations = self._lineations_at(attitude)

        self.net_window.setWindowTitle(
            f"{NET_TITLE} - {station} -- {plane}"
            + (f", {len(lineations)} lineation(s)" if lineations else "")
        )
        self.net.show_attitude(
            plane.dip_dir,
            plane.dip,
            lineations,
            color=PROVENANCE_TINT["misurata"],
        )

    def _clear_net(self):
        """Nothing on the net, and a title that says what would put it there."""

        self._netted = None
        self.net_window.setWindowTitle(NET_EMPTY_TITLE)
        self.net.show_attitude(None, None)

    def _lineations_at(self, attitude):
        """
        The lineations read at the same place as one plane, as `(trend, plunge)`.

        By station code where the lineation carries one, and by distance along the
        trace only where it does not. That order is not a preference: `off` runs to
        69.7 m in `merid_faults`, so two readings made standing in one spot can
        snap to progressives seventy metres apart, and a rule that went by
        distance first would fail on exactly the outcrops the tooltip exists to
        warn about. `SAME_OUTCROP_M` is what is left for a lineation that says
        nothing about where it was read.

        Plural on purpose. `merid_faults` has a station whose note reads "strie
        osservate: vedi foglio (pitch 1 30 deg, p. 2 80 deg)": two sets of striae
        on one surface, which is two movements and the reason this returns a list
        rather than the one lineation a station usually has. It currently returns
        an empty list for all 23 of them, there being no `lineation` record in the
        file at all -- the striae that were read are prose inside `note=`, and
        turning prose into a record is curation and not something to guess at
        while drawing.
        """

        structure = self.document.dataset.structures[self.index]
        station = attitude.attrs.get("station")
        found = []

        for lineation in structure.lineations:
            if lineation.trend is None or lineation.plunge is None:
                continue

            labelled = lineation.attrs.get("station")

            if labelled is not None:
                near = station is not None and labelled == station
            else:
                near = (
                    lineation.s is not None
                    and attitude.s is not None
                    and abs(lineation.s - attitude.s) <= SAME_OUTCROP_M
                )

            if near:
                found.append((lineation.trend, lineation.plunge))

        return found

    def _station_tip(self, attitude):
        """
        What a station dot is, in the order somebody pointing at it wants it.

        The answer first -- which station, what plane -- then where along the
        trace, and then the one thing the dot cannot say about itself. `off` is
        how far the reading was from the line it has been snapped onto, and over
        the 23 measurements of `merid_faults` it runs from 0.0 to 69.7 m. Seventy
        metres is inside the width of the line at 1:25000 and a visible lie at
        1:5000; the dot looks the same either way, and the number is not in the
        block either -- it is an attribute nobody reads unless it is put in front
        of them.

        Then a refusal, where the curation has one. A rejected measurement stays
        drawn, because somebody did stand there and that does not stop being
        true, but it is not what holds -- and until now the band in the panel was
        the only thing that said so, which is a different window.
        """

        attrs = attitude.attrs
        station = attrs.get("station") or "no station code"
        structure = self.document.dataset.structures[self.index]

        lines = [
            station if attitude.plane is None else f"{station} -- {attitude.plane}",
            f"{attitude.s:.0f} m along {structure.ident}",
        ]

        off = _as_number(attrs.get("off"))

        if off is not None:
            lines.append(
                "read on the trace"
                if off == 0.0
                else f"read {off:.1f} m off the trace, and snapped onto it"
            )

        # `attitude_at` at the anchor's own progressive: no measurement can be
        # nearer to it than it is to itself, so the only answer that is not this
        # one is a refusal.
        _, said = structure.attitude_at(attitude.s, self.panel.max_gap)

        if said.startswith("rifiutata"):
            lines.append(f"overruled here -- {said}")

        stamp = " ".join(
            part for part in (attrs.get("src", ""), attrs.get("date", "")) if part
        )

        if stamp:
            lines.append(stamp)

        for key in ("note", "site_note"):
            if attrs.get(key):
                lines.append(textwrap.fill(attrs[key], TIP_WRAP))

        return "\n".join(lines)

    # -- the map -----------------------------------------------------------

    def _on_map_pressed(self, x, y):
        modifiers = QtWidgets.QApplication.keyboardModifiers()

        self.pick(
            x, y,
            anchor=bool(modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier),
        )

    def pick(self, x, y, anchor=False):
        """
        A click on the map: the trace it was aimed at, or an anchor on this one.

        Shift-click projects onto the structure that is *selected*, whichever
        trace is nearest, because the anchor is about to be written into that
        block and an anchor snapped onto a neighbour would read back as a
        progressive on a fault it was never measured on.
        """

        here = self.in_file(x, y)
        reach = self._reach_in_metres()

        if anchor:
            if self.index is None:
                self.say("nothing selected to anchor on")
                return

            structure = self.document.dataset.structures[self.index]

            if len(structure.path) < 2:
                self.say(
                    f"{structure.ident} has no path to anchor on"
                )
                return

            s, distance = place_on(structure.path, *here)
            snapped = point_on(structure.path, s)

            self.panel.insert_anchor(*snapped)

            drawn = self.on_map([snapped])[0]
            self.picked.set_data([drawn[0]], [drawn[1]])
            self.map_view.blit()

            self.say(
                f"@{snapped[0]:.2f},{snapped[1]:.2f} -- {structure.ident} at "
                f"{s:.0f} m, {distance:.0f} m from where you clicked"
            )
            return

        found = nearest_structure(self.document.dataset, *here, within=reach)

        if found is None:
            self.say("no trace within reach of the click")
            return

        index, s, _ = found

        self.select(index)
        self._report_at(index, s)

    def _reach_in_metres(self):
        """`PICK_RADIUS_PX` as ground metres at the framing now on screen."""

        axes = self.map_view.axes
        left, right = axes.get_xlim()
        width = axes.bbox.width or 1.0

        return abs(right - left) / width * PICK_RADIUS_PX

    def _report_at(self, index, s):
        structure = self.document.dataset.structures[index]
        plane, said = structure.attitude_at(s, self.panel.max_gap)

        self.say(
            f"{structure.ident} at {s:.0f} m of {structure.length:.0f} -- {said}"
            + ("" if plane is None else f", {plane}")
        )

    # -- the file ----------------------------------------------------------

    def _retitle(self):
        mark = "*" if self.document.dirty else ""

        self.setWindowTitle(f"gSurf - trace editor - {self.document.path.name}{mark}")

        # The panel carries the file's name and the star as well, because Save is
        # in the panel and so is the typing: a window that holds unwritten work
        # should say so on its own frame, not only on the map's.
        self.panel_window.setWindowTitle(self._panel_title())

    def save(self):
        """Writes the document to the file it came from."""

        return self._write(None)

    def save_as(self):
        # Over the panel and not over the map: Qt puts a dialog on the screen its
        # parent is on, and both of these are about the file -- which is what the
        # panel window is. Asked for from the map by Ctrl+S, the answer still
        # belongs beside the buttons that otherwise ask it.
        name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.panel_window,
            "Save the file as",
            str(self.document.path),
            "gstruct (*.gstruct)",
        )

        return self._write(name) if name else False

    def _write(self, target):
        try:
            written = self.document.save(target)
        except OSError as err:
            QtWidgets.QMessageBox.critical(self.panel_window, "Not written", str(err))
            return False

        self._retitle()
        self.say(f"written to {written}")

        return True

    def closeEvent(self, event):
        """
        A document with unapplied or unwritten work is asked about, once.

        Two different losses, and the one that catches people is the first: a
        block typed into the box and never applied is not in the document, so
        Save would write the file without it and report success.
        """

        pending = (
            self.panel.index is not None
            and self.panel.text.toPlainText() != self.document.text_of(self.panel.index)
        )

        if self.document.dirty or pending:
            said = "This file has changes that are not written."

            if pending:
                said += (
                    "\n\nThe block on screen has also not been applied: Apply "
                    "reads it into the document, and only then does Save write it."
                )

            answer = QtWidgets.QMessageBox.question(
                self,
                "Not written",
                said + "\n\nWrite it now?",
                QtWidgets.QMessageBox.StandardButton.Save
                | QtWidgets.QMessageBox.StandardButton.Discard
                | QtWidgets.QMessageBox.StandardButton.Cancel,
            )

            if answer == QtWidgets.QMessageBox.StandardButton.Cancel:
                event.ignore()
                return

            if answer == QtWidgets.QMessageBox.StandardButton.Save and not self.save():
                event.ignore()
                return

        # After the question and not before it: a close that was cancelled is not
        # an arrangement anybody finished working in, and writing the geometry
        # there would save the windows as they stood in front of a dialog.
        self.group.save_geometry()

        super().closeEvent(event)


def build(session, chosen, legend="beside"):
    """The window, on a session somebody else has already opened."""

    spec = chosen.get("traces") or {}
    path = spec.get("path")

    if not is_gstruct(path):
        QtWidgets.QMessageBox.critical(
            None,
            "Nothing to edit",
            f"{path}\n\nThis edits a .gstruct, which is a text format with one "
            f"fact per line. A layer has no text to edit and no way to hold a "
            f"span or a fit.\n\nTo get one from a layer: Import - lines to "
            f".gstruct, on the launcher. It asks what the columns mean and "
            f"writes a file, which this then opens.",
        )
        return None

    try:
        document = Document(path)
    except Exception as err:
        QtWidgets.QMessageBox.critical(
            None, "Unreadable", f"{path}\n\n{type(err).__name__}: {err}"
        )
        return None

    if not document.dataset.structures:
        QtWidgets.QMessageBox.critical(
            None, "Nothing to edit", f"{path}\n\nno structure in the file"
        )
        return None

    # Refused at the door rather than handled inside, because there is nothing
    # here that would only half work: the band along the top is metres, the
    # reach is metres, and an anchor written to two decimals of a degree lands
    # somewhere else entirely. A tool that opened anyway would be a tool that
    # writes the file wrong.
    said = degrees_not_metres(document.dataset, getattr(session, "crs", None))

    if said:
        QtWidgets.QMessageBox.critical(
            None, "Degrees, not metres", f"{path}\n\n{said}"
        )
        return None

    window = EditorWindow(session, document, legend=legend)

    # The panel follows from the map's own `showEvent`, so whichever of these two
    # shows it brings the group up with it. `tiled` is the third case: nothing
    # remembered, but a screen the two fit across, so the map has a size already
    # and `fit_to_screen` would undo it -- see `_place_unremembered`.
    if window.group.restore_geometry() or window.tiled:
        window.show()
    else:
        fit_to_screen(window, *MAP_WINDOW_PX)

    return window
