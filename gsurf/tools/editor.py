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

# How near the cursor has to come, in screen pixels, for a click to have been
# aimed at a trace. In pixels and not in metres so that it means the same thing
# at every zoom.
PICK_RADIUS_PX = 14

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

# Above this fraction of the trace, a bar has room for its own word in it.
LABEL_FRACTION = 0.14


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

    def __init__(self, document, parent=None):
        super().__init__(parent)

        self.document = document
        self.index = None
        self._filling = False

        self.chooser = QtWidgets.QComboBox()
        self.chooser.setEditable(True)
        self.chooser.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.chooser.completer().setCompletionMode(
            QtWidgets.QCompleter.CompletionMode.PopupCompletion
        )
        self.chooser.setToolTip(
            "Type an ident to find it. The tally is what the structure carries: "
            "attitudes, fits, spans."
        )
        self.chooser.currentIndexChanged.connect(self._chosen)

        self.carrying = QtWidgets.QCheckBox("only the ones carrying a plane")
        self.carrying.setToolTip(
            "45 of the 393 faults of merid_faults carry an attitude or a fit. "
            "The rest are mapped contacts nobody has read a plane off yet, and "
            "they are still editable -- this only shortens the list."
        )
        self.carrying.toggled.connect(lambda _: self._fill_chooser())

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
        self.gap_spin.valueChanged.connect(lambda _: self._redraw_view())

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
        self.apply_button.setShortcut("Ctrl+Return")
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
        heading.addWidget(self.chooser, stretch=1)
        heading.addWidget(self.carrying)

        gap = QtWidgets.QHBoxLayout()
        gap.addWidget(QtWidgets.QLabel("a measurement answers for"))
        gap.addWidget(self.gap_spin)
        gap.addStretch(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addLayout(heading)
        layout.addLayout(gap)
        layout.addWidget(self.view, stretch=3)
        layout.addWidget(self.text, stretch=2)
        layout.addWidget(self.problem)
        layout.addLayout(buttons)

        self._fill_chooser()

    # -- the list ----------------------------------------------------------

    def _tally(self, structure):
        """What a structure carries, short enough to sit in a combo entry."""

        counts = (
            (len(structure.attitudes), "att"),
            (len(structure.fits), "fit"),
            (len(structure.spans), "span"),
        )
        said = ", ".join(f"{n} {word}" for n, word in counts if n)

        return f"{structure.ident}   {said}" if said else structure.ident

    def _fill_chooser(self):
        held = self.index

        self._filling = True
        self.chooser.clear()

        for index, structure in enumerate(self.document.dataset.structures):
            if self.carrying.isChecked() and not (structure.attitudes or structure.fits):
                continue

            self.chooser.addItem(self._tally(structure), index)

        at = self.chooser.findData(held)
        self.chooser.setCurrentIndex(max(at, 0))
        self._filling = False

        # Only where the filter has taken away what was open. Rebuilding the
        # list is not a reason to change which structure is being worked on, and
        # the row it sits on moving is not a change of structure.
        if at < 0 and self.chooser.count():
            self._chosen(0)

    def _chosen(self, _row):
        if self._filling:
            return

        index = self.chooser.currentData()

        if index is None:
            return

        if not self.show_index(index):
            self._point_combo_at(self.index)
            return

        self.selected.emit(index)

    def _point_combo_at(self, index):
        """Puts the combo back on whatever is actually on screen."""

        at = self.chooser.findData(index)

        if at >= 0 and at != self.chooser.currentIndex():
            self._filling = True
            self.chooser.setCurrentIndex(at)
            self._filling = False

    # -- one structure -----------------------------------------------------

    def show_index(self, index):
        """
        Puts a structure on screen, whoever asked. False if the old one was kept.

        Selecting is the one gesture that would throw away work without being
        asked to: a block typed and not applied lives only in the box, and a
        click on another trace would have redrawn over it. So leaving is a
        question, and the caller has to be able to hear no -- the map holds its
        highlight where it was, and the combo goes back to the row it was on.
        """

        if index == self.index:
            return True

        if not self._may_leave():
            return False

        self.index = index

        self._point_combo_at(index)
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

        self._build_ui(legend)
        self._draw_base_map()

        self.select(0)

        self._retitle()
        self.statusBar().showMessage(self._opening())

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
        self.map_view.status.connect(self.statusBar().showMessage)

        self.panel = EditorPanel(self.document)
        self.panel.selected.connect(self.select)
        self.panel.applied.connect(self._on_applied)

        self.save_button = QtWidgets.QPushButton("Save")
        self.save_button.setShortcut("Ctrl+S")
        self.save_button.setToolTip(
            "Write the file, replacing the lines of the structures that were "
            "edited and leaving every other byte of it alone."
        )
        self.save_button.clicked.connect(self.save)

        self.save_as_button = QtWidgets.QPushButton("Save as...")
        self.save_as_button.clicked.connect(self.save_as)

        writing = QtWidgets.QHBoxLayout()
        writing.addWidget(self.save_button)
        writing.addWidget(self.save_as_button)
        writing.addStretch(1)

        side = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.addWidget(self.panel, stretch=1)
        side_layout.addLayout(writing)
        side.setMinimumWidth(PANEL_WIDTH_PX)

        map_side = QtWidgets.QWidget()
        map_layout = QtWidgets.QHBoxLayout(map_side)
        map_layout.setContentsMargins(0, 0, 0, 0)
        map_layout.addWidget(self.map_view, stretch=1)
        map_layout.addWidget(LegendControls(self.map_view, placement=legend))

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.addWidget(map_side)
        split.addWidget(side)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)

        self.setCentralWidget(split)

    def _draw_base_map(self):
        self.map_view.draw_base_map()

        axes = self.map_view.axes

        # A collection and not 393 lines: they never change, so they belong in
        # the background the blitting is cut from, and one artist is one draw.
        self.traces = LineCollection(
            [path for path in self._drawn if len(path) > 1],
            colors="#555555", linewidths=0.8, zorder=3,
        )
        axes.add_collection(self.traces)

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

        self.map_view.anchor_home()

    def _legend_handles(self):
        return [
            Line2D([], [], color="#555555", lw=0.8, label="traces in the file"),
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

        return f"{said}. Click a trace to select it, shift-click to pick an anchor."

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

        self.highlight.set_data([x for x, _ in drawn], [y for y, _ in drawn])

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
        places = [
            point_on(structure.path, attitude.s)
            for attitude in structure.attitudes
            if attitude.s is not None
        ]
        drawn = self.on_map(places)

        self.marks.set_data([x for x, _ in drawn], [y for y, _ in drawn])

    def _on_applied(self, index):
        """A block that parsed: the map has to agree with it again."""

        self._drawn[index] = self.on_map(self.document.dataset.structures[index].path)

        # The static collection holds a copy of the geometry, so a path edited
        # in the box has to be handed to it again -- and then a full draw, which
        # is what recaptures the background the rest is blitted over.
        self.traces.set_segments([path for path in self._drawn if len(path) > 1])
        self.map_view.canvas.draw()

        self.select(index)
        self._retitle()
        self.statusBar().showMessage(
            f"{self.document.dataset.structures[index].ident} applied; "
            f"the file is written by Save"
        )

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
                self.statusBar().showMessage("nothing selected to anchor on")
                return

            structure = self.document.dataset.structures[self.index]

            if len(structure.path) < 2:
                self.statusBar().showMessage(
                    f"{structure.ident} has no path to anchor on"
                )
                return

            s, distance = place_on(structure.path, *here)
            snapped = point_on(structure.path, s)

            self.panel.insert_anchor(*snapped)

            drawn = self.on_map([snapped])[0]
            self.picked.set_data([drawn[0]], [drawn[1]])
            self.map_view.blit()

            self.statusBar().showMessage(
                f"@{snapped[0]:.2f},{snapped[1]:.2f} -- {structure.ident} at "
                f"{s:.0f} m, {distance:.0f} m from where you clicked"
            )
            return

        found = nearest_structure(self.document.dataset, *here, within=reach)

        if found is None:
            self.statusBar().showMessage("no trace within reach of the click")
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

        self.statusBar().showMessage(
            f"{structure.ident} at {s:.0f} m of {structure.length:.0f} -- {said}"
            + ("" if plane is None else f", {plane}")
        )

    # -- the file ----------------------------------------------------------

    def _retitle(self):
        mark = "*" if self.document.dirty else ""

        self.setWindowTitle(f"gSurf - trace editor - {self.document.path.name}{mark}")

    def save(self):
        """Writes the document to the file it came from."""

        return self._write(None)

    def save_as(self):
        name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save the file as", str(self.document.path), "gstruct (*.gstruct)"
        )

        return self._write(name) if name else False

    def _write(self, target):
        try:
            written = self.document.save(target)
        except OSError as err:
            QtWidgets.QMessageBox.critical(self, "Not written", str(err))
            return False

        self._retitle()
        self.statusBar().showMessage(f"written to {written}")

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

    fit_to_screen(window, 1340, 880)

    return window
