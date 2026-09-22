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
"""

from __future__ import annotations

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

PANEL_MIN_PX = 130
PANEL_WIDTH_PX = 430

# What the satellites come up as, the first time and nothing being remembered.
# The section is wider and taller than the dock it replaces because it no
# longer has to leave room for a map underneath it: 520 px is three panels of
# a bundle before the scroll area has anything to do.
SECTION_WINDOW_PX = (900, 520)
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

        self.polygons, self.lines, self.overlay_dropped = self._overlay_geometry()

        self.setWindowTitle(f"gSurf - sections - {session.label}")
        self._build_ui(legend)
        self._draw_base_map()

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

        self.section_window = SatelliteWindow(
            "gSurf - section", scroll, SECTION_WINDOW_PX, parent=self
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

        self.count_spin = QtWidgets.QSpinBox()
        self.count_spin.setRange(1, 41)
        self.count_spin.setSingleStep(2)
        self.count_spin.setValue(self.num_profiles)
        self.count_spin.setToolTip(
            "How many parallel profiles the bundle holds. Recomputed on "
            "release, not while dragging."
        )
        self.count_spin.valueChanged.connect(self._on_count_changed)

        self.offset_spin = QtWidgets.QDoubleSpinBox()
        self.offset_spin.setRange(10.0, 20000.0)
        self.offset_spin.setSingleStep(100.0)
        self.offset_spin.setDecimals(0)
        self.offset_spin.setSuffix(" m")
        self.offset_spin.setValue(self.offset)
        self.offset_spin.setToolTip("Spacing between the parallel profiles.")
        self.offset_spin.valueChanged.connect(self._on_offset_changed)

        bar.addWidget(QtWidgets.QLabel("  profiles "))
        bar.addWidget(self.count_spin)
        bar.addWidget(QtWidgets.QLabel("  spacing "))
        bar.addWidget(self.offset_spin)

        if self.traces is not None:
            self.reach_spin = QtWidgets.QDoubleSpinBox()
            self.reach_spin.setRange(0.0, 50000.0)
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

        front = QtGui.QAction("Bring all to &front", self)
        front.triggered.connect(self._raise_group)
        menu.addAction(front)

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

    def closeEvent(self, event):
        # Saved on the way out rather than as each window moves: the arrangement
        # worth keeping is the one the work ended on, and a window dragged
        # across a screen would otherwise write settings on every frame of it.
        self.save_geometry()

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
        self._draw_reach()

        self.map_view.refresh_legend()

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
            polygon_intersections=self._polygon_colors(),
            polygon_intersections_legend=False,
        )

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


def _dock_style():
    """
    Fault attitudes at a size a dock can show.

    The library's defaults were chosen for a figure the width of a page: a
    linewidth of 10 draws a dip tick as a band rather than a line.

    `segment_scale_factor` is a **divisor** -- `create_segment_for_plot` takes
    `profile_length / factor` -- so the number goes up to make the tick
    shorter, not down. The default of 3 puts a tick a third of the section long
    across it; 20 makes it a twentieth, which on a seven-kilometre section is
    360 m: long enough to read an angle off, short enough to sit at the place
    it is reporting.
    """

    from geogst.plots.parameters import LineAttitudePlotParams

    return LineAttitudePlotParams(
        color="black",
        width=1.5,
        alpha=0.9,
        markersize=4,
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
