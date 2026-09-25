"""
The tools, and the sources each one is opened on.

Picking the tool first is what decides everything else here. A tool knows what
it can be given -- the plane cannot run without a DEM and has no use for
attitudes, the fold axes the other way round -- so once one is picked, what to
open is a question about four or five named things, with the ones it cannot run
without marked as such. Asked before the tool is known, the same question has
to cover the union of everything any tool might want: a longer dialog that says
less, and that is how this started.

What has been answered is kept, slot by slot, and put back the next time: going
from one tool to the other re-proposes the files already named rather than
asking for them again. The session is kept as well, and reopened only when the
answer has actually changed -- opening a DEM is the slow part of starting a
tool, and running the same one twice on the same data should not pay it twice.

That keeping now outlives the run, through `recent`. The launcher is the only
place it has to happen: all three tools are started from here and all three ask
through the same dialog, so a store wired in at this one point is a store every
tool has. What comes back from it seeds `chosen`, which is the same slot-by-slot
dictionary the tools were already being handed -- from below, nothing has
changed except that the first dialog of a run is no longer empty.
"""

from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from .recent import Recent
from .sources import SourcesDialog, describe, open_session
from .tools import TOOLS, load


class Launcher(QtWidgets.QMainWindow):
    """The window you come back to: the tools, and what is open behind them."""

    def __init__(self, chosen=None, session=None, recent=None):
        super().__init__()

        # What earlier runs left. Handed in by the checks, which want one that
        # is theirs; taken from this machine otherwise, and off-screen that is
        # an empty one that never writes.
        self.recent = recent if recent is not None else Recent.load()

        # Every slot ever answered, whichever tool asked. A tool is shown what
        # it wants out of this; the rest stays here for the tool that wants it.
        # Seeded from the store, so the first tool of a run is asked the same
        # question the last tool of the last run was.
        self.chosen = dict(chosen) if chosen else self.recent.proposed()

        # The session, and the subset of the choices it was opened on -- which
        # is what says whether it can be handed to the next tool as it is.
        self.session = session
        self.session_chosen = dict(self.chosen) if session is not None else None

        self.tool_window = None

        self.setWindowTitle("gSurf")

        self.summary_label = QtWidgets.QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )

        tools_box = QtWidgets.QGroupBox("Tools")
        tools_layout = QtWidgets.QVBoxLayout(tools_box)

        for entry in TOOLS:
            # The ellipsis is the promise that picking this asks before it runs.
            button = QtWidgets.QPushButton(f"{entry['name']}...")
            button.setMinimumHeight(34)
            button.clicked.connect(lambda _, e=entry: self.start(e))

            caption = QtWidgets.QLabel(entry["summary"])
            caption.setWordWrap(True)
            caption.setStyleSheet("color: gray; font-size: 10px;")
            caption.setIndent(4)

            tools_layout.addWidget(button)
            tools_layout.addWidget(caption)

        session_box = QtWidgets.QGroupBox("Session")
        session_layout = QtWidgets.QVBoxLayout(session_box)
        session_layout.addWidget(self.summary_label)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.addWidget(tools_box)
        layout.addWidget(self._import_box())
        layout.addWidget(session_box)
        layout.addStretch(1)

        self.setCentralWidget(central)
        self.setMinimumWidth(460)

        self._describe()

    # -- getting data in at all --------------------------------------------

    def _import_box(self):
        """
        The import, which is here because it is not a tool and has no session.

        It draws no map and returns no window: it reads a layer, asks what its
        columns mean, and writes a file. Put among the tools it would be the one
        button that does not lead to a map, and the launcher's whole shape is
        that picking a tool decides what it then asks for.

        It will open a DEM if one is named, to read a plane off the topography
        along each trace -- but for the file being written and not for a session
        to be held open on, which is the distinction that keeps this out of the
        tools box rather than the absence of a raster.

        It sits here rather than nowhere because the editor's slot takes a
        `.gstruct` and nothing else, which is correct and which left a mapped
        layer with no way in at all -- and a conversion nobody can reach is a
        conversion that does not exist.
        """

        box = QtWidgets.QGroupBox("Import")
        layout = QtWidgets.QVBoxLayout(box)

        button = QtWidgets.QPushButton("Lines to .gstruct...")
        button.setMinimumHeight(28)
        button.clicked.connect(self.import_lines)

        caption = QtWidgets.QLabel(
            "Transcribe a line layer into the format the trace editor reads, "
            "with attitudes attached and planes read off a DEM if you name them."
        )
        caption.setWordWrap(True)
        caption.setStyleSheet("color: gray; font-size: 10px;")
        caption.setIndent(4)

        layout.addWidget(button)
        layout.addWidget(caption)

        return box

    def import_lines(self):
        """
        Writes a `.gstruct` from a layer, and remembers it where the editor asks.

        Remembered rather than merely written, which is the whole of why this
        returns a path: the file is new, so it is in no history and no project,
        and the next dialog would offer everything except the thing just made.
        It goes in as a spec and not a bare path because that is the shape the
        `traces` slot restores from -- with no columns named, the format saying
        what everything is being the reason a `.gstruct` needs none.
        """

        from .imports import run

        written = run(self)

        if written is None:
            return None

        spec = dict(
            path=written,
            role="lines",
            layer="structures",
            dip_dir_field=None,
            dip_field=None,
            is_rhr_strike=False,
        )

        self.chosen["traces"] = spec
        self.recent.remember(dict(traces=spec))

        return written

    # -- the session -------------------------------------------------------

    def ask(self, entry, wants, only=None):
        """What to open for this tool, filled in from the last answer. None if cancelled."""

        dialog = SourcesDialog(
            self,
            wants=wants,
            only=only,
            chosen=self.chosen,
            recent=self.recent,
            title=f"gSurf - {entry['name'].lower()}",
        )

        # A remembered choice the dialog could not put back is a dead entry in
        # the store, and it is dropped before it can be offered again. Only
        # ones that got past `Recent.entries`, which means the file is still
        # there and is no longer readable as what it was.
        for slot in dialog.refused:
            self.recent.forget(slot, self.chosen.pop(slot, None))

        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None

        chosen = dialog.choices()

        # Remembered for the next tool, which may want what this one did not
        # ask about: only the slots that were on screen can have changed.
        self.chosen.update(chosen)

        # And for the next run. Cancelling is deliberately above this: a
        # question backed out of was not an answer.
        self.recent.remember(chosen)

        return chosen

    def session_for(self, chosen):
        """
        The session these choices call for, opened unless the live one already is it.

        The old one is closed only once the new one is up: a DEM that turns out
        to be unreadable would otherwise leave the launcher holding nothing,
        with the session it had thrown away for it.
        """

        if self.session is not None and chosen == self.session_chosen:
            return self.session

        session = open_session(chosen)

        if self.session is not None:
            self.session.close()

        self.session = session
        self.session_chosen = dict(chosen)

        print(f"session: {session.summary()}")

        if session.overlay:
            print(f"vectors: {session.overlay.summary()}")

        return session

    def _describe(self):
        """What is open right now, under the tools."""

        if self.session is None:
            self.summary_label.setText(
                "Nothing yet. Pick a tool and it asks for what it needs."
            )
            self.adjustSize()
            return

        lines = [f"<b>{self.session.label}</b>", self.session.summary()]
        lines += describe(self.session_chosen or {})

        self.summary_label.setText("<br>".join(lines))
        self.adjustSize()

    # -- the tools ---------------------------------------------------------

    def start(self, entry):
        """
        Asks what to open, opens it, and hides until the tool window is closed.

        One tool at a time: two windows on one session would share a DEM handle
        and a set of readers, and nothing in them was written to be touched
        from two places. Hiding rather than closing is what keeps this window's
        own state -- and the session it holds -- alive underneath.
        """

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)

        try:
            module = load(entry)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        chosen = self.ask(entry, module.WANTS, getattr(module, "ONLY", None))

        if chosen is None:
            return

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)

        try:
            window = module.build(self.session_for(chosen), chosen)
        except Exception as err:
            QtWidgets.QMessageBox.critical(
                self, entry["name"], f"{type(err).__name__}: {err}"
            )
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self._describe()

        # A tool that has already said why it cannot run -- unusable attitudes,
        # a session in degrees -- returns nothing, having shown the reason.
        if window is None:
            return

        self.tool_window = window

        window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        window.destroyed.connect(self._tool_closed)
        window.show()

        self.hide()

    def _tool_closed(self):
        self.tool_window = None

        self._describe()
        self.show()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event):
        if self.session is not None:
            self.session.close()

        super().closeEvent(event)
