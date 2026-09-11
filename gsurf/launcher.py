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
"""

from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from .sources import SourcesDialog, describe, open_session
from .tools import TOOLS, load


class Launcher(QtWidgets.QMainWindow):
    """The window you come back to: the tools, and what is open behind them."""

    def __init__(self, chosen=None, session=None):
        super().__init__()

        # Every slot ever answered, whichever tool asked. A tool is shown what
        # it wants out of this; the rest stays here for the tool that wants it.
        self.chosen = dict(chosen or {})

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
        layout.addWidget(session_box)
        layout.addStretch(1)

        self.setCentralWidget(central)
        self.setMinimumWidth(460)

        self._describe()

    # -- the session -------------------------------------------------------

    def ask(self, entry, wants):
        """What to open for this tool, filled in from the last answer. None if cancelled."""

        dialog = SourcesDialog(
            self,
            wants=wants,
            chosen=self.chosen,
            title=f"gSurf - {entry['name'].lower()}",
        )

        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None

        chosen = dialog.choices()

        # Remembered for the next tool, which may want what this one did not
        # ask about: only the slots that were on screen can have changed.
        self.chosen.update(chosen)

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

        chosen = self.ask(entry, module.WANTS)

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
