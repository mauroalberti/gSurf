"""
Windows that travel together: a map, and the panels taken out of it.

A tool here is one entry in the taskbar and several windows on the screen. The
map is the tool -- it carries the toolbar, the status bar and the menu -- and a
panel that does not want to live beside it gets a window of its own, parented to
the map. That arrangement asks for four small things, and all four are easy to
get quietly wrong: where the windows were left, who brings them back, what
happens to the application when one of them is the last one closed, and what a
run with no desktop under it should inherit.

The section tool answered them first and this is its code, lifted out of it
unchanged in behaviour. What makes it shared rather than copied is that none of
the answers are about sections: they are about Qt, and about desktops.

**A satellite is flagged `Window` and not left as a floating dock.** Qt makes a
floating `QDockWidget` a `Qt::Tool`, which on X11 stays above its parent and out
of the taskbar -- correct for something you glance at, wrong for something you
put on the other screen and leave there. A dock is still the better answer where
the panel is small and wants to go back into the side, which is why the fold
tool's stereonet is one.

**And it is parented anyway**, for two reasons neither of which shows in the
window: Qt destroys it with the tool, so there is no lifetime to keep track of,
and a window with a parent does not count towards `quitOnLastWindowClosed` --
which matters, because while a tool runs the launcher is hidden underneath it,
and closing an unparented satellite would be the last window closed and would
take the application with it.
"""

from __future__ import annotations

import os

from PyQt6 import QtCore, QtGui, QtWidgets

# The narrowest desktop a satellite is placed on by hand. Below it the window
# manager's own placement is the better guess: on a 1920-wide screen with the
# map maximised there is no arrangement that does not cover it, and a panel
# dropped on top of the map is worse than one wherever the desktop put it.
ROOM_TO_PLACE_PX = 1600


def settings_for(name):
    """
    Where a tool's windows were left, if there is a desktop they were left on.

    Off-screen there is no window manager, no screen to be placed against and no
    session to continue -- and a layout restored there is one real run's
    arrangement pushed into a run that measures pixels. `checks/run.py` works in
    exactly that mode, and a section window last left as a strip would make its
    ink counts meaningless. So off-screen the windows come up at their own size
    and nothing is written back.

    `name` is the tool's, not the group's: one tool's arrangement has nothing to
    say about another's, and the geometry keys inside are shared out by the
    window's own name within the tool.
    """

    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        return None

    return QtCore.QSettings("gSurf", name)


class SatelliteWindow(QtWidgets.QWidget):
    """
    A panel given a window of its own, which is the tool's and not the desktop's.

    The flag and the parent are argued in this module's docstring; what is left
    is the closing.

    **Closing hides.** What is inside is expensive -- a bundle's panels are
    rebuilt only when the section's length moves enough to matter, a table of
    the whole file costs a fill -- and a window shut by accident should not cost
    that. The way back is the Windows menu, and the menu follows the window
    rather than the other way round, so a close from the title bar unticks its
    own box.
    """

    visibility_changed = QtCore.pyqtSignal(bool)

    def __init__(self, title, content, size, parent=None):
        super().__init__(parent)

        self.setWindowFlag(QtCore.Qt.WindowType.Window, True)
        self.setWindowTitle(title)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(content)

        # Asked for, but never larger than the screen it will come up on. What
        # sits at the bottom of a panel is its buttons, and a window taller than
        # the desktop puts them under the edge, where -- unlike the panel's own
        # contents -- there is nothing to scroll to reach them. The desktop this
        # is written on is 1366x741 of usable area, which is less than two of
        # these three windows ask for.
        available = self.screen().availableGeometry()

        self.resize(
            min(size[0], available.width()), min(size[1], available.height())
        )

    def closeEvent(self, event):
        event.ignore()
        self.hide()

    def hideEvent(self, event):
        super().hideEvent(event)
        self.visibility_changed.emit(False)

    def showEvent(self, event):
        super().showEvent(event)
        self.visibility_changed.emit(True)


class WindowGroup:
    """
    A map and its satellites, kept in step: shown together, saved together.

    Keyed, because geometry is saved and restored under these names, and because
    a tool can have a different number of windows from one run to the next -- the
    section tool opened without traces has two rather than three. `map` is the
    main window's key and is reserved; the rest are whatever the tool calls them.

    What the group does not do is decide when to show anything. The tool's own
    `showEvent` calls `show_satellites`, so that a window built directly -- which
    is how the checks build one -- gets its whole group from `show()`, with
    nothing to remember to do.
    """

    def __init__(self, main, name, satellites, placer=None):
        self.main = main
        self.name = name
        self.windows = {"map": main, **satellites}

        # Where the windows go with nothing remembered about them is the tool's
        # own question where the tool has an answer: three windows that fit
        # nowhere and two that fit side by side are not the same problem.
        self._placer = placer
        self._up = False

    @property
    def satellites(self):
        """The group without the main window, in the order it was given."""

        return {
            name: window for name, window in self.windows.items() if name != "map"
        }

    def settings(self):
        """This tool's settings, or nothing where there is no desktop."""

        return settings_for(self.name)

    # -- the menu ----------------------------------------------------------

    def actions_into(self, menu, labels=None):
        """
        One checkable entry per satellite, keyed as the group is keyed.

        Returned by name so a tool can reach its own entries -- a shortcut, or a
        state to restore -- without walking the menu.
        """

        labels = labels or {}
        made = {}

        for name, window in self.satellites.items():
            action = QtGui.QAction(labels.get(name, name.title()), self.main)
            action.setCheckable(True)

            # Checked because the window is up, or because it is about to be:
            # the menu is built before `show_satellites` has run.
            action.setChecked(window.isVisible() or not self._up)
            action.triggered.connect(
                lambda shown, w=window: self.show_one(w, shown)
            )

            # The window is the authority on whether it is up: closed from its
            # own title bar it has to untick this box itself, or the menu would
            # claim a window that is not there.
            window.visibility_changed.connect(action.setChecked)

            menu.addAction(action)
            made[name] = action

        return made

    def show_one(self, window, shown):
        window.setVisible(shown)

        if shown:
            # Shown is not the same as seen: a window put back from the menu can
            # come up behind the map it was asked for from.
            window.raise_()
            window.activateWindow()

    def raise_all(self):
        for window in self.windows.values():
            if window.isVisible():
                window.raise_()

    # -- on screen ---------------------------------------------------------

    def show_satellites(self):
        """
        Brings the satellites up with the map -- the first time, and only then.

        After that their being on screen is the user's business: a window closed
        on purpose must not come back because the map happened to be
        un-minimised, and `showEvent` fires for that too.
        """

        if self._up:
            return

        self._up = True

        for window in self.satellites.values():
            window.show()

    def restore_geometry(self):
        """
        Puts the group back where it was left, and says whether the map was.

        The map's own answer is the one the caller needs: `fit_to_screen` sizes a
        window that has nothing remembered about it, and calling it over a
        restored geometry would undo the restoring. The satellites have no such
        competition and are simply put back.
        """

        settings = self.settings()

        if settings is None:
            self.place_unremembered()
            return False

        restored = set()

        for name, window in self.windows.items():
            saved = settings.value(f"geometry/{name}")

            if isinstance(saved, QtCore.QByteArray) and window.restoreGeometry(saved):
                restored.add(name)

        # Whatever was not remembered -- a first run, a tool opened with traces
        # for the first time -- still has to be put somewhere.
        if not restored.issuperset(set(self.satellites)):
            self.place_unremembered(skip=restored)

        return "map" in restored

    def place_unremembered(self, skip=()):
        """
        Where a satellite goes with nothing remembered about it.

        Down the right edge of the screen, and only if the screen has the room
        for it -- see `ROOM_TO_PLACE_PX`. This runs once in the life of an
        installation: after it, there is something remembered.

        A tool with an opinion of its own supplies a `placer` and this is not
        what runs; what it is handed is the same `skip`, being the windows that
        were remembered and must not be moved.
        """

        if self._placer is not None:
            self._placer(skip)
            return

        available = self.main.screen().availableGeometry()

        if available.width() < ROOM_TO_PLACE_PX:
            return

        top = available.top() + 40

        for name, window in self.satellites.items():
            if name in skip:
                continue

            window.move(available.right() - window.width() - 20, top)
            top += window.height() + 40

    def save_geometry(self):
        """
        Where the windows ended up, under the names they were given.

        Saved on the way out rather than as each window moves: the arrangement
        worth keeping is the one the work ended on, and a window dragged across
        a screen would otherwise write settings on every frame of it.
        """

        settings = self.settings()

        if settings is None:
            return

        for name, window in self.windows.items():
            settings.setValue(f"geometry/{name}", window.saveGeometry())
