"""
What has been opened before, kept between one run and the next.

The launcher already carries the answers from tool to tool: `chosen` holds
every slot ever filled, and the dialog puts them back. What it could not do was
survive being closed, so a morning started where the first morning did -- naming
the same DEM again, pointing at the same two columns again -- and the work is
almost always the same area for weeks at a time.

So the answers are written down. Per slot and not per session: `chosen` is
already a dictionary keyed by slot, the dialog is already a box per slot, and
keeping a list for each is what lets yesterday's DEM be opened under today's
attitudes without either of them having been foreseen as a pair.

Two things this deliberately does *not* do.

It does not open anything. An entry is a path and the choices made around it,
and it is checked for existence and no further -- `Path.exists()` and not a
`rasterio.open`, because the store is read while the dialog is being built and
a run must not pay for a list of eight rasters to show a list of eight names.
Whether a file is still readable is found out when it is restored, by the
picker, which is where it was always found out.

And it does not write while the checks run. `profiles.remembered` already set
that rule for window geometry, for the reason that applies here word for word:
`checks/run.py` works off-screen, and a check that inherited whatever was last
opened on this machine would be testing the machine. Off-screen the store is
present but empty, and nothing goes back into it -- so the checks start from
the blank slate they were written against, and can still be handed a store of
their own to exercise the remembering itself.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from PyQt6 import QtCore

# How many of each slot to keep. Long enough to hold the two or three areas a
# piece of work moves between, short enough that the list stays a list and not
# a file manager -- past that, Browse is the better way to find something.
DEPTH = 8

# One key per slot under this group, each holding a JSON string. A string
# rather than a QSettings list because QSettings has no faithful round trip for
# a list of dictionaries -- and reads a one-element list back as a scalar,
# which would turn "one DEM remembered" into a different shape from "two".
GROUP = "recent"


def default_settings():
    """
    Where this machine keeps them, or nothing while the checks are running.

    Returning None rather than a scratch file is what makes the off-screen case
    unmistakable: there is no store, so there is nothing to accidentally read.
    """

    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        return None

    return QtCore.QSettings("gSurf", "sources")


def _identity(entry):
    """
    What makes two entries the same choice, so the newer replaces the older.

    The file and the layer inside it, which is finer than the file alone on
    purpose: one geopackage commonly holds the polygons and the faults both,
    and those are two entries and not one. The fields chosen around them are
    not part of it -- naming the same layer again with a different dip column
    is a correction, and a correction should move the entry rather than sit
    beside it.
    """

    if isinstance(entry, str):
        return (entry, None)

    return (entry.get("path"), entry.get("layer"))


def _path_of(entry):
    return entry if isinstance(entry, str) else entry.get("path")


class Recent:
    """
    The lists, one per slot, most recent first.

    Built on a QSettings that can be handed in, which is how the checks get one
    that is theirs: a real run takes `default_settings`, and off-screen that is
    None and every list is empty for as long as the run lasts.
    """

    def __init__(self, settings=None):
        self.settings = settings
        self.slots = self._read()

    @classmethod
    def load(cls):
        return cls(default_settings())

    # -- reading -----------------------------------------------------------

    def _read(self):
        if self.settings is None:
            return {}

        slots = {}

        self.settings.beginGroup(GROUP)

        try:
            for slot in self.settings.childKeys():
                raw = self.settings.value(slot)

                try:
                    entries = json.loads(raw) if raw else []
                except (TypeError, ValueError):
                    # A settings file edited by hand, or written by a version
                    # that shaped this differently. One unreadable slot is not
                    # a reason to refuse to start.
                    continue

                if isinstance(entries, list):
                    slots[slot] = entries
        finally:
            self.settings.endGroup()

        return slots

    def entries(self, slot):
        """
        What has been opened in this slot, newest first, minus what is gone.

        The filtering happens here rather than when writing: a file on a drive
        that was not mounted this morning is not a mistake to be forgotten, it
        is a file on a drive that is not mounted, and it comes back by itself
        when the drive does.
        """

        return [
            entry
            for entry in self.slots.get(slot, [])
            if _path_of(entry) and Path(_path_of(entry)).exists()
        ]

    def proposed(self):
        """
        The most recent of each slot, shaped like `chosen` so it can go straight in.

        Slots whose files have all gone simply do not appear, which is the same
        thing as never having filled them -- the dialog shows an empty box and
        colours it if the tool needs it.
        """

        proposed = {}

        for slot in self.slots:
            entries = self.entries(slot)

            if entries:
                proposed[slot] = entries[0]

        return proposed

    # -- writing -----------------------------------------------------------

    def remember(self, chosen):
        """
        Puts an answer at the front of each slot it filled.

        Only what was actually chosen: a dialog reports every slot it asked
        about, with None for the ones left empty, and an empty slot is not a
        choice to be remembered. Clearing a slot therefore leaves its history
        alone, which is what lets a DEM be taken off the fold axes for one run
        and still be there for the next.

        With no store this does nothing at all, rather than building a list
        that will never be written: "no store" has to mean one thing, and a
        `Recent` that accumulated in memory while persisting nothing would be
        an in-memory history nobody asked for -- carrying answers across the
        tools of one run is `Launcher.chosen`'s job, and it already does it.
        """

        if self.settings is None or not chosen:
            return

        touched = False

        for slot, entry in chosen.items():
            if not entry:
                continue

            wanted = _identity(entry)
            kept = [
                other
                for other in self.slots.get(slot, [])
                if _identity(other) != wanted
            ]

            self.slots[slot] = [entry] + kept[: DEPTH - 1]
            touched = True

        if touched:
            self.save()

    def forget(self, slot, entry):
        """
        Drops one entry, for when restoring it has just failed.

        A file that still exists and can no longer be read -- a truncated
        geopackage, a raster on a share that answers but does not serve -- gets
        past `entries` and fails in the picker. Leaving it in the list would
        offer it again every time.
        """

        if self.settings is None or not entry:
            return

        wanted = _identity(entry)
        kept = [
            other for other in self.slots.get(slot, []) if _identity(other) != wanted
        ]

        if len(kept) != len(self.slots.get(slot, [])):
            self.slots[slot] = kept
            self.save()

    def save(self):
        if self.settings is None:
            return

        self.settings.beginGroup(GROUP)

        try:
            for slot, entries in self.slots.items():
                self.settings.setValue(slot, json.dumps(entries))
        finally:
            self.settings.endGroup()

        self.settings.sync()
