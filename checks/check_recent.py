"""
Do the answers survive the run they were given in?

The launcher already carried them from tool to tool. What is checked here is
the part that outlives a close: that an answer goes into the store, that a new
launcher built on that store asks its first question with the slots already
filled, and that the three things which should *not* be remembered are not --
a cancelled dialog, an empty slot, and anything at all while the checks run.

That last one is the reason this file can exist at all. `Recent` takes the
QSettings it works on, so a check gets one in a temporary directory and never
touches the one this machine keeps; `Recent.load()` in a real run takes the
machine's, and off-screen takes none. Both paths are exercised below, because
the interesting failure is the one where a check starts passing for a reason
that belongs to whoever ran it.

The dead-entry cases are the ones worth having. A remembered file can be gone,
which the store sees by looking; or still there and no longer readable, which
only the picker finds out, and which has to leave the slot empty without
opening a warning box over a dialog nobody has finished reading yet.

    QT_QPA_PLATFORM=offscreen python check_recent.py
"""

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(os.environ.get("GSURF_REPO", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(REPO))

from check_launcher import attitude_layer  # noqa: E402
from synthetic import synthetic_dem  # noqa: E402

FAILURES = []


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}{'   ' + detail if detail else ''}")
    if not condition:
        FAILURES.append(label)


def store_in(directory, name="recent.ini"):
    """A settings file of this check's own, so the machine's is never touched."""

    from PyQt6 import QtCore

    return QtCore.QSettings(
        str(Path(directory) / name), QtCore.QSettings.Format.IniFormat
    )


def dismiss_next_dialog(tries=200):
    """
    Cancels whatever modal opens next, so that `ask` can be called from here.

    `ask` builds its dialog inside itself and blocks on `exec`, which starts an
    event loop and keeps it running until something answers -- off-screen very
    much included, there being no relation between having a screen and having a
    loop. `check_launcher` sidesteps this by replacing `ask` wholesale, but what
    is being checked here lives *inside* `ask`: the dead entry is dropped there,
    and a cancelled answer has to get past `exec` to prove it recorded nothing.

    So the cancel is queued before the call. The timer first fires inside that
    loop, which is the one moment the dialog exists and can be reached through
    `activeModalWidget`; it retries because the first tick can land between the
    loop starting and the dialog being shown.
    """

    from PyQt6 import QtCore, QtWidgets

    state = dict(left=tries)

    def tick():
        modal = QtWidgets.QApplication.activeModalWidget()

        if modal is not None:
            modal.reject()
            return

        state["left"] -= 1

        # Giving up rather than spinning: a dialog that never opens is a
        # failure to report, not a reason to hang the whole suite.
        if state["left"] > 0:
            QtCore.QTimer.singleShot(5, tick)

    QtCore.QTimer.singleShot(0, tick)


def main():
    from PyQt6 import QtWidgets

    from gsurf.launcher import Launcher
    from gsurf.recent import DEPTH, Recent, default_settings
    from gsurf.sources import SourcesDialog
    from gsurf.tools import TOOLS, load

    app = QtWidgets.QApplication(sys.argv)

    entries = {entry["name"]: entry for entry in TOOLS}
    modules = {name: load(entry) for name, entry in entries.items()}

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        dem_path = str(synthetic_dem(directory / "synthetic.tif", side=1200))
        other_dem = str(synthetic_dem(directory / "another.tif", side=800))
        layer_path = str(attitude_layer(directory))

        attitudes = dict(
            path=layer_path,
            role="points",
            layer="giaciture",
            dip_dir_field="Immersione",
            dip_field="Inclinazione",
            is_rhr_strike=False,
        )

        # -- nothing is written while the checks run -------------------------

        print("\n-- off-screen, the store stays out of it --")

        check(
            "off-screen there is no settings file to read",
            default_settings() is None,
            "QT_QPA_PLATFORM=" + os.environ.get("QT_QPA_PLATFORM", "(unset)"),
        )

        blank = Recent.load()
        blank.remember(dict(dem=dem_path))

        check("so a remembered answer goes nowhere", blank.proposed() == {})
        check(
            "and a launcher built the ordinary way starts empty",
            Launcher().chosen == {},
        )

        # -- one run to the next ---------------------------------------------

        print("\n-- what one run leaves the next --")

        recent = Recent(store_in(directory))
        recent.remember(dict(dem=dem_path, attitudes=attitudes))

        again = Recent(store_in(directory))

        check(
            "the DEM comes back in a store built from scratch",
            again.proposed().get("dem") == dem_path,
        )
        check(
            "and the attitudes come back whole, fields and convention and all",
            again.proposed().get("attitudes") == attitudes,
            str(again.proposed().get("attitudes")),
        )

        launcher = Launcher(recent=again)

        check(
            "a launcher opens on them without being handed anything",
            launcher.chosen.get("dem") == dem_path,
        )

        dialog = SourcesDialog(wants=modules["Plane on a DEM"].WANTS, chosen=launcher.chosen)

        check("so the first dialog of the run is already filled", dialog.boxes["dem"].value() == dem_path)
        check("and nothing was refused", dialog.refused == [])
        check(
            "Open is offered straight away",
            dialog.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Open).isEnabled(),
        )

        # -- the list, newest first -------------------------------------------

        print("\n-- the list, and what it keeps --")

        recent.remember(dict(dem=other_dem))

        check("the newest answer is proposed", recent.proposed()["dem"] == other_dem)
        check(
            "and the one before it is still on the list",
            recent.entries("dem") == [other_dem, dem_path],
            str([Path(p).name for p in recent.entries("dem")]),
        )

        recent.remember(dict(dem=dem_path))

        check(
            "naming one again moves it up rather than doubling it",
            recent.entries("dem") == [dem_path, other_dem],
            str([Path(p).name for p in recent.entries("dem")]),
        )

        deep = Recent(store_in(directory, "deep.ini"))

        for n in range(DEPTH + 4):
            deep.remember(dict(dem=str(directory / f"nowhere_{n}.tif")))

        check(
            f"the list stops at {DEPTH}",
            len(deep.slots["dem"]) == DEPTH,
            f"{len(deep.slots['dem'])} kept",
        )

        # -- the list, as the dialog shows it ------------------------------------

        print("\n-- the list on screen --")

        listed = Recent(store_in(directory, "listed.ini"))
        listed.remember(dict(dem=dem_path, attitudes=attitudes))
        listed.remember(dict(dem=other_dem))

        shown = SourcesDialog(
            wants=modules["Plane on a DEM"].WANTS,
            chosen=listed.proposed(),
            recent=listed,
        )
        combo = shown.boxes["dem"].path_combo

        check(
            "both DEMs are on the list",
            combo.count() == 2,
            str([combo.itemText(n) for n in range(combo.count())]),
        )
        check(
            "the one being opened on is the one selected",
            combo.currentIndex() == 0 and combo.itemData(0) == other_dem,
            combo.currentText(),
        )
        check(
            "and named by its file rather than its path",
            combo.itemText(0) == Path(other_dem).name,
            combo.itemText(0),
        )

        # Reaching for the older one by hand, which is what the list is for:
        # `activated` is what a click emits, and only a click emits it.
        combo.setCurrentIndex(1)
        combo.activated.emit(1)

        check("picking an older one opens it", shown.boxes["dem"].value() == dem_path)
        check(
            "and it moves to the top without doubling",
            combo.count() == 2 and combo.itemData(0) == dem_path,
            str([combo.itemText(n) for n in range(combo.count())]),
        )

        folds_shown = SourcesDialog(
            wants=modules["Fold axes"].WANTS, chosen=listed.proposed(), recent=listed
        )
        remembered = folds_shown.boxes["attitudes"]

        check(
            "a layer entry keeps its fields for the next time it is picked",
            remembered.path_combo.itemData(0) == attitudes,
            str(remembered.path_combo.itemData(0)),
        )
        check("and is restored whole", remembered.value() == attitudes)

        bare = SourcesDialog(wants=modules["Plane on a DEM"].WANTS)

        check(
            "a slot with no history is the empty box it always was",
            bare.boxes["dem"].path_combo.count() == 0
            and bare.boxes["dem"].value() is None,
        )

        # -- what is not an answer ---------------------------------------------

        print("\n-- what does not get remembered --")

        empty = Recent(store_in(directory, "empty.ini"))
        empty.remember(dict(dem=dem_path, polygons=None, lines=None))

        check("an empty slot is not a choice", "polygons" not in empty.slots)

        standing = Recent(store_in(directory, "cancel.ini"))
        standing.remember(dict(dem=dem_path))

        cancelled = Launcher(recent=standing)

        dismiss_next_dialog()
        answer = cancelled.ask(entries["Plane on a DEM"], modules["Plane on a DEM"].WANTS)

        check("cancelling gives the launcher nothing", answer is None)
        check(
            "and a cancelled question changes nothing",
            standing.entries("dem") == [dem_path],
            str([Path(p).name for p in standing.entries("dem")]),
        )

        # -- entries that have gone bad ------------------------------------------

        print("\n-- a remembered file that is no longer there --")

        vanishing = directory / "vanishing.tif"
        synthetic_dem(vanishing, side=400)

        gone = Recent(store_in(directory, "gone.ini"))
        gone.remember(dict(dem=str(vanishing)))
        vanishing.unlink()

        check("a file that has gone is not offered", gone.entries("dem") == [])
        check("and not proposed", "dem" not in gone.proposed())
        check(
            "but it is still written down, for a drive that comes back",
            len(gone.slots["dem"]) == 1,
        )

        print("\n-- a remembered file that is there and will not open --")

        broken = directory / "broken.tif"
        broken.write_bytes(b"this is not a raster")

        rotten = Recent(store_in(directory, "rotten.ini"))
        rotten.remember(dict(dem=str(broken)))

        check(
            "the store cannot tell, and offers it",
            rotten.entries("dem") == [str(broken)],
        )

        refusing = Launcher(recent=rotten)

        check("so the launcher opens on it", refusing.chosen.get("dem") == str(broken))

        # No warning box can be waited for here: the point of the quiet restore
        # is that there is nothing to dismiss. Off-screen a modal would block
        # this check forever, which is itself the assertion.
        rejected = SourcesDialog(
            wants=modules["Plane on a DEM"].WANTS, chosen=refusing.chosen
        )

        check("the dialog refuses it without a word", rejected.refused == ["dem"])
        check("leaving the slot empty", rejected.boxes["dem"].value() is None)
        check(
            "and still coloured as the thing that is missing",
            not rejected.boxes["dem"].is_filled,
        )

        dismiss_next_dialog()
        refusing.ask(entries["Plane on a DEM"], modules["Plane on a DEM"].WANTS)

        check(
            "and asking again drops it from the list for good",
            rotten.entries("dem") == [] and rotten.slots["dem"] == [],
        )
        check("the slot going with it", "dem" not in refusing.chosen)

        launcher.close()
        cancelled.close()
        refusing.close()

    print()

    if FAILURES:
        print(f"FAILED: {len(FAILURES)}")
        for label in FAILURES:
            print(f"  {label}")
        return 1

    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
