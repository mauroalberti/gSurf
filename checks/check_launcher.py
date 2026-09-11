"""
Does the launcher ask the picked tool's question, and then get out of the way?

The order is the whole design. A tool is picked first and only then is anything
asked, so what the dialog shows, what it marks as required and what it refuses
to open on all come from that tool: the plane is asked for a DEM and never for
attitudes, the fold axes the other way round. What was answered is kept -- the
files go back into the next dialog -- and so is the session, which is handed on
as it is when the answer has not changed and rebuilt when it has.

The height is checked too, and it is not a detail. A dialog cannot be dragged
shorter than the sum of its parts, so a plain stack of boxes on a short screen
puts its own buttons under the bottom edge, where nothing can reach them: what
is asserted here is that it can be shrunk, which is what the scroll area buys.

The field guessing is checked against the names an Italian survey writes --
`Immersione`, `Inclinazione` -- because those are the layers this is used on,
and a picker that has to be told them twice has not saved anybody anything.

    QT_QPA_PLATFORM=offscreen python check_launcher.py
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("GSURF_REPO", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(REPO))

from check_folds import as_layer, cylindrical_fold  # noqa: E402
from synthetic import synthetic_dem  # noqa: E402

FAILURES = []


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}{'   ' + detail if detail else ''}")
    if not condition:
        FAILURES.append(label)


def attitude_layer(directory):
    """A cylindrical fold, laid out over the synthetic DEM's own extent."""

    dip_dirs, dips = cylindrical_fold(120.0, 10.0, n=150, noise_deg=4.0, seed=3)

    rng = np.random.default_rng(11)
    xy = np.c_[
        rng.uniform(600500.0, 605500.0, len(dips)),
        rng.uniform(4414500.0, 4419500.0, len(dips)),
    ]

    return as_layer(directory, "giaciture", xy, dip_dirs, dips)


def main():
    from PyQt6 import QtCore, QtWidgets

    from gsurf.launcher import Launcher
    from gsurf.sources import FILLED_COLOUR, MISSING_COLOUR, SourcesDialog
    from gsurf.tools import TOOLS, load

    app = QtWidgets.QApplication(sys.argv)

    entries = {entry["name"]: entry for entry in TOOLS}
    modules = {name: load(entry) for name, entry in entries.items()}
    open_button = QtWidgets.QDialogButtonBox.StandardButton.Open

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        dem_path = str(synthetic_dem(directory / "synthetic.tif", side=1200))
        layer_path = attitude_layer(directory)

        attitudes = dict(
            path=str(layer_path),
            role="points",
            layer="giaciture",
            dip_dir_field="Immersione",
            dip_field="Inclinazione",
            is_rhr_strike=False,
        )

        # -- what each tool is asked for -----------------------------------

        print("\n-- one question per tool --")

        check(
            "every tool says what it can be given",
            all(hasattr(module, "WANTS") for module in modules.values()),
        )

        plane = SourcesDialog(wants=modules["Plane on a DEM"].WANTS)
        folds = SourcesDialog(wants=modules["Fold axes"].WANTS)

        check("the plane is asked for a DEM", plane.boxes["dem"].level == "required")
        check("and never for attitudes", "attitudes" not in plane.boxes)
        check(
            "the fold axes are asked for attitudes",
            folds.boxes["attitudes"].level == "required",
        )
        check("with the DEM demoted to backdrop", folds.boxes["dem"].level == "optional")
        check(
            "and no points slot to put the same layer in twice",
            "points" not in folds.boxes,
        )
        check(
            "the required slot comes first",
            list(folds.boxes)[0] == "attitudes" and list(plane.boxes)[0] == "dem",
        )

        # -- and how that is shown ------------------------------------------

        print("\n-- required, and whether it is there --")

        dem_box = plane.boxes["dem"]

        check("the required slot says which it is", dem_box.title().endswith("(required)"))
        check("and is coloured while it is empty", MISSING_COLOUR in dem_box.styleSheet())
        check(
            "an optional one is left in the theme's own colours",
            plane.boxes["lines"].styleSheet() == "",
        )
        check("with nothing chosen, Open is refused", not plane.buttons.button(open_button).isEnabled())

        dem_box.set_path(dem_path)

        check("filling it turns the colour", FILLED_COLOUR in dem_box.styleSheet())
        check("and offers Open", plane.buttons.button(open_button).isEnabled())

        folds.boxes["dem"].set_path(dem_path)

        check(
            "a DEM alone is not enough for the fold axes",
            not folds.buttons.button(open_button).isEnabled(),
        )

        # -- a short screen --------------------------------------------------

        print("\n-- fitting on the screen it is opened on --")

        available = app.primaryScreen().availableGeometry().height()

        for name, dialog in (("plane", plane), ("fold axes", folds)):
            smallest = dialog.minimumSizeHint().height()

            check(
                f"the {name} dialog can be dragged shorter than its contents",
                smallest <= 320,
                f"{smallest} px smallest, {dialog.height()} px as opened",
            )
            check(
                f"and the {name} buttons open above the bottom edge",
                dialog.height() <= available,
                f"{dialog.height()} px of {available}",
            )

        # -- the fields, guessed rather than asked twice ---------------------

        print("\n-- the attitude picker --")

        picker = folds.boxes["attitudes"]

        check("the point layer is accepted", picker.set_path(str(layer_path), "giaciture"))
        check(
            "the dip direction is guessed",
            picker.dip_dir_combo.currentText() == "Immersione",
            picker.dip_dir_combo.currentText(),
        )
        check(
            "the dip angle is guessed",
            picker.dip_combo.currentText() == "Inclinazione",
            picker.dip_combo.currentText(),
        )
        check("so the slot is already filled", picker.is_filled)
        check(
            "and read as a dip direction, not a strike",
            picker.value()["is_rhr_strike"] is False,
        )
        check("which is what Open was waiting for", folds.buttons.button(open_button).isEnabled())

        picker.dip_combo.setCurrentText("(choose)")

        check(
            "unsetting a field withdraws it again",
            not folds.buttons.button(open_button).isEnabled(),
        )

        # -- what was answered once ------------------------------------------

        print("\n-- the answers, put back --")

        launcher = Launcher(chosen=dict(dem=dem_path, attitudes=attitudes))
        again = SourcesDialog(wants=modules["Fold axes"].WANTS, chosen=launcher.chosen)

        check("a slot the other tool filled comes back", again.boxes["dem"].value() == dem_path)
        check(
            "and the attitudes come back whole, fields and all",
            again.boxes["attitudes"].value() == attitudes,
            str(again.boxes["attitudes"].value()),
        )

        first = launcher.session_for(dict(dem=dem_path))

        check(
            "an unchanged answer is handed the session it already had",
            launcher.session_for(dict(dem=dem_path)) is first,
        )

        changed = launcher.session_for(dict(attitudes=attitudes))

        check("a changed one gets a session of its own", changed is not first)
        check(
            "which knows where it is with no DEM under it",
            changed.crs is not None and changed.epsg == 25833,
            f"EPSG:{changed.epsg}",
        )

        # -- handing the session to a tool, and back -------------------------

        print("\n-- the tool window --")

        launcher = Launcher()
        launcher.show()

        # The one blocking call stands in for the dialog. Everything else in
        # `start` runs as it does in earnest: the session is opened here, the
        # module is imported here, the window is built and shown here.
        launcher.ask = lambda entry, wants: dict(attitudes=attitudes)
        launcher.start(entries["Fold axes"])
        app.processEvents()

        window = launcher.tool_window
        fold_session = launcher.session

        check("the tool window opened", window is not None)
        check("on the session the launcher opened", window is not None and window.session is fold_session)
        check(
            "reading the attitudes it was framed on",
            window is not None and len(window.attitudes) > 0,
            f"{len(window.attitudes)} attitudes" if window else "",
        )
        check("and the launcher stepped aside", launcher.isHidden())

        window.close()
        app.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        app.processEvents()

        check("closing the tool brings the launcher back", not launcher.isHidden())
        check("with nothing left holding the window", launcher.tool_window is None)
        check("and the session still open underneath", launcher.session is fold_session)

        launcher.ask = lambda entry, wants: dict(dem=dem_path)
        launcher.start(entries["Plane on a DEM"])
        app.processEvents()

        check(
            "the other tool opens on the session its own answer called for",
            launcher.tool_window is not None
            and launcher.session is not fold_session
            and launcher.session.dem is not None,
        )

        launcher.tool_window.close()
        app.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        app.processEvents()

        standing = launcher.session
        launcher.ask = lambda entry, wants: None
        launcher.start(entries["Fold axes"])

        check(
            "cancelling the question starts nothing, and disturbs nothing",
            launcher.tool_window is None and launcher.session is standing,
        )

        launcher.close()

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
