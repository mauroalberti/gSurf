"""
The trace editor: what holds along a trace, and what saving leaves alone.

Two things are worth checking here and they pull in opposite directions.

The first is that the editor writes to the file it opened, which nothing else in
this project does. What makes that safe is that it never rewrites the file: it
replaces the lines of the structure that was edited and leaves every other byte
where it was. That is measured against the real `curation.gstruct`, whose ten
lines of comment are the argument for why five thrusts are `exposed` -- and
which the format's own writer deletes, having nowhere to keep them. A Save that
loses those is the failure this whole arrangement exists to prevent, so the
check asserts both halves: that a whole-file rewrite would lose them, and that
saving an edited block does not.

The second is the picture. Precedence in this format is a computation over
several lines at once, so the synthetic file is built to make all five of its
answers come out at places that can be named. One trace runs due east for a
kilometre with a vertex every hundred metres; a fit covers the whole of it, a
stretch between 200 m and 400 m is rejected, and a compass reading sits at
800 m. So at 100 m the fit answers, at 300 m the refusal does, at 800 m the
measurement does, and at 500 m -- three hundred metres from the reading, past
the reach -- the fit takes it back. Any other answer is a number to explain.

    python check_editor.py
"""

import difflib
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(os.environ.get("GSURF_REPO", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(REPO))

FAILURES = []

AOI = Path("/home/mauro/Documenti/Ricerca/AppenninoMeridionale/gstruct")

# The trace everything exact is measured along: due east, a vertex every 100 m.
X0, Y0 = 600000.0, 4420000.0

SOURCE = """gstruct 0.2
crs EPSG:25833
project "a synthetic file, built so the five answers are known"

# This comment belongs to no block. It sits between the header and the first
# structure, and the point of it is that editing either neighbour leaves it
# exactly here.

structure F001 "Alpha" fid=1
  kind fault
  span certainty * * certain
  span use @600200.00,4420000.00 @600400.00,4420000.00 rejected reason="prova"
  attitude @600800.00,4420000.00 plane 90/30 station=S1 src=field
  fit plane * * 100/40 from=trace-dem verdict=indipendente-dal-versante
  path 11
{alpha}

structure F002 "Beta" fid=2
  kind fault
  fit plane * * 12/88 from=trace-dem verdict=traccia-rettilinea
  path 2
    602000.00 4420000.00
    602500.00 4420000.00

structure F003 "Gamma" fid=3
  kind fault
  span use @603000.00,4420000.00 @603000.00,4420100.00 rejected reason="bordo"
  attitude @603000.00,4420000.00 plane 270/60 station=S2 src=field off=12.5 note="{note}"
  lineation @603000.00,4420070.00 206.6 37.8 station=S2 note="prima generazione"
  lineation @603000.00,4420001.00 333.4 37.8
  lineation @603000.00,4420010.00 90.0 10.0
  path 2
    603000.00 4420000.00
    603000.00 4421000.00

structure F004 "Delta" fid=4
  kind fault
  span certainty * * inferred
  path 2
    605000.00 4420000.00
    605400.00 4420000.00
"""


# The DEM the fit is read off: one plane and nothing else, so what a best fit
# through a trace draped on it comes back as is known from the arithmetic rather
# than from any of this code. `check_imports` builds the same raster for the same
# reason and each check builds its own fixtures, which is the arrangement here --
# a check that can only run with another check's help is not a script.
RELIEF_DIP, RELIEF_DIP_DIR = 30.0, 90.0
CELL = 5.0

# Where the V turns, and so the only stretch of it that can carry an attitude.
APEX_S = (1200.0 ** 2 + 600.0 ** 2) ** 0.5


def plane_dem(directory, name="plane.tif", crs="EPSG:25833"):
    """A DEM dipping 30 degrees due east, over the ground the traces are on."""

    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    cols, rows = int(2700 / CELL), int(1600 / CELL)

    # Cell centres, so a sample anywhere inside a cell is off by at most half a
    # cell of gradient rather than by a systematic half-cell shift.
    x = X0 + (np.arange(cols) + 0.5) * CELL
    z = 2000.0 - np.tan(np.radians(RELIEF_DIP)) * (x - X0)

    where = Path(directory) / name

    with rasterio.open(
        where, "w", driver="GTiff", width=cols, height=rows, count=1,
        dtype="float32", crs=crs,
        transform=from_origin(X0, Y0 + 1400.0, CELL, CELL),
    ) as out:
        out.write(np.repeat(z[None, :], rows, axis=0).astype("float32"), 1)

    return where


def densified(points, step=20.0):
    """A vertex every `step` along the corners, which is what digitising gives."""

    out = []

    for (ax, ay), (bx, by) in zip(points, points[1:]):
        count = max(2, int(((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5 / step))
        out.extend(
            (ax + t * (bx - ax), ay + t * (by - ay))
            for t in (n / count for n in range(count))
        )

    out.append(points[-1])

    return out


def fitted_source():
    """
    Four traces on that plane, one for each thing the button has to do.

    VEE turns once, so exactly the stretch around the bend carries a plane and
    the rest carries none -- which is what makes it the test that a fit is
    written over the stretch that held. ZIG turns everywhere, so its fit runs to
    an end of the path and is written `*`, which is the token nothing may aim at.
    EAST is dead straight and has to come back with nothing at all. TAKEN is a V
    that already carries a fit off a table, for the precedence the panel reports
    and does not settle.
    """

    corners = {
        "VEE": [(X0 + 100.0, Y0), (X0 + 1300.0, Y0 + 600.0), (X0 + 2500.0, Y0)],
        "ZIG": [(X0 + 100.0, Y0 + 700.0)] + [
            (X0 + 200.0 + tooth * 200.0, Y0 + 700.0 + 100.0 * (tooth % 2))
            for tooth in range(6)
        ],
        "EAST": [(X0 + 200.0, Y0 + 1200.0), (X0 + 1800.0, Y0 + 1200.0)],
        "TAKEN": [(X0 + 100.0, Y0 + 300.0), (X0 + 1300.0, Y0 + 900.0),
                  (X0 + 2500.0, Y0 + 300.0)],
    }

    out = ["gstruct 0.2", "crs EPSG:25833", ""]

    for ident, points in corners.items():
        path = densified(points)

        out.append(f'structure {ident} ""')
        out.append("  kind fault")

        if ident == "TAKEN":
            out.append("  fit plane * * 100/40 from=table src=gsurf")

        out.append(f"  path {len(path)}")
        out.extend(f"    {x:.2f} {y:.2f}" for x, y in path)
        out.append("")

    return "\n".join(out)


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}{'   ' + detail if detail else ''}")
    if not condition:
        FAILURES.append(label)


def written(directory, name, text):
    path = Path(directory) / name
    path.write_text(text, encoding="utf8")
    return path


# Gamma's note, out here because it is 67 characters and has to be: the tooltip
# folds a note at `TIP_WRAP`, and a note shorter than that would leave the folding
# untested. Notes in `merid_faults` run to 65, so this is the length real ones
# reach and not an invented extreme.
GAMMA_NOTE = "read on a slickenside and not on the plane itself, which is covered"

# Gamma's three lineations, and which two of them belong to its station. All are
# written on one plane, 270/60, and the numbers are not invented: the first two
# are that plane's rake -45 and rake -135 computed by `geogst.Fault`, so they lie
# on the great circle the net draws and a marker off it would be visible as the
# error it is. The third does not, and does not belong to the station either.
#
# The one that matches is anchored 70 m from the plane it was read with, which is
# the point of it: `off` in `merid_faults` runs to 69.7 m, so two readings made at
# one outcrop can snap to progressives that far apart, and a rule based on
# distance along the trace would miss the pair. The one at 1 m carries no station
# code and is matched by distance, which is the weaker rule and the fallback.
GAMMA_LINEATIONS = [(206.6, 37.8), (333.4, 37.8)]


def source_text():
    alpha = "\n".join(f"    {X0 + n * 100.0:.2f} {Y0:.2f}" for n in range(11))

    return SOURCE.format(alpha=alpha, note=GAMMA_NOTE)


def comment_lines(text):
    return [line for line in text.splitlines() if line.lstrip().startswith("#")]


def changed_lines(before, after):
    """The lines a save added or took away, and nothing about the rest."""

    return [
        line
        for line in difflib.unified_diff(
            before.splitlines(), after.splitlines(), lineterm="", n=0
        )
        if line[:1] in "+-" and line[:3] not in ("+++", "---")
    ]


def main():
    from PyQt6 import QtCore, QtWidgets

    import gstruct
    import mplstereonet
    import numpy as np

    from gsurf.curation import (
        Document,
        nearest_structure,
        place_on,
        point_on,
        provenance_of,
        stretch,
    )
    from gsurf.session import Session
    from gsurf.tools import editor as tool

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841

    with tempfile.TemporaryDirectory() as tmp:
        path = written(tmp, "synthetic.gstruct", source_text())
        original = path.read_text(encoding="utf-8")

        # -- the document, and what it does not touch ---------------------

        print("\n-- the document --\n")

        document = Document(path)

        check("every structure in the text is a structure in the model",
              len(document.blocks) == len(document.dataset.structures) == 4,
              f"{len(document.blocks)} block(s)")

        check("a block is the lines of one structure, geometry and all",
              document.text_of(0).startswith('structure F001 "Alpha"')
              and document.text_of(0).rstrip().endswith("601000.00 4420000.00")
              and "F002" not in document.text_of(0))

        check("and the comment above the first one belongs to neither",
              "This comment belongs to no block" not in document.text_of(0))

        document.save()

        check("saving a file nobody edited gives back the same bytes",
              path.read_text(encoding="utf-8") == original)

        # -- the terminators the file came with ----------------------------

        print("\n-- line endings --\n")

        # The check above is the right assertion against the wrong file: this
        # fixture is a Python literal, so its endings are LF and it cannot see a
        # save that puts LF back where something else was. These are the shapes
        # a text file arrives in that the fixture is not, and before they were
        # written the loss was thirty carriage returns out of thirty.

        def as_saved(name, raw, index=0, added=None):
            """One block spliced back into `raw`, and the bytes that came out."""

            where = Path(tmp) / name
            where.write_bytes(raw)

            spliced = Document(where)
            block = spliced.text_of(index)
            spliced.replace(index, block if added is None else block + "\n" + added)
            spliced.save()

            return where.read_bytes()

        # Named, because a backslash cannot go inside an f-string expression
        # until 3.12 and the floor here is 3.9.
        CR, LF = b"\r", b"\n"
        CRLF = CR + LF

        lf = source_text().encode("utf-8")
        crlf = source_text().replace("\n", "\r\n").encode("utf-8")

        out = as_saved("crlf.gstruct", crlf)
        carriage = out.count(CR)
        stray = out.count(LF) - out.count(CRLF)

        check("a CRLF file comes back CRLF, byte for byte",
              out == crlf,
              f"{carriage} CR and {stray} bare LF, in {len(out)} of {len(crlf)} bytes")

        out = as_saved(
            "crlf-added.gstruct", crlf, added='  span use * * rejected reason="x"'
        )
        now, before = out.count(CRLF), crlf.count(CRLF)

        check("and a line written into it takes the file's ending, not this file's",
              out.count(LF) == now == before + 1,
              f"{now} CRLF where there were {before}, {out.count(LF) - now} bare LF")

        out = as_saved("no-newline.gstruct", lf.rstrip(LF))

        check("a file that ends without a newline is not handed one",
              out == lf.rstrip(LF),
              "it gained one" if out.endswith(LF) else "it ends as it did")

        # Mixed endings are not a hypothetical -- a Windows editor appending to a
        # Unix file makes one -- and they are the case that says whether the
        # terminators are kept line by line or guessed once for the whole file.
        mixed = source_text().replace("\n", "\r\n", 4).encode("utf-8")
        out = as_saved("mixed.gstruct", mixed, index=1)

        check("and in a file of mixed endings the lines nobody edited keep theirs",
              out == mixed,
              f"{out.count(CR)} CR where there were {mixed.count(CR)}")

        # -- a block that will not go in ----------------------------------

        print("\n-- refusals --\n")

        was = document.text_of(0)

        refusals = (
            ("an empty block", ""),
            ("a block that starts with a span", "  span use * * rejected"),
            ("a `structure` line that is indented", "  structure F001"),
            ("two structures in one block", was + "\n\nstructure F009"),
            ("an observation inside one",
             was + "\nobservation S9 @1,2 plane 55/70"),
            ("an anchor that is not one", 'structure F001\n  span use @nope * rejected'),
        )

        for label, text in refusals:
            try:
                document.replace(0, text)
                check(f"{label} is refused", False, "it went in")
            except ValueError as err:
                check(f"{label} is refused", True, str(err)[:58])

        check("and a refused block leaves the document as it was",
              document.text_of(0) == was and not document.dirty)

        # The header a block is read under is not a gate, and `_header` used to
        # say it was. The `+ span` button writes a `use` axis, which is gstruct
        # 0.2, and both curations in the AOI declare 0.1 -- nothing stops it,
        # because `loads` refuses a file ahead of the library and nothing else.
        aged = Document(written(
            tmp, "old.gstruct",
            'gstruct 0.1\ncrs EPSG:25833\n\nstructure F001 "Alpha" fid=1\n'
            '  kind fault\n  path 2\n'
            f'    {X0:.2f} {Y0:.2f}\n    {X0 + 1000.0:.2f} {Y0:.2f}\n'))

        added = (f'  span use @{X0 + 200.0:.2f},{Y0:.2f} '
                 f'@{X0 + 400.0:.2f},{Y0:.2f} rejected reason="due"')
        aged.replace(0, aged.text_of(0) + "\n" + added)

        check("a 0.2 axis goes into a file that declares 0.1, and answers there",
              aged._header().startswith("gstruct 0.1")
              and aged.dataset.structures[0].attitude_at(300.0, 250.0)[1]
              == "rifiutata:due",
              aged.dataset.structures[0].attitude_at(300.0, 250.0)[1])

        # -- what holds along a trace -------------------------------------

        print("\n-- what holds --\n")

        alpha, beta, gamma, _delta = document.dataset.structures

        def said_at(structure, s):
            return structure.attitude_at(s, 250.0)[1]

        check("a fit covering the trace answers where nothing else does",
              said_at(alpha, 100.0) == "fit:indipendente-dal-versante",
              said_at(alpha, 100.0))

        check("a rejected stretch answers with the reason it was rejected for",
              said_at(alpha, 300.0) == "rifiutata:prova", said_at(alpha, 300.0))

        check("a measurement within reach beats the fit under it",
              said_at(alpha, 800.0) == "misurata:S1@0m", said_at(alpha, 800.0))

        check("and past its reach the fit takes it back",
              said_at(alpha, 500.0).startswith("fit:"), said_at(alpha, 500.0))

        check("a fit off a straight trace holds nothing at all",
              said_at(beta, 250.0) == "assente", said_at(beta, 250.0))

        check("a measurement with no fit under it still answers, further off",
              said_at(gamma, 1000.0) == "misurata-lontana:1000m",
              said_at(gamma, 1000.0))

        # The band the panel paints, over the same trace: one sample a metre,
        # so a run's edges are the stretch that was written down.
        runs = tool.runs_of(provenance_of(alpha, samples=1001, max_gap=250.0))
        rejected = [run for run in runs if run[2] == "rifiutata"]

        check("the band's own run of refusal is the stretch in the file",
              len(rejected) == 1
              and abs(rejected[0][0] - 200.0) <= 1.0
              and abs(rejected[0][1] - 400.0) <= 1.0,
              str([(round(a), round(b), kind) for a, b, kind, _ in runs]))

        check("and every class it paints is one the format answers with",
              {run[2] for run in runs} <= set(tool.PROVENANCE_TINT),
              str({run[2] for run in runs}))

        # -- a click, and an anchor ---------------------------------------

        print("\n-- the click --\n")

        found = nearest_structure(document.dataset, X0 + 300.0, Y0 + 50.0)

        check("a click finds the trace under it, and where along it",
              found is not None and found[0] == 0
              and abs(found[1] - 300.0) < 1e-6 and abs(found[2] - 50.0) < 1e-6,
              str(found))

        check("and finds nothing where nothing is within reach",
              nearest_structure(
                  document.dataset, X0 + 300.0, Y0 + 50.0, within=10.0
              ) is None)

        # The anchor is snapped onto the trace and not written where the mouse
        # was: `resolve` projects it back to get `s`, so a point fifty metres
        # off the line would read the same and say something false about where
        # anybody stood.
        s, _ = place_on(alpha.path, X0 + 300.0, Y0 + 50.0)
        snapped = point_on(alpha.path, s)

        check("a picked anchor is snapped onto the trace it names",
              abs(snapped[0] - (X0 + 300.0)) < 1e-6 and abs(snapped[1] - Y0) < 1e-6,
              str(snapped))

        check("a rejected stretch draws over the ground it covers",
              [round(x) for x, _ in stretch(alpha.path, 200.0, 400.0)]
              == [600200, 600300, 600400],
              str(stretch(alpha.path, 200.0, 400.0)))

        # -- the tool -----------------------------------------------------

        print("\n-- the tool --\n")

        check("it declares the file it edits as the thing it cannot run without",
              tool.WANTS["traces"] == "required" and tool.WANTS["dem"] == "optional")

        spec = dict(path=str(path), role="traces")
        session = Session.open(frame_layers=[spec])
        window = tool.build(session, {"traces": spec})

        check("the file opens as a window", window is not None)

        # The precondition the unapplied-work guard rests on: what is in the box
        # and what is in the document are the same string until somebody types.
        check("the box holds the block exactly as the document has it",
              window.panel.text.toPlainText() == window.document.text_of(0))

        # -- three windows, and what keeps them one tool -------------------
        #
        # The map is the tool; the panel and the net are windows of their own,
        # which is the section tool's arrangement and now the same code:
        # `gsurf.windows`. Two things about it are worth holding down and
        # neither shows in the window. Each is parented to the map, which has Qt
        # destroy it with the tool and -- the part that does not show at all --
        # keeps closing it from counting as the last window closed: while a tool
        # runs the launcher is hidden underneath, so an unparented panel would
        # take the application down instead of handing it back. And closing one
        # hides it, because a table of the whole file costs a fill and a window
        # shut by accident should not.

        print("\n-- three windows --\n")

        panel_window = window.panel_window

        check("the map, the panel and the net are three windows",
              sorted(window.group.windows) == ["map", "net", "panel"],
              ", ".join(sorted(window.group.windows)))

        check("the panel is a window in its own right, not a pane of the map",
              panel_window.isWindow() and window.centralWidget() is not window.panel)

        check("and it is parented to the map, so closing it cannot quit the app",
              panel_window.parent() is window)

        check("showing the map brought it up with it", panel_window.isVisible())

        panel_action = window.window_actions["panel"]

        # Closed the way the window manager closes it, not hidden behind its back.
        panel_window.close()

        check("closing it hides it and leaves the table standing",
              not panel_window.isVisible() and window.panel.table.rowCount() == 4)

        check("and the menu stops claiming a window that is not there",
              not panel_action.isChecked())

        panel_action.trigger()

        check("the menu puts it back",
              panel_window.isVisible() and panel_action.isChecked())

        # Both shortcuts used to be the buttons' own, and a button's shortcut
        # reaches only the window the button is in -- which was the only window
        # there was. Ctrl+S over the map is the case that would have gone quiet:
        # clicking anchors along a trace and then writing the file is one motion.
        #
        # The net is in this too, and it is the one nobody types in: it is a
        # canvas, clicking it gives it the keyboard, and a Ctrl+S that depends on
        # which window was last clicked fails silently and only sometimes.
        def shortcuts_of(widget):
            return {
                action.shortcut().toString()
                for action in widget.actions()
                if not action.shortcut().isEmpty()
            }

        check("Save and Apply answer from any of the three windows",
              shortcuts_of(window) == shortcuts_of(panel_window)
              == shortcuts_of(window.net_window) == {"Ctrl+S", "Ctrl+Return"},
              " and ".join(sorted(shortcuts_of(window.net_window))))

        check("and the buttons carry none of their own, so neither is ambiguous",
              window.save_button.shortcut().isEmpty()
              and window.panel.apply_button.shortcut().isEmpty())

        window.say("a line of news")

        check("what the tool says is said in both windows",
              window.statusBar().currentMessage() == window.echo.text()
              == window.echo.toolTip() == "a line of news")

        # The guard that keeps a real desktop's arrangement out of this run:
        # without it a panel last left as a strip would come back as one here.
        check("off-screen there is no layout to inherit",
              window.group.settings() is None
              and window.group.restore_geometry() is False)

        # What a first run makes of a screen, on two screens this run does not
        # have. The laptop this was written on is 1366x741 of usable area, which
        # the group's own rule -- satellites down the right edge, 1600 and up --
        # calls too small to place anything on. Two of these three fit there
        # anyway: 840 of map beside 520 of panel is the arrangement the splitter
        # used to make, and a map maximised under a floating panel would have
        # been worse than what this replaces.
        was_area = window.main_screen_area
        was_map, was_panel = window.geometry(), panel_window.geometry()
        was_net = window.net_window.geometry()

        def placed_on(width, height):
            """The three geometries a screen of this size leads to."""

            window.main_screen_area = lambda: QtCore.QRect(0, 0, width, height)
            window.tiled = False
            window.group.place_unremembered()

            return (
                window.tiled,
                window.geometry(),
                panel_window.geometry(),
                window.net_window.geometry(),
            )

        tiled, on_map, on_panel, on_net = placed_on(1366, 741)

        check("on one laptop screen map and panel are laid side by side",
              tiled and on_map.width() == 840 and on_panel.width() == 520,
              f"{on_map.width()} of map, {on_panel.width()} of panel")

        check("filling it across, with the panel against the right edge",
              on_map.left() == 0
              and on_panel.left() == on_map.right() + 1 + tool.TILE_GAP_PX
              and on_panel.right() == 1365,
              f"map to {on_map.right()}, panel from {on_panel.left()}")

        check("and neither of them taller than the desktop they are on",
              on_map.height() == on_panel.height() == 741 - tool.FRAME_ALLOWANCE_PX,
              f"{on_map.height()} px of 741")

        # There is no third rectangle: two windows fill the screen across at any
        # width, because the panel is a fixed 520 and the map takes the rest. So
        # the net goes over the map, in the corner furthest from the panel, and
        # covers 420x440 of an 840x701 map -- 31% of it. That is the cost of it
        # being readable at all, and it is paid once: this is a window, one drag
        # puts it where it should be, and `save_geometry` keeps it there.
        check("and the net over the map's far corner, clear of the status bar",
              on_net.left() == tool.TILE_GAP_PX
              and on_net.width() == tool.NET_WINDOW_PX[0]
              and on_map.contains(on_net)
              and on_net.bottom()
              < on_map.bottom() - window.statusBar().sizeHint().height(),
              f"net {on_net.width()}x{on_net.height()} at "
              f"({on_net.left()}, {on_net.top()}), map to {on_map.bottom()}")

        narrow, _, _, narrow_net = placed_on(1024, 720)

        check("on a screen too narrow for both, the map is not cut to fit one",
              not narrow, "left to fit_to_screen, and the panel over it")

        # The net is still placed there. It is small enough to land on any
        # screen, and the branch that gives up on tiling gives up before the map
        # has a size -- so a net placed against the map would have nothing to be
        # placed against.
        check("and the net is placed anyway, against the screen",
              narrow_net.left() == tool.TILE_GAP_PX and narrow_net.bottom() < 720,
              f"at ({narrow_net.left()}, {narrow_net.top()}) on 1024x720")

        # Put back, all three: everything below draws on this map, and an extent
        # is read off a canvas whose aspect is the window's. A probe that leaves
        # the window a different shape is a probe that decides what the framing
        # checks are measuring.
        window.main_screen_area = was_area
        window.setGeometry(was_map)
        panel_window.setGeometry(was_panel)
        window.net_window.setGeometry(was_net)

        # -- finding the one to open ---------------------------------------

        print("\n-- the table --\n")

        table = window.panel.table

        check("every structure is a row, not only the ones carrying a plane",
              table.rowCount() == 4, f"{table.rowCount()} row(s)")

        # Qt starts a header's sort indicator on section 0 *descending*, so a
        # table that enables sorting and fills comes out reversed and says
        # nothing about it. This is the assertion that the indicator was set.
        idents = [table.item(row, 0).text() for row in range(4)]

        check("and the rows are in ident order, which Qt does not do by itself",
              idents == ["F001", "F002", "F003", "F004"], str(idents))

        check("`carries` is a plane having been read, not a span having been said",
              tool.carries(alpha) and not tool.carries(_delta)
              and len(_delta.spans) == 1,
              f"delta: {len(_delta.spans)} span(s), "
              f"{len(_delta.attitudes)} attitude(s)")

        window.panel.carrying.setChecked(True)

        # Delta drops out: a mapped contact nobody has read a plane off yet,
        # which is 348 of the 393 traces of the real fault layer. It is still
        # editable -- the filter shortens the list, it does not lock anything.
        shown = [row for row in range(4) if not table.isRowHidden(row)]

        check("the filter leaves the ones something was read on, and counts them",
              len(shown) == 3 and window.panel.shown.text() == "3 of 4",
              window.panel.shown.text())

        # The combo this replaced was rebuilt on every filter change and had to
        # notice when what was open had dropped out of it. A hidden row is still
        # the open row, so there is nothing to notice and nothing to get wrong.
        check("and it hides rather than removes, so what is open stays open",
              table.rowCount() == 4 and window.panel.index == 0
              and table.index_of(table.currentRow()) == 0,
              f"index {window.panel.index}")

        window.panel.carrying.setChecked(False)

        # Lengths as numbers and not as text. 400, 500, 1000, 1000 sorts to
        # "1000", "1000", "400", "500" the other way -- which is the order that
        # hides the long faults at the top of the sort meant to find them.
        table.sortItems(1, QtCore.Qt.SortOrder.AscendingOrder)
        lengths = [int(table.item(row, 1).text()) for row in range(4)]

        check("a length column sorts as numbers", lengths == sorted(lengths),
              str(lengths))

        check("and the structure being worked on survives the sort",
              table.index_of(table.currentRow()) == window.panel.index == 0,
              f"row {table.currentRow()} -> "
              f"{table.index_of(table.currentRow())}")

        # The two lengths are two columns and say which is which in their names.
        # `m` said neither, and a plan length and a draped one differ by a sixth
        # on the real sheet -- 654 km against 613 -- which is not a rounding
        # anybody would spot in a column headed `m`.
        check("the lengths are named, in plan and over the topography",
              tool.StructureTable.COLUMNS[1:3] == ("length_2d", "length_3d"),
              str(tool.StructureTable.COLUMNS))

        # This session has no DEM, and the column is empty rather than zero:
        # zero is a length a trace could have and this is the absence of one.
        # The header says so, which is where a reader looks when a whole column
        # is blank.
        drapes = [table.item(row, 2).text() for row in range(4)]

        check("with no DEM the draped column is blank, and the header says why",
              drapes == ["", "", "", ""]
              and "No DEM" in table.horizontalHeaderItem(2).toolTip(),
              table.horizontalHeaderItem(2).toolTip()[:52])

        table.sortItems(0, QtCore.Qt.SortOrder.AscendingOrder)

        # What the table says in one word, against the same computation done in
        # metres of run rather than in samples. Two different arithmetics that
        # have to agree about which class covers most of the trace.
        kind, fraction = tool.holds_along(alpha, max_gap=250.0)
        widths = Counter()

        for s0, s1, said_kind, _ in tool.runs_of(
            provenance_of(alpha, samples=tool.HOLDS_SAMPLES, max_gap=250.0)
        ):
            widths[said_kind] += s1 - s0

        check("the table's one word is the class covering most of the trace",
              kind == widths.most_common(1)[0][0],
              f"{kind} {fraction:.0%}, against "
              f"{widths.most_common(1)[0][0]} by metre")

        check("and a fit off a straight trace holds nothing along any of it",
              tool.holds_along(beta, max_gap=250.0) == ("assente", 1.0),
              str(tool.holds_along(beta, max_gap=250.0)))

        # -- and on the map ------------------------------------------------

        check("the map draws the ones carrying a plane apart from the ones not",
              len(window.traces[True].get_segments()) == 3
              and len(window.traces[False].get_segments()) == 1,
              f"{len(window.traces[True].get_segments())} carrying, "
              f"{len(window.traces[False].get_segments())} bare")

        # This tool asked for its legend handles and never asked for the legend,
        # so the four it built were made on a rebuild that never came: the window
        # opened with no legend until somebody moved the placement combo. Two
        # weights on the map are worth nothing without the entry naming them.
        entries = (
            [] if window.map_view.legend is None
            else [text.get_text() for text in window.map_view.legend.get_texts()]
        )

        check("and the legend is built, and says which weight is which",
              "carrying a plane" in entries and "mapped, nothing read" in entries,
              str(entries))

        # A click on the northward trace, half way up it.
        window.pick(603000.0, 4420500.0)

        check("a click on the map selects the trace it landed on",
              window.index == 2, f"index {window.index}")

        check("and says what holds there, in the format's own words",
              "misurata-lontana:500m" in window.statusBar().currentMessage(),
              window.statusBar().currentMessage())

        # The template puts the cursor somewhere known; the two shift-clicks
        # fill the two anchors in turn.
        window.panel.add_line('  span use * * rejected reason="check"')
        window.pick(603000.0, 4420200.0, anchor=True)
        window.pick(603000.0, 4420600.0, anchor=True)

        # The line the template added, and not every `span use` in the block:
        # Gamma carries one of its own from the file -- the refusal the hover has
        # to report -- and the anchors being checked are the two this one left
        # open.
        typed = [
            line for line in window.panel.text.toPlainText().splitlines()
            if 'reason="check"' in line
        ]

        check("two picked anchors fill the two the template left open",
              typed == [
                  '  span use @603000.00,4420200.00 @603000.00,4420600.00 '
                  'rejected reason="check"'
              ],
              str(typed))

        check("and what was typed is not in the document until it is applied",
              window.panel.text.toPlainText()
              != window.document.text_of(window.index)
              and not window.document.dirty)

        # An anchor goes on the selected trace whatever it was aimed at, which is
        # the rule that keeps a progressive off a fault nobody measured -- and the
        # rule the dots put at risk, now that a green dot on a neighbour is a
        # visible thing to aim at. Gamma is open; this click is 10 m from Beta and
        # 1200 m from Gamma, and it is written on Gamma, correctly and silently.
        window.pick(602200.0, 4420010.0, anchor=True)
        aimed = window.statusBar().currentMessage()

        check("an anchor aimed nearer another trace still goes on the open one",
              "F003 at" in aimed and "F002 is nearer" in aimed, aimed)

        # And the same click aimed at the trace it is on says nothing extra, or the
        # warning would be furniture rather than a warning.
        window.pick(603000.0, 4420300.0, anchor=True)
        aimed = window.statusBar().currentMessage()

        check("and one aimed at the open trace is confirmed and not warned about",
              "F003 at" in aimed and "is nearer" not in aimed, aimed)

        window.panel._redraw()
        window.panel.add_line('  span use * * rejected reason="check"')
        window.pick(603000.0, 4420200.0, anchor=True)
        window.pick(603000.0, 4420600.0, anchor=True)

        check("applying it reads it in", window.panel.apply_block())

        after_edit = window.document.dataset.structures[2]
        answers = {
            kind for _, _, _, kind in provenance_of(after_edit, samples=201)
        }

        check("and the refusal is in force along the ground it named",
              after_edit.attitude_at(400.0, 250.0)[1] == "rifiutata:check"
              and "rifiutata" in answers,
              str(sorted(answers)))

        check("the map has the rejected stretch to draw now",
              len(window.refused.get_xdata()) > 0,
              f"{len(window.refused.get_xdata())} point(s)")

        # Where the highlight is, and not only that there is one. Both of these
        # read `[y for y, _ in drawn]` from the day the tool was written -- which
        # binds the *first* of the pair, whatever the name -- so the selected
        # trace and the dots marking the measurements were drawn at
        # (easting, easting): off the map, on a diagonal no extent here covers.
        # Selecting a trace worked perfectly and showed nothing, and a count of
        # points was true the whole time.
        highlight = list(
            zip(window.highlight.get_xdata(), window.highlight.get_ydata())
        )
        trace = window.document.dataset.structures[window.index].path

        check("and the highlight is drawn along the trace, not beside it",
              [(round(x, 2), round(y, 2)) for x, y in highlight]
              == [(round(x, 2), round(y, 2)) for x, y in trace],
              f"{highlight[:1]} against {trace[:1]}")

        marks = list(zip(window.marks.get_xdata(), window.marks.get_ydata()))
        gamma_at = point_on(trace, gamma.attitudes[0].s)

        check("and the measured dot sits where the plane was measured",
              len(marks) == 1
              and abs(marks[0][0] - gamma_at[0]) < 0.01
              and abs(marks[0][1] - gamma_at[1]) < 0.01,
              f"{marks} against {[gamma_at]}")

        # And every other station in the file is drawn too, which is what makes the
        # map answer "where has anything been read" once rather than 393 times.
        # Alpha's S1 is on the map while Gamma is the fault open in the box.
        everywhere = list(zip(window.stations.get_xdata(), window.stations.get_ydata()))
        alpha = window.document.dataset.structures[0]
        alpha_at = point_on(alpha.path, alpha.attitudes[0].s)

        check("every station in the file has a dot, not only the open fault's",
              len(everywhere) == 2
              and any(
                  abs(x - alpha_at[0]) < 0.01 and abs(y - alpha_at[1]) < 0.01
                  for x, y in everywhere
              ),
              f"{everywhere} against {[alpha_at, gamma_at]}")

        # Drawn twice where they are the selection's, larger on top. Size and not
        # hue, because every hue on this map already means something -- and the two
        # artists have to land on the same point or the pair reads as two stations
        # a metre apart rather than as one.
        check("the open fault's own are the same dots, drawn bigger over the rest",
              window.stations.get_markersize() < window.marks.get_markersize()
              and window.stations.get_color() == window.marks.get_color()
              and any(
                  abs(x - gamma_at[0]) < 0.01 and abs(y - gamma_at[1]) < 0.01
                  for x, y in everywhere
              ),
              f"{window.stations.get_markersize()} under "
              f"{window.marks.get_markersize()}, "
              f"{window.stations.get_color()} and {window.marks.get_color()}")

        # In the background with the traces, because they change when a block is
        # applied and at no other time. Which means an applied block has to hand
        # them over again -- an `attitude` line added to Beta is a dot that was not
        # there, on a fault that had none.
        window.select(1)
        was = window.document.text_of(1)
        window.panel.text.setPlainText(
            was.rstrip("\n")
            + "\n  attitude @602100.00,4420000.00 plane 10/20 station=S9 src=check"
        )
        window.panel.apply_block()

        check("a block that adds a station adds its dot, the file over again",
              len(window.stations.get_xdata()) == 3
              and len(window.marks.get_xdata()) == 1,
              f"{len(window.stations.get_xdata())} in the file, "
              f"{len(window.marks.get_xdata())} on the open fault")

        window.panel.text.setPlainText(was)
        window.panel.apply_block()

        check("and taking the line back takes the dot back with it",
              len(window.stations.get_xdata()) == 2
              and len(window.marks.get_xdata()) == 0,
              f"{len(window.stations.get_xdata())} in the file, "
              f"{len(window.marks.get_xdata())} on the open fault")

        window.select(2)

        # -- what the cursor is resting on ---------------------------------

        print("\n-- the hover --\n")

        # Every probe below is in screen pixels, so the map has to have a data
        # area for them to be pixels of. This is asserted and not assumed because
        # for a while it was false: the net was a dock taking 276 px out of the
        # window, and under about 1000 px of window the map's constrained layout
        # gave up -- matplotlib says `axes sizes collapsed to zero` in a warning
        # nothing reads -- and put the whole map into 24 px. Every dot was then
        # within 10 px of every other and the reach checks passed on nonsense,
        # which is how this was found. The net is a window now and the map has
        # its width back: the number this prints went from 600.7 px to 885.4.
        window.resize(1280, 900)
        QtWidgets.QApplication.processEvents()
        window.map_view.canvas.draw()

        axes = window.map_view.axes

        check("the map has a data area for a reach in pixels to mean anything",
              axes.get_window_extent().width > 400.0,
              f"{axes.get_window_extent().width:.1f} px wide")

        kept_view = (axes.get_xlim(), axes.get_ylim())

        # Gamma's, found by asking which fault each dot is on rather than by
        # position. `_marked` is every station in the file now and Alpha's S1 comes
        # first in it, so the dot at index 0 is no longer the dot of the fault that
        # happens to be selected.
        at_gamma = next(
            where for where, (_, _, whose, _) in enumerate(window._marked)
            if whose == 2
        )
        dot = window._marked[at_gamma][0]

        window._on_map_hover(*dot)
        tip = window.map_view.canvas.toolTip().splitlines()

        check("resting on a station dot names the station and the plane",
              tip[:1] == ["S2 -- 270/60"], str(tip[:1]))

        check("and says how far the reading was from the line it is drawn on",
              any("12.5 m off the trace" in line for line in tip), str(tip))

        # The one thing the dot cannot show about itself. It is drawn because
        # somebody stood there, which stays true; it is not what holds, and until
        # this the band in the panel was the only thing that said so -- and the
        # panel is a different window now.
        check("and that the curation overrules it here, which the dot does not",
              any("overruled here -- rifiutata:bordo" in line for line in tip),
              str(tip))

        check("and folds the note rather than opening a tooltip wider than the map",
              GAMMA_NOTE not in tip
              and " ".join(tip[-2:]) == GAMMA_NOTE
              and max(len(line) for line in tip) <= tool.TIP_WRAP,
              f"longest line {max(len(line) for line in tip)}")

        window._on_map_hover(dot[0] + 5000.0, dot[1])

        check("and the cursor five kilometres away is resting on nothing",
              window.map_view.canvas.toolTip() == "",
              repr(window.map_view.canvas.toolTip()))

        # A reach in metres would be a target a kilometre wide framed on the AOI
        # and unhittable framed on one fault. Two framings three orders of
        # magnitude apart, and the answer has to follow the pixels both times.
        def metres_per_pixel():
            inverse = axes.transData.inverted()
            (x0, _), (x1, _) = inverse.transform([(0.0, 0.0), (1.0, 0.0)])

            return abs(x1 - x0)

        def reach_at(half_width):
            axes.set_xlim(dot[0] - half_width, dot[0] + half_width)
            axes.set_ylim(dot[1] - half_width, dot[1] + half_width)
            window.map_view.canvas.draw()

            scale = metres_per_pixel()

            return (
                scale,
                window._station_near(dot[0] + (tool.HOVER_RADIUS_PX - 1) * scale, dot[1]),
                window._station_near(dot[0] + (tool.HOVER_RADIUS_PX + 2) * scale, dot[1]),
                window._station_near(dot[0] + 400.0, dot[1]),
            )

        close_in, far_out = reach_at(500.0), reach_at(50000.0)

        check("the reach is in pixels: just inside it hits at either framing",
              close_in[1] == at_gamma and far_out[1] == at_gamma,
              f"{close_in[1]} at {close_in[0]:.1f} m/px, "
              f"{far_out[1]} at {far_out[0]:.1f} m/px")

        check("and just outside it misses at either framing",
              close_in[2] is None and far_out[2] is None,
              f"{close_in[2]}, {far_out[2]}")

        check("while a fixed 400 m is a miss zoomed in and a hit zoomed out",
              close_in[3] is None and far_out[3] == at_gamma,
              f"{close_in[3]} at {close_in[0]:.1f} m/px, "
              f"{far_out[3]} at {far_out[0]:.1f} m/px")

        axes.set_xlim(*kept_view[0])
        axes.set_ylim(*kept_view[1])
        window.map_view.canvas.draw()

        # Held on the dot across a change of structure. The tooltip is dropped on a
        # selection because the cursor has not moved and the answer may have: what
        # it said a moment ago was about a fault that was open and is not now.
        window._on_map_hover(*dot)
        named = window.map_view.canvas.toolTip().splitlines()[:1]
        window.select(0)

        check("opening another trace drops the tooltip the cursor is still on",
              named == ["S2 -- 270/60"] and window.map_view.canvas.toolTip() == "",
              f"{named} then {window.map_view.canvas.toolTip()!r}")

        # And the dot is still there to ask again, which is the change: it used to
        # be swept off the map with its fault, so there was nothing under the
        # cursor to answer. What it answers is the same station and one line more.
        window._on_map_hover(*dot)
        again = window.map_view.canvas.toolTip().splitlines()

        check("but the dot is still on the map, and says whose trace it is on",
              again[:2] == [
                  "S2 -- 270/60",
                  "0 m along F003, which is not the selected trace",
              ],
              str(again[:2]))

        # The one line in the tooltip that named a trace named `self.index` -- so
        # before the dots were global it was right by accident, and the moment they
        # were it would have put the open fault's name on another fault's station.
        window.select(2)
        window._on_map_hover(*dot)

        check("and drops the caveat once that trace is the one selected",
              window.map_view.canvas.toolTip().splitlines()[1] == "0 m along F003",
              str(window.map_view.canvas.toolTip().splitlines()[1]))

        # One motion event, two meanings. The gesture had this event to itself
        # until now, and a hover firing while a handle is being dragged would put
        # a tooltip over the thing being moved.
        class Motion:
            def __init__(self, inaxes, x, y):
                self.inaxes, self.xdata, self.ydata = inaxes, x, y

        seen = Counter()
        window.map_view.dragged.connect(lambda x, y: seen.update(["dragged"]))
        window.map_view.hovered.connect(lambda x, y: seen.update(["hovered"]))
        window.map_view.hover_off.connect(lambda: seen.update(["off"]))

        window.map_view._pressing = True
        window.map_view._on_motion(Motion(axes, *dot))
        window.map_view._on_motion(Motion(None, None, None))
        window.map_view._pressing = False

        check("a moved cursor with the button down is a drag and not a hover",
              seen["dragged"] == 1 and seen["hovered"] == 0,
              f"{dict(seen)}")

        window.map_view._on_motion(Motion(axes, *dot))
        window.map_view._on_motion(Motion(None, None, None))

        check("and free it is a hover, and leaving the axes says so",
              seen["hovered"] == 1 and seen["off"] == 1, f"{dict(seen)}")

        window.map_view.toolbar.mode = "pan/zoom"
        window.map_view._on_motion(Motion(axes, *dot))
        window.map_view.toolbar.mode = ""

        check("and with a navigation mode on there is no hover to have",
              seen["hovered"] == 1 and seen["off"] == 2, f"{dict(seen)}")

        window._tip_on(None)

        # -- the planes on the net -----------------------------------------
        #
        # The figure the tooltip cannot be. `270/60` written out is a plane you
        # have to picture; the great circle is the picture, and the thing neither
        # number shows is where a striation sits inside the plane -- down the dip
        # or along the strike. And the thing no *pair* of numbers shows is whether
        # two planes are the same surface, which is why the net carries the whole
        # of one fault and not the one reading under the cursor.

        print("\n-- the net --\n")

        window.select(2)

        check("selecting a fault puts its plane on the net, no cursor needed",
              len(window.net.measured.get_xdata()) > 2,
              f"{len(window.net.measured.get_xdata())} points on the circle")

        # And that is all of the plane that is drawn. Its pole was there too at
        # first, on the argument that a pole is how this net would be compared
        # with the fold tool's -- but that is a reason to draw one where there is
        # a population to see the shape of. Here the markers inside the primitive
        # circle are the striae, which is what the picture is read for, and a pole
        # is a mark inside that circle which is not a striation.
        check("and nothing else inside the circle, the pole having been dropped",
              len(window.net.poles.get_xdata()) == 0,
              f"{len(window.net.poles.get_xdata())} poles on one fault's planes")

        check("and the window says whose planes they are, the net having no label",
              window.net_window.windowTitle()
              == f"{tool.NET_TITLE} - F003 -- 1 measured, 2 lineation(s)",
              window.net_window.windowTitle())

        # Alpha is the case the whole change is for: a reading and a fit on one
        # fault, which in `merid_faults` is six of the forty-five that carry
        # anything. Solid and dashed, because one is a plane somebody put a compass
        # on and the other is a surface least-squares fitted to a scatter, and a
        # picture that drew them alike would invite them to be read as the same
        # kind of claim -- on F0074 the two disagree by 86 degrees in dip
        # direction and the file says why in a `caveat`.
        window.select(0)

        check("a fault carrying both draws the reading and the fit together",
              len(window.net.measured.get_xdata()) > 2
              and len(window.net.fitted.get_xdata()) > 2,
              f"{len(window.net.measured.get_xdata())} measured points, "
              f"{len(window.net.fitted.get_xdata())} fitted")

        check("and the two are told apart by the line and not by the colour alone",
              window.net.measured.get_linestyle() == "-"
              and window.net.fitted.get_linestyle() != "-"
              and window.net.measured.get_color() != window.net.fitted.get_color(),
              f"measured {window.net.measured.get_linestyle()!r} "
              f"{window.net.measured.get_color()}, fitted "
              f"{window.net.fitted.get_linestyle()!r} "
              f"{window.net.fitted.get_color()}")

        check("the title counts both kinds",
              window.net_window.windowTitle()
              == f"{tool.NET_TITLE} - F001 -- 1 measured, 1 fitted",
              window.net_window.windowTitle())

        # Beta is the other half of it, and the measurement that decided the
        # question. It carries a fit and no reading, so there is no station dot on
        # it and under the old rule -- the net filled by resting on a dot -- its
        # net was blank and stayed blank. Twenty-nine faults of `merid_faults` are
        # this shape: a net that showed only readings was empty on 373 selections
        # out of 393, and counting the fits that falls to 348.
        window.select(1)

        check("a fault with a fit and no reading is no longer a blank net",
              len(window.net.fitted.get_xdata()) > 2
              and len(window.net.measured.get_xdata()) == 0
              and window.net_window.windowTitle()
              == f"{tool.NET_TITLE} - F002 -- 1 fitted",
              f"{len(window.net.measured.get_xdata())} measured, "
              f"{len(window.net.fitted.get_xdata())} fitted, "
              f"{window.net_window.windowTitle()!r}")

        # And Delta, which carries neither. An empty net has to name the fault it
        # is empty about: 348 of the 393 selections fill it with nothing, so a
        # caption that said only `stereonet` could not be told from a window that
        # had not been pointed at yet.
        window.select(3)

        check("a fault with nothing read empties the net and names itself",
              len(window.net.measured.get_xdata()) == 0
              and len(window.net.fitted.get_xdata()) == 0
              and window.net_window.windowTitle()
              == f"{tool.NET_TITLE} - F004 - nothing read",
              f"{window.net_window.windowTitle()!r}")

        # Several planes go on one artist with a break between them, which is what
        # lets any number of them cost two Line2D instead of two per fault. Without
        # the break the last point of one circle joins the first point of the next
        # by a chord across the net -- a line nobody measured, drawn in the colour
        # of a measurement. Asserted on the widget rather than through a fixture,
        # because giving a structure a second attitude moves the table, the
        # provenance band and `carries` along with it.
        window.net.show_planes(measured=[(90.0, 30.0), (270.0, 60.0)])
        xs = np.asarray(window.net.measured.get_xdata(), dtype=float)

        check("several planes on one artist are broken apart, not joined up",
              np.isnan(xs).sum() == 2 and np.isnan(xs[-1]),
              f"{np.isnan(xs).sum()} breaks in {len(xs)} points")

        window.select(2)

        # A window in the group and not a dock, which is what makes where it was
        # left something that is written down: `WindowGroup.save_geometry` walks
        # the group, a floating dock's position is not in it, and a net dragged
        # out to be read would have had to be dragged out again every run.
        check("the net is one of the tool's windows, so where it is put is kept",
              window.group.satellites.get("net") is window.net_window
              and "net" in window.window_actions,
              f"{sorted(window.group.satellites)}, "
              f"menu {sorted(window.window_actions)}")

        # Two of Gamma's three lineations, by two different rules: one carries
        # `station=S2` and is anchored 70 m away, one carries no station and is
        # anchored 1 m away. The third is 10 m away with no station and is out.
        drawn = sorted(zip(*window.net.lineation.get_data()))
        wanted = sorted(
            zip(*mplstereonet.line(
                [p for _, p in GAMMA_LINEATIONS], [t for t, _ in GAMMA_LINEATIONS]
            ))
        )

        check("the lineations read on that plane are on it too, and only those",
              len(drawn) == 2
              and all(
                  abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9
                  for a, b in zip(drawn, wanted)
              ),
              f"{len(drawn)} drawn, wanted {len(wanted)}")

        gamma = window.document.dataset.structures[2]

        check("a station code outranks distance along the trace, 70 m apart",
              window._lineations_at(gamma, window._marked[at_gamma][1])[0]
              == GAMMA_LINEATIONS[0],
              f"{window._lineations_at(gamma, window._marked[at_gamma][1])}")

        # -- and which of them the cursor is on ----------------------------
        #
        # What the hover chooses now is not the net's contents but which circle on
        # it answers. Drawn again over itself, thicker, with everything else
        # dimmed: dimming the rest and not recolouring the one, because the colour
        # here says what kind of claim a circle is and a hover must not spend it.
        window._on_map_hover(*dot)

        check("resting on a station dot points at that station's circle",
              len(window.net.marked.get_xdata()) > 2
              and window.net.marked.get_color() == window.net.measured.get_color(),
              f"{len(window.net.marked.get_xdata())} points, "
              f"{window.net.marked.get_color()}")

        check("and the rest of the net steps back while it does",
              window.net.measured.get_alpha() == window.net.DIMMED
              and window.net.fitted.get_alpha() == window.net.DIMMED,
              f"measured {window.net.measured.get_alpha()}, "
              f"fitted {window.net.fitted.get_alpha()}")

        # Both answers are about where the cursor is and both die with it. That is
        # the asymmetry this used to have and no longer does: the net was filled by
        # the cursor and cleared by a change of trace, so `hover_off` had to leave
        # it alone. What the cursor owns now goes away with the cursor, and what it
        # does not own is not its to clear.
        title = window.net_window.windowTitle()
        window.map_view.hover_off.emit()

        check("the cursor leaving takes the tooltip and the pointing with it",
              window.map_view.canvas.toolTip() == ""
              and len(window.net.marked.get_xdata()) == 0
              and window.net.measured.get_alpha() == 1.0,
              f"tip {window.map_view.canvas.toolTip()!r}, "
              f"{len(window.net.marked.get_xdata())} marked, "
              f"alpha {window.net.measured.get_alpha()}")

        check("and leaves the planes and the caption alone, those being the fault's",
              len(window.net.measured.get_xdata()) > 2
              and window.net_window.windowTitle() == title,
              f"{len(window.net.measured.get_xdata())} points, "
              f"{window.net_window.windowTitle()!r}")

        # A dot is drawn where somebody stood and a circle where they read a plane,
        # and all 23 stations of `merid_faults` are both -- so the two lists run in
        # step today and the pairing is stored anyway. An `attitude` carrying `at=`
        # and no plane is a legal record: it would be a dot with no circle, and a
        # net indexed by counting dots would answer with the next station's plane.
        # Counted within a fault, so that it survives the selection moving.
        check("a dot knows which circle is its own, and not by counting dots",
              [(whose, on_net) for _, _, whose, on_net in window._marked]
              == [(0, 0), (2, 0)],
              f"{[(whose, on_net) for _, _, whose, on_net in window._marked]}")

        # And a dot on a fault that is not open points at no circle at all, because
        # the net is not showing that fault. The tooltip still answers -- which is
        # the whole reason the dot is drawn -- and the net saying nothing is the
        # true answer to "which of these circles is this one".
        window.select(0)
        window._on_map_hover(*dot)

        check("a dot on another fault is readable and points at nothing",
              window.map_view.canvas.toolTip() != ""
              and window._netted is None
              and len(window.net.marked.get_xdata()) == 0,
              f"netted {window._netted}, "
              f"{len(window.net.marked.get_xdata())} points pointed at")

        window.select(2)

        # Closed, it goes on being filled -- which is the opposite of what the
        # fold tool does with this same widget. There the net redraws on every
        # frame of a drag and a hidden canvas would cost frame budget and make
        # the frame cost it reports a measurement of something nobody can see.
        # Here it redraws when the cursor crosses onto another dot, and the
        # saving would cost the thing it is for: a net put back would show
        # whichever station it happened to have been closed on.
        window.net_window.close()

        check("closing the net unticks its own box in the menu",
              not window.window_actions["net"].isChecked())

        window.select(0)
        window.window_actions["net"].trigger()

        check("and a net put back shows the fault now open, not a stale one",
              window.net_window.isVisible()
              and len(window.net.measured.get_xdata()) > 2
              and window.net_window.windowTitle()
              == f"{tool.NET_TITLE} - F001 -- 1 measured, 1 fitted",
              f"{window.net_window.windowTitle()!r}, "
              f"{len(window.net.measured.get_xdata())} points")

        # The four letters round the edge, proven on the pixels rather than on
        # the text having been set -- it was set from the day the widget was
        # written and was never drawn. mplstereonet keeps the azimuth labels on a
        # hidden polar axes underneath, and a figure sized to a widget inflates
        # the stereonet axes until its own background covers them.
        #
        # Calibrated against the bug rather than against a number: the same net
        # is drawn twice, once with the opaque patch it used to have, and every
        # label has to come out darker without it. A threshold would be a claim
        # about this machine's fonts.
        def ink_on_labels():
            window.net.canvas.draw()
            rgba = np.asarray(window.net.canvas.buffer_rgba())
            height = rgba.shape[0]

            return [
                int(
                    (
                        rgba[
                            int(height - label.get_window_extent().y1):
                            int(height - label.get_window_extent().y0),
                            int(label.get_window_extent().x0):
                            int(label.get_window_extent().x1),
                            :3,
                        ].sum(axis=2) < 3 * 128
                    ).sum()
                )
                for label in window.net.axes._polar.get_xticklabels()
            ]

        shown = ink_on_labels()
        window.net.axes.patch.set_alpha(1.0)
        hidden = ink_on_labels()
        window.net.axes.patch.set_alpha(0.0)

        check("the net says which way is north, which it never used to",
              [label.get_text() for label in window.net.axes._polar.get_xticklabels()]
              == ["N", "E", "S", "W"]
              and all(now > was for now, was in zip(shown, hidden)),
              f"{shown} drawn against {hidden} behind an opaque background")

        # And the widget is shared with the fold tool, which puts a population on
        # it. Neither use may leave its artists behind for the other to draw, and
        # the second direction matters more than it used to: one fault's planes are
        # meant to have nothing inside the circle but their striae, so a population
        # of poles left behind would be twenty marks that are not striae.
        window.net.show_planes(measured=[(270.0, 60.0)])
        window.net.show_window([120.0, 130.0], [30.0, 40.0])
        after_population = (
            len(window.net.poles.get_xdata()),
            len(window.net.measured.get_xdata()),
            len(window.net.fitted.get_xdata()),
            len(window.net.lineation.get_xdata()),
        )

        window.net.show_planes(measured=[(270.0, 60.0)], fitted=[(100.0, 40.0)])
        after_planes = (
            len(window.net.poles.get_xdata()),
            len(window.net.girdle.get_xdata()),
            len(window.net.measured.get_xdata()),
        )

        check("the widget's two uses clear each other, both ways round",
              after_population[:3] == (2, 0, 0)
              and after_planes[0] == 0
              and after_planes[2] > 2,
              f"a population leaves {after_population} as poles, measured, "
              f"fitted, lineations; planes after it leave {after_planes[0]} poles")

        # And a population must not leave the planes behind it *pointable* either.
        # `mark` draws one of the planes on the net a second time, and the net it
        # was handed them by is gone: an index that still resolved would put a
        # circle from another tool's question over a fold's poles.
        window.net.show_planes(measured=[(270.0, 60.0)])
        window.net.show_window([120.0, 130.0], [30.0, 40.0])
        window.net.mark(0)

        check("a population also forgets the planes, so nothing can point into it",
              len(window.net.marked.get_xdata()) == 0,
              f"{len(window.net.marked.get_xdata())} points pointed at")

        window.select(2)

        # -- where the view is ---------------------------------------------

        print("\n-- the framing --\n")

        # Gamma runs 1000 m due north from (603000, 4420000), so the margin comes
        # to 250 m: a quarter of the longer side, which for a trace with no width
        # is the only side there is.
        framed = tool.framing_for(gamma.path)

        check("a framing has the trace inside it with room around it",
              framed == [602750.0, 603250.0, 4419750.0, 4421250.0], str(framed))

        # The floor, which is what a polyline adds to `sections.framing_for`. The
        # shortest fault in `merid_faults` is 12 m against a median of 1048, and a
        # window fitted to that one says nothing about where you are.
        short = tool.framing_for([(0.0, 0.0), (12.0, 0.0)])

        check("and a very short one gets ground around it, not a close-up",
              abs((short[1] - short[0]) - tool.FRAME_MIN_SPAN_M) < 1e-6,
              f"{short[1] - short[0]:.0f} m across")

        whole = window.map_view.framing

        # What a click on the map must not do. You are already looking at what you
        # clicked, and a view that jumped to it would be taking the ground around
        # it away as the reward for having found it.
        window.pick(X0 + 300.0, Y0)

        check("a click on the map selects without moving the view",
              window.index == 0 and window.map_view.framing == whole
              and not window._frame_timer.isActive(),
              f"index {window.index}")

        def pick_row(index):
            """What a click on a row of the table does, without a mouse."""

            table.setCurrentCell(table._row_of(index), 0)

        pick_row(2)

        check("picking a row selects it and asks for the view to follow",
              window.index == 2 and window._framing == 2
              and window._frame_timer.isActive(),
              f"index {window.index}, framing on {window._framing}")

        # Asked for and not done: arrow-keying down the table is a row a
        # keystroke, and each framing is a full redraw plus a hillshade reread.
        check("and asking is not doing, so a run of them costs one move",
              window.map_view.framing == whole)

        window.frame_now()
        moved = window.map_view.framing

        check("and when it moves, the trace is in the view and the view is smaller",
              moved[0] < 603000.0 < moved[1]
              and moved[2] < 4420000.0 and 4421000.0 < moved[3]
              and (moved[1] - moved[0]) < (whole[1] - whole[0]),
              f"{(moved[1] - moved[0]) / 1000.0:.1f} km across, "
              f"was {(whole[1] - whole[0]) / 1000.0:.1f}")

        # The way back. `restore_framing` pushes onto the navigation stack over
        # the home the base map anchored, so the bar's back arrow is the way out
        # of a framing this made -- the same as out of a zoom made by hand.
        window.map_view.toolbar.back()

        check("the bar's back arrow is the way out of a framing",
              [round(v) for v in window.map_view.framing]
              == [round(v) for v in whole],
              str([round(v) for v in window.map_view.framing]))

        window.panel.frame_wanted.setChecked(False)
        pick_row(0)

        check("and with the box unticked a row picked leaves the view alone",
              window.index == 0 and not window._frame_timer.isActive(),
              f"index {window.index}, framing on {window._framing}")

        window.panel.frame_wanted.setChecked(True)
        window.select(2)

        # -- and the one gesture that could throw work away ----------------

        print("\n-- leaving a block --\n")

        asked = []
        QtWidgets.QMessageBox.question = staticmethod(
            lambda *args, **rest: asked.append(args[2] if len(args) > 2 else "")
            or QtWidgets.QMessageBox.StandardButton.Cancel
        )

        window.panel.text.appendPlainText("  span exposure * * covered")
        window.pick(X0 + 300.0, Y0)

        check("clicking another trace with a block typed and not applied asks",
              len(asked) == 1 and "not applied" in asked[0],
              f"asked {len(asked)} time(s)")

        check("and saying no leaves the map and the text on the same structure",
              window.index == 2 and window.panel.index == 2
              and "span exposure * * covered" in window.panel.text.toPlainText(),
              f"index {window.index}")

        QtWidgets.QMessageBox.question = staticmethod(
            lambda *args, **rest: QtWidgets.QMessageBox.StandardButton.Discard
        )

        window.pick(X0 + 300.0, Y0)

        check("and saying yes moves on, with the document none the wiser",
              window.index == 0
              and "span exposure * * covered" not in window.document.text_of(2),
              f"index {window.index}")

        # The reach decides what holds, not what is written, so turning it must
        # redraw the picture and leave the box alone -- the same loss as above,
        # from a dial instead of a click.
        window.panel.text.appendPlainText("  span exposure * * covered")
        window.panel.gap_spin.setValue(600.0)

        check("turning the reach dial redraws the picture and not the text",
              "span exposure * * covered" in window.panel.text.toPlainText())

        def measured(runs):
            return sum(b - a for a, b, kind, _ in runs if kind == "misurata")

        narrow = tool.runs_of(provenance_of(alpha, samples=201, max_gap=250.0))
        wide = tool.runs_of(provenance_of(alpha, samples=201, max_gap=600.0))

        check("and a wider reach is one measurement answering for more trace",
              measured(wide) > measured(narrow),
              f"{measured(narrow):.0f} m -> {measured(wide):.0f} m")

        window.panel.gap_spin.setValue(250.0)
        window.panel._redraw()

        # -- two projections, and the one place each crosses ---------------

        print("\n-- projections --\n")

        check("one projection for the file and the map means no transform",
              window._forward is None and window._back is None)

        from pyproj import CRS, Transformer

        theirs, ours = CRS.from_epsg(25833), CRS.from_epsg(32632)
        window._forward = Transformer.from_crs(theirs, ours, always_xy=True)
        window._back = Transformer.from_crs(ours, theirs, always_xy=True)

        here = (X0 + 300.0, Y0)
        there = window.on_map([here])[0]
        back = window.in_file(*there)

        check("two projections put the map somewhere else entirely",
              abs(there[0] - here[0]) > 100000.0, f"{there[0] - here[0]:.0f} m")

        check("and a point out to the map and back lands where it started",
              abs(back[0] - here[0]) < 0.001 and abs(back[1] - here[1]) < 0.001,
              str(back))

        window._forward = window._back = None

        # -- saving -------------------------------------------------------

        print("\n-- saving --\n")

        before = path.read_text(encoding="utf-8")
        window.save()
        saved = path.read_text(encoding="utf-8")

        check("saving writes the line that was added and nothing else",
              changed_lines(before, saved) == [
                  '+  span use @603000.00,4420200.00 @603000.00,4420600.00 '
                  'rejected reason="check"'
              ],
              str(changed_lines(before, saved))[:70])

        check("and the file still reads as what it was",
              len(gstruct.loads(saved).structures) == 4)

        # -- the real file, and the reason any of this is done this way ----

        print("\n-- the reasoning that would have been lost --\n")

        real = AOI / "curation.gstruct"

        if not real.exists():
            print(f"(not here: {real})")
        else:
            copy = Path(tmp) / "curation.gstruct"
            shutil.copy(real, copy)

            kept = copy.read_text(encoding="utf-8")
            rebuilt = gstruct.dumps(gstruct.loads(kept))

            check("the format's own writer has nowhere to keep a comment",
                  len(comment_lines(kept)) == 10 and not comment_lines(rebuilt),
                  f"{len(comment_lines(kept))} lines in, "
                  f"{len(comment_lines(rebuilt))} out")

            curation = Document(copy)
            curation.replace(
                0,
                curation.text_of(0)
                + '\n  span use * * rejected src=check reason="prova"',
            )
            curation.save()

            written_back = copy.read_text(encoding="utf-8")

            check("editing a block of the real file changes that one line",
                  changed_lines(kept, written_back) == [
                      '+  span use * * rejected src=check reason="prova"'
                  ],
                  str(changed_lines(kept, written_back))[:70])

            check("and the ten lines saying why it says what it says stay put",
                  comment_lines(written_back) == comment_lines(kept))

        # -- the topography, read along one trace --------------------------

        print("\n-- the fit off the DEM --\n")

        from gsurf import traces as traces_module
        from gsurf.fits import FROM_DEM, PLANE_DECIMALS, as_line

        # The session this check has been driving all along has no DEM, the slot
        # being optional here, and that is the first of the three answers the
        # button can give before it is pressed.
        check("with no DEM the button is off, and its reason is the DEM",
              window.panel.dem is None
              and not window.panel.fit_button.isEnabled()
              and "DEM" in window.panel.fit_button.toolTip(),
              window.panel.fit_button.toolTip()[:58])

        relief = plane_dem(tmp)
        elsewhere = plane_dem(tmp, "utm32.tif", crs="EPSG:25832")

        traced = written(tmp, "fitted.gstruct", fitted_source())
        traced_spec = dict(path=str(traced), role="traces")

        # The second: a DEM that cannot be sampled for these traces at all. The
        # trace would be sampled in the DEM's grid and the anchors written in the
        # file's, so the dip direction that came out would be measured from one
        # north and written against another -- which is not something a
        # reprojection on the way in would fix.
        crossed = tool.build(
            Session.open(dem_path=str(elsewhere), frame_layers=[traced_spec]),
            {"traces": traced_spec},
        )

        check("a DEM in another projection turns it off too, naming both",
              crossed is not None
              and not crossed.panel.fit_button.isEnabled()
              and "25832" in (crossed.panel.dem_said or "")
              and "25833" in (crossed.panel.dem_said or ""),
              (crossed.panel.dem_said or "nothing said")[:58])

        crossed.close()

        session = Session.open(dem_path=str(relief), frame_layers=[traced_spec])
        fitting = tool.build(session, {"traces": traced_spec})
        panel = fitting.panel

        check("and a DEM these traces can be read against turns it on",
              panel.fit_button.isEnabled() and panel.dem_said is None)

        rows = {
            structure.ident: n
            for n, structure in enumerate(fitting.document.dataset.structures)
        }

        # -- the trace over the topography ---------------------------------

        # The arithmetic, on the one trace whose answer is a closed form: EAST
        # runs straight down the dip of a plane at 30 degrees, so hanging it on
        # that plane lengthens it by exactly 1/cos(30) and by nothing else. The
        # agreement is to six figures, which is what says the draping is
        # measuring the ground rather than the grid -- nearest-cell sampling on
        # a 5 m cell at a 5 m step would show as a staircase in this digit.
        east = fitting.document.dataset.structures[rows["EAST"]]
        down_dip, _ = traces_module.draped_length(east.path, panel.dem)

        check("a trace straight down a 30 degree dip is longer by 1/cos(30)",
              abs(down_dip / east.length - 1.0 / math.cos(math.radians(30.0)))
              < 1e-6,
              f"{east.length:.2f} m in plan, {down_dip:.3f} m draped, "
              f"ratio {down_dip / east.length:.6f}")

        # And the cell is that number, rounded to the metre. Separate from the
        # assertion above because they are two different claims: that the
        # arithmetic is right, and that the column is showing this arithmetic
        # and not some second one of its own.
        east_row = next(
            row for row in range(panel.table.rowCount())
            if panel.table.item(row, 0).text() == "EAST"
        )

        check("and the cell is that, to the metre, beside the plan length",
              panel.table.item(east_row, 2).text() == str(round(down_dip))
              and panel.table.item(east_row, 1).text() == str(round(east.length)),
              f"{panel.table.item(east_row, 1).text()} / "
              f"{panel.table.item(east_row, 2).text()}")

        # And the other end of the same statement: along the strike of that
        # plane there is no rise, so the draped length is the plan length to the
        # metre. A draping that invented relief where the ground is level -- by
        # counting the cell edges it steps over -- would fail here and pass the
        # check above, since a staircase inflates both.
        strike = [(X0 + 500.0, Y0 + n * 10.0) for n in range(101)]
        level, level_over = traces_module.draped_length(strike, panel.dem)

        check("and along the strike of it there is no rise, so no extra length",
              abs(level - 1000.0) < 0.5 and abs(level_over - 1000.0) < 0.5,
              f"{level:.4f} m over {level_over:.1f} m of a 1000 m trace")

        # A trace that runs off the DEM, which is the case the `~` exists for:
        # half of this one is on the raster, so its draped length comes back
        # *shorter* than its plan length. Printed bare that is a subtraction
        # anybody would read as a bug in the draping, so the metres never travel
        # without the metres they cover.
        edge = [(X0 + 2000.0, Y0), (X0 + 3400.0, Y0)]
        part, part_over = traces_module.draped_length(edge, panel.dem)

        check("a trace running off the DEM is measured on the part that is on it",
              part < 1400.0 and abs(part_over - 695.0) < 10.0
              and abs(part / part_over - 1.0 / math.cos(math.radians(30.0))) < 1e-3,
              f"{part:.1f} m over {part_over:.1f} m of 1400")

        # Dropped and not closed up. `trace_points` hands back the samples it
        # kept, so the two either side of a hole are neighbours in the array and
        # the straight line between them is gap and not trace. Counting it would
        # put the whole 1400 m in the answer, which is the number this is not.
        check("and the stretch it is off the DEM for is dropped, not spanned",
              part_over < 1400.0 * tool.WHOLE_TRACE,
              f"covers {part_over / 1400.0:.0%}")

        check("a trace with no DEM under it at all comes back with nothing",
              traces_module.draped_length(
                  [(X0 + 5000.0, Y0), (X0 + 5400.0, Y0)], panel.dem
              ) == (None, 0.0))

        # The step is the DEM's own cell, and the header says which -- because
        # the number depends on it and does not converge. On `merid_faults`
        # against the 5 m DTM the total is 634 km at a 50 m step and 685 km at
        # 2.5 m, so a draped length quoted without its step is not reproducible.
        check("the header says what the column was walked at",
              f"{CELL:g} m" in panel.table.horizontalHeaderItem(2).toolTip(),
              panel.table.horizontalHeaderItem(2).toolTip()[:60])

        # Sorted on the metres, not on the text: `~802` is a length with a
        # caveat in front of it and an empty cell is no length at all, and
        # neither compares as the number it stands for. The empties go below
        # every real length rather than to zero, which is a length.
        panel.table.sortItems(2, QtCore.Qt.SortOrder.AscendingOrder)
        order = [
            panel.table.item(row, 2).text().lstrip("~")
            for row in range(panel.table.rowCount())
        ]

        check("the draped column sorts on the metres behind the text",
              order == sorted(order, key=int),
              str(order))

        # The three kinds of cell against each other, which no one file puts in
        # one column: a plain length, a length with the `~` of a trace that runs
        # off the DEM, and the blank of a trace with no DEM under it at all.
        # Sorted on their text `~802` would follow `1848` and `` would lead, and
        # every one of the three would be in the wrong place.
        mixed = sorted([
            tool._Ranked("1848", 1847.5),
            tool._Ranked("", tool.UNMEASURED),
            tool._Ranked("~802", 802.5),
        ])

        check("blank last and `~` in its place, whatever the text would do",
              [cell.text() for cell in mixed] == ["", "~802", "1848"],
              str([cell.text() for cell in mixed]))

        panel.table.sortItems(0, QtCore.Qt.SortOrder.AscendingOrder)

        # The floor, end to end, and the reason this file is the case that found
        # it: every vertex in the format is written to two decimals, so these
        # densified traces report their own storage -- four millimetres -- as the
        # roughness of the hand that drew them. Three times that is a lever floor
        # of a centimetre, which admits every window of every trace.
        gate, sigma = panel._gate()

        check("the gate keeps its own floor: a file's storage is not a pen",
              sigma is None and gate == traces_module.TraceGate(),
              f"lever {gate.min_lever:.1f} m, measured sigma "
              + ("none" if sigma is None else f"{sigma * 1000.0:.1f} mm"))

        panel.show_index(rows["VEE"])

        before = panel.text.toPlainText()
        reading = panel.fit_off_dem()
        added = [
            line for line in panel.text.toPlainText().splitlines()
            if line not in before.splitlines()
        ]

        check("the V is read, and what comes of it lands in the box and nowhere else",
              reading is not None and len(reading.fits) == 1
              and len(added) == 1
              and added[0].strip().startswith("fit plane")
              and fitting.document.text_of(rows["VEE"]) == before,
              f"{len(reading.fits)} fit(s), {len(added)} line(s) written")

        fit = reading.fits[0]

        # The DEM is one plane and the trace lies on it, so this is the
        # arithmetic that built the raster. Two degrees of slack for the DEM's
        # own cell quantisation, which is 1.4 m of height over 5 m of ground.
        check("and the plane written is the one the DEM was built from",
              abs(fit.plane.dip - RELIEF_DIP) <= 2.0
              and abs(fit.plane.dip_dir - RELIEF_DIP_DIR) <= 2.0,
              f"{fit.plane.dip_dir:.1f}/{fit.plane.dip:.1f} against "
              f"{RELIEF_DIP_DIR:.0f}/{RELIEF_DIP:.0f}")

        # The whole of `as_line`'s contract in one assertion: the anchors, the
        # attributes and their quoting, and the decimals. Read back rather than
        # matched against a string, because what has to hold is that the line
        # says what the fit says -- not that it is spelled a particular way.
        reread = gstruct.loads(
            'gstruct 0.2\ncrs EPSG:25833\n\nstructure X ""\n'
            + added[0] + "\n  path 2\n"
            + f"    {X0:.2f} {Y0:.2f}\n    {X0 + 1000.0:.2f} {Y0:.2f}\n"
        ).structures[0].fits[0]

        check("the written line reads back as the fit that was written",
              reread.plane.dip_dir == round(fit.plane.dip_dir, PLANE_DECIMALS)
              and reread.plane.dip == round(fit.plane.dip, PLANE_DECIMALS)
              and reread.attrs == fit.attrs
              and reread.start == (None if fit.start is None
                                   else tuple(round(v, 2) for v in fit.start)),
              f"{reread.plane} with {len(reread.attrs)} attribute(s)")

        # And why the line is built in `fits` rather than taken from `dumps`: the
        # format's own writer prints whole degrees, and a plane computed over a
        # swept window is a computed number.
        rounded = gstruct.Plane(140.5, 31.2)

        check("which `dumps` could not have done, printing whole degrees",
              str(rounded) == "140/31"
              and "140.5/31.2" in as_line(gstruct.Fit(plane=rounded)),
              f"`{rounded}` against `{as_line(gstruct.Fit(plane=rounded)).strip()}`")

        check("and Apply takes it, which is the only way it reaches the model",
              panel.apply_block()
              and len(fitting.document.dataset.structures[rows["VEE"]].fits) == 1,
              fitting.document.text_of(rows["VEE"]).splitlines()[2][:58])

        # Apply rewrites the row, and the draped length has to come back with
        # it. It is the one cell on the row that is not read off the structure
        # -- it was measured once at the door, because it is a raster read per
        # trace -- so a rewrite that forgot it would blank the column one row at
        # a time, on exactly the traces being worked on.
        vee_row = next(
            row for row in range(panel.table.rowCount())
            if panel.table.item(row, 0).text() == "VEE"
        )

        check("and the rewritten row keeps the length nobody recomputed",
              panel.table.item(vee_row, 2).text()
              == str(round(traces_module.draped_length(
                  fitting.document.dataset.structures[rows["VEE"]].path,
                  panel.dem,
              )[0])),
              f"VEE: {panel.table.item(vee_row, 2).text()} m")

        held = fitting.document.dataset.structures[rows["VEE"]].fits[0]

        check("on the stretch that turned, and not on the trace it is on",
              held.s0 < APEX_S < held.s1
              and (held.s1 - held.s0) < 2.0 * APEX_S / 10.0,
              f"{held.s0:.0f}..{held.s1:.0f} m, apex at {APEX_S:.0f}")

        # `add_written` and not `add_line`, and the `*` is the whole difference:
        # it is the format's word for an end of the path, and `add_line` selects
        # the next one so that a shift-click on the map fills it in. Aiming at
        # this one would offer to overwrite the only token on the line that is
        # not a coordinate, and the next pick anywhere would land there.
        panel.show_index(rows["ZIG"])
        panel.fit_off_dem()

        reaching = [
            line for line in panel.text.toPlainText().splitlines()
            if line.strip().startswith("fit plane")
        ]

        check("a fit running to the end of its path writes `*`, and nothing aims at it",
              len(reaching) == 1 and " * " in reaching[0]
              and not panel.text.textCursor().hasSelection(),
              reaching[0].strip()[:58] if reaching else "no fit")

        panel.apply_block()

        # The load-bearing negative, here as well as in the import: a plane
        # through a straight trace is arbitrary rather than imprecise, so nothing
        # is written -- and the panel says what was walked instead of saying
        # nothing, because an empty answer and a refused one look the same.
        panel.show_index(rows["EAST"])

        before = panel.text.toPlainText()
        nothing = panel.fit_off_dem()

        check("a dead straight trace writes no line at all, and says what it walked",
              not nothing.fits and nothing.silent
              and panel.text.toPlainText() == before
              and "nothing held" in nothing.describe(),
              nothing.describe()[:66])

        # Precedence, reported and not settled -- `attitude_at` takes the *first*
        # fit covering a progressive, and these lines go after the ones the block
        # already had. Asserted as what it does rather than as what it should do,
        # so that changing it has to change this line too.
        said = []
        panel.said.connect(said.append)

        panel.show_index(rows["TAKEN"])
        panel.fit_off_dem()
        panel.apply_block()

        now = fitting.document.dataset.structures[rows["TAKEN"]].fits

        check("a computed fit is written after the ones the block already carried",
              len(now) == 2 and now[0].attrs.get("from") == "table"
              and now[-1].attrs.get("from") == FROM_DEM,
              ", ".join(one.attrs.get("from", "?") for one in now))

        check("so the panel says which of them will answer, rather than leaving it",
              any("already carries" in one for one in said)
              and any("first" in one for one in said),
              (said[-1] if said else "nothing said")[:76])

        # And the file, which is the invariant every other assertion here rests
        # on: the lines of the structures that were edited, and not one byte more.
        was = traced.read_text(encoding="utf-8")
        fitting.save()
        now_text = traced.read_text(encoding="utf-8")

        changed = changed_lines(was, now_text)

        check("saving writes the three fits and leaves every other byte alone",
              len(changed) == 3
              and all(one.startswith("+  fit plane") for one in changed),
              f"{len(changed)} line(s) changed")

        fitting.close()

        # -- what it will not open ----------------------------------------

        print("\n-- what it will not open --\n")

        # `build` says why it will not run by putting a box on screen, and a
        # modal box in a check is a check that hangs. The refusal is the thing
        # being tested, so the box is replaced by something that writes it down.
        shown = []
        QtWidgets.QMessageBox.critical = staticmethod(
            lambda *args, **rest: shown.append(args[2] if len(args) > 2 else "")
        )

        layer = written(tmp, "not-a-format.gpkg", "")

        # The second half is the assertion, and it has been wrong twice in the
        # same way: this box used to name `export_gsurf.py`, which runs the other
        # way and had produced the very file being refused, and then
        # `export_geology.py`, which is a script for one survey and not a way in
        # for anybody else. What a refusal owes the reader is the thing to do
        # instead, and there is now one in the program -- so the assertion is
        # that it points there and at no script at all.
        check("a layer is not a thing this edits, and it says where to convert one",
              tool.build(session, {"traces": dict(path=str(layer))}) is None
              and shown
              and "Import" in shown[-1]
              and ".gstruct" in shown[-1]
              and ".py" not in shown[-1],
              (shown[-1] if shown else "nothing said").split("\n")[-1][:60])

        empty = written(tmp, "empty.gstruct", "gstruct 0.2\ncrs EPSG:25833\n")

        check("nor is a file with no structure in it",
              tool.build(session, {"traces": dict(path=str(empty))}) is None
              and "no structure" in shown[-1],
              shown[-1].split("\n")[-1][:60])

        ahead = written(
            tmp, "ahead.gstruct", "gstruct 9.9\ncrs EPSG:25833\n\nstructure F1\n"
        )

        check("nor one written by a gstruct that does not exist yet",
              tool.build(session, {"traces": dict(path=str(ahead))}) is None
              and "9.9" in shown[-1],
              shown[-1].split("\n")[-1][:60])

        # The first vertex of `merid_faults.gstruct` taken to EPSG:4326. The
        # refusal is not about the projection being unusual: it is that an anchor
        # is written to two decimals, which here is the 472 m the message quotes,
        # and that every number along a trace -- the reach, a fit's window,
        # `DEFAULT_MAX_GAP` -- is metres. Half of this tool cannot work in
        # degrees, so none of it opens.
        degrees = written(
            tmp, "degrees.gstruct",
            "gstruct 0.2\ncrs EPSG:4326\n\nstructure F1\n  path 2\n"
            "    16.270831 39.915793\n    16.280000 39.920000\n",
        )

        check("nor one whose ruler is degrees, and it says how far that is",
              tool.build(session, {"traces": dict(path=str(degrees))}) is None
              and "472 m" in shown[-1],
              shown[-1].split(": ")[-1][:70])

    print()

    if FAILURES:
        print(f"FAILED: {', '.join(FAILURES)}")
        return 1

    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
