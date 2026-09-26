"""
A section written to a file, and opened again somewhere else.

The tool already carried the last section from one run to the next. This is the
other thing, and a different claim: a section that was named and kept, brought
back onto a DEM that may not be the one it was drawn on, in a projection that
may not be the one it was written in.

What is under test:

  - **A file round-trips.** Not just the numbers: the bundle laid on the ground
    after opening is the same array of lines, to floating point, as the one that
    was there when it was saved. A section is recomputed from the trace and the
    DEM, so that is the only claim worth making about a stored one.

  - **The ends survive a change of projection, and what does not is measured.**
    Two ends carried across projections come back exactly; the straight line
    between them does not, a straight line in one projection not being one in
    another. So the middle of it is measured and reported rather than assumed
    small -- and how small it is depends on the length, which is why the number
    is here and not in a comment.

  - **What cannot be met is refused whole.** A section of ground this session is
    not open on is the case that matters: half-applying it would put somebody's
    trace on a hillside it says nothing about, and a message with the two
    extents in it is what tells you the DEM is wrong rather than the file.

The ground is the synthetic DEM, known by construction: EPSG:25833, 600000 to
610000 by 4410000 to 4420000, cells of 5 m.

    QT_QPA_PLATFORM=offscreen python checks/check_section_files.py
"""

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("GSURF_REPO", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(REPO))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

FAILURES = []

# A fixed instant, so "the same section written twice is the same file" is a
# claim about the writer and not about the clock.
WHEN = datetime(2026, 9, 26, 10, 30, tzinfo=timezone.utc)


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}{'   ' + detail if detail else ''}")
    if not condition:
        FAILURES.append(label)


def reprojected(trace, epsg):
    """The two ends as another projection would have written them."""

    from pyproj import CRS, Transformer

    transformer = Transformer.from_crs(
        CRS.from_epsg(25833), CRS.from_epsg(epsg), always_xy=True
    )
    xs, ys = transformer.transform([x for x, _ in trace], [y for _, y in trace])

    return [[float(x), float(y)] for x, y in zip(xs, ys)]


def main():
    from PyQt6 import QtCore, QtWidgets

    from checks.synthetic import synthetic_dem
    from gsurf import sections
    from gsurf.session import Session
    from gsurf.tools import profiles as tool

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    work = Path(tempfile.mkdtemp())
    dem_path = synthetic_dem(work / "synthetic.tif")

    session = Session.open(dem_path=str(dem_path))
    print(f"synthetic: {session.summary()}")

    window = tool.ProfilesWindow(session, num_profiles=5)
    window.resize(1400, 950)
    window.show()
    app.processEvents()
    window.map_view.canvas.draw()

    def bundle_ends():
        """Where each profile of the bundle starts and where it ends."""

        return np.array([
            [line.coords[0, :2], line.coords[-1, :2]]
            for line in window.geoprofiles.profilers.lines
        ])

    # A section nobody would arrive at by default: off-centre, not west to east,
    # an odd count that is not the one the window opened on, and a spacing that
    # is not the default either. Everything restored has to be distinguishable
    # from what a fresh window does on its own.
    window.trace = [(601000.0, 4412000.0), (609000.0, 4418500.0)]
    window._redraw_trace()
    window.count_spin.setValue(7)
    window.offset_spin.setValue(650.0)
    app.processEvents()

    saved_trace = [tuple(end) for end in window.trace]
    saved_bundle = bundle_ends()
    saved_length = window._length()

    print(f"\n-- the file --")

    path = window.save_section(work / "montealpi_section.json")
    text = path.read_text(encoding="utf8")

    check("saving writes the file the dialog was answered with",
          path.exists() and path.name == "montealpi_section.json",
          f"{path.stat().st_size} bytes")

    check("it says what it is on its first line",
          text.splitlines()[1].strip().startswith(f'"{sections.MARKER}"'),
          text.splitlines()[1].strip())

    check("it is indented, so moving the trace is a readable diff",
          "\n  " in text and text.endswith("\n"),
          f"{len(text.splitlines())} lines")

    # The rule the trace editor arrived at the hard way, kept here because it
    # costs nothing: text mode translates on the way out, and a file every other
    # tool in this project writes LF would come out CRLF on Windows.
    check("and it is LF, on any platform",
          b"\r" not in path.read_bytes())

    again = sections.write(work / "twice.json", window.current_state(),
                           crs=session.crs, when=WHEN)
    once = sections.write(work / "once.json", window.current_state(),
                          crs=session.crs, when=WHEN)

    check("the same section written twice is the same bytes",
          again.read_bytes() == once.read_bytes())

    stored = json.loads(text)

    check("the projection goes in twice: the code, and a WKT to build on",
          stored["epsg"] == 25833 and "ETRS" in stored["crs"],
          f"EPSG:{stored['epsg']}, {len(stored['crs'])} chars of WKT")

    check("the DEM it was drawn over is named, as provenance",
          stored["source"] == str(dem_path))

    check("and the numbers are the section, not the profiles computed from it",
          set(stored) == {sections.MARKER, "written", "source", "epsg", "crs",
                          "trace", "profiles", "offset", "reach", "extent",
                          "legend"},
          ", ".join(k for k in stored if k not in (sections.MARKER, "written")))

    print(f"\n-- opened again, on the DEM it was drawn on --")

    # Dragged somewhere else first, and the bundle changed too: coming back has
    # to be a restore and not a window that never moved.
    window.trace = [(602000.0, 4419000.0), (603000.0, 4411000.0)]
    window._redraw_trace()
    window.count_spin.setValue(3)
    app.processEvents()

    # Counted through the instance, which is what `take_on` and every spin box
    # signal go through: four restored numbers must be one bundle, not four.
    bundles = []
    computed = window.update_bundle
    window.update_bundle = lambda: (bundles.append(1), computed())[1]

    opened = window.open_section(path)
    app.processEvents()

    check("the trace comes back exactly, not to the nearest metre",
          [tuple(end) for end in window.trace] == saved_trace,
          f"{saved_length / 1000.0:.3f} km")

    check("and the bundle with it, in the boxes as well as the attributes",
          window.num_profiles == 7 and window.count_spin.value() == 7
          and window.offset == 650.0 and window.offset_spin.value() == 650.0,
          f"{window.num_profiles} profiles at {window.offset:.0f} m")

    check("the section on the ground is the one that was saved, line for line",
          np.allclose(bundle_ends(), saved_bundle),
          f"{len(saved_bundle)} profiles, to floating point")

    check("and it was computed once for the whole file",
          len(bundles) == 1, f"{len(bundles)} bundle(s)")

    # Removed and not put back: assigning the bound method would leave an
    # instance attribute holding a method of the window it lives on, and a
    # window kept alive by a cycle of its own is one Qt has already taken apart
    # by the time the collector reaches it.
    del window.update_bundle

    check("opening reports what it did rather than leaving it to be noticed",
          "km" in opened.summary() and opened.moved is None,
          opened.summary().splitlines()[0])

    # This window was opened on bare topography, and the file carries a reach.
    # There is nothing for it to be applied to, which is a thing to say rather
    # than a number to take in and drop.
    check("and a reach with no traces to reach along is said, not swallowed",
          any("no traces layer" in note for note in opened.notes),
          "opened without a traces slot")

    print(f"\n-- onto another DEM, and onto another projection --")

    # Same ground, same projection, different file. `applicable` refuses this --
    # it is silent and prefers the middle of the DEM to a guess -- and a named
    # file must not: a trace is metres in a projection, not a property of a
    # raster.
    #
    # Built rather than opened, and that is the point of the case as well as a
    # way of not opening a second raster: what `open_onto` is allowed to depend
    # on is the projection, the extent and the name of the source -- nothing
    # that requires the DEM to be there at all.
    other_dem = shutil.copy(dem_path, work / "same_ground.tif")
    elsewhere_session = Session(
        session.crs, session.bounds, base_path=other_dem
    )

    landed = sections.open_onto(sections.read(path), elsewhere_session)

    check("a section opens onto another DEM covering the same ground",
          [tuple(end) for end in landed.state["trace"]] == saved_trace,
          "the same metres, a different raster")

    check("while the silent restore still refuses it",
          "trace" not in tool.applicable(elsewhere_session, stored),
          "`applicable` is for continuing, not for keeping")

    # Written under another projection, which is the case a file has and a conf
    # does not: the numbers in it are somewhere else entirely -- a 1114548
    # easting for a DEM that stops at 610000 -- and cannot be taken as they come.
    for epsg, why in ((32633, "the same grid, another datum"),
                      (32632, "a zone over"),
                      (4326, "degrees")):
        moved_file = dict(stored, epsg=epsg, trace=reprojected(saved_trace, epsg))
        moved_file["crs"] = None
        moved_file.pop("extent")

        carried = sections.open_onto(moved_file, session)
        ends = [tuple(end) for end in carried.state["trace"]]
        off = max(
            float(np.hypot(x - sx, y - sy))
            for (x, y), (sx, sy) in zip(ends, saved_trace)
        )

        check(f"a trace written in EPSG:{epsg} lands back on the same ground",
              off < 1e-6,
              f"{why}: ends within {off * 1e9:.1f} nm, middle moved "
              f"{carried.moved:.2f} m")

    # The one number in this that grows: the deviation goes as the square of the
    # length -- 1.9 m over 10 km, 69 m over 60 -- so the threshold is crossed
    # exactly where a section stops being one. Below it nothing is said, the
    # shift being under the cell of the DEM being sampled.
    said = [note for note in carried.notes if "reprojected" in note]

    check("a section's worth of that is not worth a sentence",
          not said and carried.moved < sections.DEVIATION_WORTH_SAYING_M,
          f"{carried.moved:.2f} m over {saved_length / 1000.0:.1f} km, "
          f"under a 5 m cell")

    transect = [(600500.0, 4410500.0), (600500.0 + 48000.0, 4410500.0 + 36000.0)]
    long_file = dict(stored, epsg=4326, crs=None,
                     trace=reprojected(transect, 4326))
    long_file.pop("extent")

    stretched, moved = sections._onto_crs(
        [tuple(end) for end in long_file["trace"]],
        sections._crs_of(long_file),
        session.crs,
    )

    check("a transect's worth of it is, and is measured rather than bounded",
          moved > sections.DEVIATION_WORTH_SAYING_M,
          f"{moved:.0f} m over 60 km, against {carried.moved:.2f} m over 10")

    print(f"\n-- the framing, which is about looking and not about the section --")

    check("a framing written on this projection is carried",
          np.allclose(landed.state["extent"], stored["extent"]))

    across = dict(stored, epsg=4326, crs=None, trace=reprojected(saved_trace, 4326))
    reframed = sections.open_onto(across, session)
    left, right, bottom, top = reframed.state["extent"]

    check("one written under another is replaced by one made from the trace",
          not np.allclose(reframed.state["extent"], stored["extent"])
          and all(left < x < right and bottom < y < top
                  for x, y in reframed.state["trace"]),
          "a rectangle in another projection is not a rectangle here")

    check("and the swap is said, not done quietly",
          any("framing" in note for note in reframed.notes))

    print(f"\n-- what is refused, and what is only reported --")

    refusals = (
        ("not JSON at all", "this is a section, honestly", sections.NotASection),
        ('JSON with no marker in it', json.dumps({"trace": stored["trace"]}),
         sections.NotASection),
        ("a format from a later gSurf",
         json.dumps(dict(stored, **{sections.MARKER: sections.FORMAT + 1})),
         sections.NotASection),
    )

    for what, content, expected in refusals:
        bad = work / "bad.json"
        bad.write_text(content, encoding="utf8")

        try:
            sections.read(bad)
        except expected as err:
            spoken = str(err)
        else:
            spoken = ""

        check(f"refused: {what}", bool(spoken) and bad.name in spoken, spoken[:90])

    left, bottom, right, top = session.bounds
    away = dict(stored, trace=[[left - 50000.0, bottom - 50000.0],
                               [left - 40000.0, bottom - 40000.0]])

    try:
        sections.open_onto(away, session)
    except sections.SectionElsewhere as err:
        spoken = str(err)
    else:
        spoken = ""

    # The message has to be actionable without the file being opened in an
    # editor: the usual cause is the right section over the wrong DEM, and the
    # two extents beside each other say so at a glance.
    check("refused: a section of ground this session is not open on",
          bool(spoken) and f"{left:.0f}" in spoken and "550000" in spoken,
          spoken[:120] if spoken else "nothing raised")

    try:
        sections.open_onto(dict(stored, trace=[[601000.0, 4412000.0],
                                               [601000.2, 4412000.2]]), session)
    except sections.SectionError as err:
        spoken = str(err)
    else:
        spoken = ""

    # In metres of this ground, which is the point of where the test sits: the
    # floor is applied after the reprojection, so a section written in degrees
    # -- a tenth of one long, and ten kilometres of ground -- is not refused as
    # a trace with no length in it. It is the same check that catches this.
    check("refused: two ends with no length between them",
          bool(spoken) and "0.28 m" in spoken,
          spoken[:90] if spoken else "nothing raised")

    # Not refusals. A number no box could hold is the file being partly wrong
    # about a habit, and the trace is still the thing that was asked for.
    edited = sections.open_onto(
        dict(stored, profiles=4, offset=1e9, reach=-3.0), session
    )

    check("a count no bundle can have is left alone and said, not refused",
          "profiles" not in edited.state and "offset" not in edited.state
          and "reach" not in edited.state
          and len(edited.notes) == 3
          and [tuple(end) for end in edited.state["trace"]] == saved_trace,
          "; ".join(note.split(" in the file")[0] for note in edited.notes))

    nameless = dict(stored, epsg=None, crs=None)

    check("a file that does not say which projection it is in is taken as this one",
          [tuple(end) for end in sections.open_onto(nameless, session).state["trace"]]
          == saved_trace
          and any("nothing about which projection" in note
                  for note in sections.open_onto(nameless, session).notes))

    print(f"\n-- the two doors on the menu --")

    # Off the menu bar's own actions and the two the tool holds -- and not by
    # enumerating the `QMenu`s under the bar, which is worth a note because it
    # is not obvious and it cost an afternoon. `findChildren(QMenu)` here makes
    # this run segfault on the way out about half the time: a teardown ordering
    # between PyQt, GDAL and the off-screen platform, with nothing to do with
    # what is being checked, and it arrives as a return code with every
    # assertion in the file already printed as passed. Which is the worst shape
    # a failure can have -- `run.py` reports this script as failed and the
    # output says it passed.
    bar = [action.text() for action in window.menuBar().actions()]

    check("the section gets a menu of its own, before the windows one",
          bar == ["&Section", "&Windows"], ", ".join(bar))

    check("with both directions on it, under the keys a hand reaches for",
          window.save_section_action.text() == "&Save section as..."
          and window.save_section_action.shortcut().toString() == "Ctrl+S"
          and window.open_section_action.text() == "&Open section..."
          and window.open_section_action.shortcut().toString() == "Ctrl+O",
          "Ctrl+S to keep one, Ctrl+O to come back to it")

    # Closed, and then what closing deferred is actually carried out -- the same
    # three lines `check_launcher.py` ends each of its handovers on. Restoring a
    # section rebuilds the bundle's canvas, which retires the old one through
    # `deleteLater`, and a widget still waiting to be deleted when the
    # interpreter exits is destroyed somewhere in the teardown of PyQt, GDAL and
    # the off-screen platform rather than by the event loop that was asked to do
    # it. The session is deliberately left open: closing it would pull the DEM
    # out from under a window Qt has yet to destroy.
    window.close()
    app.processEvents()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    app.processEvents()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for label in FAILURES:
            print(f"  - {label}")
        return 1

    print("all checks passed")
    return 0


if __name__ == "__main__":
    code = main()

    # Out without the interpreter's own teardown, which is a decision and not a
    # workaround, so here is the whole of it.
    #
    # This run segfaults on the way out, some of the time, after the last
    # assertion has been printed and the verdict is known. There is no Python
    # frame in it: it is the destruction of PyQt6, GDAL and PROJ objects in
    # whatever order the interpreter gets to them, in an off-screen process that
    # has built and retired more canvases than any other check here. Measured,
    # over fourteen runs each: half of them crashed, one in seven with the menu
    # read off the bar instead of by enumerating `QMenu`s, and one in seven
    # again with `deleteLater` flushed before closing. Both of those are worth
    # keeping and neither is the cause.
    #
    # What it costs to leave is worse than what it costs to cut: `run.py` runs
    # each check as a subprocess and reads its return code, so a teardown crash
    # is reported as this script failing while its own output says every
    # assertion passed -- the one failure shape that teaches the reader to
    # distrust the suite. Nothing after this line is under test, and the verdict
    # above it is complete.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
