"""
The sections tool: the bargain it strikes between a drag and a bundle.

The claims under test are the three the tool is built on. That a dragged trace
recomputes one profile and a released one recomputes the bundle. That the reach
of an anchored measurement is set here and changes what the section crosses.
And that the trace handed to the profiler is the one drawn -- two points --
because that is the difference between forty milliseconds and thirteen seconds.

Run against real data when it is there, against a synthetic DEM otherwise:

    QT_QPA_PLATFORM=offscreen python checks/check_sections.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from time import perf_counter

REPO = Path(os.environ.get("GSURF_REPO", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(REPO))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

FAILURES = []


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}{'   ' + detail if detail else ''}")
    if not condition:
        FAILURES.append(label)


def mouse(map_view, name, x, y):
    from matplotlib.backend_bases import MouseEvent

    px, py = map_view.axes.transData.transform((x, y))
    MouseEvent(name, map_view.canvas, int(px), int(py), button=1)._process()


def typed(box, digits):
    """The digits into a spin box, over whatever it held, as a hand puts them."""

    from PyQt6 import QtCore, QtGui, QtWidgets

    box.lineEdit().selectAll()

    for digit in digits:
        QtWidgets.QApplication.sendEvent(
            box.lineEdit(),
            QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress,
                QtCore.Qt.Key.Key_0 + int(digit),
                QtCore.Qt.KeyboardModifier.NoModifier,
                digit,
            ),
        )


def main():
    from PyQt6 import QtCore, QtWidgets

    from checks.synthetic import synthetic_dem
    from gsurf.session import Session
    from gsurf.tools import profiles as tool

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    # -- the data ---------------------------------------------------------

    real = Path(
        "/home/mauro/Documenti/Ricerca/AppenninoMeridionale/gstruct/misure_montealpi.gpkg"
    )
    dem_path = Path(
        "/home/mauro/Documenti/Ricerca/AppenninoMeridionale/DEM_5m/"
        "DEM_5m_Sirino_Raparo_Alpi_Pollino_TimpaSanLorenzo_Sellaro_Cerchiara.tif"
    )

    if real.exists() and dem_path.exists():
        spec = dict(
            path=str(real),
            layer="faglie_con_giacitura",
            role="traces",
            category_field="code",
            dip_dir_field="dip_dir",
            dip_field="dip",
        )
        session = Session.open(dem_path=str(dem_path), frame_layers=[spec])
        source = tool.read_traces(session, spec)
        print(f"Monte Alpi: {session.summary()}")
    else:
        # Left where it is written rather than in a `with`: the session holds
        # the DEM open for the whole run, and a temporary directory taken down
        # at the end of this branch would take the raster with it.
        session = Session.open(
            dem_path=str(synthetic_dem(Path(tempfile.mkdtemp()) / "synthetic.tif"))
        )
        source, spec = None, None
        print(f"synthetic: {session.summary()}")

    check("a session opens on the DEM", session.dem is not None, session.label)

    # The claim that nothing is written back is only worth making if it is
    # checked against the file itself.
    mtime = source.path.stat().st_mtime if source is not None else None

    if source is not None:
        check("the traces come in with their anchors",
              any(r.anchor is not None for r in source.traces),
              f"{sum(1 for r in source.traces if r.anchor is not None)} anchored "
              f"of {len(source.traces)}")

    window = tool.ProfilesWindow(session, traces=source, num_profiles=5)

    # Sized before it is shown: offscreen the default geometry collapses the
    # map axes to zero width, and then every map coordinate transforms to the
    # same pixel -- which makes the two ends of the trace one handle and the
    # gesture test meaningless rather than failing honestly.
    window.resize(1400, 950)
    window.show()
    app.processEvents()
    window.map_view.canvas.draw()

    check("the tool declares a required DEM", tool.WANTS["dem"] == "required")
    check("and takes traces without needing them", tool.WANTS["traces"] == "optional")

    # -- what a drag costs, against what a release costs -------------------

    map_view = window.map_view
    (x0, y0), (x1, y1) = window.trace

    apart = np.hypot(*(np.asarray(map_view.display_xy(x1, y1))
                       - np.asarray(map_view.display_xy(x0, y0))))
    check("the two ends are separate handles on screen",
          apart > 2 * window.HANDLE_RADIUS_PX, f"{apart:.0f} px")

    mouse(map_view, "button_press_event", x1, y1)
    check("an end is grabbed rather than a new trace started",
          window.dragging == "end", str(window.dragging))

    times = []
    for step in range(1, 6):
        start = perf_counter()
        mouse(map_view, "motion_notify_event", x1, y1 + step * 60.0)
        app.processEvents()
        times.append((perf_counter() - start) * 1000.0)

    single = sorted(times)[len(times) // 2]

    check("a drag redraws one profile",
          window.stack.currentWidget() is window.single,
          f"{len(window.single.view.axes):d} panel")

    start = perf_counter()
    mouse(map_view, "button_release_event", x1, y1 + 300.0)
    app.processEvents()
    bundle = (perf_counter() - start) * 1000.0

    check("and the release the bundle",
          window.stack.currentWidget() is window.bundle,
          f"{len(window.bundle.view.axes):d} panels")

    # -- one section in a panel built for one -------------------------------
    #
    # The blitting background must come out of the full draw with no data on
    # it. Captured with the profile in it, it is a background with a section
    # printed on it: every later frame restores that and draws the new curve on
    # top, and the panel shows two sections at once -- the live one and one
    # from wherever the trace stood when the background was taken.

    def ink_of(canvas):
        """The pixels the panel has drawn on, whatever colour they came out."""

        buf = np.asarray(canvas.buffer_rgba()).copy()

        return int((buf[:, :, :3].astype(int).sum(axis=2) < 700).sum())

    for which in ("single", "bundle"):
        canvas_panel = getattr(window, which)

        if canvas_panel is None or canvas_panel.background is None:
            continue

        drawn = canvas_panel.view.data_artists()

        check(f"the {which} panel keeps its data out of the full draw",
              drawn and all(a.get_animated() for a in drawn),
              f"{sum(a.get_animated() for a in drawn)} of {len(drawn)} animated")

        canvas_panel.canvas.restore_region(canvas_panel.background)
        in_background = ink_of(canvas_panel.canvas)

        for artist in drawn:
            artist.set_visible(False)
        canvas_panel.canvas.draw()
        without_section = ink_of(canvas_panel.canvas)

        for artist in drawn:
            artist.set_visible(True)
        canvas_panel.canvas.draw()
        with_section = ink_of(canvas_panel.canvas)

        section_px = with_section - without_section
        baked_px = in_background - without_section

        check(f"and nothing of the {which} section is baked into its background",
              section_px > 0 and baked_px <= 0.1 * section_px,
              f"{baked_px} px of the {section_px} the section draws")

    print(f"      drag frame {single:6.1f} ms   release {bundle:6.1f} ms")

    check("the drag is the cheaper of the two", single < bundle,
          f"{bundle / single:.1f}x")

    # -- three windows, and what keeps them one tool -----------------------
    #
    # The section and the records are top-level windows rather than docks. Two
    # things about that are worth holding down, and neither is visible in the
    # window. They are parented to the map, which is what has Qt destroy them
    # with the tool and -- the part that does not show at all -- what keeps
    # closing one from counting as the last window closed: while a tool runs
    # the launcher is hidden underneath, so a satellite without a parent would
    # take the application down instead of handing it back. And closing one
    # hides it, because the canvases inside are rebuilt at a price and a window
    # shut by accident should not cost that.

    group = window.window_group
    satellites = {name: w for name, w in group.items() if name != "map"}

    check("the map, the section and the records are three windows",
          sorted(group) == ["map", "section", "traces"], ", ".join(sorted(group)))

    check("each of them is a window in its own right",
          all(w.isWindow() for w in satellites.values()),
          f"{sum(w.isWindow() for w in satellites.values())} of {len(satellites)}")

    check("and each is parented to the map, so closing one cannot quit the app",
          all(w.parent() is window for w in satellites.values()))

    check("showing the map brings the group up with it",
          all(w.isVisible() for w in satellites.values()),
          f"{sum(w.isVisible() for w in satellites.values())} of {len(satellites)} up")

    def menu_action(text):
        """The entry of the Windows menu that switches one satellite."""

        for entry in window.menuBar().actions():
            if entry.menu() is None:
                continue
            for action in entry.menu().actions():
                if action.text().replace("&", "") == text:
                    return action

        return None

    section_window, action = group["section"], menu_action("Section")

    check("the Windows menu offers one entry per satellite",
          action is not None and menu_action("Traces") is not None)

    # Closed the way the window manager closes it, not hidden behind its back.
    section_window.close()

    check("closing a window hides it and leaves what is inside standing",
          not section_window.isVisible()
          and window.single is not None and window.bundle is not None,
          "the single panel and the bundle both survive")

    check("and the menu stops claiming a window that is not there",
          not action.isChecked())

    action.trigger()

    check("the menu puts it back", section_window.isVisible() and action.isChecked())

    # The guard that keeps a real desktop's arrangement out of this run. Without
    # it, a section window last left as a strip would come back as one here and
    # every ink count above would be measured on it.
    check("off-screen there is no layout to inherit",
          tool.remembered() is None and window.restore_geometry() is False)

    check("and nothing to inherit about the section either",
          tool.read_state(tool.remembered()) == {})

    # -- turning the section round -----------------------------------------

    def bundle_ends():
        """Where each profile of the bundle starts and where it ends."""

        return np.array([
            [line.coords[0, :2], line.coords[-1, :2]]
            for line in window.geoprofiles.profilers.lines
        ])

    before = [tuple(end) for end in window.trace]
    length_before = window._length()
    panels_before = window.bundle
    original_bundle = bundle_ends()

    window.reverse()

    check("reverse swaps the two ends",
          [tuple(end) for end in window.trace] == [before[1], before[0]])

    check("and leaves the section the same length",
          abs(window._length() - length_before) < 1e-6,
          f"{length_before / 1000.0:.2f} km")

    # A reversal is the one edit that cannot change the length, so the fitted
    # axis still fits and the panels must be reused rather than rebuilt --
    # which is what makes the button cheap enough to press twice in a row.
    check("the panels are kept, not rebuilt", window.bundle is panels_before)

    flipped = bundle_ends()

    # The bundle is laid out central around the trace -- half the profiles
    # offset to its left and half to its right -- so reversing the trace makes
    # each side the other. The set of lines on the ground is the same one; what
    # changes is which panel it is and which way it is read. Both at once, and
    # that is the whole claim: profile order reversed, and each profile's two
    # ends swapped.
    check("reversing mirrors the bundle: each side becomes the other",
          np.allclose(flipped, original_bundle[::-1, ::-1]),
          f"{len(flipped)} profiles, order and direction both turned round")

    window.reverse()

    check("and reversing twice comes back to where it started",
          [tuple(end) for end in window.trace] == before
          and np.allclose(bundle_ends(), original_bundle))

    check("the start end is drawn, so the direction is on the map",
          window.start_marker.get_xydata().tolist() == [list(window.trace[0])])

    # -- what the count box will hold --------------------------------------
    #
    # A central bundle has a middle, so `Profilers` refuses an even count
    # rather than choose a side for you, and the step of two keeps the arrows
    # off one. The box can still be typed into. What makes that worth a check
    # of its own is where the refusal lands: `_on_count_changed` is a Qt slot,
    # and an exception out of a slot under PyQt6 is qFatal -- not a traceback
    # over a status bar but SIGABRT, between one keystroke and the next, with
    # the section and the curation in the window at the time. Nothing here
    # could catch that if it happened, which is the point of checking: what is
    # under test is that it does not get that far.

    count_box = window.count_spin
    count_box.setValue(3)
    app.processEvents()

    handed_on = []
    count_box.valueChanged.connect(handed_on.append)

    typed(count_box, "4")
    app.processEvents()

    check("an even count typed in is never handed on to the section",
          not handed_on and count_box.value() == 3
          and len(window.geoprofiles.profilers.lines) == 3,
          f"the box reads {count_box.text()!r}, the map is still on "
          f"{len(window.geoprofiles.profilers.lines)}")

    # Allowed to stand while it is being typed, though, rather than refused
    # keystroke by keystroke: every count in the twenties begins with an even
    # digit and so does 41, and a box that would not hold one for a moment
    # would put half of its own range out of reach of the keyboard.
    check("but it may stand in the box on the way to an odd one",
          count_box.text() == "4", f"{count_box.text()!r} held, not swallowed")

    typed(count_box, "21")
    app.processEvents()

    check("and a count typed through an even digit arrives",
          count_box.value() == 21 and window.num_profiles == 21
          and len(window.geoprofiles.profilers.lines) == 21,
          f"{len(window.geoprofiles.profilers.lines)} profiles")

    # The end of the edit is what settles it, and it goes up. Dropping back to
    # whatever the box held before would be a keystroke that looked as though
    # it had never landed; one fewer profile than was asked for would be a box
    # that agrees with the map about a number neither was given.
    typed(count_box, "4")
    count_box.interpretText()       # Enter, or the focus going elsewhere
    app.processEvents()

    check("and an even one left standing is rounded up, not discarded",
          count_box.value() == 5 and window.num_profiles == 5
          and len(window.geoprofiles.profilers.lines) == 5,
          f"4 typed, {count_box.value()} drawn")

    count_box.valueChanged.disconnect(handed_on.append)

    # The keyboard is one door into the value and `setValue` is the other,
    # and that one does not pass the validator -- Qt only clamps it to the
    # range. It is the door `num_profiles` comes through as an argument, so
    # the count is read back off the box afterwards rather than the box being
    # set from it and the two left free to disagree about the number the
    # section was computed from.
    fresh = tool.OddSpinBox()
    fresh.setRange(*tool.BUNDLE_RANGE)
    fresh.setValue(4)

    check("an even count set on the box rather than typed is rounded as well",
          fresh.value() == 5 and window.count_spin.value() == window.num_profiles,
          f"setValue(4) -> {fresh.value()}")

    # -- what is carried from one run to the next --------------------------

    # Zoomed in before the state is read, and that is not decoration: over the
    # whole DEM "the framing came back" is the claim that the whole DEM comes
    # back as the whole DEM, which it would whether anything was restored or
    # not. Here the remembered framing is a twelve-kilometre window on an
    # eighty-kilometre mosaic and only a restore can produce it.
    cx, cy = session.center()
    window.map_view.axes.set_xlim(cx - 6000.0, cx + 6000.0)
    window.map_view.axes.set_ylim(cy - 5000.0, cy + 5000.0)
    window.map_view.canvas.draw()

    state = window.current_state()

    check("the framing read back is the one on screen, not the DEM's",
          state["extent"] != session.extent,
          f"{(state['extent'][1] - state['extent'][0]) / 1000.0:.1f} km "
          f"of {(session.extent[1] - session.extent[0]) / 1000.0:.1f}")

    check("the state written is the section as it stands",
          [tuple(end) for end in state["trace"]] == before
          and state["profiles"] == window.num_profiles
          and state["extent"] == window.map_view.framing,
          f"{state['profiles']} profiles, {state['offset']:.0f} m apart")

    settings = QtCore.QSettings(
        str(Path(tempfile.mkdtemp()) / "sections.conf"),
        QtCore.QSettings.Format.IniFormat,
    )
    settings.setValue(tool.STATE_KEY, json.dumps(state))

    check("and it survives the round trip through the store",
          tool.applicable(session, tool.read_state(settings)).get("trace")
          == [tuple(end) for end in state["trace"]])

    # The gate. Coordinates are metres in a projection: the same pair is
    # somewhere else under another one and nowhere at all on another DEM.
    elsewhere = dict(state, source="/somewhere/else.tif")
    rotated = dict(state, epsg=(session.epsg or 0) + 1)

    check("a trace from another source is not restored onto this one",
          "trace" not in tool.applicable(session, elsewhere)
          and "extent" not in tool.applicable(session, elsewhere))

    check("nor one written under another projection",
          "trace" not in tool.applicable(session, rotated))

    check("but the way of working carries across anyway",
          tool.applicable(session, elsewhere).get("profiles") == state["profiles"]
          and tool.applicable(session, elsewhere).get("offset") == state["offset"],
          "profiles, spacing and reach are habits, not places")

    left, bottom, right, top = session.bounds
    off_dem = dict(state, trace=[[left - 5000.0, bottom - 5000.0], [right, top]])

    check("a trace that is no longer on the DEM is dropped, not shown empty",
          "trace" not in tool.applicable(session, off_dem))

    check("and so is a state that was never written",
          tool.applicable(session, {}) == {})

    # The conf is a file a hand can reach, and a window is built on it before
    # there is an application running to show an error in. An even count is the
    # one that matters: `Profilers` raises on it during construction, so a
    # number this tool never wrote would be a window that cannot be opened at
    # all without finding and deleting the file.
    check("an even count from an edited file is dropped, not built on",
          "profiles" not in tool.applicable(session, dict(state, profiles=4)))

    check("and so is a spacing the spin box could not hold",
          "offset" not in tool.applicable(session, dict(state, offset=1e9))
          and "offset" not in tool.applicable(session, dict(state, offset=0.0)),
          "a box clamped to 20 km over a section computed at 1000 km")

    check("but a reach of none is a value, not a missing one",
          tool.applicable(session, dict(state, reach=None)).get("reach", "gone")
          is None,
          "'whole trace' has to survive the round trip")

    check("whether the legend is up is a habit, not a place",
          tool.applicable(session, dict(elsewhere, legend=False)).get("legend")
          is False
          and "legend" not in tool.applicable(session, dict(state, legend="yes")),
          "carried across sources; anything but a boolean dropped")

    # -- a window that comes up on what was left behind --------------------

    kept = dict(state, profiles=3, offset=750.0, reach=180.0)
    continued = tool.ProfilesWindow(session, traces=source, state=kept)
    continued.resize(1400, 950)
    continued.show()
    app.processEvents()
    continued.map_view.canvas.draw()

    check("a new window opens on the remembered trace",
          [tuple(end) for end in continued.trace] == before)

    check("and on its bundle, with the controls saying so",
          continued.num_profiles == 3 and continued.offset == 750.0
          and continued.count_spin.value() == 3
          and continued.offset_spin.value() == 750.0)

    # The reach reaches the source and not just the box: it is the source the
    # section is computed from, and a spin box agreeing with a state nobody
    # applied would be the failure this cannot tell apart otherwise.
    check("and the reach is the source's again, not just the box's",
          continued.traces.half_span == 180.0
          and continued.reach_spin.value() == 180.0,
          "180 m either side")

    # The bar holds one action and no icon, and a toolbar left on its default
    # style shows icons: a button that came out as an empty 20 pixels would
    # pass every other check here.
    section_bar = next(
        bar for bar in continued.findChildren(QtWidgets.QToolBar)
        if bar.windowTitle() == "section"
    )
    reverse_button = section_bar.widgetForAction(continued.reverse_action)

    check("the reverse button is on the bar, wide enough to read",
          reverse_button is not None
          and reverse_button.text() == "Reverse"
          and reverse_button.sizeHint().width() > 40
          and continued.reverse_action.shortcut().toString() == "Ctrl+R",
          f"{reverse_button.sizeHint().width()} px, Ctrl+R")

    check("the remembered framing is what the map comes up on",
          np.allclose(continued.map_view.framing, state["extent"], atol=1.0),
          f"{continued.map_view.framing[1] - continued.map_view.framing[0]:.0f} m wide")

    # Home has to be the DEM and not the corner the last run ended on, or a
    # restored zoom would be one there is no way back out of.
    continued.map_view.toolbar.home()

    check("and home is still the whole area, not that framing",
          np.allclose(continued.map_view.framing, session.extent, atol=1.0),
          "the bar's home is anchored before the framing is pushed over it")

    continued.close()

    # -- the trace goes to the profiler as drawn ---------------------------

    profilers = window._profilers(1)
    vertices = profilers.lines[0].num_points()

    check("the profiler gets the trace drawn, not a sampled one",
          vertices == 2, f"{vertices} vertices")

    # -- reach ------------------------------------------------------------

    if source is not None:
        def trace_metres():
            return sum(
                line.length_2d()
                for records in source.records.values()
                for _, lines in records
                for line in lines
            )

        source.set_half_span(None)
        whole = trace_metres()

        source.set_half_span(50.0)
        short = trace_metres()

        check("shortening the reach shortens what is intersected",
              short < whole, f"{whole:.0f} m -> {short:.0f} m")

        anchored = [r for r in source.traces if r.anchor is not None]
        fitted = [r for r in source.traces if r.anchor is None]

        check("and leaves the traces that carry no anchor whole",
              all(r.is_whole(50.0) for r in fitted),
              f"{len(fitted)} fitted, {len(anchored)} anchored")

        source.set_half_span(250.0)
        window.update_bundle()

    # -- the panel: what a section is allowed to be argued with ------------

    if window.panel is not None:
        panel = window.panel

        check("every record has a row", panel.table.rowCount() == len(source.traces),
              f"{panel.table.rowCount()} rows")

        anchored = next(i for i, r in enumerate(source.traces) if r.anchor is not None)
        record = source.traces[anchored]

        before = (record.plane.dipazim, record.plane.dipang)
        panel.table.item(anchored, 1).setText("123/45")

        check("an attitude typed into the table reaches the record",
              (record.plane.dipazim, record.plane.dipang) == (123.0, 45.0),
              f"{before[0]:.0f}/{before[1]:.0f} -> "
              f"{record.plane.dipazim:.0f}/{record.plane.dipang:.0f}")

        panel.table.item(anchored, 1).setText("nonsense")
        check("and one that is not an attitude is refused, not stored",
              (record.plane.dipazim, record.plane.dipang) == (123.0, 45.0),
              f"{record.plane.dipazim:.0f}/{record.plane.dipang:.0f}")

        panel.table.item(anchored, 4).setText("80")
        s0, s1 = record.extent(source.half_span)
        check("a reach typed into the table becomes that record's own",
              record.span is not None and abs((s1 - s0) / 2.0 - 80.0) < 1e-6,
              f"{(s1 - s0) / 2.0:.0f} m either side of {record.anchor:.0f}")

        text, _ = panel.curation_text()
        check("a reach that was set is an assertion", "span reach" in text)

        panel.table.item(anchored, 4).setText("")
        check("and emptying it hands the record back to the default",
              record.span is None, str(record.span))

        text, _ = panel.curation_text()
        check("a reach left on the default is not one",
              "span reach" not in text,
              "nobody decided 250 m, so nobody says so")

        panel.table.item(anchored, 0).setCheckState(QtCore.Qt.CheckState.Unchecked)
        check("unchecking a row takes it out of the section",
              not record.enabled and all(
                  record.plane is not plane
                  for records in source.records.values()
                  for plane, _ in records
              ),
              f"{sum(1 for r in source.traces if not r.enabled)} held out")

        text, written = panel.curation_text()
        check("the edits come out as a gstruct fragment",
              written >= 1 and "span use * * excluded" in text,
              f"{written} structure(s), {len(text.splitlines())} lines")
        check("and the source layer was never written to",
              source.path.stat().st_mtime == mtime, source.path.name)

    # -- the backdrop, as surveyors actually leave it ----------------------
    #
    # A mapped unit does not always arrive as a polygon. The Conglomerato di
    # Santa Croce, in the `carbonates` layer of geology.gpkg, is stored as a
    # GeometryCollection of its polygon plus two dangling edges 4 and 56 m
    # long, and GeometryCollection is the one container whose name does not
    # begin with 'Multi' -- so a test on the outer type walked straight into
    # `.exterior` and took the tool down before the window opened.

    from shapely.geometry import (
        GeometryCollection, LineString, MultiLineString, MultiPolygon, Point,
    )
    from shapely import wkt as shapely_wkt

    square = shapely_wkt.loads("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")
    holed = shapely_wkt.loads(
        "POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0), (2 2, 2 4, 4 4, 4 2, 2 2))"
    )
    dangled = GeometryCollection(
        [LineString([(0, 0), (0, 4)]), LineString([(9, 9), (9, 65)]), square]
    )

    kept, skipped = tool.single_parts(dangled, "Polygon")
    check("a unit stored as a collection still yields its polygon",
          [p.geom_type for p in kept] == ["Polygon"] and skipped == 2,
          f"{len(kept)} kept, {skipped} left out")

    kept, skipped = tool.single_parts(
        GeometryCollection([MultiPolygon([square]), Point(0, 0)]), "Polygon"
    )
    check("and the containers are unwrapped however they nest",
          len(kept) == 1 and skipped == 1, f"{len(kept)} kept, {skipped} left out")

    kept, _ = tool.single_parts(holed, "Polygon")
    check("a polygon keeps the holes in it", len(kept[0].interiors) == 1,
          f"{len(kept[0].interiors)} interior ring(s)")

    kept, skipped = tool.single_parts(LineString([(0, 0), (1, 1)]), "Polygon")
    check("a stray line in a polygon layer is counted, not converted",
          kept == [] and skipped == 1, f"{len(kept)} kept, {skipped} left out")

    check("what was left out is said, not swallowed",
          isinstance(window.overlay_dropped, dict),
          str(window.overlay_dropped) or "nothing left out")

    # -- and the same, for the traces the attitudes are read off ------------
    #
    # A trace layer meets the same geometry. Before, anything not named after a
    # line was filtered out of the frame without a word, so a fault that came
    # back from a cleaning pass as a collection vanished with its measurement.

    import geopandas as gpd

    from gsurf.attitudes import TraceAttitudeSource

    rows = [
        ("plain", LineString([(0, 0), (1000, 0)]), 120.0, 40.0),
        ("multi", MultiLineString([[(0, 500), (400, 500)], [(600, 500), (1000, 500)]]),
         130.0, 45.0),
        ("cleaned", GeometryCollection(
            [LineString([(0, 1000), (1000, 1000)]), Point(500, 1000)]), 140.0, 50.0),
        ("no line", GeometryCollection([Point(0, 1500), Point(10, 1500)]), 150.0, 55.0),
        ("polygon", shapely_wkt.loads("POLYGON ((0 2000, 10 2000, 10 2010, 0 2000))"),
         160.0, 60.0),
        ("nothing", None, 170.0, 65.0),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        mixed = Path(tmp) / "mixed_traces.gpkg"

        gpd.GeoDataFrame(
            {"code": [r[0] for r in rows],
             "dip_dir": [r[2] for r in rows],
             "dip": [r[3] for r in rows]},
            geometry=[r[1] for r in rows],
            crs="EPSG:25833",
        ).to_file(mixed, layer="traces", driver="GPKG")

        mixed_source = TraceAttitudeSource(
            mixed, crs="EPSG:25833", layer="traces",
            category_field="code", dip_dir_field="dip_dir", dip_field="dip",
        )

    kept_categories = {r.category for r in mixed_source.traces}

    check("a trace stored as a collection is used, not filtered away",
          "cleaned" in kept_categories,
          f"{sorted(kept_categories)}")
    check("a trace interrupted stays several lines under one record",
          len(next(r for r in mixed_source.traces
                   if r.category == "multi").lines) == 2)
    check("a collection with no line in it leaves no empty record",
          "no line" not in kept_categories)
    check("a polygon and a null are not traces",
          {"polygon", "nothing"}.isdisjoint(kept_categories))
    check("and what was left out is counted on both grounds",
          mixed_source.dropped.get("with no line geometry") == 2
          and mixed_source.dropped.get("not a line") == 3,
          str(mixed_source.dropped))

    # -- what the section says it crosses -----------------------------------

    print("\n-- the legend beside the panels --")

    from shapely.geometry import box

    with tempfile.TemporaryDirectory() as tmp:
        units_path, lines_path = Path(tmp) / "unita.gpkg", Path(tmp) / "tracce.gpkg"

        left, bottom, right, top = session.bounds
        cx, cy = session.center()

        # Two units on the line and one off it, and the same for the lines. The
        # claim is that the column names what this section goes through, and a
        # backdrop every profile crosses cannot tell that apart from a backdrop
        # listed whole. Four kilometres north is far enough to be another
        # section and near enough to stay on the smallest DEM this runs on.
        band, away = 400.0, 4000.0

        gpd.GeoDataFrame(
            {"unita": ["crossed one", "crossed two", "elsewhere"]},
            geometry=[
                box(cx - 3000.0, cy - band, cx - 1000.0, cy + band),
                box(cx - 500.0, cy - band, cx + 2000.0, cy + band),
                box(cx - 3000.0, cy + away - band, cx + 2000.0, cy + away + band),
            ],
            crs=session.crs,
        ).to_file(units_path, layer="unita", driver="GPKG")

        gpd.GeoDataFrame(
            {"tipo": ["met", "missed"]},
            geometry=[
                LineString([(cx, cy - 1500.0), (cx, cy + 1500.0)]),
                LineString([(cx, cy + away - 1500.0), (cx, cy + away + 1500.0)]),
            ],
            crs=session.crs,
        ).to_file(lines_path, layer="tracce", driver="GPKG")

        backdrop = Session.open(
            dem_path=str(session.dem.path) if session.dem is not None else None,
            vectors=[
                dict(path=str(units_path), role="polygons", layer="unita",
                     category_field="unita"),
                dict(path=str(lines_path), role="lines", layer="tracce",
                     category_field="tipo"),
            ],
        )

        shown = tool.ProfilesWindow(backdrop, num_profiles=1)
        shown.resize(1000, 700)
        shown.show()
        app.processEvents()

        shown.trace = [(cx - 3500.0, cy), (cx + 2500.0, cy)]
        shown._redraw_trace()
        shown.update_bundle()

        entries = shown._section_legend_entries(shown.geoprofiles)
        patches = {label for kind, label, _ in entries if kind == "patch"}
        notes = [label for kind, label, _ in entries if kind == "note"]
        palette = shown._polygon_colors()

        check("the legend names the units this section crosses",
              patches == {"crossed one", "crossed two"},
              f"{len(palette)} in the palette, {len(patches)} on the panel")

        # The reason this cannot be taken from the library: the profiler hands
        # back an entry per category it was given, crossed or not, so the ids
        # alone say every unit in the window is on the section.
        from geogst.plots.profiles import polygon_intersections_categories

        check("which is fewer than the ids the intersections carry",
              len(polygon_intersections_categories(shown.geoprofiles.polygons_intersections))
              > len(patches),
              "categories with no piece in them are in there too")

        check("each swatch is the colour the panel drew that unit in",
              all(color == tool._as_color(palette[label])
                  for kind, label, color in entries if kind == "patch"))

        dots = [(label, color) for kind, label, color in entries if kind == "dot"]

        check("the line crossings claim one colour between them",
              len(dots) == 1 and dots[0][0].startswith("crossings ("),
              dots[0][0] if dots else "no entry")

        check("and the categories are named under it, the ones met only",
              any(note.startswith("met") for note in notes)
              and not any(note.startswith("missed") for note in notes),
              ", ".join(notes))

        check("with no traces loaded there is nothing to say about attitudes",
              not any(kind == "tick" for kind, _, _ in entries))

        # -- the column itself

        legend = shown.section_legend
        body = legend.widget()

        def rows():
            return [w for w in body.findChildren(QtWidgets.QWidget)
                    if w.parent() is body]

        legend.set_entries([("heading", "one", None),
                            ("patch", "a", (0.0, 0.0, 1.0))])
        app.processEvents()
        legend.set_entries([("heading", "two", None)])

        # Before `processEvents`, deliberately: `deleteLater` has not run yet,
        # so this is the check that the old rows were unparented and not merely
        # taken out of the layout. Left to the deletion alone, the column shows
        # two legends at once for as many frames as a drag has.
        check("a rebuilt column has none of the old rows left in it",
              len(rows()) == 1, f"{len(rows())} rows for 1 entry")

        standing = [id(w) for w in rows()]
        legend.set_entries([("heading", "two", None)])

        check("and asking for the same legend again rebuilds nothing",
              [id(w) for w in rows()] == standing,
              "the comparison is what keeps this off the frame budget")

        # -- put away, and caught up on the way back

        shown.legend_action.setChecked(False)
        current = legend._entries

        shown.trace = [(cx - 3500.0, cy + away), (cx + 2500.0, cy + away)]
        shown._redraw_trace()
        shown.update_bundle()

        check("put away, it is not worked out either",
              legend.isHidden() and legend._entries == current,
              "the walk is cheap but it is per frame")

        shown.legend_action.setChecked(True)
        moved = {label for kind, label, _ in legend._entries if kind == "patch"}

        check("and it catches up on the section it comes back to",
              not legend.isHidden() and moved == {"elsewhere"},
              ", ".join(sorted(moved)) or "nothing")

        shown.close()
        backdrop.close()

    if source is not None:
        # Put across a trace rather than left wherever the checks above ended,
        # so that the branch under test is the one that has something to say.
        # Across and not along: a section parallel to a fault can run its whole
        # length beside it without ever meeting it.
        record = next(r for r in source.traces if r.enabled and r.lines)
        ends = record.lines[0].coords

        (x0, y0), (x1, y1) = ends[0, :2], ends[-1, :2]
        dx, dy = x1 - x0, y1 - y0
        across = np.array([-dy, dx]) / np.hypot(dx, dy) * 1500.0
        middle = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])

        # Whole trace: the reach is centred on the anchor and the middle of a
        # line need not be inside it.
        source.set_half_span(None)

        window.trace = [tuple(middle - across), tuple(middle + across)]
        window._redraw_trace()
        window.update_bundle()

        crossings = tool._attitude_crossings(window.geoprofiles)
        ticks = [label for kind, label, _ in
                 window._section_legend_entries(window.geoprofiles) if kind == "tick"]

        check("the attitudes are named where the section meets any",
              (len(ticks) == 1 and f"({crossings})" in ticks[0])
              if crossings else not ticks,
              f"{crossings} crossings")

    # -- a backdrop nobody categorised --------------------------------------

    print("\n-- a backdrop layer taken as it comes --")

    with tempfile.TemporaryDirectory() as tmp:
        plain = Path(tmp) / "plain.gpkg"
        left, bottom, right, top = session.bounds

        gpd.GeoDataFrame(
            {"nota": ["a", "b"]},
            geometry=[
                LineString([(left + 200, bottom + 200), (right - 200, top - 200)]),
                LineString([(left + 200, top - 200), (right - 200, bottom + 200)]),
            ],
            crs=session.crs,
        ).to_file(plain, layer="faglie", driver="GPKG")

        # No category field, which for a line backdrop is not a corner: the
        # picker guesses one only for polygons, so a fault layer arrives this
        # way unless somebody goes and chooses. The frame then has no category
        # column at all, and reading one ended the tool here with a KeyError
        # that the launcher could only show verbatim.
        bare = Session.open(
            dem_path=str(session.dem.path) if session.dem is not None else None,
            vectors=[dict(path=str(plain), role="lines", layer="faglie")],
        )

        try:
            second = tool.ProfilesWindow(bare, num_profiles=1)
            opened, reason = True, ""
        except Exception as err:
            second, opened, reason = None, False, f"{type(err).__name__}: {err}"

        check("the tool opens on it", opened, reason)

        if second is not None:
            check(
                "and the layer is one category, named after itself",
                list(second.lines) == ["faglie"],
                str(list(second.lines)),
            )
            second.close()

        bare.close()

    window.close()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed: " + ", ".join(FAILURES))
        return 1

    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
