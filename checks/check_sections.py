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

import os
import sys
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
        import tempfile

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
    import tempfile

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

    window.close()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed: " + ", ".join(FAILURES))
        return 1

    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
