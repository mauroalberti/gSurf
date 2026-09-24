"""
Does a line layer with no attitude columns open, and stay honest until it is fitted?

A mapped contact carries a plane in its geometry and the ground under it, so a
layer that names no dip columns is not an incomplete layer -- it is the one the
fit exists for. What has to hold is that it opens, that it says plainly it has
read nothing, that it puts nothing in a section until something reads it, and
that the fit then gives back the plane the ground was built on.

The DEM here *is* a plane of known attitude, and the trace is a V laid on it in
plan. That way the answer is known by construction at both ends: every point of
the trace lies on the plane, so the attitude coming back off it is the one the
surface was made with, and the whole path from `fit_records` down is exercised
rather than the pieces under it.

    python check_bare_traces.py
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("GSURF_REPO", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(REPO))

FAILURES = []

# The plane everything here is built on, and the ground it sits on.
DIP_DIRECTION, DIP = 130.0, 55.0
ORIGIN_X, ORIGIN_Y = 600200.0, 4419900.0
CELL = 5.0


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}{'   ' + detail if detail else ''}")
    if not condition:
        FAILURES.append(label)


def pole(dip_direction, dip):
    """(East, North, Up) of the upward normal, as `check_traces.py` builds it."""

    a, d = np.radians(dip_direction), np.radians(dip)

    return np.array([np.sin(a) * np.sin(d), np.cos(a) * np.sin(d), np.cos(d)])


def separation(one, other):
    """Degrees between two planes, as the angle between their poles."""

    dot = float(np.clip(np.dot(pole(*one), pole(*other)), -1.0, 1.0))

    return np.degrees(np.arccos(abs(dot)))


def planar_dem(path, width=140, height=300, z0=2000.0):
    """A DEM that is the plane itself, so a trace on it has an attitude exactly."""

    import rasterio
    from rasterio.transform import from_origin

    n = pole(DIP_DIRECTION, DIP)

    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs = ORIGIN_X + (cols + 0.5) * CELL
    ys = ORIGIN_Y + (height - rows - 0.5) * CELL

    z = (z0 - (n[0] * (xs - ORIGIN_X) + n[1] * (ys - ORIGIN_Y)) / n[2]).astype("float32")

    profile = dict(
        driver="GTiff", height=height, width=width, count=1, dtype="float32",
        crs="EPSG:25833", nodata=-9999.0,
        transform=from_origin(ORIGIN_X, ORIGIN_Y + height * CELL, CELL, CELL),
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(z, 1)

    return path


def vee(x0=600400.0, y0=4420050.0, width=250.0, length=1300.0, step=10.0):
    """A plan-view V, which is what a plane crossing a valley draws on a map."""

    ys = np.arange(0.0, length + step, step)
    xs = x0 + width * np.abs(ys / (length / 2.0) - 1.0)

    return np.column_stack([xs, y0 + ys])


def main():
    import geopandas as gpd
    from shapely.geometry import LineString, Point

    from gsurf.attitudes import TraceAttitudeSource, TraceRecord

    print("-- a layer that names no angles --\n")

    plan = vee()

    # Two contacts under one name, which is the case that says whether records
    # are pooled: with attitudes they would be if the attitudes agreed, and
    # here there are none to agree.
    rows = [
        ("contatto", LineString(plan)),
        ("contatto", LineString([(600250, 4420100), (600600, 4420100)])),
        ("faglia", LineString([(600250, 4421100), (600600, 4421400)])),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bare.gpkg"

        gpd.GeoDataFrame(
            {"Tipo": [r[0] for r in rows], "OBJECTID": [1, 2, 3]},
            geometry=[r[1] for r in rows],
            crs="EPSG:25833",
        ).to_file(path, layer="limiti", driver="GPKG")

        bare = TraceAttitudeSource(
            path, crs="EPSG:25833", layer="limiti", category_field="Tipo",
        )

        half = TraceAttitudeSource(
            path, crs="EPSG:25833", layer="limiti", category_field="Tipo",
            dip_dir_field="OBJECTID",
        )

    check("a layer with no angle fields opens", bare.problem is None, str(bare.problem))
    check("and knows it is carrying no attitudes", bare.has_attitudes is False)
    check("one record per feature, none of them pooled",
          len(bare.traces) == 3, f"{len(bare.traces)} records from 3 features")
    check("and not one of them has a plane",
          all(r.plane is None for r in bare.traces) and bare.unread == 3)
    check("the geometry is read as usual",
          all(len(r.lines) == 1 for r in bare.traces)
          and abs(bare.traces[1].length - 350.0) < 1.0,
          f"{bare.traces[1].length:.0f} m")
    check("the other columns still come through as attributes",
          bare.traces[0].attrs.get("OBJECTID") == 1, str(bare.traces[0].attrs))

    check("nothing reaches the section until something is read",
          bare.records == {} and bare.attitudes() == [], str(bare.records))
    check("and the summary says so rather than counting planes",
          "no attitude read" in bare.summary(), bare.summary())

    check("one angle field named and the other not is refused",
          half.problem is not None and "one angle" in half.problem, str(half.problem))

    # -- the picker, which is where these layers were being turned away -----

    print("\n-- the dialog --\n")

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PyQt6 import QtWidgets

    from gsurf.sources import AttitudePicker, TracePicker

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bare.gpkg"

        gpd.GeoDataFrame(
            {"Tipo": [r[0] for r in rows], "OBJECTID": [1, 2, 3]},
            geometry=[r[1] for r in rows],
            crs="EPSG:25833",
        ).to_file(path, layer="limiti", driver="GPKG")

        traces, attitudes = TracePicker(), AttitudePicker()

        traces.set_path(str(path), "limiti", quiet=True)
        attitudes.set_path(str(path), "limiti", quiet=True)

        check("a trace layer fills its slot with no angles named",
              traces.is_filled is True)
        check("and the spec it hands over says so in both fields",
              traces.value()["dip_dir_field"] is None
              and traces.value()["dip_field"] is None,
              str({k: v for k, v in traces.value().items() if "field" in k}))

        # The same file in the attitudes slot, where a layer naming no angles
        # is a set of stations with nothing measured at them.
        check("an attitude layer still will not", attitudes.is_filled is False)

        traces.dip_dir_combo.setCurrentText("OBJECTID")

        check("half an answer fills nothing", traces.is_filled is False
              and traces.value() is None)

        traces.dip_dir_combo.setCurrentText(traces.NO_FIELD)

        # The round trip that decides whether a deliberate blank survives. A
        # null in the spec has to read as an answer: guessed at again, a layer
        # opened bare on Monday comes back with columns on Tuesday.
        restored = TracePicker()
        restored.restore(dict(traces.value()))

        check("a blank answered is put back blank, not guessed at again",
              restored.value()["dip_dir_field"] is None
              and restored.value()["dip_field"] is None,
              str({k: v for k, v in restored.value().items() if "field" in k}))

    # -- the fit, which is the only way such a layer says anything ----------

    print("\n-- read off the trace --\n")

    from gsurf.dem import Dem
    from gsurf.traces import describe_fit, fit_records

    with tempfile.TemporaryDirectory() as tmp:
        dem = Dem(planar_dem(Path(tmp) / "plane.tif"))

        surveyed = [r for r in bare.traces if r.category == "contatto"]
        fitted, report = fit_records(surveyed, dem)

        check("the V determines a plane and the straight line does not",
              report["fitted"] == 1 and report["silent"] == 1,
              f"{report['fitted']} fitted, {report['silent']} silent")
        check("and it is the plane the ground was built on",
              fitted and separation(
                  (DIP_DIRECTION, DIP),
                  (fitted[0].plane.dipazim, fitted[0].plane.dipang)) < 2.0,
              f"{fitted[0].plane.dipazim:.1f}/{fitted[0].plane.dipang:.1f}"
              if fitted else "nothing fitted")

        # Into the source, which is what the panel's `apply_fit` does.
        bare.set_traces(fitted)

        check("and now the section has something to draw",
              list(bare.records) == ["contatto"], str(list(bare.records)))

        # -- stopped part way ------------------------------------------------

        print("\n-- stopped half way --\n")

        stopped_at = []

        def stop_at_two(done, total):
            stopped_at.append(done)
            return done < 2

        many = [
            TraceRecord(category="c", plane=None, lines=list(surveyed[0].lines),
                        length=surveyed[0].length)
            for _ in range(5)
        ]

        got, report = fit_records(many, dem, progress=stop_at_two)

    check("a stopped fit says it was stopped", report["stopped"] is True)
    check("and gave up where it was told to", max(stopped_at) == 2, str(stopped_at))
    check("describe_fit leads with that, not with a tally",
          "Stopped" in describe_fit(report), describe_fit(report)[:60])

    # -- the panel, where a record with no plane has to be shown as one ------

    print("\n-- the trace panel --\n")

    from gsurf.tools.profiles import UNREAD, TracePanel

    blank = TraceRecord(category="contatto", plane=None, lines=[], length=100.0)
    given = TraceRecord(category="faglia", plane=fitted[0].plane, lines=[], length=100.0)

    check("a record with no plane reads as unread, not as a blank cell",
          TracePanel._attitude_text(blank) == UNREAD,
          TracePanel._attitude_text(blank))
    check("and one with a plane still reads as its attitude",
          "/" in TracePanel._attitude_text(given), TracePanel._attitude_text(given))

    # -- cutting the layer down to the ground the section covers -------------

    print("\n-- the ground being worked on --\n")

    from geogst.core.geometries.shapes.lines import Ln

    from gsurf.attitudes import within
    from gsurf.tools.profiles import describe_scope, section_swath

    def line_record(name, points):
        line = Ln(np.asarray(points, dtype=float))

        return TraceRecord(category=name, plane=None, lines=[line],
                           length=float(line.length_2d()))

    # A section along y = 0 from x = 0 to 1000, three profiles 100 m apart: the
    # bundle reaches 100 m either side and the margin is one more spacing, so
    # everything within 200 m of the line is in.
    area = section_swath([(0.0, 0.0), (1000.0, 0.0)], 3, 100.0)

    check("the swath reaches the bundle's width plus one more spacing",
          area.contains(Point(500.0, 199.0)) and not area.contains(Point(500.0, 201.0)),
          f"{(3 + 1) / 2 * 100.0:.0f} m either side")

    check("a bundle with no spacing has no swath", section_swath(
        [(0.0, 0.0), (1000.0, 0.0)], 3, 0.0) is None)

    inside = line_record("in", [(400.0, -50.0), (600.0, 50.0)])
    outside = line_record("out", [(400.0, 500.0), (600.0, 500.0)])
    straddling = line_record("half", [(500.0, 100.0), (500.0, 900.0)])

    # Bounding box over the swath's, geometry nowhere near it: the assertion
    # that the cheap first test has not quietly become the answer.
    corner = line_record("corner", [(1150.0, 150.0), (1190.0, 190.0)])

    scattered = TraceRecord(
        category="scattered", plane=None, length=0.0,
        lines=[Ln(np.array([[3000.0, 3000.0], [3100.0, 3000.0]])),
               Ln(np.array([[400.0, -10.0], [500.0, 10.0]]))],
    )

    candidates = [inside, outside, straddling, corner, scattered]
    kept = within(candidates, area)

    check("a trace inside is kept and one outside is not",
          inside in kept and outside not in kept)

    check("a bounding box over the swath is not enough",
          corner not in kept,
          f"its box is {corner.lines[0].coords[:, 0].min():.0f}-"
          f"{corner.lines[0].coords[:, 0].max():.0f} east, the swath's ends "
          f"{area.bounds[2]:.0f}")

    check("a record is kept for any one of its lines",
          scattered in kept, "one line 3 km away, one crossing the section")

    # The one the whole rule turns on. Clipping here would put an endpoint on
    # the trace where the selection stops, and the fit would then read a plane
    # off a bend that is the edge of a decision rather than of a contact.
    check("a trace crossing the edge is kept whole, not cut at it",
          straddling in kept and abs(straddling.length - 800.0) < 1e-9
          and len(straddling.lines[0].coords) == 2,
          f"{straddling.length:.1f} m, {len(straddling.lines[0].coords)} vertices")

    check("and the records that survive are the very same objects",
          all(any(k is c for c in candidates) for k in kept))

    # -- the panel, where the cut decides what gets fitted -------------------

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scoped.gpkg"

        # Two on the section's ground -- the V, which determines a plane, and a
        # straight line, which does not -- and one a kilometre east of both.
        gpd.GeoDataFrame(
            {"Tipo": ["contatto", "contatto", "faglia"]},
            geometry=[
                LineString(vee()),
                LineString([(600250, 4420100), (600600, 4420100)]),
                LineString([(601500, 4420100), (601800, 4420100)]),
            ],
            crs="EPSG:25833",
        ).to_file(path, layer="limiti", driver="GPKG")

        source = TraceAttitudeSource(
            path, crs="EPSG:25833", layer="limiti", category_field="Tipo",
        )

        dem = Dem(planar_dem(Path(tmp) / "plane2.tif"))
        swath = section_swath([(600500, 4420000), (600500, 4421400)], 3, 100.0)

        panel = TracePanel(source, dem=dem, swath=lambda: swath)

        # Held in a name: a panel built inside the call is collected before the
        # button is read, and Qt takes the C++ widget with it.
        unsectioned = TracePanel(source, dem=dem)

        check("with no section to cut to the button is not offered",
              unsectioned.scope_button.isEnabled() is False)

        check("the panel opens on the whole layer",
              panel.table.rowCount() == 3 and not panel.scoped)

        report = panel.set_scope(swath)

        check("the cut leaves only what the section reaches",
              panel.table.rowCount() == 2 and panel.scoped
              and report["kept"] == 2 and report["total"] == 3,
              f"{report['kept']} of {report['total']}")

        check("and the button now offers the way back",
              panel.scope_button.text() == "All of the layer",
              panel.scope_button.text())

        panel.apply_fit()

        check("the fit runs on what is in scope, not on the layer",
              panel.fitted and panel.table.rowCount() == 1
              and all(r.attrs.get("fitted") for r in source.traces),
              f"{panel.table.rowCount()} fitted from 2 in scope")

        panel.restore_surveyed()

        check("and undoing it comes back to the scope, not to the layer",
              panel.table.rowCount() == 2 and panel.scoped and not panel.fitted,
              f"{panel.table.rowCount()} rows")

        # A fit is a claim about a set of records, so a different set ends it.
        panel.apply_fit()
        dropped = panel.set_scope(swath)

        check("cutting again drops the fit and says it did",
              dropped["dropped_fit"] and not panel.fitted
              and "fit did not come across" in describe_scope(dropped),
              describe_scope(dropped).splitlines()[-1][:60])

        elsewhere = section_swath([(700000, 4500000), (701000, 4500000)], 3, 100.0)
        empty = panel.set_scope(elsewhere)

        check("a cut that would empty the table changes nothing",
              empty["empty"] and panel.table.rowCount() == 2 and panel.scoped
              and "nothing was changed" in describe_scope(empty))

        whole = panel.set_scope(None)

        check("and letting the scope go brings the whole layer back",
              panel.table.rowCount() == 3 and not panel.scoped
              and whole["kept"] == 3
              and panel.scope_button.text() == "Near the section only")

    print()

    if FAILURES:
        print(f"FAILED: {', '.join(FAILURES)}")
        return 1

    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
