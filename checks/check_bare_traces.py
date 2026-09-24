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
    from shapely.geometry import LineString

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

    print()

    if FAILURES:
        print(f"FAILED: {', '.join(FAILURES)}")
        return 1

    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
