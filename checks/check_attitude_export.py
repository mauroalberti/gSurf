"""
Does a computed attitude get out of the tool, and land where it was read?

A fit that exists only on a screen is not a result. `curation_text` will not
carry one and should not -- that file declares every line in it a human
assertion -- so the way out is a point layer, and a point layer is only worth
writing if the point is right. What has to hold is that the point sits on the
trace at the progressive the plane was read at, that the span and the window
travel with it, that the gate that admitted it travels too, and that the file
can be read back with all of it intact.

The DEM here *is* a plane of known attitude and the trace is a V laid on it, as
in `check_bare_traces.py`, so the dip that comes back out of the written file
is checkable against the surface the whole thing was built on.

    python check_attitude_export.py
"""

import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("GSURF_REPO", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_bare_traces import DIP, DIP_DIRECTION, planar_dem, separation, vee

FAILURES = []


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}{'   ' + detail if detail else ''}")
    if not condition:
        FAILURES.append(label)


def off_trace(point, coords):
    """How far a point lies from a polyline, in metres."""

    coords = np.asarray(coords)[:, :2]
    steps = np.diff(coords, axis=0)
    t = np.clip(
        ((np.asarray(point[:2]) - coords[:-1]) * steps).sum(1) / (steps ** 2).sum(1),
        0.0, 1.0,
    )

    return float(np.hypot(*(coords[:-1] + t[:, None] * steps - point[:2]).T).min())


def main():
    warnings.simplefilter("ignore")

    import geopandas as gpd
    from geogst.core.geology.orientations import Plane
    from geogst.core.geometries.shapes.lines import Ln

    from gsurf.attitudes import TraceRecord, point_at
    from gsurf.dem import Dem
    from gsurf.tools.profiles import attitudes_frame
    from gsurf.traces import fit_records

    plan = vee()
    lines = [Ln(plan)]
    length = float(sum(line.length_2d() for line in lines))

    # -- point_at: the walk, and where it stops ---------------------------

    check("a progressive comes back as a point on the trace",
          off_trace(point_at(lines, 400.0), plan) < 1e-6,
          f"{off_trace(point_at(lines, 400.0), plan):.2e} m off")

    check("past the end of the trace there is no point",
          point_at(lines, length + 1.0) is None)

    check("before the start of the trace there is no point",
          point_at(lines, -5.0) is None)

    check("the two ends are the trace's own ends",
          np.allclose(point_at(lines, 0.0)[:2], plan[0])
          and np.allclose(point_at(lines, length)[:2], plan[-1]))

    # The one that matters for a located measurement. A trace digitised every
    # 100 m read back against its nearest vertex puts the attitude up to 50 m
    # from where it was computed, which is the size of the stretch it holds on.
    coarse = Ln(np.column_stack([np.zeros(5), np.arange(0.0, 500.0, 100.0)]))
    inside = point_at([coarse], 250.0)

    check("a progressive inside a segment is interpolated, not snapped",
          abs(inside[1] - 250.0) < 1e-9,
          f"y = {inside[1]:.6f}, vertices at 200 and 300")

    # -- anchor_point: which of the three questions it answers -------------

    anchored = TraceRecord("c", Plane(130.0, 55.0), lines, length, anchor=300.0)
    spanned = TraceRecord("c", Plane(130.0, 55.0), lines, length, span=(100.0, 300.0))
    neither = TraceRecord("c", Plane(130.0, 55.0), lines, length)

    check("an anchor is where the record sits",
          np.allclose(anchored.anchor_point()[:2], point_at(lines, 300.0)[:2]))

    check("with only a span, the middle of the span",
          np.allclose(spanned.anchor_point()[:2], point_at(lines, 200.0)[:2]))

    check("with neither, the middle of the trace",
          np.allclose(neither.anchor_point()[:2], point_at(lines, length / 2.0)[:2]))

    # `extent` widens an anchor with no half-span to the whole trace, so its
    # midpoint is the trace's middle and not the reading. Taking the point from
    # there would silently move every un-spanned attitude to mid-trace.
    check("an anchor is not read through `extent`",
          not np.allclose(anchored.anchor_point()[:2],
                          neither.anchor_point()[:2]),
          "anchor at 300 m, trace middle at "
          f"{length / 2.0:.0f} m")

    with tempfile.TemporaryDirectory() as tmp:
        dem = Dem(planar_dem(Path(tmp) / "plane.tif"))

        bare = TraceRecord("contatto", None, lines, length)
        fitted, report = fit_records([bare], dem)

        check("the gate travels in the report", report.get("gate") is not None)

        record = fitted[0]

        check("the window is a number, not prose to be parsed",
              isinstance(record.attrs.get("window"), float),
              f"window={record.attrs.get('window')!r}, src={record.attrs.get('src')!r}")

        check("a fitted record sits in the middle of the stretch it held",
              abs(record.anchor_point()[1] - point_at(lines, record.anchor)[1]) < 1e-9)

        # -- the frame ----------------------------------------------------

        frame = attitudes_frame(fitted + [bare], "EPSG:25833",
                                gate=report["gate"], dem=dem)

        check("a record with no plane is not exported",
              len(frame) == len(fitted),
              f"{len(fitted) + 1} records in, {len(frame)} rows out")

        check("every column name fits a shapefile",
              not [c for c in frame.columns if c != "geometry" and len(c) > 10],
              str([c for c in frame.columns if c != "geometry" and len(c) > 10]))

        check("the gate is on the row",
              frame["min_lever"].iloc[0] == report["gate"].min_lever
              and frame["min_pts"].iloc[0] == report["gate"].min_points)

        check("the span is there in metres and as an interval",
              abs(frame["span_m"].iloc[0]
                  - (frame["span_s1"].iloc[0] - frame["span_s0"].iloc[0])) < 1e-9)

        # The line's own third value is the digitiser's, not a measurement. A
        # trace carrying a wrong z must not put that z in the file.
        lying = Ln(np.column_stack([plan, np.zeros(len(plan))]))
        misread = TraceRecord("c", Plane(130.0, 55.0), [lying],
                              float(lying.length_2d()), anchor=300.0)
        got = attitudes_frame([misread], "EPSG:25833", dem=dem)["elev_m"].iloc[0]

        check("elevation comes from the DEM, not from the line's own z",
              got > 1000.0, f"{got:.1f} m, line says 0.0")

        check("with no DEM the elevation is empty, not zero",
              attitudes_frame([misread], "EPSG:25833")["elev_m"].isna().all())

        # -- the file -----------------------------------------------------

        mixed = fitted + [anchored, neither]

        for suffix in (".gpkg", ".shp"):
            path = Path(tmp) / f"attitudes{suffix}"
            frame = attitudes_frame(mixed, "EPSG:25833",
                                    gate=report["gate"], dem=dem)
            frame.to_file(path)
            back = gpd.read_file(path)

            check(f"{suffix}: every row survives the round trip",
                  len(back) == len(frame), f"{len(frame)} out, {len(back)} back")

            check(f"{suffix}: the CRS survives", back.crs.to_epsg() == 25833)

            # Columns empty for a table record and full for a fit: the mix is
            # what a shapefile's typing is most likely to refuse.
            check(f"{suffix}: an empty span stays empty and a full one full",
                  int(back["span_m"].notna().sum()) == len(fitted),
                  f"{int(back['span_m'].notna().sum())} of {len(back)} rows spanned")

            row = back[back["fitted"]].iloc[0]
            gap = separation((row["dipdir"], row["dip"]), (DIP_DIRECTION, DIP))

            check(f"{suffix}: the plane read back is the one the ground was built on",
                  gap < 1.5, f"{row['dipdir']:.1f}/{row['dip']:.1f}, {gap:.2f} deg off")

            check(f"{suffix}: the point read back is still on the trace",
                  off_trace((row.geometry.x, row.geometry.y), plan) < 0.01,
                  f"{off_trace((row.geometry.x, row.geometry.y), plan):.2e} m off")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed: {', '.join(FAILURES)}")
        return 1

    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
