"""
The session, and the map with no DEM under it.

This is the path the fold-axis module will arrive on: a projection and an area
that come from the layers rather than from a raster. It has no user yet, so it
is exercised here rather than left to be discovered later.

    QT_QPA_PLATFORM=offscreen python check_session.py
"""

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(os.environ.get("GSURF_REPO", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(REPO))

from checks.synthetic import synthetic_dem  # noqa: E402

FAILURES = []


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}{'   ' + detail if detail else ''}")
    if not condition:
        FAILURES.append(label)


def field_data(directory):
    """Two layers in two CRSs, over the same ground as the synthetic DEM."""

    import geopandas as gpd
    from shapely.geometry import LineString, Point

    # The DEM sits at EPSG:25833, 600000/4420000, 5 m cells.
    stations = gpd.GeoDataFrame(
        {"code": ["S1", "S2", "S3"], "dip_dir": [120.0, 135.0, 118.0], "dip": [32.0, 40.0, 28.0]},
        geometry=[Point(601000, 4416000), Point(603500, 4413000), Point(605000, 4410500)],
        crs="EPSG:25833",
    )
    stations_path = directory / "stations.gpkg"
    stations.to_file(stations_path, layer="stations", driver="GPKG")

    # The same area written in geographic coordinates, so the union of the two
    # extents has to cross a projection to be taken.
    faults = gpd.GeoDataFrame(
        {"name": ["F1"]},
        geometry=[LineString([(601500, 4417000), (606000, 4409000)])],
        crs="EPSG:25833",
    ).to_crs("EPSG:4326")
    faults_path = directory / "faults.gpkg"
    faults.to_file(faults_path, layer="faults", driver="GPKG")

    return [
        dict(path=str(stations_path), role="points", layer="stations"),
        dict(path=str(faults_path), role="lines", layer="faults"),
    ]


def main():
    from PyQt6 import QtWidgets

    from app.mapview import MapView
    from app.session import Session

    app = QtWidgets.QApplication(sys.argv)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        vectors = field_data(tmp)

        # -- a session with no DEM at all -------------------------------------
        session = Session.open(vectors=vectors)

        check("a session opens on vectors alone", session.dem is None)
        check("the CRS comes from the first layer", session.epsg == 25833,
              f"EPSG:{session.epsg}")

        left, bottom, right, top = session.bounds
        check("the extent covers the projected layer",
              left <= 601000 and right >= 605000 and bottom <= 4410500 and top >= 4416000,
              f"{left:.0f} {bottom:.0f} {right:.0f} {top:.0f}")
        check("and the geographic one, reprojected onto it",
              left <= 601500 and right >= 605990 and bottom <= 4409010 and top >= 4416990,
              f"{right - left:.0f} x {top - bottom:.0f} m")

        cx, cy = session.center()
        check("the centre is the centre of that", abs(cx - (left + right) / 2) < 1e-6)
        check("the label names the first source", session.label == "stations.gpkg",
              session.label)
        check("exports are suggested beside the source",
              session.suggested_name(".shp", tag="axes").name == "stations_axes.shp",
              session.suggested_name(".shp", tag="axes").name)
        check("both layers were read",
              len(session.overlay.sources) == 2, session.overlay.summary())

        # A layer can say where we are without being drawn: a tool that reads
        # its own data does not want it a second time as a backdrop.
        framed = Session.open(vectors=[vectors[1]], frame_layers=[vectors[0]])
        check("a frame layer is not drawn",
              len(framed.overlay.sources) == 1, framed.overlay.summary())
        check("but it still counts for the extent",
              framed.bounds[0] <= 601000 and framed.bounds[2] >= 605000,
              f"{framed.bounds[0]:.0f} .. {framed.bounds[2]:.0f}")
        framed.close()

        # -- and a map on it ----------------------------------------------------
        view = MapView(session)
        view.resize(900, 700)
        view.draw_base_map()
        view.refresh_legend()
        view.anchor_home()
        view.show()
        app.processEvents()

        check("no hillshade is drawn", view.shade_image is None)
        check("the limits are the session's area",
              abs(view.axes.get_xlim()[0] - left) < 1.0
              and abs(view.axes.get_ylim()[1] - top) < 1.0,
              f"{view.axes.get_xlim()[0]:.0f} .. {view.axes.get_xlim()[1]:.0f}")
        check("the axis still names the projection",
              "EPSG:25833" in view.axes.get_xlabel(), view.axes.get_xlabel())

        view.schedule_shade_refresh()
        check("no background reload is scheduled", not view.shade_timer.isActive())

        limits_before = view.axes.get_xlim()
        view._refresh_shade()
        check("and asking for one outright does nothing",
              view.axes.get_xlim() == limits_before)

        check("the legend shows the backdrop", view.legend is not None,
              f"{len(view.legend.get_texts())} entries")

        # An artist over a map with no raster under it still blits.
        (probe,) = view.axes.plot([], [], "-", color="red")
        view.add_animated(probe)
        probe.set_data([left, right], [bottom, top])
        view.blit()
        check("blitting works without a background raster", probe.get_animated())

        out = tmp / "vectors_only.png"
        view.rendered_figure(out, dpi=60)
        check("and the figure saves", out.exists() and out.stat().st_size > 3000,
              f"{out.stat().st_size} bytes")

        session.close()

        # -- with a DEM, the DEM decides ---------------------------------------
        dem_path = synthetic_dem(tmp / "synthetic.tif", side=1200)
        with_dem = Session.open(dem_path=dem_path, vectors=vectors)

        check("with a DEM the frame is the DEM's",
              tuple(round(v) for v in with_dem.bounds) == (600000, 4414000, 606000, 4420000),
              str(tuple(round(v) for v in with_dem.bounds)))
        check("the label names the DEM", with_dem.label == "synthetic.tif", with_dem.label)
        check("layers off the DEM are set aside, not fatal",
              len(with_dem.overlay.sources) + len(with_dem.overlay.rejected) == 2,
              with_dem.overlay.summary())

        with_dem.close()

        # -- and nothing at all is an error ------------------------------------
        try:
            Session.open()
            refused = False
        except ValueError:
            refused = True

        check("a session with no source is refused", refused)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed: {', '.join(FAILURES)}")
        return 1

    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
