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
    from shapely.geometry import LineString, Point, box

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

    # Fifteen units, three more than the legend lists: what falls past the cut
    # is still on the map, and the entry that stands for the remainder is the
    # only thing that can take it off.
    units = gpd.GeoDataFrame(
        [
            dict(
                code=f"U{i:02d}",
                name=f"unit {i}",
                geometry=box(
                    600000 + i * 400, 4409000,
                    600000 + i * 400 + 300 * (1 + i % 3), 4418000,
                ),
            )
            for i in range(15)
        ],
        crs="EPSG:25833",
    )
    units_path = directory / "units.gpkg"
    units.to_file(units_path, layer="units", driver="GPKG")

    # Last, so that the session's label and its suggested filenames still come
    # from the stations: which layer names the session is a separate question
    # from which layers are drawn, and this file already checks it.
    return [
        dict(path=str(stations_path), role="points", layer="stations"),
        dict(path=str(faults_path), role="lines", layer="faults"),
        dict(path=str(units_path), role="polygons", layer="units", category_field="code"),
    ]


def main():
    from PyQt6 import QtWidgets

    from gsurf.mapview import MapView
    from gsurf.session import Session

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
        check("every layer was read",
              len(session.overlay.sources) == 3, session.overlay.summary())

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

        # -- the categories, switched off one at a time --------------------------
        print("\n-- what the legend switches --")

        units = next(s for s in session.overlay.sources if s.role == "polygons")

        check("a categorised layer is one artist per category, not one for the lot",
              len(units.artists) == len(units.category_order) == 15,
              f"{len(units.artists)} artists, {len(units.category_order)} categories")

        check("and they are drawn heaviest first, as the legend lists them",
              list(units.artists) == units.category_order,
              f"{units.category_order[0]} first")

        # -- the layer's own entry, over its categories ---------------------------
        entries = [t.get_text() for t in view.legend.get_texts()]
        headings = [
            t for t in view.legend.get_texts() if t.get_fontweight() == "bold"
        ]

        # Bold reads as "a layer" and plain as "a category inside the one
        # above", which is the only thing saying where one layer's entries end.
        check("a categorised layer is headed by its own name",
              entries[0] == "units" and headings[0].get_text() == "units",
              f"first entry {entries[0]!r}, {len(headings)} in bold")
        check("and every layer is marked as one, categorised or not",
              [t.get_text() for t in headings] == ["units", "faults", "stations"],
              str([t.get_text() for t in headings]))

        # The point of the heading: twenty units cost twenty-one clicks to clear
        # without it, and clearing the backdrop is what one actually does.
        units.toggle(units.category_order)
        check("and clicking it takes the whole layer off in one go",
              not any(a.get_visible() for a in units.artists.values()),
              f"{len(units.hidden)} of {len(units.artists)} off")

        view.refresh_legend()
        heading = view.legend.get_texts()[0]
        check("the heading greys with the layer it stands for",
              heading.get_color() == "#9a9a9a", heading.get_color())

        units.toggle(units.category_order)
        check("and a second click brings the layer back whole",
              all(a.get_visible() for a in units.artists.values()) and not units.hidden)

        # A layer with no categories is already one entry under its own name, so
        # it gets no second one: a heading over a single line would say the same
        # thing twice.
        lines = next(s for s in session.overlay.sources if s.role == "lines")
        check("an uncategorised layer is its own heading, not given a second",
              len(lines.legend_handles()) == 1
              and lines.legend_handles()[0].get_label() == "faults",
              lines.legend_handles()[0].get_label())

        first, second = units.category_order[0], units.category_order[1]

        units.toggle((first,))
        check("switching one takes it off the map",
              not units.artists[first].get_visible(), f"{first} off")
        check("and leaves the others on it", units.artists[second].get_visible())

        # The rebuild is where the state is easiest to lose: it is remembered in
        # the layer and read back out, not carried by the legend entry itself.
        view.refresh_legend()
        greyed = [t for t in view.legend.get_texts() if t.get_text().startswith(first)]
        check("the entry comes back greyed after the legend is rebuilt",
              len(greyed) == 1 and greyed[0].get_color() == "#9a9a9a",
              greyed[0].get_color() if greyed else "no entry")
        check("and it is still off the map", not units.artists[first].get_visible())

        placements = [mode for _, mode in view.LEGEND_PLACEMENTS]
        for mode in ("inside", "hidden", "beside"):
            view.set_legend_placement(mode)
        check("moving the legend about does not bring it back",
              not units.artists[first].get_visible() and units.hidden == {first},
              f"hidden {sorted(units.hidden)}, placements {placements}")

        units.toggle((first,))
        check("switching it again puts it back", units.artists[first].get_visible())

        # -- the block past the twelfth entry ------------------------------------
        rest = units.category_order[units.MAX_LEGEND_ENTRIES:]
        overflow = [t.get_text() for t in view.legend.get_texts() if t.get_text().startswith("+")]

        check("the categories past the cut get one entry between them",
              len(rest) == 3 and overflow == [f"+3 more in {units.role}"],
              f"{len(rest)} past the cut, entry {overflow}")

        units.toggle(rest)
        check("which switches them together",
              not any(units.artists[v].get_visible() for v in rest),
              f"{len(rest)} off at once")
        check("and brings them back together, not one per click",
              units.toggle(rest) and all(units.artists[v].get_visible() for v in rest))

        # -- and the way out of the corner ---------------------------------------
        units.toggle((first,))
        units.toggle(rest)
        check("the map counts what is off", view.hidden_count() == 4,
              f"{view.hidden_count()} hidden")

        view.show_all_categories()
        check("and can put all of it back at once",
              view.hidden_count() == 0
              and all(a.get_visible() for a in units.artists.values()))

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
              len(with_dem.overlay.sources) + len(with_dem.overlay.rejected) == 3,
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
