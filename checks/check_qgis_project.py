"""
Is a QGIS project read for what it actually says?

Against a project written here rather than against one on this machine. The
project used while writing this holds a hundred layers and four megabytes, and
a check that read it would be checking that one file -- it would pass for
whoever has it and fail for everybody else, and it would go on passing after
the reader stopped understanding anything but that file.

So the fixture below is small and deliberately awkward: both spellings of a
colour property, both container formats, a path that climbs out of the project
directory, a category QGIS uses as a catch-all, a category switched off, and
three kinds of layer that have to be passed over rather than mishandled. Each
of those is something the real projects turned out to contain.

    QT_QPA_PLATFORM=offscreen python check_qgis_project.py

Qt is never started here: the reader is the standard library and nothing else,
which is most of the reason it is worth having as its own module.
"""

import os
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(os.environ.get("GSURF_REPO", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(REPO))

FAILURES = []


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}{'   ' + detail if detail else ''}")
    if not condition:
        FAILURES.append(label)


# A symbol in the spelling QGIS 3.30 and later use.
def option_symbol(name, colour, key="color"):
    return f"""
        <symbol name="{name}" type="fill" alpha="1">
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
            </Option>
          </data_defined_properties>
          <layer class="SimpleFill" enabled="1">
            <Option type="Map">
              <Option name="{key}" type="QString" value="{colour}" />
              <Option name="outline_color" type="QString" value="0,0,0,255" />
            </Option>
          </layer>
        </symbol>"""


# And in the one everything before it used.
def prop_symbol(name, colour, key="color"):
    return f"""
        <symbol name="{name}" type="fill" alpha="1">
          <layer class="SimpleFill">
            <prop k="{key}" v="{colour}" />
            <prop k="outline_color" v="0,0,0,255" />
          </layer>
        </symbol>"""


def categorized(attr, categories, symbols):
    entries = "\n".join(
        f'<category value="{value}" symbol="{symbol}" label="{label}" render="{render}" />'
        for value, symbol, label, render in categories
    )

    return f"""
      <renderer-v2 type="categorizedSymbol" attr="{attr}">
        <categories>
          {entries}
        </categories>
        <symbols>
          {"".join(symbols)}
        </symbols>
      </renderer-v2>"""


def maplayer(identifier, title, provider, datasource, geometry=None, renderer=""):
    attribute = f' geometry="{geometry}"' if geometry else ""

    return f"""
    <maplayer{attribute}>
      <id>{identifier}</id>
      <layername>{title}</layername>
      <provider>{provider}</provider>
      <datasource>{datasource}</datasource>
      {renderer}
    </maplayer>"""


def project_xml():
    """
    A project holding one of each thing the reader has to get right.

    The layers are stored in one order and shown in another, which is the case
    in every real project and the reason the tree is read at all.
    """

    units = categorized(
        "nome",
        [
            ("Flysch Rosso", "0", "Flysch Rosso", "true"),
            ("Argille Varicolori", "1", "Argille Varicolori (AV)", "true"),
            ("Conglomerati", "2", "Conglomerati", "false"),
            ("", "3", "all other values", "true"),
        ],
        [
            option_symbol("0", "212,50,45,255,hsv:0.005,0.78,0.83,1"),
            option_symbol("1", "#8ce321"),
            option_symbol("2", "29,119,216,255"),
            option_symbol("3", "200,200,200,255"),
        ],
    )

    faults = categorized(
        "tipo_geo",
        [("diretta", "0", "diretta", "true"), ("inversa", "1", "inversa", "true")],
        [
            # A line symbol names its colour differently, which is the trap:
            # read as `color` a whole categorised fault layer comes back grey.
            prop_symbol("0", "255,0,0,255", key="line_color"),
            prop_symbol("1", "0,0,255,255", key="line_color"),
        ],
    )

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<qgis version="4.0.3-Norrkoping">
  <homePath path="" />
  <layer-tree-group>
    <layer-tree-group name="Geologia">
      <layer-tree-layer id="units_1" name="Sheet 489 - units" />
      <layer-tree-layer id="faults_1" name="Sheet 489 - faults" />
    </layer-tree-group>
    <layer-tree-group name="Terrain">
      <layer-tree-layer id="dem_1" name="DEM 5 m" />
    </layer-tree-group>
    <layer-tree-layer id="stations_1" name="Stations" />
  </layer-tree-group>
  <projectlayers>
    {maplayer("stations_1", "Stations", "ogr",
              "./survey.gpkg|layername=stations|geometrytype=Point", "Point")}
    {maplayer("dem_1", "DEM 5 m", "gdal", "../rasters/dem5m.tif")}
    {maplayer("faults_1", "Sheet 489 - faults", "ogr",
              "./sheet489.gpkg|layername=geologia_linee", "Line", faults)}
    {maplayer("units_1", "Sheet 489 - units", "ogr",
              "./sheet489.gpkg|layername=geologia_poligoni", "Polygon", units)}
    {maplayer("table_1", "attitudes table", "ogr",
              "./sheet489.gpkg|layername=fault_attitudes", "No geometry")}
    {maplayer("basemap_1", "OpenStreetMap", "wms", "type=xyz&amp;url=https://example.invalid")}
    {maplayer("broken_1", "a layer with nothing behind it", "ogr", "")}
  </projectlayers>
</qgis>
"""


def main():
    from gsurf.qgis_project import Project, read

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        directory = root / "projects"
        directory.mkdir()
        (root / "rasters").mkdir()

        plain = directory / "area.qgs"
        plain.write_text(project_xml())

        zipped = directory / "area.qgz"
        with zipfile.ZipFile(zipped, "w") as archive:
            archive.writestr("area.qgs", project_xml())

        project = read(plain)

        # -- what came through, and what did not ------------------------------

        print("\n-- the layers, and the ones passed over --")

        check("the four usable layers are read", len(project.layers) == 4, project.summary())
        check(
            "and the three unusable ones are set aside with a reason",
            len(project.skipped) == 3,
            str(sorted(reason for _, reason in project.skipped)),
        )

        reasons = dict(project.skipped)

        check(
            "a service is not a file",
            "wms" in reasons.get("OpenStreetMap", ""),
        )
        check(
            "a table fills no slot",
            "geometry" in reasons.get("attitudes table", ""),
        )
        check(
            "and a layer with no datasource says so",
            reasons.get("a layer with nothing behind it") == "no datasource",
        )

        # -- where the files actually are --------------------------------------

        print("\n-- datasources resolved --")

        units = next(layer for layer in project.layers if layer.geometry == "Polygon")
        dem = next(layer for layer in project.layers if layer.is_raster)
        stations = next(layer for layer in project.layers if layer.geometry == "Point")

        check(
            "a relative path is resolved against the project",
            units.path == str(directory / "sheet489.gpkg"),
            units.path,
        )
        check(
            "and one that climbs out of it still lands where it points",
            dem.path == str(root / "rasters" / "dem5m.tif"),
            dem.path,
        )
        check("the layer inside the file is taken", units.layer == "geologia_poligoni")
        check(
            "and what follows it is not mistaken for one",
            stations.layer == "stations",
            stations.layer or "(none)",
        )

        # -- the panel, not the file -------------------------------------------

        print("\n-- the order and the groups of the layer panel --")

        check(
            "the layers come back in the order the panel shows them",
            [layer.title for layer in project.layers]
            == ["Sheet 489 - units", "Sheet 489 - faults", "DEM 5 m", "Stations"],
            str([layer.title for layer in project.layers]),
        )
        check("and each knows its group", units.group == "Geologia", units.group)
        check(
            "a layer outside every group has none rather than a wrong one",
            stations.group == "",
        )

        # -- the categories -----------------------------------------------------

        print("\n-- the categories, and the colours they are drawn in --")

        check("the field is taken from the renderer", units.category_field == "nome")
        check(
            "three categories, the catch-all not among them",
            sorted(units.colors) == ["Argille Varicolori", "Conglomerati", "Flysch Rosso"],
            str(sorted(units.colors)),
        )
        check(
            "the colour is read as matplotlib takes it",
            units.colors["Flysch Rosso"] == (212 / 255, 50 / 255, 45 / 255),
            str(units.colors["Flysch Rosso"]),
        )
        check(
            "hex is read too",
            units.colors["Argille Varicolori"] == (0x8C / 255, 0xE3 / 255, 0x21 / 255),
            str(units.colors["Argille Varicolori"]),
        )
        check(
            "a category switched off in the project stays off",
            units.hidden == ["Conglomerati"],
            str(units.hidden),
        )
        check(
            "a label is kept only when it says more than the value",
            units.labels == {"Argille Varicolori": "Argille Varicolori (AV)"},
            str(units.labels),
        )

        faults = next(layer for layer in project.layers if layer.geometry == "Line")

        check(
            "a line symbol is read under the name it keeps its colour by",
            faults.colors == {"diretta": (1.0, 0.0, 0.0), "inversa": (0.0, 0.0, 1.0)},
            str(faults.colors),
        )
        check(
            "and the older spelling of a property is read as well as the current one",
            bool(faults.colors) and bool(units.colors),
        )

        # -- the slots ----------------------------------------------------------

        print("\n-- what each slot is offered --")

        for slot, expected in (
            ("dem", ["DEM 5 m"]),
            ("polygons", ["Sheet 489 - units"]),
            ("lines", ["Sheet 489 - faults"]),
            ("points", ["Stations"]),
            ("attitudes", ["Stations"]),
            ("traces", ["Sheet 489 - faults"]),
        ):
            found = [layer.title for layer in project.for_slot(slot)]
            check(f"the {slot} slot is offered what fits it", found == expected, str(found))

        entry = units.entry("polygons")

        check(
            "a backdrop entry carries the colours",
            entry["colors"] == units.colors and entry["hidden"] == ["Conglomerati"],
        )
        check(
            "and the field they are keyed to",
            entry["category_field"] == "nome" and entry["role"] == "polygons",
        )
        check(
            "and the name QGIS shows, which is the one that is recognised",
            entry["title"] == "Sheet 489 - units",
        )

        angles = faults.entry("traces")

        check(
            "an angle slot is given the file and nothing invented",
            set(angles) == {"path", "layer", "title"},
            str(sorted(angles)),
        )

        check(
            "the DEM slot takes the bare path its picker restores",
            units.entry("dem") == units.path and isinstance(dem.entry("dem"), str),
        )

        # -- the same project, zipped -------------------------------------------

        print("\n-- a .qgz reads as the .qgs inside it --")

        archive = read(zipped)

        check(
            "the layers are the same",
            [layer.title for layer in archive.layers]
            == [layer.title for layer in project.layers],
        )
        check(
            "and so are the paths, resolved against the archive itself",
            [layer.path for layer in archive.layers]
            == [layer.path for layer in project.layers],
        )

        # -- what is not a project ----------------------------------------------

        print("\n-- refusals --")

        empty = directory / "empty.qgz"
        with zipfile.ZipFile(empty, "w") as handle:
            handle.writestr("readme.txt", "no project in here")

        try:
            read(empty)
            refused = False
        except ValueError:
            refused = True

        check("an archive with no project in it is refused", refused)

        nonsense = directory / "nonsense.qgs"
        nonsense.write_text("this is not xml at all")

        try:
            read(nonsense)
            raised = False
        except Exception:
            raised = True

        check("and a file that is not XML does not come back as an empty project", raised)

        blank = Project("nowhere.qgs")

        check("a project with no layers offers none", blank.entries_for("polygons") == [])

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
