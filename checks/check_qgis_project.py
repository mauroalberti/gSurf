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

The first half starts no Qt at all -- the reader is the standard library and
nothing else, which is most of the reason it is worth having as its own module.
The second half is about what the dialog then does with it, and about the one
thing that can go wrong afterwards: a description outliving what it described.
A palette is one colour per value of one column, and the column can be changed
in the box next to it.
"""

import os
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(os.environ.get("GSURF_REPO", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(REPO))

FAILURES = []
APP = None


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


def make_data(root, directory):
    """
    The files the fixture project names, so that the pickers can open them.

    The reader never opens anything and does not need these; the dialog does,
    and a slot that refuses the layer would make every assertion after it pass
    for the wrong reason.
    """

    import geopandas as gpd
    from shapely.geometry import LineString, Point, Polygon as Shape

    from synthetic import synthetic_dem

    dem = synthetic_dem(root / "rasters" / "dem5m.tif", side=400)

    with __import__("rasterio").open(dem) as handle:
        left, bottom, right, top = handle.bounds
        crs = handle.crs

    def square(fx, fy, size=0.15):
        x = left + (right - left) * fx
        y = bottom + (top - bottom) * fy
        side = min(right - left, top - bottom) * size

        return Shape([(x, y), (x + side, y), (x + side, y + side), (x, y + side)])

    sheet = directory / "sheet489.gpkg"

    gpd.GeoDataFrame(
        {
            # The three the project colours, and one it has never heard of:
            # a sheet revised after the project was last saved is the ordinary
            # case, and the unnamed unit has to come out of the wheel rather
            # than out of nothing.
            "nome": ["Flysch Rosso", "Argille Varicolori", "Conglomerati", "Brecce"],
            "sigla": ["FYR", "AV", "CGL", "BR"],
        },
        geometry=[square(0.2, 0.2), square(0.4, 0.4), square(0.6, 0.2), square(0.2, 0.6)],
        crs=crs,
    ).to_file(sheet, layer="geologia_poligoni", driver="GPKG")

    gpd.GeoDataFrame(
        {"nome": ["terrazzo"]},
        geometry=[square(0.7, 0.7)],
        crs=crs,
    ).to_file(sheet, layer="geomorfologia_poligoni", driver="GPKG")

    gpd.GeoDataFrame(
        {"tipo_geo": ["diretta", "inversa"]},
        geometry=[
            LineString([(left + 100, bottom + 100), (right - 100, top - 100)]),
            LineString([(left + 100, top - 100), (right - 100, bottom + 100)]),
        ],
        crs=crs,
    ).to_file(sheet, layer="geologia_linee", driver="GPKG")

    gpd.GeoDataFrame(
        {"Immersione": [120.0, 300.0], "Inclinazione": [35.0, 20.0]},
        geometry=[
            Point(left + (right - left) * 0.3, bottom + (top - bottom) * 0.3),
            Point(left + (right - left) * 0.5, bottom + (top - bottom) * 0.5),
        ],
        crs=crs,
    ).to_file(directory / "survey.gpkg", layer="stations", driver="GPKG")

    return str(dem)


def check_awkward_layers(directory):
    """
    The layers a container describes in the two ways that used to hide them.

    Not hypothetical, either of them. Every section trace drawn here is a
    `LineString Z`, because a trace is surveyed on a DEM and comes back with
    heights on it; and every ViDEPI layer is `Unknown`, because the archive
    publishes KML and KML declares nothing. Between them they took 23 of the
    92 vector layers of this area's project out of the dialog -- offered on the
    list, because a project says what its layers are, and then refused on
    opening with "no suitable layer", which is the worst of both ways to fail.
    """

    import geopandas as gpd
    import pyogrio
    from shapely.geometry import LineString, Point, Polygon as Shape

    print("\n-- layers whose type is written awkwardly --")

    awkward = directory / "awkward.gpkg"

    square = Shape([(0, 0), (10, 0), (10, 10), (0, 10)])

    # With heights, which is what a trace drawn over a DEM comes back carrying.
    gpd.GeoDataFrame(
        {"nome": ["sezione"]},
        geometry=[LineString([(0, 0, 100), (10, 10, 250)])],
        crs="EPSG:25833",
    ).to_file(awkward, layer="traccia_z", driver="GPKG")

    gpd.GeoDataFrame(
        {"nome": ["cava"]},
        geometry=[Shape([(0, 0, 5), (10, 0, 5), (10, 10, 5), (0, 10, 5)])],
        crs="EPSG:25833",
    ).to_file(awkward, layer="poligoni_z", driver="GPKG")

    declared = dict((str(n), str(g)) for n, g in pyogrio.list_layers(awkward))

    check(
        "a layer with heights on it is declared with a dimension",
        declared.get("traccia_z", "").endswith(" Z"),
        str(declared),
    )

    # Told nothing, which is what a KML import writes.
    for layer, geometry in (
        ("titoli", [square]),
        ("linee", [LineString([(0, 0), (10, 10)])]),
        ("pozzi", [Point(5, 5)]),
    ):
        pyogrio.write_dataframe(
            gpd.GeoDataFrame({"nome": ["x"]}, geometry=geometry, crs="EPSG:25833"),
            awkward,
            layer=layer,
            driver="GPKG",
            geometry_type="Unknown",
            append=True,
        )

    # Told nothing and holding nothing: there is no first feature to judge it
    # by, and a layer that would draw nothing belongs in no slot.
    pyogrio.write_dataframe(
        gpd.GeoDataFrame({"nome": []}, geometry=[], crs="EPSG:25833"),
        awkward,
        layer="vuoto",
        driver="GPKG",
        geometry_type="Unknown",
        append=True,
    )

    # A plain table, which has contents but no geometry at all.
    gpd.pd.DataFrame({"pozzo": ["A", "B"], "quota": [1, 2]}).to_csv(
        directory / "tabella.csv", index=False
    )

    from gsurf.vectors import VectorSource

    check(
        "the dimension does not hide a line",
        VectorSource.geometry_role("LineString Z") == "lines"
        and VectorSource.geometry_role("MultiPolygon ZM") == "polygons"
        and VectorSource.geometry_role("Point M") == "points",
    )
    check(
        "and the ordinary spellings still read as they did",
        VectorSource.geometry_role("MultiLineString") == "lines"
        and VectorSource.geometry_role("Polygon") == "polygons"
        and VectorSource.geometry_role("MultiPoint") == "points",
    )
    check(
        "a table is no geometry and not a shape that was not named",
        VectorSource.geometry_role(None) is None
        and VectorSource.geometry_role("Unknown") is None,
    )

    lines = VectorSource.candidate_layers(awkward, "lines")
    polygons = VectorSource.candidate_layers(awkward, "polygons")
    points = VectorSource.candidate_layers(awkward, "points")

    check("the trace with heights is offered as a line", "traccia_z" in lines, str(lines))
    check(
        "and the quarry with them as a polygon", "poligoni_z" in polygons, str(polygons)
    )
    check(
        "a layer that declares nothing is offered for what it holds",
        polygons == ["poligoni_z", "titoli"]
        and lines == ["traccia_z", "linee"]
        and points == ["pozzi"],
        f"polygons {polygons}, lines {lines}, points {points}",
    )
    check(
        "an empty one is offered nowhere, there being nothing to judge it by",
        not any("vuoto" in found for found in (polygons, lines, points)),
    )

    # Reading the contents is the expensive half, and the dialog asks the same
    # file once per role: the second question must not reopen it.
    from time import perf_counter

    VectorSource._CONTENTS.clear()

    started = perf_counter()
    VectorSource.candidate_layers(awkward, "polygons")
    cold = perf_counter() - started

    started = perf_counter()
    VectorSource.candidate_layers(awkward, "lines")
    warm = perf_counter() - started

    check(
        "and what was read is not read again for the next role",
        warm < cold / 2,
        f"{cold * 1000:.0f} ms then {warm * 1000:.0f} ms",
    )


def check_the_dialog(root, project_path, dem_path):
    """What the dialog does with a project, and what it stops doing with it."""

    from PyQt6 import QtWidgets

    from gsurf.sources import SourcesDialog, open_session
    from gsurf.tools import TOOLS, load

    # Held in a name that outlives this call: built and dropped, PyQt takes the
    # C++ object down with the last reference and every widget after it dies
    # complaining that no application was ever constructed.
    global APP

    APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    wants = load(next(entry for entry in TOOLS if entry["name"] == "Sections")).WANTS

    print("\n-- the dialog, offered a project --")

    dialog = SourcesDialog(wants=wants)
    units = dialog.boxes["polygons"]

    check("before a project there is nothing on the list", units.path_combo.count() == 0)

    project = dialog.use_project(str(project_path))

    check("the project is read", project is not None, project.summary() if project else "")

    # The count says some were passed over; the reasons say which, and are the
    # difference between a limitation and a layer that looks lost.
    hint = dialog.project_label.toolTip()

    check(
        "what was passed over is named, with its reason, behind the count",
        all(title in hint and why in hint for title, why in project.skipped),
        hint.replace("\n", " | "),
    )

    combo = units.path_combo

    check(
        "the polygon slot is offered the one polygon layer",
        combo.count() == 1,
        str([combo.itemText(n) for n in range(combo.count())]),
    )
    check(
        "under the name the project gives it",
        combo.itemText(0) == "Sheet 489 - units",
        combo.itemText(0),
    )
    from PyQt6 import QtCore

    check(
        "with the file it is in still readable, on hover",
        str(combo.itemData(0, QtCore.Qt.ItemDataRole.ToolTipRole)).endswith("sheet489.gpkg"),
        str(combo.itemData(0, QtCore.Qt.ItemDataRole.ToolTipRole)),
    )
    check(
        "both line slots are offered the line layer, being unable to tell them apart",
        dialog.boxes["lines"].path_combo.count() == 1
        and dialog.boxes["traces"].path_combo.count() == 1,
    )
    check("and the DEM slot the raster", dialog.boxes["dem"].path_combo.count() == 1)
    check(
        "nothing was filled in by it",
        all(box.value() is None for box in dialog.boxes.values()),
    )

    # -- picking one ------------------------------------------------------

    print("\n-- what a choice from a project carries --")

    combo.activated.emit(0)

    spec = units.value()

    check("picking it opens it", spec is not None)
    check(
        "the field is the project's, not a guess",
        spec.get("category_field") == "nome",
        str(spec.get("category_field")),
    )
    check(
        "the colours come with it",
        spec.get("colors", {}).get("Flysch Rosso") == (212 / 255, 50 / 255, 45 / 255),
        str(spec.get("colors")),
    )
    check("as does what the project switched off", spec.get("hidden") == ["Conglomerati"])
    check("and the name", spec.get("title") == "Sheet 489 - units")

    # -- and what stops being true ----------------------------------------

    print("\n-- a description outliving what it described --")

    units.category_combo.setCurrentText("sigla")
    moved = units.value()

    check("under another field the palette is not offered", "colors" not in moved)
    check("nor what it switched off", "hidden" not in moved)
    check("the name survives, being about the layer", moved.get("title") == "Sheet 489 - units")

    units.category_combo.setCurrentText("nome")

    check("and it all comes back when the field does", units.value().get("colors"))

    units.layer_combo.setCurrentText("geomorfologia_poligoni")
    elsewhere = units.value()

    check(
        "another layer keeps none of it",
        "colors" not in elsewhere and "title" not in elsewhere,
        str(sorted(elsewhere)),
    )

    # -- the angle slots ---------------------------------------------------

    print("\n-- the slots a project cannot answer for --")

    stations = dialog.boxes["attitudes"]
    stations.path_combo.activated.emit(0)
    reading = stations.value()

    check("an angle slot opens on the layer", reading is not None)
    check(
        "with its two columns guessed as they always were",
        reading and reading.get("dip_dir_field") == "Immersione",
        str(reading and reading.get("dip_field")),
    )
    check("it takes the name", reading and reading.get("title") == "Stations")
    check("and no palette, having nothing to draw with one", "colors" not in (reading or {}))

    # -- two lists, one box -------------------------------------------------

    print("\n-- the history and the project, in one list --")

    mixed = SourcesDialog(wants=wants)
    mixed.boxes["polygons"].offer([dict(path=str(root / "elsewhere.gpkg"), layer="units")])

    def separators(box):
        """A separator is an item with nothing behind it, which nothing selects."""

        return sum(
            1 for n in range(box.path_combo.count()) if box.path_combo.itemData(n) is None
        )

    check("a history alone is not divided", separators(mixed.boxes["polygons"]) == 0)

    mixed.use_project(str(project_path))

    check(
        "with a project it is, once",
        separators(mixed.boxes["polygons"]) == 1,
        str([mixed.boxes["polygons"].path_combo.itemText(n)
             for n in range(mixed.boxes["polygons"].path_combo.count())]),
    )
    check(
        "the project's layers first, the history under it",
        mixed.boxes["polygons"].path_combo.itemText(0) == "Sheet 489 - units",
    )

    # -- and through to what is drawn ---------------------------------------

    print("\n-- the colours, where they are actually used --")

    # Back onto the layer they were said of. That alone is not enough: the
    # picker guesses a field for every layer it lands on and guesses `sigla`
    # here, `sigla` coming before `nome` in its list -- so the palette stays
    # out until the field it is keyed to is back as well, which is the rule
    # doing its job rather than getting in the way.
    units.layer_combo.setCurrentText("geologia_poligoni")

    check(
        "coming back to the layer is not yet coming back to the palette",
        "colors" not in units.value(),
        str(units.value().get("category_field")),
    )

    units.category_combo.setCurrentText("nome")

    check("and with the field back, so is it", units.value().get("colors"))

    session = open_session(dict(dem=dem_path, polygons=units.value()))
    source = session.overlay.sources[0]

    check(
        "the layer is drawn in the project's colours",
        source.colors["Flysch Rosso"] == (212 / 255, 50 / 255, 45 / 255),
        str(source.colors.get("Flysch Rosso")),
    )
    check(
        "a unit the project never saw still gets one",
        source.colors.get("Brecce") is not None
        and source.colors["Brecce"] not in (
            source.colors["Flysch Rosso"],
            source.colors["Argille Varicolori"],
        ),
        str(source.colors.get("Brecce")),
    )
    check(
        "and one switched off in the project starts off the map",
        source.hidden == {"Conglomerati"},
        str(source.hidden),
    )

    session.close()
    dialog.close()
    mixed.close()


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

        check_awkward_layers(directory)

        check_the_dialog(root, plain, make_data(root, directory))

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
