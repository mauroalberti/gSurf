"""
The layers of a QGIS project, and the colours they are drawn in there.

A project is the fourth way of filling a slot, after Browse, the command line
and the list of what was opened before. It fills the same slots with the same
specs -- nothing below this knows where a choice came from -- and it answers
two questions the other three cannot.

**Which field the categories are.** `VectorPicker` guesses it from a list of
names, and on the sheets this is used against it guesses `nome` right and
`tipo_geo`, `legenda` and `DESCRIZION` wrong. A categorised renderer says so:
the field is an attribute of the renderer, not a thing to be inferred from it.

**What colour each category is.** This is the one that matters. The wheel in
`VectorSource` is `tab20` and `tab20b` end to end, forty colours; the CASMEZ
sheet of Calabria carries 181 units and the official 1:50,000 of Moliterno and
Lauria 118, so past the fortieth the wheel comes round and two formations are
drawn the same. Worse, the section was never coloured from the same wheel at
all -- `geogst` spreads its own hue ramp over whatever it was handed -- so one
unit had one colour on the map and another in the section below it. A project
answers both at once, because it holds one colour per category and that colour
is the one already being looked at in QGIS.

What is deliberately *not* read is the symbology. A geological map is drawn
with hatches, pattern fills and SVG markers with teeth on them, and matplotlib
draws none of those; trying would be a project of its own and would still end
in something that is not what QGIS shows. The fill colour is taken and the rest
is left, which is honest about what it can reproduce.

No PyQGIS. A `.qgs` is XML and a `.qgz` is a zip with one inside, and the
standard library reads both -- 141 ms for the 4.9 MB project this was written
against, which is a button press and not a startup cost. Importing PyQGIS into
a PyQt6 process would mean two Qt bindings in one interpreter, which is a
worse problem than the one it would solve.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

# What a project declares as a layer's geometry, and the role it can fill.
# The names are QGIS's, and they cover the multi-part cases too: a layer of
# MultiPolygons is declared `Polygon` here, the part count being a property of
# the features and not of the layer.
ROLE_OF_GEOMETRY = {"Polygon": "polygons", "Line": "lines", "Point": "points"}

# Which geometry each slot takes. Two slots take lines and two take points,
# which is exactly why a project cannot fill them by itself: `lines` is a
# backdrop and `traces` is data the tool reads, and nothing in the geometry
# says which of the two a layer was meant to be. The dialog asks.
GEOMETRY_OF_SLOT = {
    "polygons": "Polygon",
    "lines": "Line",
    "points": "Point",
    "attitudes": "Point",
    "traces": "Line",
}

# The providers that name a file. `wms` and the rest name a service, and a
# service is not something any of these tools can open -- they are listed as
# skipped rather than passed on to fail later with a worse message.
FILE_PROVIDERS = ("ogr", "gdal")

# The property holding the colour that fills the symbol, under each of the
# names the symbol layer classes give it: `SimpleFill` and `SimpleMarker` say
# `color`, `SimpleLine` says `line_color`, and a few say `single_color`.
# `outline_color` is not here on purpose -- it is the colour of the edge, and
# a map read at section scale is read by its fills.
COLOUR_KEYS = ("color", "line_color", "single_color")


def _document(path):
    """
    The project XML and the directory its relative paths are relative to.

    A `.qgz` is a zip holding one `.qgs`, and the paths inside it are relative
    to the zip, not to anything within it.
    """

    path = Path(path)

    if path.suffix.lower() == ".qgz":
        with zipfile.ZipFile(path) as archive:
            inner = [name for name in archive.namelist() if name.endswith(".qgs")]

            if not inner:
                raise ValueError("the archive holds no .qgs")

            root = ET.fromstring(archive.read(inner[0]))
    else:
        root = ET.parse(path).getroot()

    return root, path.parent


def _home(root, beside):
    """
    Where a relative datasource starts from.

    `homePath` when the project sets one, and the project's own directory when
    it does not -- which is the usual case, and the one that matters: written
    this way a project keeps working when the whole tree is moved or copied,
    which is why QGIS defaults to it.
    """

    element = root.find("homePath")
    declared = element.get("path") if element is not None else None

    return Path(declared) if declared else Path(beside)


def _resolve(datasource, home):
    """
    The file a datasource names, and the layer inside it.

    The forms that turn up are `file|layername=x` and a bare path; the pipe
    also carries `geometrytype` and `subset`, which are about what to read and
    not about where, so everything after the first pipe is looked at by name
    rather than by position.

    The path is resolved but not checked. A project can name a layer on a drive
    that is not mounted this morning, and that is the same non-event it is in
    `recent`: the entry is built, and whether it opens is found out by whoever
    opens it.
    """

    head, _, rest = datasource.partition("|")

    layer = None

    for piece in rest.split("|"):
        key, sep, value = piece.partition("=")

        if sep and key == "layername":
            layer = value

    path = Path(head)

    if not path.is_absolute():
        path = (home / path).resolve()

    return str(path), layer


def _colour(text):
    """
    One colour as matplotlib takes it, or None if it cannot be read.

    QGIS writes `212,50,45,255` and appends its own HSV form after it, so the
    first three numbers are the answer and everything past them is a
    restatement. Hex turns up in hand-edited projects. Anything else -- a named
    constant, an expression -- is left alone rather than guessed at, and the
    category falls back to the wheel.

    The alpha is read and dropped. `VectorSource` fixes its own, and for a
    reason that survives the import: a backdrop is there to be seen *through*,
    and twenty opaque tints over a field of fold axes hide the thing the map
    was opened for.
    """

    if not text:
        return None

    text = text.strip()

    if text.startswith("#"):
        try:
            return tuple(int(text[i : i + 2], 16) / 255 for i in (1, 3, 5))
        except ValueError:
            return None

    pieces = text.split(",")

    if len(pieces) < 3:
        return None

    try:
        return tuple(int(piece) / 255 for piece in pieces[:3])
    except ValueError:
        return None


def _symbol_colour(symbol):
    """
    The colour of one symbol, from the first of its layers that has one.

    Symbols stack -- a pattern fill over a plain one is two layers -- and the
    first that yields a colour is the one taken. Properties are read as direct
    children rather than by descending the whole subtree, because a symbol
    layer also carries a `data_defined_properties` map that has an entry called
    `name` and would otherwise be walked into.

    Both spellings are handled. QGIS 3.30 and later write `<Option name=... />`
    and everything before wrote `<prop k=... />`; the project this was written
    against is 4.0 and has 664 of the first and none of the second, which is
    exactly the kind of thing that makes an older project fail silently.
    """

    for layer in symbol.findall("layer"):
        properties = {}

        for option in layer.findall("Option/Option"):
            properties[option.get("name")] = option.get("value")

        for prop in layer.findall("prop"):
            properties[prop.get("k")] = prop.get("v")

        for key in COLOUR_KEYS:
            colour = _colour(properties.get(key))

            if colour is not None:
                return colour

    return None


def _symbol_colours(renderer):
    """Every symbol of a renderer, by the name its categories refer to it by."""

    colours = {}

    for symbol in renderer.findall("symbols/symbol"):
        colour = _symbol_colour(symbol)

        if colour is not None:
            colours[symbol.get("name")] = colour

    return colours


def _categories(renderer):
    """
    The field, the colour of each value, the labels, and what is switched off.

    A category with no value is QGIS's catch-all for everything not listed, and
    it is dropped: it is a rule rather than a value, and putting it in the map
    under the empty string would colour one unit with it.

    `render="false"` is a category turned off in the project. It is carried
    over rather than ignored, because a layer is nearly always brought into a
    project with everything on and then quietened down to the few units being
    worked on, and arriving here with all 118 back on would undo that.
    """

    colours = _symbol_colours(renderer)

    values, labels, hidden = {}, {}, []

    for category in renderer.findall("categories/category"):
        value = category.get("value")

        if not value:
            continue

        colour = colours.get(category.get("symbol"))

        if colour is not None:
            values[value] = colour

        label = category.get("label")

        if label and label != value:
            labels[value] = label

        if category.get("render") == "false":
            hidden.append(value)

    return values, labels, hidden


def _tree_order(root):
    """
    Where each layer sits in the panel, and under which group.

    The order layers are stored in is not the order they are shown in, and the
    shown one is the only one anybody recognises: a project of a hundred layers
    is navigated by its groups -- "Geologia — carte", "Tettonica regionale" --
    and a list that arrives in storage order arrives shuffled.
    """

    order = {}

    def walk(node, group):
        for child in node:
            if child.tag == "layer-tree-group":
                walk(child, child.get("name") or group)
            elif child.tag == "layer-tree-layer":
                order[child.get("id")] = (len(order), group)

    tree = root.find("layer-tree-group")

    if tree is not None:
        walk(tree, "")

    return order


class Layer:
    """One layer of a project, in the shape the slots take."""

    def __init__(self, title, path, layer=None, geometry=None, group=""):
        self.title = title
        self.path = path
        self.layer = layer
        self.geometry = geometry
        self.group = group

        self.category_field = None
        self.colors = {}
        self.labels = {}
        self.hidden = []

        # Where the panel shows it, filled in once the tree has been walked.
        self.position = 0

    @property
    def is_raster(self):
        return self.geometry is None

    @property
    def role(self):
        return ROLE_OF_GEOMETRY.get(self.geometry)

    def entry(self, slot):
        """
        This layer as a choice the picker for `slot` can restore.

        The DEM slot takes a bare path, every other slot a dictionary -- which
        is the shape the pickers already had, and the reason a project needs no
        say anywhere below here.

        The angle slots are given the file and the layer and nothing else: a
        project knows nothing about which column holds a dip, and the pickers
        have guessed that from the field names since before any of this. The
        colours go only where they are drawn from, which is the backdrop roles.
        """

        if slot == "dem":
            return self.path

        entry = dict(path=self.path, layer=self.layer, title=self.title)

        if slot in ROLE_OF_GEOMETRY.values():
            entry["role"] = self.role

            if self.category_field:
                entry["category_field"] = self.category_field

            if self.colors:
                entry["colors"] = self.colors
                entry["labels"] = self.labels
                entry["hidden"] = self.hidden

        return entry


class Project:
    """
    A QGIS project, read for its layers alone.

    Nothing is opened: a datasource is resolved to a path and handed on, and
    whether it is still readable is found out by the picker, which is where it
    was always found out.
    """

    def __init__(self, path, layers=(), skipped=()):
        self.path = str(path)
        self.layers = list(layers)

        # What was passed over, and why -- shown rather than swallowed. A
        # project of a hundred layers that offers eighty is a thing the user
        # should be able to check, not wonder about.
        self.skipped = list(skipped)

    @property
    def name(self):
        return Path(self.path).name

    def for_slot(self, slot):
        """The layers that could fill one slot, in the order the panel shows them."""

        if slot == "dem":
            return [layer for layer in self.layers if layer.is_raster]

        geometry = GEOMETRY_OF_SLOT.get(slot)

        return [layer for layer in self.layers if layer.geometry == geometry]

    def entries_for(self, slot):
        return [layer.entry(slot) for layer in self.for_slot(slot)]

    def summary(self):
        vectors = sum(1 for layer in self.layers if not layer.is_raster)
        rasters = len(self.layers) - vectors
        coloured = sum(1 for layer in self.layers if layer.colors)

        text = f"{self.name}: {vectors} vector and {rasters} raster layers"

        if coloured:
            text += f", {coloured} with categories"

        if self.skipped:
            text += f"; {len(self.skipped)} skipped"

        return text


def read(path):
    """
    A project read into layers, skipping what none of these tools can open.

    Three things are passed over, and each is counted with its reason: layers
    on a service rather than a file, layers with no geometry at all -- a
    geopackage commonly carries a plain table, `fault_attitudes` in the one
    used here -- and layers a project declares without a usable datasource,
    which is what a broken or embedded entry comes through as.
    """

    root, beside = _document(path)
    home = _home(root, beside)
    order = _tree_order(root)

    layers, skipped = [], []

    for element in root.findall(".//maplayer"):
        title = element.findtext("layername") or "(unnamed)"
        provider = element.findtext("provider")
        datasource = element.findtext("datasource")

        if provider not in FILE_PROVIDERS:
            skipped.append((title, f"on {provider or 'no provider'}, not a file"))
            continue

        if not datasource:
            skipped.append((title, "no datasource"))
            continue

        geometry = element.get("geometry")

        if geometry is not None and geometry not in ROLE_OF_GEOMETRY:
            # "No geometry": a table, and a table fills no slot here.
            skipped.append((title, f"geometry is {geometry.lower()}"))
            continue

        path_of, layer_of = _resolve(datasource, home)

        # The id is a child element here and an attribute in the tree, and the
        # two are the same string: read as an attribute it comes back None,
        # every layer lands on the same default, and the panel order and the
        # group names are quietly lost rather than reported missing.
        position, group = order.get(element.findtext("id"), (len(order), ""))

        found = Layer(title, path_of, layer_of, geometry, group)
        found.position = position

        renderer = element.find("renderer-v2")

        if renderer is not None and renderer.get("type") == "categorizedSymbol":
            found.category_field = renderer.get("attr") or None
            found.colors, found.labels, found.hidden = _categories(renderer)

        layers.append(found)

    layers.sort(key=lambda found: found.position)

    return Project(path, layers, skipped)
