"""
What to open, asked once a tool has been picked.

This used to live inside the intersection tool, because that was the only tool
there was. None of it is about intersecting a plane: it is about turning files
into the specs `Session.open` wants, and every tool needs the same thing.

A tool says which slots it can be given as `WANTS` -- a slot name mapped to
"required" or "optional" -- and the dialog shows those and no others. Asked for
what it cannot use, a tool would be demanding data in order to ignore it, and
the two slots that matter are exactly the ones that swap: the plane cannot run
without a DEM and has no use for attitudes, the fold axes are the other way
round. The required ones are coloured, so that what is missing can be seen
rather than inferred from an Open button that stays grey.

The choices come back as one dictionary keyed by slot -- `dem` a path, the rest
specs -- and that same shape goes into `open_session` and on into the tool. One
shape rather than three is what keeps the command line and the dialog from
drifting into framing a session differently.

Nothing here loads data. Layers are listed and fields are read off the
metadata, which is what lets the dialog filter a half-gigabyte geopackage while
the user is still choosing -- in the polygon slot the faults simply never
appear, and a layer whose azimuth field is text is never offered as one.
"""

from __future__ import annotations

from pathlib import Path

import rasterio

from PyQt6 import QtCore, QtWidgets

from .attitudes import CONVENTIONS, numeric_fields
from .recent import identity
from .session import Session
from .vectors import VectorSource

VECTOR_FILTER = (
    "Vector (*.gpkg *.shp *.geojson *.json *.gml *.kml *.sqlite *.fgb);;"
    "All files (*)"
)

RASTER_FILTER = "Raster (*.tif *.tiff *.vrt *.asc *.img *.dt2 *.hgt);;All files (*)"

PROJECT_FILTER = "QGIS project (*.qgs *.qgz);;All files (*)"

# The same two the fold axes already use for a verdict: green once a required
# slot holds something, dark orange while it does not.
FILLED_COLOUR = "#1a7f37"
MISSING_COLOUR = "#a03000"


def _first_match(candidates, fields):
    """The first candidate name present among the fields, case-insensitively."""

    lowered = {str(field).lower(): str(field) for field in fields}

    for name in candidates:
        if name in lowered:
            return lowered[name]

    return None


def vectors_of(chosen):
    """The backdrop layers among a set of choices, in drawing order."""

    return [chosen[role] for role in VectorSource.ROLES if chosen.get(role)]


# The slots that are a tool's own data: they say where we are without being
# drawn as backdrop, because the tool draws them itself and two symbols on one
# feature is worse than none.
OWN_DATA_SLOTS = ("attitudes", "traces")


def open_session(chosen):
    """
    A session on a set of choices, whoever made them.

    A tool's own layer frames the session without being drawn by it: with no
    DEM it is what says where we are, but the tool draws it itself. Both ways
    in come through here, so that a command line and a dialog cannot end up
    framing the same files differently.
    """

    return Session.open(
        dem_path=chosen.get("dem"),
        vectors=vectors_of(chosen),
        frame_layers=[chosen[slot] for slot in OWN_DATA_SLOTS if chosen.get(slot)],
    )


class SlotBox(QtWidgets.QGroupBox):
    """
    One thing a tool can be given, and whether it has to be.

    The requirement is drawn rather than only enforced: a required slot is
    coloured while it is empty and goes green once it is filled, which says
    what the disabled Open button cannot.
    """

    # Emitted whenever the choice changes in a way that could decide whether
    # the dialog has enough to open on.
    changed = QtCore.pyqtSignal()

    def __init__(self, title, parent=None):
        super().__init__(title, parent)

        self.base_title = title
        self.level = "optional"

        # What this slot has held before, newest first, in whatever shape this
        # picker's own `restore` takes. Empty until the dialog offers them.
        self.entries = []

        # And what a QGIS project holds that would fit it. Kept apart rather
        # than merged into one list, because the two answer different questions
        # -- what was opened here, against what this project has -- and a list
        # that mixed them would make the second unfindable in the first.
        self.project_entries = []

        self.changed.connect(self._mark)

    # -- what was opened here before ---------------------------------------

    def new_path_combo(self):
        """
        The path widget: what is chosen now, over the list of what was before.

        A combo box and not the read-only line edit this used to be, which is
        the whole of the visible change -- Browse still browses, and a slot
        with nothing behind it looks as it always did, a box reading "none".

        `activated` and not `currentIndexChanged`, because only a choice made
        by hand should reopen a file. Every other way the index moves here is
        this class moving it, having already opened what it is moving to.
        """

        combo = QtWidgets.QComboBox()
        combo.setPlaceholderText("none")
        combo.setMinimumContentsLength(28)
        # Qt6 kept only the "WithIcon" spelling of this policy; there are no
        # icons here, so it is the plain one under a longer name.
        combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        combo.activated.connect(self._recent_chosen)

        return combo

    def offer(self, entries):
        """The remembered choices for this slot, before anything is restored."""

        self.entries = [entry for entry in entries if entry]
        self._relist()

    def offer_project(self, entries):
        """
        The layers a project holds that would fit this slot, over the history.

        Whatever is chosen stays chosen: loading a project is asking to be
        shown what is in it, not asking for the slot to be emptied. It is read
        off the list rather than from the picker because only the list knows
        the whole entry -- `value` gives back what the widgets hold, which for
        a half-filled angle slot is nothing at all.
        """

        self.project_entries = [entry for entry in entries if entry]

        # Said on the closed box, because nothing else on it changes. A slot
        # that offers nothing and a slot that now offers twenty-eight both read
        # "none" until the list is opened, and a button whose effect is
        # invisible until you go looking for it is a button that did nothing.
        if self.project_entries:
            self.path_combo.setPlaceholderText(
                f"none - {len(self.project_entries)} in the project"
            )

        self._relist(self.path_combo.currentData())

    def _add_item(self, entry):
        self.path_combo.addItem(self._label(entry), entry)

        # The whole path on hover: the list shows file names, and two surveys
        # of the same area name their DEM the same thing.
        self.path_combo.setItemData(
            self.path_combo.count() - 1,
            identity(entry)[0],
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )

    def _relist(self, current=None):
        """
        Refills the list: what is chosen now first, then where it could come from.

        Built rather than appended to, so that one path cannot appear twice
        under two spellings of the same choice -- the file just browsed to is
        very often the one at the top already, and a project very often holds
        the layer that was opened from it yesterday.

        The separators appear only once a project has been loaded. Until then
        there is one list and nothing to divide, and a rule drawn across a
        history of three files would be furniture.
        """

        seen = set() if current is None else {identity(current)}
        groups = []

        for group in (self.project_entries, self.entries):
            unique = []

            for entry in group:
                key = identity(entry)

                if key in seen:
                    continue

                seen.add(key)
                unique.append(entry)

            groups.append(unique)

        divided = bool(self.project_entries)

        with QtCore.QSignalBlocker(self.path_combo):
            self.path_combo.clear()

            if current is not None:
                self._add_item(current)

            for group in groups:
                if not group:
                    continue

                if divided and self.path_combo.count():
                    self.path_combo.insertSeparator(self.path_combo.count())

                for entry in group:
                    self._add_item(entry)

            self.path_combo.setCurrentIndex(0 if current is not None else -1)

    def _recent_chosen(self, index):
        entry = self.path_combo.itemData(index)

        if entry is not None:
            self.restore(entry)

    def _label(self, entry):
        """
        The name to show. The file alone, unless the layer is what tells them apart.

        One geopackage commonly holds the polygons and the faults both, and a
        list offering the same filename twice says nothing about which is
        which.

        A layer that came from a project is named as the project names it.
        `CARG_489_MarsicoNuovo - geologia_poligoni` is what has been read in
        the layer panel for weeks, and the file it sits in is a path nobody
        looks at -- it is still there, on hover.
        """

        if isinstance(entry, dict) and entry.get("title"):
            return entry["title"]

        path, layer = identity(entry)
        name = Path(path).name

        return f"{name} - {layer}" if layer else name

    def set_requirement(self, level):
        self.level = level
        self._mark()

    def _mark(self):
        required = self.level == "required"

        self.setTitle(f"{self.base_title} ({self.level})")

        if not required:
            self.setStyleSheet("")
            return

        colour = FILLED_COLOUR if self.is_filled else MISSING_COLOUR

        # Only the title is styled. Give a group box any rule of its own and Qt
        # hands the whole widget to the stylesheet engine, frame included, and
        # the border it then draws is not the one the desktop theme draws.
        self.setStyleSheet(f"QGroupBox::title {{ color: {colour}; }}")

    @property
    def is_filled(self):
        return self.value() is not None

    def value(self):
        """What was chosen here, or None while the slot is empty."""

        raise NotImplementedError


class RasterPicker(SlotBox):
    """The DEM slot: one file, described from its header as soon as it is named."""

    def __init__(self, parent=None):
        super().__init__("DEM", parent)

        self._path = None

        self.path_combo = self.new_path_combo()

        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(self._browse)

        self.clear_button = QtWidgets.QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear)
        self.clear_button.setEnabled(False)

        self.info_label = QtWidgets.QLabel()
        self.info_label.setStyleSheet("color: gray; font-size: 10px;")

        grid = QtWidgets.QGridLayout(self)
        grid.addWidget(self.path_combo, 0, 0)
        grid.addWidget(browse, 0, 1)
        grid.addWidget(self.clear_button, 0, 2)
        grid.addWidget(self.info_label, 1, 0, 1, 3)
        grid.setColumnStretch(0, 1)

    def _browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose the DEM", "", RASTER_FILTER
        )

        if path:
            self.set_path(path)

    def set_path(self, path, quiet=False):
        """
        Opens the DEM for its header alone, and says what it is from that.

        Which also catches a file that is not a raster here, rather than after
        the dialog has closed and the window is already being built.
        """

        if not path:
            return False

        try:
            with rasterio.open(path) as src:
                epsg = src.crs.to_epsg() if src.crs else None
                info = (
                    f"{src.width}x{src.height} "
                    f"({src.width * src.height / 1e6:.1f} Mpx), "
                    f"EPSG:{epsg or '?'}, cell {abs(src.transform.a):g} m"
                )
        except Exception as err:
            if not quiet:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Unreadable DEM",
                    f"{Path(path).name}\n\n{str(err).splitlines()[0]}",
                )
            return False

        self._path = str(path)
        self._relist(self._path)
        self.info_label.setText(info)
        self.clear_button.setEnabled(True)

        self.changed.emit()

        return True

    def clear(self):
        self._path = None
        self._relist()
        self.info_label.clear()
        self.clear_button.setEnabled(False)

        self.changed.emit()

    def restore(self, path):
        return self.set_path(path, quiet=True)

    def value(self):
        return self._path


class LayerPicker(SlotBox):
    """
    Choosing one file and one layer inside it, filtered on geometry.

    What depends on the chosen layer -- a category field here, two numeric
    fields there -- is left to the subclass: it adds its widgets in
    `_add_layer_rows` and fills them in `_on_layer_changed`.
    """

    # What a project said about a choice, beyond the file and the layer: the
    # name it shows it under, and for a backdrop the colour of each category.
    DESCRIBED = ("title", "colors", "labels", "hidden")

    def __init__(self, role, title=None, parent=None):
        super().__init__(title or role.capitalize(), parent)

        self.role = role
        self._path = None

        # What a project said about what is in this slot, kept together with
        # what it was said *about*. Checked against that when read rather than
        # cleared wherever it could go stale: the layer combo and the category
        # combo can both move without passing through here, and a description
        # that is verified where it is used cannot be left behind by a path
        # neither of them took.
        self.described = {}

        self.path_combo = self.new_path_combo()

        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(self._browse)

        self.clear_button = QtWidgets.QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear)
        self.clear_button.setEnabled(False)

        self.layer_combo = QtWidgets.QComboBox()
        self.layer_combo.setEnabled(False)
        self.layer_combo.currentTextChanged.connect(self._on_layer_changed)

        self.grid = QtWidgets.QGridLayout(self)
        self.grid.addWidget(self.path_combo, 0, 0, 1, 2)
        self.grid.addWidget(browse, 0, 2)
        self.grid.addWidget(self.clear_button, 0, 3)
        self.grid.addWidget(QtWidgets.QLabel("layer"), 1, 0)
        self.grid.addWidget(self.layer_combo, 1, 1, 1, 3)
        self.grid.setColumnStretch(1, 1)

        self._add_layer_rows(self.grid, 2)

    # -- what a subclass fills in -----------------------------------------

    def _add_layer_rows(self, grid, row):
        """Widgets for whatever inside the layer this picker also chooses."""

    def _on_layer_changed(self, layer, preferred=None):
        """Refill those widgets for the layer just chosen."""

    def _clear_layer_rows(self):
        """Empty them again when the file is cleared."""

    # -- the file ----------------------------------------------------------

    def _browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Choose the file: {self.role}", "", VECTOR_FILTER
        )

        if path:
            self.set_path(path)

    def describe(self, spec):
        """
        Keeps what a project said, against the layer and field it said it of.

        The layer is taken as it ended up rather than as the spec asked for it:
        a shapefile names no layer, and the one the picker settled on is the
        one the description is now about.
        """

        said = {key: spec[key] for key in self.DESCRIBED if spec.get(key)}

        if said:
            said["layer"] = self.layer
            said["category_field"] = spec.get("category_field")

        self.described = said

    def described_now(self, field=None):
        """
        The part of it that still describes what is chosen here.

        The name survives a change of category field, being about the layer.
        The colours do not: they are one colour per value of one column, and
        under another column they are a palette for values that are not there.
        """

        said = self.described

        if not said or said.get("layer") != self.layer:
            return {}

        kept = {"title": said["title"]} if said.get("title") else {}

        if field is not None and said.get("category_field") == field:
            for key in ("colors", "labels", "hidden"):
                if said.get(key):
                    kept[key] = said[key]

        return kept

    def set_path(self, path, layer=None, preferred=None, quiet=False):
        """Loads the list of layers fit for the role. Returns False if there are none."""

        # Said of the file that was here, which this is not. `restore` puts it
        # back afterwards, being the only caller that has anything to put.
        self.described = {}

        try:
            candidates = VectorSource.candidate_layers(path, self.role)
        except Exception as err:
            if not quiet:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Unreadable file",
                    f"{Path(path).name}\n\n{str(err).splitlines()[0]}",
                )
            return False

        if not candidates:
            if not quiet:
                QtWidgets.QMessageBox.information(
                    self,
                    "No suitable layer",
                    f"{Path(path).name} holds no {self.role} layer.",
                )
            return False

        self._path = Path(path)
        self.clear_button.setEnabled(True)

        with QtCore.QSignalBlocker(self.layer_combo):
            self.layer_combo.clear()
            self.layer_combo.addItems(candidates)

            if layer and layer in candidates:
                self.layer_combo.setCurrentText(layer)

        self.layer_combo.setEnabled(True)

        # After the layer is settled, because the list names an entry by both
        # and one of the two was only just decided.
        self._relist(dict(path=str(path), layer=self.layer_combo.currentText()))
        self._on_layer_changed(self.layer_combo.currentText(), preferred=preferred)

        self.changed.emit()

        return True

    def clear(self):
        self._path = None
        self._relist()
        self.clear_button.setEnabled(False)
        self.layer_combo.clear()
        self.layer_combo.setEnabled(False)
        self._clear_layer_rows()

        self.changed.emit()

    @property
    def path(self):
        return None if self._path is None else str(self._path)

    @property
    def layer(self):
        return self.layer_combo.currentText() or None


class VectorPicker(LayerPicker):
    """One backdrop layer for one role: file, layer within the file, categories."""

    # The names a category column usually goes by. The Italian ones are kept
    # alongside the English: the geological maps this is used on are surveyed
    # in Italy, and their attribute tables say `sigla` and `unita`. On polygons
    # categorisation starts on because it is almost always what you want; on
    # lines and points it starts off, since twenty tints over four hundred
    # faults cannot be read.
    PREFERRED_FIELDS = ("code", "sigla", "unit", "unita", "type", "tipo", "name", "nome")

    def _add_layer_rows(self, grid, row):
        self.category_combo = QtWidgets.QComboBox()
        self.category_combo.setEnabled(False)

        grid.addWidget(QtWidgets.QLabel("categories"), row, 0)
        grid.addWidget(self.category_combo, row, 1, 1, 3)

    def _on_layer_changed(self, layer, preferred=None):
        if not self._path or not layer:
            return

        try:
            fields = VectorSource.text_fields(self._path, layer)
        except Exception:
            fields = []

        with QtCore.QSignalBlocker(self.category_combo):
            self.category_combo.clear()
            self.category_combo.addItem("(none)")
            self.category_combo.addItems(fields)

            chosen = None

            if preferred and preferred in fields:
                chosen = preferred
            elif self.role == "polygons":
                chosen = next((f for f in self.PREFERRED_FIELDS if f in fields), None)

            self.category_combo.setCurrentText(chosen or "(none)")

        self.category_combo.setEnabled(bool(fields))

    def _clear_layer_rows(self):
        self.category_combo.clear()
        self.category_combo.setEnabled(False)

    def restore(self, spec):
        if not self.set_path(
            spec["path"], spec.get("layer"), spec.get("category_field"), quiet=True
        ):
            return False

        self.describe(spec)

        # `set_path` listed what it knew, which is the file and the layer. The
        # whole spec is better: picked out of the list again later, it comes
        # back with its category field rather than with a fresh guess at one.
        self._relist(spec)

        return True

    def value(self):
        """The chosen role as a dictionary, or None if the slot is empty."""

        if self._path is None:
            return None

        field = self.category_combo.currentText()
        field = None if field in ("", "(none)") else field

        spec = dict(
            path=str(self._path),
            role=self.role,
            layer=self.layer,
            category_field=field,
        )

        # The colours ride along in the choice, which is what carries them to
        # the map and to the section without a second way in. They are written
        # down with it too: a palette is a few kilobytes of JSON next to a
        # path, and paying that once is better than reparsing a four-megabyte
        # project on every startup to draw the same units the same way.
        spec.update(self.described_now(field))

        return spec


class AnglePicker(LayerPicker):
    """
    A layer whose features carry an orientation, and which columns hold it.

    Only numeric fields are offered for the two angles, and they are guessed
    from their names before the user is asked: an Italian survey writes
    `Immersione` and `Inclinazione`, an English-language one `dipdir` and
    `dip`, and either way the guess is checked against the field list rather
    than assumed.

    The convention is a choice and not a checkbox because it changes what the
    azimuth *means* -- a right-hand-rule strike read as a dip direction is 90
    degrees wrong on every feature, and wrong in a way that still plots.

    Two layers answer this description and the difference is the geometry: a
    point is where a plane was measured, a line is where it crops out. What
    they want asked is the same, so it is asked in one place.
    """

    DIP_DIR_FIELDS = (
        "immersione", "dipdir", "dip_dir", "dipdirection", "dip_direction",
        "azimuth", "azimut", "dir", "strike",
    )

    DIP_FIELDS = ("inclinazione", "dip", "dipangle", "dip_angle", "angolo", "incl")

    def _add_layer_rows(self, grid, row):
        self.dip_dir_combo = QtWidgets.QComboBox()
        self.dip_combo = QtWidgets.QComboBox()

        for combo in (self.dip_dir_combo, self.dip_combo):
            combo.setEnabled(False)
            combo.currentTextChanged.connect(lambda _: self.changed.emit())

        self.convention_combo = QtWidgets.QComboBox()
        self.convention_combo.addItems([label for label, _ in CONVENTIONS])

        grid.addWidget(QtWidgets.QLabel("azimuth"), row, 0)
        grid.addWidget(self.dip_dir_combo, row, 1)
        grid.addWidget(QtWidgets.QLabel("dip"), row, 2)
        grid.addWidget(self.dip_combo, row, 3)
        grid.addWidget(QtWidgets.QLabel("read as"), row + 1, 0)
        grid.addWidget(self.convention_combo, row + 1, 1, 1, 3)

    def _on_layer_changed(self, layer, preferred=None):
        if not self._path or not layer:
            return

        try:
            fields = numeric_fields(self._path, layer)
        except Exception:
            fields = []

        wanted = dict(preferred or {})

        for key, combo, guesses in (
            ("dip_dir_field", self.dip_dir_combo, self.DIP_DIR_FIELDS),
            ("dip_field", self.dip_combo, self.DIP_FIELDS),
        ):
            with QtCore.QSignalBlocker(combo):
                combo.clear()
                combo.addItem("(choose)")
                combo.addItems(fields)

                asked = wanted.get(key)
                chosen = asked if asked in fields else _first_match(guesses, fields)
                combo.setCurrentText(chosen or "(choose)")

            combo.setEnabled(bool(fields))

        # A field named `strike` is being called a strike by whoever wrote it,
        # and that is better evidence of the convention than the default is.
        if (self.dip_dir_combo.currentText() or "").lower().startswith("strike"):
            self.convention_combo.setCurrentIndex(1)

        self.changed.emit()

    def _clear_layer_rows(self):
        for combo in (self.dip_dir_combo, self.dip_combo):
            combo.clear()
            combo.setEnabled(False)

    @property
    def is_filled(self):
        """A layer is not enough: both angles have to have been pointed at."""

        if self._path is None:
            return False

        return all(
            combo.currentText() not in ("", "(choose)")
            for combo in (self.dip_dir_combo, self.dip_combo)
        )

    def restore(self, spec):
        """Puts back a choice made earlier, convention included."""

        if not self.set_path(spec["path"], spec.get("layer"), spec, quiet=True):
            return False

        # After `set_path`, which may have guessed the convention off a field
        # called `strike`: whoever says which one it is has the better claim,
        # including when what they say is the default.
        self.convention_combo.setCurrentIndex(1 if spec.get("is_rhr_strike") else 0)

        self.describe(spec)

        # And the whole spec into the list, over the file and layer `set_path`
        # put there: chosen again later it keeps its two columns and its
        # convention, which is most of what made it worth remembering.
        self._relist(spec)

        return True

    def value(self):
        """The choice as a dictionary, or None while it is incomplete."""

        if not self.is_filled:
            return None

        spec = dict(
            path=str(self._path),
            role=self.role,
            layer=self.layer,
            dip_dir_field=self.dip_dir_combo.currentText(),
            dip_field=self.dip_combo.currentText(),
            is_rhr_strike=CONVENTIONS[self.convention_combo.currentIndex()][1],
        )

        # Only the name: these layers are drawn by the tool that asked for
        # them, symbol by symbol, and a backdrop palette has nothing to say
        # about a dip tick.
        spec.update(self.described_now())

        return spec


class AttitudePicker(AnglePicker):
    """The point layer a structural tool reads: one station, one measurement."""

    def __init__(self, parent=None):
        super().__init__("points", title="Attitudes", parent=parent)


class TracePicker(AnglePicker):
    """
    The line layer a profile is cut against: outcrop traces carrying a plane.

    One field more than the attitudes want, and it is the one that says which
    lines belong together. A fault mapped across a sheet arrives as a dozen
    fragments, and what a section labels is the fault, not the fragment; with
    no field chosen the whole layer is one system, which is the right answer
    for a file holding one.
    """

    # The same names a backdrop layer is categorised by: it is the same
    # question asked of the same attribute tables.
    CATEGORY_FIELDS = VectorPicker.PREFERRED_FIELDS + ("source", "sorgente", "fault", "faglia")

    def __init__(self, parent=None):
        super().__init__("lines", title="Traces with attitudes", parent=parent)

    def _add_layer_rows(self, grid, row):
        self.category_combo = QtWidgets.QComboBox()
        self.category_combo.setEnabled(False)
        self.category_combo.currentTextChanged.connect(lambda _: self.changed.emit())

        grid.addWidget(QtWidgets.QLabel("grouped by"), row, 0)
        grid.addWidget(self.category_combo, row, 1, 1, 3)

        super()._add_layer_rows(grid, row + 1)

    def _on_layer_changed(self, layer, preferred=None):
        if not self._path or not layer:
            return

        try:
            fields = VectorSource.text_fields(self._path, layer)
        except Exception:
            fields = []

        asked = (preferred or {}).get("category_field")

        with QtCore.QSignalBlocker(self.category_combo):
            self.category_combo.clear()
            self.category_combo.addItem("(none)")
            self.category_combo.addItems(fields)

            chosen = asked if asked in fields else _first_match(self.CATEGORY_FIELDS, fields)
            self.category_combo.setCurrentText(chosen or "(none)")

        self.category_combo.setEnabled(bool(fields))

        super()._on_layer_changed(layer, preferred)

    def _clear_layer_rows(self):
        self.category_combo.clear()
        self.category_combo.setEnabled(False)

        super()._clear_layer_rows()

    def value(self):
        spec = super().value()

        if spec is None:
            return None

        field = self.category_combo.currentText()
        spec["role"] = "traces"
        spec["category_field"] = None if field in ("", "(none)") else field

        return spec


def picker_for(slot):
    """The widget that fills one slot."""

    if slot == "dem":
        return RasterPicker()

    if slot == "attitudes":
        return AttitudePicker()

    if slot == "traces":
        return TracePicker()

    return VectorPicker(slot)


class SourcesDialog(QtWidgets.QDialog):
    """
    What to open, asked for one tool and filled in from the last answer.

    `wants` decides everything visible: which slots appear, in the order the
    tool lists them -- the required one first, in both tools, because that is
    the one being asked for -- and which are coloured. `chosen` is whatever was
    picked before, per slot, so
    that going from one tool to the other does not mean naming the same files
    again; slots this tool does not want are simply not shown, and the caller
    keeps them for the tool that does.

    Everything is inside a scroll area, and the buttons are outside it. Not
    decoration: a dialog cannot be resized below the sum of its parts, so on a
    short screen a plain stack of boxes pushes the buttons under the bottom
    edge and leaves no way to get at them -- what cannot be resized cannot be
    scrolled either.
    """

    def __init__(
        self, parent=None, wants=None, chosen=None, recent=None, title="gSurf - sources"
    ):
        super().__init__(parent)

        self.setWindowTitle(title)
        self.setMinimumWidth(560)
        self.setMinimumHeight(240)

        self.wants = dict(wants or {})
        chosen = dict(chosen or {})

        self.boxes = {}
        self.project = None

        content = QtWidgets.QWidget()
        stack = QtWidgets.QVBoxLayout(content)
        stack.setContentsMargins(0, 0, 0, 0)

        stack.addLayout(self._project_row())

        for slot, level in self.wants.items():
            box = picker_for(slot)
            box.set_requirement(level)
            box.changed.connect(self._refresh_ok)

            # Before `restore`, which puts the choice being put back at the top
            # of this list rather than beside itself somewhere down it.
            if recent is not None:
                box.offer(recent.entries(slot))

            self.boxes[slot] = box
            stack.addWidget(box)

        stack.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setWidget(content)

        # An explicit minimum beats the one the contents ask for, which is what
        # lets the window be dragged shorter than everything it holds.
        scroll.setMinimumHeight(120)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Open
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        # Qt's standard buttons already read "Open" and "Cancel" untranslated,
        # so nothing has to be written over them here.

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(scroll, 1)
        layout.addWidget(self.buttons)

        # The slots that were offered a remembered choice and could not take
        # it. Kept rather than only acted on, because the dialog is built
        # before the caller gets it back and the caller is who owns the list
        # the choice came from.
        self.refused = self.restore(chosen)

        self._fit(content, layout)

    # -- a whole project at once -------------------------------------------

    def _project_row(self):
        """
        The one control that is not about a single slot.

        Above the boxes rather than inside any of them, because what it fills
        is all of them: a project is where a DEM and the sheets drawn over it
        were last seen together, and offering it per slot would make choosing
        it five times the way to use it once.
        """

        self.project_button = QtWidgets.QPushButton("From a QGIS project...")
        self.project_button.clicked.connect(self._browse_project)

        self.project_label = QtWidgets.QLabel()
        self.project_label.setStyleSheet("color: gray; font-size: 10px;")
        self.project_label.setWordWrap(True)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.project_button)
        row.addWidget(self.project_label, 1)

        return row

    def _browse_project(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose a QGIS project", "", PROJECT_FILTER
        )

        if path:
            self.use_project(path)

    def use_project(self, path):
        """
        Reads a project and offers what it holds to every slot on show.

        Offers and does not fill. Two slots take lines and two take points, and
        nothing in a layer says whether it was meant as a backdrop or as the
        data a tool reads -- the geometry narrows a hundred layers to the
        twenty-eight that could be the polygons, and which one it is stays the
        question the dialog was opened to ask.

        A project that cannot be read is reported and changes nothing, unlike a
        remembered choice that quietly does not come back: this one was asked
        for, just now, by hand.
        """

        from . import qgis_project

        try:
            project = qgis_project.read(path)
        except Exception as err:
            QtWidgets.QMessageBox.warning(
                self,
                "Unreadable project",
                f"{Path(path).name}\n\n{str(err).splitlines()[0]}",
            )
            return None

        self.project = project

        for slot, box in self.boxes.items():
            box.offer_project(project.entries_for(slot))

        self.project_label.setText(project.summary())

        return project

    def restore(self, chosen):
        """
        Fills the slots from an earlier answer, ignoring what is not asked for.

        Quietly, and that is the part that matters once answers outlive the
        run they were given in. Inside one session every remembered file was
        opened minutes ago and a failure is worth a box; across runs a file
        that has moved, a share not mounted this morning or a layer since
        renamed are all ordinary, and a gSurf that opened onto a stack of
        warnings would be reporting the weather. Picking a file is an act that
        deserves an answer, having one put back for you is not -- so a refused
        slot is left empty, coloured if the tool needs it, and named in the
        list this returns.
        """

        refused = []

        for slot, box in self.boxes.items():
            if not chosen.get(slot):
                continue

            if box.restore(chosen[slot]) is False:
                refused.append(slot)

        self._refresh_ok()

        return refused

    def _fit(self, content, layout):
        """As tall as it needs to be, and never taller than the screen."""

        screen = self.screen() or QtWidgets.QApplication.primaryScreen()

        wanted = (
            content.sizeHint().height()
            + self.buttons.sizeHint().height()
            + layout.contentsMargins().top()
            + layout.contentsMargins().bottom()
            + layout.spacing()
        )

        if screen is not None:
            wanted = min(wanted, int(screen.availableGeometry().height() * 0.85))

        self.resize(max(self.minimumWidth(), content.sizeHint().width()), wanted)

    def _refresh_ok(self):
        """
        Open stays disabled until what this tool needs is actually there.

        The last clause is the one that is not about any tool: a session takes
        its projection and its extent from what was opened, and with nothing
        opened there is no answer to where we are. `Session.open` raises on it,
        and a raise after the dialog closes is a worse way to be told.
        """

        ready = any(box.is_filled for box in self.boxes.values()) and all(
            box.is_filled for box in self.boxes.values() if box.level == "required"
        )

        self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Open).setEnabled(ready)

    def choices(self):
        """What was chosen, one key per slot this dialog asked about."""

        return {slot: box.value() for slot, box in self.boxes.items()}


def describe(chosen):
    """The choices in one line each, for a window that has to show them."""

    lines = []

    if chosen.get("dem"):
        lines.append(f"DEM: {Path(chosen['dem']).name}")

    for slot in OWN_DATA_SLOTS:
        spec = chosen.get(slot)

        if not spec:
            continue

        name = spec.get("layer") or Path(spec["path"]).name
        detail = f"{spec['dip_dir_field']}, {spec['dip_field']}"

        if spec.get("category_field"):
            detail += f", by {spec['category_field']}"

        lines.append(f"{slot}: {name} ({detail})")

    for spec in vectors_of(chosen):
        name = spec.get("layer") or Path(spec["path"]).name
        lines.append(f"{spec['role']}: {name}")

    return lines
