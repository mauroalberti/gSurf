"""
A line layer written out as a file this format can hold: the way in that was missing.

Two directions existed and neither was this one. `export_gsurf.py` runs out of
gstruct and into a GeoPackage; `export_geology.py` runs into gstruct out of one,
and is a script written for a single survey -- its `type` column, its six
spellings of "maybe", its 100 m attachment radius -- so every other layer had no
way in at all. The trace editor made that visible rather than caused it: a tool
whose slot takes one suffix is a tool you cannot reach with the data you have.

**What an importer may not do is already written down**, as four rules in
FORMAT.md, and they are the shape of this module rather than a checklist bolted
to it: the source string is kept beside anything normalised from it as `raw.*`;
what the source does not say comes out `unknown` with a `reason=`, never a
plausible default; what does not attach is not dropped; and synonyms do not
become grades.

**The fourth rule is why this asks so little.** `Tipologia` on a CARG tectonic
sheet holds `certo`, `incerto` and `sepolto` -- 10027, 2205 and 486 of them over
the eight sheets of the AOI -- and those three words are two axes mixed.
FORMAT.md uses this very column as its example of the damage: "certain but not
exposed" has no box to be written in, so `incerto` and `sepolto` come out
mutually exclusive when they are answers to different questions. Deciding here
which axis each word belongs on would be that damage done silently, in a dialog.
So `certainty` and `exposure` go out `unknown` with the reason, the column is
preserved beside them, and the curator says which is which in the editor -- which
is the tool this exists to feed.

**And why there is no kind column.** `kind` is a vocabulary, and `dumps` writes
it unquoted while the parser reads back `pos[0]`: a value with a space in it is
truncated on the round trip, so the file would not say what it appears to say.
No line layer in the AOI has a column that would survive anyway -- `Tipo` is
free Italian text in 28 spellings, several carrying a parenthetical instruction
to the cartographer inside the value. So the kind is one token typed once for the
layer, or left unsaid; on a sheet whose 24717 lines are 16506 stratigraphic
contacts and 2152 faults, unsaid is the true answer and the editor is where it
stops being.

**No DEM and no fits.** A plane comes out of this only where a column already
holds one, and then it is a `fit` carrying `from=table`: FORMAT.md defines an
`attitude` as an observation *at a point*, and a column that speaks for a whole
trace has no point to be at -- anchored to the midpoint it would claim a place
nobody stood. `fit` is the construct for a plane over an interval, `from=` is
what says which producer made it, and no diagnostics are written because none
were computed. Reading a plane off the topography is `traces.fit_records`, and
it belongs to whoever asks for it with a DEM open.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PyQt6 import QtCore, QtWidgets

from .attitudes import CONVENTIONS, admissible, normalised_azimuth, numeric, numeric_fields
from .curation import SUFFIX, degrees_not_metres, gstruct_plane, module
from .sources import first_match
from .vectors import VectorSource, single_parts

# What a plane read off the attribute table says it came from. Not decoration:
# `from=` is the format's own way of telling one producer's fits from another's,
# and this one carries none of `export_geology.py`'s diagnostics because it
# computed nothing -- the number was in the table.
FROM = "table"

# The ident a feature gets when the source has no name for it. A handle and not
# a fact, which is why minting one is allowed at all: without it the structure
# cannot be named, and a structure that cannot be named cannot be curated.
MINTED = "L"

# What separates a source name from the part of it this had to write. One
# suffix, two causes -- a multipart feature, or an ident the source repeated --
# and both mean the same thing to whoever reads the file: this name was not
# unique, here is which one.
PART = "."

# The reason attached to an axis the source does not speak to. FORMAT.md's rule
# 2 is that the value is `unknown` and the reason is written, and these are the
# two reasons a layer can give: the column is not there, or it is there and this
# does not presume to read it.
ABSENT = "assente-in-sorgente"
UNREAD = "non-interpretato"


@dataclass
class Imported:
    """What the layer turned into, and what of it did not travel."""

    features: int = 0        # rows that held a line
    structures: int = 0      # blocks written
    split: int = 0           # features that became more than one block
    minted: int = 0          # idents this made up
    collided: int = 0        # idents the source used more than once
    planes: int = 0          # fits written off the table
    dropped: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def summary(self):
        text = f"{self.structures} structure(s) from {self.features} feature(s)"

        if self.planes:
            text += f", {self.planes} with a plane off the table"

        for count, said in (
            (self.split, "multipart"),
            (self.collided, "sharing a name"),
            (self.minted, "unnamed"),
        ):
            if count:
                text += f"; {count} {said}"

        if self.dropped:
            detail = ", ".join(f"{reason} x{n}" for reason, n in self.dropped.items())
            text += f"; not carried: {detail}"

        return text


@dataclass
class Mapping:
    """Which column means what, which is the whole of what this asks."""

    layer: str = None
    ident_field: str = None
    label_field: str = None
    kind: str = ""
    dip_dir_field: str = None
    dip_field: str = None
    is_rhr_strike: bool = False

    # The columns carried through untouched, as `raw.*`. They are source strings
    # that nothing here interpreted, and the prefix is the format's own mark for
    # exactly that -- which is also what keeps invented top-level attribute
    # names out of the file. This is "the relevant data", and it is asked
    # because only the surveyor knows which columns are.
    keep_fields: tuple = ()

    @property
    def has_attitudes(self):
        """Both angle columns, or neither: half an answer is not one."""

        return bool(self.dip_dir_field) and bool(self.dip_field)


def fields_of(path, layer=None):
    """The columns on offer, as `(all, numeric)`, with neither read as data."""

    try:
        every = VectorSource.text_fields(path, layer) or []
    except Exception:
        every = []

    try:
        numbers = numeric_fields(path, layer) or []
    except Exception:
        numbers = []

    # `text_fields` is the text ones, and an ident is very often an integer:
    # `OBJECTID` is the only unique column on either CARG line sheet.
    merged = list(every) + [name for name in numbers if name not in every]

    return merged, list(numbers)


def guess(path, layer=None):
    """A mapping with the columns this can recognise already filled in."""

    every, numbers = fields_of(path, layer)

    return Mapping(
        layer=layer,
        ident_field=first_match(IDENT_FIELDS, every),
        label_field=first_match(LABEL_FIELDS, every),
        dip_dir_field=first_match(DIP_DIR_FIELDS, numbers),
        dip_field=first_match(DIP_FIELDS, numbers),
        keep_fields=tuple(every),
    )


# What an ident and a label are called in the tables this opens. `objectid` is
# last rather than first: it is a row number the file format assigned, and a
# survey's own code for a fault is a better name whenever there is one.
IDENT_FIELDS = ("ident", "code", "codice", "id", "fid", "objectid")

LABEL_FIELDS = ("label", "name", "nome", "etichetta", "denominazione")

# The same two lists the sources dialog guesses from, and deliberately the same:
# it is the same question asked of the same attribute tables.
DIP_DIR_FIELDS = (
    "immersione", "dipdir", "dip_dir", "dipdirection", "dip_direction",
    "azimuth", "azimut", "dir", "strike",
)

DIP_FIELDS = ("inclinazione", "dip", "dipangle", "dip_angle", "angolo", "incl")


def crs_of(frame):
    """
    The projection to declare, or why it cannot be written.

    EPSG or nothing, and that is the format's constraint rather than a
    preference: `crs` is read back as `pos[0]`, one whitespace-delimited token,
    so a WKT written on that line would come back as the word `PROJCRS` and the
    file would be wrong about where it is. A projection with no EPSG code is
    refused here instead, where it can be said.
    """

    if frame.crs is None:
        return None, "the layer declares no projection"

    code = frame.crs.to_epsg()

    if code is None:
        return None, (
            f"the layer's projection has no EPSG code ({frame.crs.name}), and "
            f"the `crs` line holds one word: written as WKT it would be "
            f"truncated to the first of its own"
        )

    return f"EPSG:{code}", None


def transcript_of(path, mapping, project=None):
    """
    A line layer as a file of its own, as `(text, Imported)`.

    A transcript and not a curation, which is the difference from
    `curation.curation_of` and the reason this is a second writer rather than an
    argument to that one. A curation carries no geometry and states only what was
    decided: restating its source would make it a second copy of it, and applied
    back it would double every fit it had just read. Here the second copy *is*
    the point -- there is no source file for it to be laid over, and the `path`
    is most of what is being carried across.

    **One structure per line, and no pooling.** `TraceAttitudeSource` pools
    features that share a category and an attitude, because two fragments
    carrying one plane are one plane digitised in pieces and a panel should list
    it once. A file is not a panel: pooling here would merge two faults that
    happen to dip alike into one block with one name, and the name is what a
    curation has to hold on to afterwards.

    **A structure has one path, so a multipart feature becomes several.** Joining
    the parts end to end is the one thing that cannot be done -- it invents a
    segment that is not on the ground, and an anchor near the gap would project
    onto it -- and dropping all but the longest is what `export_geology.py` does
    and is data thrown away. So the parts are written as their own structures,
    suffixed, sharing `set=` with the name they came from, which is the
    attribute that already means "these traces are one thing".
    """

    gstruct = module()

    import geopandas as gpd

    frame = gpd.read_file(path, layer=mapping.layer) if mapping.layer else gpd.read_file(path)

    crs, refusal = crs_of(frame)

    if refusal is not None:
        raise ValueError(refusal)

    report = Imported()

    dataset = gstruct.Dataset(crs=crs, meta={"version": gstruct.VERSION})

    if project:
        dataset.meta["project"] = project

    dataset.meta["source"] = (
        f"{path}{' layer=' + mapping.layer if mapping.layer else ''}"
    )
    dataset.meta["exported"] = _now()

    frame, angles = _usable(frame, mapping, report)

    if frame is None:
        return gstruct.dumps(dataset), report

    named = _named(frame, mapping, report)

    for (ident, source_name), parts, ndx in named:
        structure = gstruct.Structure(
            ident=ident,
            label=_text(frame, mapping.label_field, ndx),
            kind=mapping.kind or "unknown",
            path=[(float(x), float(y)) for x, y in parts.coords],
        )

        if source_name is not None:
            structure.attrs["set"] = source_name

        for name in mapping.keep_fields:
            said = _text(frame, name, ndx)

            if said:
                structure.attrs[f"raw.{name}"] = said

        # Both axes, always, and always `unknown`. FORMAT.md is explicit that
        # they stay separate even when one is entirely unknown, and that writing
        # the line at all is what makes the silence a statement somebody can act
        # on: an axis absent from the file and an axis nobody has decided read
        # the same to `value_at`, and only one of them says so out loud.
        for axis in ("certainty", "exposure"):
            structure.spans.append(gstruct.Span(
                axis=axis, value="unknown", start=None, end=None,
                attrs={"reason": UNREAD if mapping.keep_fields else ABSENT},
            ))

        if angles is not None:
            plane = _plane(gstruct, angles, ndx, mapping)

            if plane is not None:
                structure.fits.append(plane)
                report.planes += 1

        dataset.structures.append(structure)
        report.structures += 1

    report.notes = _notes(mapping, report)
    dataset.meta["note"] = "; ".join(report.notes)

    said = degrees_not_metres(dataset)

    if said:
        raise ValueError(said)

    return gstruct.dumps(dataset.resolve()), report


def _now():
    import datetime

    return datetime.datetime.now().isoformat(timespec="seconds")


def _notes(mapping, report):
    """What the file says about its own making, in the order it was decided."""

    notes = [
        "Trascritto da un layer di linee da gSurf: una struttura per linea, "
        "nessun raggruppamento."
    ]

    if not mapping.kind:
        notes.append(
            "`kind` non dichiarato: la sorgente non ha una colonna nel "
            "vocabolario del formato"
        )

    notes.append(
        "`certainty` e `exposure` escono unknown: questo import non "
        "interpreta le colonne della sorgente"
    )

    if report.planes:
        notes.append(
            f"{report.planes} `fit` con from={FROM}: il piano viene dalla "
            f"tabella, non da un calcolo, e non porta diagnostica"
        )

    if report.split:
        notes.append(
            f"{report.split} geometrie multiparte divise in strutture con "
            f"`set=` in comune"
        )

    if report.minted:
        notes.append(f"{report.minted} ident coniati qui ({MINTED}0001...)")

    return notes


def _usable(frame, mapping, report):
    """
    The rows that hold a line, and their angles where there are angles to read.

    The angle columns decide admissibility and the geometry does not: a row with
    no line is nothing to write, while a row whose dip is 99 -- which is how a
    CARG sheet records contorted bedding -- is a trace that is perfectly good
    and an attitude that is not. So the first is dropped and the second keeps
    its geometry and loses its plane.
    """

    usable = frame.geometry.notna() & ~frame.geometry.is_empty

    without = int((~usable).sum())

    if without:
        report.dropped["no geometry"] = without

    frame = frame[usable]

    if frame.empty:
        return None, None

    if not mapping.has_attitudes:
        return frame, None

    azimuth = numeric(frame[mapping.dip_dir_field])
    dip = numeric(frame[mapping.dip_field])

    keep, dropped = admissible(azimuth, dip)

    # Counted as an attitude not carried rather than as a row not carried, which
    # is the truth of it: the trace is written either way.
    for reason, count in dropped.items():
        report.dropped[f"plane: {reason}"] = count

    azimuth = normalised_azimuth(azimuth, dip)

    return frame, (azimuth, dip, keep)


def _named(frame, mapping, report):
    """
    Every block to be written, as `((ident, set), line, row)`, with idents unique.

    Two passes, because uniqueness is not knowable one row at a time. A name is
    suffixed where one feature became several parts *or* where the source used
    the same name twice, and those are one rule with two causes: the ident of a
    written structure is unique, and where the source's was not the `set=` says
    what it was.

    The two causes are counted apart even though they are handled alike, because
    they are different news. A multipart trace suffixed into two blocks is this
    module doing the only thing it can; the same name on two unrelated features
    is the source saying something about itself, and on a layer where the ident
    column turns out to be a category rather than a name it is the whole answer
    -- `raw` counts the rows, `set` counts what they were called.
    """

    candidates = []

    for ndx in range(len(frame)):
        parts, other = single_parts(frame.geometry.iloc[ndx], "LineString")

        if other:
            report.dropped["parts that were not lines"] = (
                report.dropped.get("parts that were not lines", 0) + other
            )

        # A one-vertex line has no length and no direction: it draws nothing,
        # measures nothing, and `path 1` would be a trace nobody can walk.
        parts = [line for line in parts if len(line.coords) > 1]

        if not parts:
            report.dropped["no line geometry"] = (
                report.dropped.get("no line geometry", 0) + 1
            )
            continue

        if len(parts) > 1:
            report.split += 1

        said = _text(frame, mapping.ident_field, ndx)

        if not said:
            report.minted += 1

        report.features += 1

        for line in parts:
            candidates.append((said, line, ndx))

    # How many blocks want each name, and how many *features* wanted it. The
    # first decides the suffix; the second is what makes it a collision rather
    # than a split, and counting one for the other would report every multipart
    # trace as a duplicate name.
    blocks, rows = {}, {}

    for said, _, ndx in candidates:
        if not said:
            continue

        blocks[said] = blocks.get(said, 0) + 1
        rows.setdefault(said, set()).add(ndx)

    report.collided = sum(
        len(seen) for said, seen in rows.items() if len(seen) > 1
    )

    minted, taken, out = 0, {}, []

    for said, line, ndx in candidates:
        if not said:
            minted += 1
            out.append(((f"{MINTED}{minted:04d}", None), line, ndx))
            continue

        if blocks[said] == 1:
            out.append(((said, None), line, ndx))
            continue

        taken[said] = taken.get(said, 0) + 1

        out.append(((f"{said}{PART}{taken[said]}", said), line, ndx))

    return out


def _text(frame, name, ndx):
    """One cell as a string, with a null reading as nothing said."""

    if not name or name not in frame.columns:
        return ""

    value = frame[name].iloc[ndx]

    if value is None or value != value:
        return ""

    said = str(value).strip()

    return "" if said.lower() in ("", "nan", "none", "<null>") else said


def _plane(gstruct, angles, ndx, mapping):
    """
    One row's plane as a `fit` over the whole trace, or None where it has none.

    `*` at both ends is the format's way of saying "the whole path", and it is
    the honest interval here: a column says the plane holds for the trace, and
    nothing in the table says where along it anybody looked.

    **No verdict, so the editor's band reads `fit:?`.** That is not a gap to be
    filled: `attitude_at` reports a fit's provenance as `fit:` plus its verdict,
    the three verdicts in FORMAT.md are all about how a plane came off a DEM, and
    this one did not come off anything -- it was in the table. A word invented to
    fill the field would be a verdict on a computation that never ran. What says
    where this plane came from is `from=`, which is in the line.
    """

    from geogst.core.geology.orientations import Plane

    azimuth, dip, keep = angles

    if not bool(keep[ndx]):
        return None

    plane = gstruct_plane(
        Plane(float(azimuth[ndx]), float(dip[ndx]), is_rhr_strike=mapping.is_rhr_strike)
    )

    # Rule 1, on the one value this normalises: the two numbers as the table
    # wrote them, beside the dip direction they were turned into. Without it a
    # strike read as a strike is indistinguishable afterwards from a dip
    # direction that was already one.
    raw = (
        f"{mapping.dip_dir_field}={azimuth[ndx]:.0f} "
        f"{mapping.dip_field}={dip[ndx]:.0f}"
    )

    return gstruct.Fit(
        plane=plane, start=None, end=None,
        attrs={"from": FROM, "src": "gsurf", "raw": raw},
    )


class ImportDialog(QtWidgets.QDialog):
    """
    What the columns mean, asked once and only about the layer being read.

    Deliberately not a `SlotBox`. The sources dialog asks what to open and keeps
    what was answered; this asks what a table means, which is a fact about one
    file and worth nothing the next time. What it does share is the guessing --
    the same two lists of column names, checked against the layer rather than
    assumed -- because that is knowledge about attribute tables and not
    plumbing.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("gSurf - lines to .gstruct")

        self._path = None
        self.written = None

        self.path_label = QtWidgets.QLabel("no file")
        self.path_label.setStyleSheet("color: gray; font-size: 10px;")

        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(self._browse)

        self.layer_combo = QtWidgets.QComboBox()
        self.layer_combo.setEnabled(False)
        self.layer_combo.currentTextChanged.connect(self._on_layer_changed)

        self.ident_combo = QtWidgets.QComboBox()
        self.label_combo = QtWidgets.QComboBox()
        self.dip_dir_combo = QtWidgets.QComboBox()
        self.dip_combo = QtWidgets.QComboBox()

        self.convention_combo = QtWidgets.QComboBox()
        self.convention_combo.addItems([label for label, _ in CONVENTIONS])

        self.kind_edit = QtWidgets.QLineEdit()
        self.kind_edit.setPlaceholderText("one word, or leave empty for unknown")

        self.keep_list = QtWidgets.QListWidget()
        self.keep_list.setMaximumHeight(120)

        self.report_label = QtWidgets.QLabel()
        self.report_label.setWordWrap(True)
        self.report_label.setStyleSheet("font-size: 10px;")

        self.write_button = QtWidgets.QPushButton("Write .gstruct...")
        self.write_button.setEnabled(False)
        self.write_button.clicked.connect(self._write)

        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.reject)

        source = QtWidgets.QGroupBox("Layer")
        grid = QtWidgets.QGridLayout(source)
        grid.addWidget(browse, 0, 0)
        grid.addWidget(self.layer_combo, 0, 1)
        grid.addWidget(self.path_label, 1, 0, 1, 2)
        grid.setColumnStretch(1, 1)

        names = QtWidgets.QGroupBox("What the columns mean")
        form = QtWidgets.QGridLayout(names)
        form.addWidget(QtWidgets.QLabel("ident"), 0, 0)
        form.addWidget(self.ident_combo, 0, 1)
        form.addWidget(QtWidgets.QLabel("label"), 0, 2)
        form.addWidget(self.label_combo, 0, 3)
        form.addWidget(QtWidgets.QLabel("azimuth"), 1, 0)
        form.addWidget(self.dip_dir_combo, 1, 1)
        form.addWidget(QtWidgets.QLabel("dip"), 1, 2)
        form.addWidget(self.dip_combo, 1, 3)
        form.addWidget(QtWidgets.QLabel("read as"), 2, 0)
        form.addWidget(self.convention_combo, 2, 1, 1, 3)
        form.addWidget(QtWidgets.QLabel("kind"), 3, 0)
        form.addWidget(self.kind_edit, 3, 1, 1, 3)
        form.setColumnStretch(1, 1)
        form.setColumnStretch(3, 1)

        keep = QtWidgets.QGroupBox("Columns to carry through, as raw.*")
        carried = QtWidgets.QVBoxLayout(keep)
        carried.addWidget(self.keep_list)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.report_label, 1)
        buttons.addWidget(self.write_button)
        buttons.addWidget(close)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(source)
        layout.addWidget(names)
        layout.addWidget(keep)
        layout.addLayout(buttons)

        self.setMinimumWidth(560)

    # -- the layer ---------------------------------------------------------

    NO_FIELD = "(none)"

    def _browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose the line layer", "",
            "Vector (*.gpkg *.shp *.geojson *.json *.gml *.kml *.sqlite *.fgb);;"
            "All files (*)",
        )

        if path:
            self.set_path(path)

    def set_path(self, path, layer=None):
        """Lists the line layers in the file, and fills the mapping from one."""

        try:
            layers = VectorSource.candidate_layers(path, "lines")
        except Exception as err:
            QtWidgets.QMessageBox.warning(
                self, "Unreadable", f"{path}\n\n{type(err).__name__}: {err}"
            )
            return False

        if not layers:
            QtWidgets.QMessageBox.information(
                self, "No lines",
                f"{path}\n\nno layer of lines in this file. An import needs "
                f"traces: what a structure has is one path, and a polygon or a "
                f"point is not one.",
            )
            return False

        self._path = path
        self.path_label.setText(str(path))

        with QtCore.QSignalBlocker(self.layer_combo):
            self.layer_combo.clear()
            self.layer_combo.addItems(layers)
            self.layer_combo.setCurrentText(
                layer if layer in layers else layers[0]
            )

        self.layer_combo.setEnabled(True)

        self._on_layer_changed(self.layer_combo.currentText())

        return True

    def _on_layer_changed(self, layer):
        if not self._path or not layer:
            return

        every, numbers = fields_of(self._path, layer)
        proposed = guess(self._path, layer)

        for combo, fields, chosen in (
            (self.ident_combo, every, proposed.ident_field),
            (self.label_combo, every, proposed.label_field),
            (self.dip_dir_combo, numbers, proposed.dip_dir_field),
            (self.dip_combo, numbers, proposed.dip_field),
        ):
            with QtCore.QSignalBlocker(combo):
                combo.clear()
                combo.addItem(self.NO_FIELD)
                combo.addItems(fields)
                combo.setCurrentText(chosen or self.NO_FIELD)

        # The same evidence the sources dialog reads: a column somebody called
        # `strike` is being called one by whoever wrote it.
        if (self.dip_dir_combo.currentText() or "").lower().startswith("strike"):
            self.convention_combo.setCurrentIndex(1)

        self.keep_list.clear()

        for name in every:
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked)
            self.keep_list.addItem(item)

        self.write_button.setEnabled(True)

        self._describe(layer)

    def _describe(self, layer):
        """How much is about to be written, before anybody waits for it."""

        try:
            import pyogrio

            info = pyogrio.read_info(self._path, layer=layer)
            count = int(info["features"])
        except Exception:
            self.report_label.setText("")
            return

        said = f"{count:,} feature(s)"

        # Said before the writing rather than after, because this is where it can
        # still be changed: 24717 lines is what a CARG sheet's `limiti_geologici`
        # holds, and every one of them becomes a block with its own name.
        if count > 5000:
            said += " - one structure each, and the editor lists them all"

        self.report_label.setText(said)

    # -- the mapping -------------------------------------------------------

    def _named(self, combo):
        text = combo.currentText()

        return None if text in ("", self.NO_FIELD) else text

    def mapping(self):
        """The choices as a `Mapping`."""

        kept = tuple(
            self.keep_list.item(row).text()
            for row in range(self.keep_list.count())
            if self.keep_list.item(row).checkState() == QtCore.Qt.CheckState.Checked
        )

        return Mapping(
            layer=self.layer_combo.currentText() or None,
            ident_field=self._named(self.ident_combo),
            label_field=self._named(self.label_combo),
            kind=self.kind_edit.text().strip(),
            dip_dir_field=self._named(self.dip_dir_combo),
            dip_field=self._named(self.dip_combo),
            is_rhr_strike=CONVENTIONS[self.convention_combo.currentIndex()][1],
            keep_fields=kept,
        )

    def refusal(self, mapping):
        """Why this mapping cannot be written, or None."""

        if len(mapping.kind.split()) > 1:
            return (
                f"`kind {mapping.kind}` is more than one word. The format writes "
                f"a kind unquoted and reads back the first token, so this would "
                f"be saved as `{mapping.kind.split()[0]}` and the file would not "
                f"say what it looks like it says. It is a vocabulary -- `fault`, "
                f"`thrust`, `contact` -- not a description."
            )

        if bool(mapping.dip_dir_field) != bool(mapping.dip_field):
            return (
                "One angle column is named and the other is not. Which half was "
                "meant cannot be guessed, and a plane needs both; leave them "
                "both empty to import the traces with no attitude on them."
            )

        return None

    # -- writing -----------------------------------------------------------

    def _write(self):
        mapping = self.mapping()

        said = self.refusal(mapping)

        if said:
            QtWidgets.QMessageBox.warning(self, "Not like that", said)
            return

        target, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Write the .gstruct", "", f"gstruct (*{SUFFIX})"
        )

        if not target:
            return

        if not str(target).lower().endswith(SUFFIX):
            target = f"{target}{SUFFIX}"

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)

        try:
            text, report = transcript_of(self._path, mapping)
        except Exception as err:
            QtWidgets.QMessageBox.critical(
                self, "Not imported", f"{type(err).__name__}: {err}"
            )
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        if not report.structures:
            QtWidgets.QMessageBox.information(
                self, "Nothing to write",
                f"no structure came out of this layer.\n\n{report.summary()}",
            )
            return

        from pathlib import Path

        Path(target).write_text(text, encoding="utf-8")

        self.written = str(target)

        QtWidgets.QMessageBox.information(
            self, "Imported",
            f"{target}\n\n{report.summary()}\n\n" + "\n".join(report.notes),
        )

        self.accept()


def run(parent=None):
    """
    The dialog, and the file it wrote if it wrote one.

    What comes back is a path, which is what the caller does something with: the
    launcher remembers it in the traces slot, so the editor proposes it next
    without anybody browsing for it again.
    """

    dialog = ImportDialog(parent)

    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None

    return dialog.written
