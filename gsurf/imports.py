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

**A plane in a column is a `fit` carrying `from=table`.** FORMAT.md defines an
`attitude` as an observation *at a point*, and a column that speaks for a whole
trace has no point to be at -- anchored to the midpoint it would claim a place
nobody stood. `fit` is the construct for a plane over an interval, `from=` is
what says which producer made it, and no diagnostics are written because none
were computed.

**A plane off the topography is a `fit` per stretch that holds**, and it is the
line FORMAT.md already had somebody else's name on: *gli intervalli ancorati ci
sono e si leggono -- gSurf ne scrive, uno per finestra che tiene -- quindi qui
manca il produttore, non il formato*. This is that producer. `traces` sweeps a
window along the trace and reports where the plane is held, loose, or merely a
line; a held run becomes one `fit` between two anchors, and a run that is only a
line becomes nothing, because a plane through a straight trace is arbitrary
rather than imprecise. The diagnostics are `window=` and `span_verdict=`, which
FORMAT.md names as this producer's own, and never `nvert=`, which counts
digitised vertices where this counts DEM samples.

**Two kinds of fit on one structure are ordered, not merged.** `attitude_at`
takes the *first* fit that covers a progressive, where `span_at` takes the
*last* -- so a local statement goes last among spans and first among fits, and
getting it the wrong way round is silent. The stretch read off the ground is
written before the column that speaks for the whole trace.

**A measured point becomes an `attitude`, or an `observation` saying why not.**
That is rule 3, and the only rule here with a threshold in it: past
`attach_within` metres the point is not a measurement on that trace, and what
the file then carries is the point, the nearest structure, the distance and the
threshold -- so the decision can be re-taken by someone who disagrees with it
rather than discovered to have been made. An attachment also changes what the
trace says about itself: a measurement inside `max_gap` outranks every fit, so
the stretch nearest the outcrop stops reporting the computation.
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

# What a plane read off the topography says it came from: FORMAT.md's own name
# for it, and deliberately the same one `export_geology.py` writes. Both sample
# a DEM along a mapped trace, so they are the same kind of derivation; what
# separates them is the diagnostic each can honestly report, and `src=gsurf`
# says which program to read those by.
FROM_DEM = "trace-dem"

# How far a measured point may sit from a trace and still be a measurement on
# it. `export_geology.py`'s `ATTACH_M`, and the same 100 m, which is a figure
# with evidence under it: of the 24 Monte Alpi field attitudes, 23 fell within
# 100 m of their fault with a median offset of half a metre, so the threshold
# is not cutting a continuum in half -- it is separating a station logged on a
# trace from one logged somewhere else.
ATTACH_M = 100.0

# Why a point did not attach. The first is FORMAT.md's own example wording, and
# both exist so that rule 3 has something to write: a point kept with the reason
# is a decision somebody can disagree with, and a point dropped is not.
UNATTACHED = "oltre-soglia"
NO_ATTITUDE = "assente-in-sorgente"

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

# Roughly what one structure costs to fit, in seconds, for the estimate the
# dialog shows before anybody waits. Measured at 7 ms over 1200 traces of
# `elementi_tettonici` on the 5 m DEM and rounded up, because the point of the
# number is to stop a click on a sheet of twenty-five thousand lines rather than
# to be right to the second -- a wait shorter than promised costs nothing.
DEM_SECONDS_EACH = 0.01


@dataclass
class Imported:
    """What the layer turned into, and what of it did not travel."""

    features: int = 0        # rows that held a line
    structures: int = 0      # blocks written
    split: int = 0           # features that became more than one block
    minted: int = 0          # idents this made up
    collided: int = 0        # idents the source used more than once
    planes: int = 0          # fits written off the table
    attached: int = 0        # points that became an attitude on a trace
    observations: int = 0    # points kept as observations instead, with the reason
    minted_points: int = 0   # station codes this made up
    fits: int = 0            # fits read off the DEM
    read: int = 0            # structures that gave at least one of them
    # The three ways a structure can carry no fit, apart, because they are three
    # different facts and lumping them reports the wrong one. Measured on
    # `elementi_tettonici`: the first 400 features of the sheet are wholly off
    # the 5 m DEM, and the median trace on it is 172 m long -- so a single
    # "determines no attitude" count would have said the topography refused
    # 12718 traces when it was never asked about most of them.
    unreached: int = 0       # no elevation under the trace at all
    too_short: int = 0       # shorter than the shortest window swept
    silent: int = 0          # sampled, long enough, and nothing held
    swept: int = 0           # structures that chose their own window length
    shortened: int = 0       # read at less than the fallback, that being all that fit
    stopped: bool = False    # the fitting was cancelled, so no fit was written
    dropped: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def summary(self):
        text = f"{self.structures} structure(s) from {self.features} feature(s)"

        if self.planes:
            text += f", {self.planes} with a plane off the table"

        if self.fits:
            text += f"; {self.fits} fit(s) off the DEM on {self.read} of them"

        for count, said in (
            (self.silent, "determining no attitude"),
            (self.too_short, "shorter than the window"),
            (self.unreached, "off the DEM"),
        ):
            if count:
                text += f", {count} {said}"

        if self.attached:
            text += f"; {self.attached} measured point(s) attached"

        if self.observations:
            text += f", {self.observations} kept as observations"

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

    # The measured points, and how far one may sit from a trace and still be a
    # measurement on it. A separate file and separate columns, because a station
    # layer is a separate survey: the two are joined here by distance, which is
    # the only thing they have in common and the reason the threshold is a field
    # rather than a constant.
    points_path: str = None
    points_layer: str = None
    points_ident_field: str = None
    points_dip_dir_field: str = None
    points_dip_field: str = None
    points_is_rhr_strike: bool = False
    attach_within: float = ATTACH_M

    # The topography, and how the window is swept along the trace. The gate is
    # not here: `TraceGate.from_traces` measures its lever floor off the layer
    # being read, which is the way round to prefer, so it is built where the
    # geometry is rather than typed in beside the file name.
    dem_path: str = None
    fit_lengths: tuple = ()
    fit_step: float = 25.0
    fit_fallback: float = 250.0
    fit_keep: str = "held"

    @property
    def has_attitudes(self):
        """Both angle columns, or neither: half an answer is not one."""

        return bool(self.dip_dir_field) and bool(self.dip_field)

    @property
    def has_points(self):
        """A point layer with both its angles, which is the only useful kind."""

        return bool(self.points_path) and bool(self.points_dip_dir_field) \
            and bool(self.points_dip_field)

    @property
    def lengths(self):
        """The sweep, defaulting to the one `traces` measured its costs on."""

        from .traces import DEFAULT_SWEEP

        return tuple(self.fit_lengths) or DEFAULT_SWEEP


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

# What the code of a measured station is called, which is not what the ident of a
# trace is called: `station_code` on `geology.gpkg`, `sigla` on a field notebook.
# Tried before `IDENT_FIELDS` and not instead of it -- a point layer with an `id`
# and nothing else still has a name to use, and `attitude_at` reports whatever
# this picks as the provenance of every attitude it finds.
STATION_FIELDS = (
    "station", "station_code", "stazione", "sigla", "codice_stazione",
    "site", "sito", "punto", "misura",
)

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


def transcript_of(path, mapping, project=None, progress=None):
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

    **Three passes, in this order, and the order is the format's.** The traces
    are written first because everything else is stated against a path: an
    anchor without one cannot be projected, and `s` is always derived by
    projection. Then the measured points, which need the paths to find the
    nearest and to resolve their own `off=`. Then the DEM, last, because it is
    the only slow phase and the only one that can be given up on -- and
    `progress` is called through it with `(done, total)`, returning False to
    stop.
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

    # Every file that was read, not only the one the traces came out of. Three
    # inputs can go into one of these, and a `source` naming the first would
    # leave the other two nowhere: the point layer a station was attached from
    # and the DEM a plane was read off are both things somebody will want to go
    # back and look at, and neither is recoverable from the rest of the file.
    read = [f"{path}{' layer=' + mapping.layer if mapping.layer else ''}"]

    if mapping.has_points:
        read.append(
            f"points={mapping.points_path}"
            f"{' layer=' + mapping.points_layer if mapping.points_layer else ''}"
        )

    if mapping.dem_path:
        read.append(f"dem={mapping.dem_path}")

    dataset.meta["source"] = " ".join(read)
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

    said = degrees_not_metres(dataset)

    # Before the two slow phases and not after, because both of them are stated
    # in metres: a threshold of 100 and a window of 250 mean nothing on a layer
    # in degrees, and finding that out at the end would mean finding it out
    # after the waiting.
    if said:
        raise ValueError(said)

    dataset.resolve()

    if mapping.has_points:
        _attach(gstruct, dataset, frame.crs, mapping, report)

    if mapping.dem_path:
        _fit_every(gstruct, dataset, frame.crs, mapping, report, progress)

    # The measured notes last, and appended rather than replaced: the two phases
    # above write what they found -- the gate's floor, a reprojection -- and a
    # plain assignment here would throw exactly those away, being the only ones
    # not derivable from the mapping.
    report.notes = _notes(mapping, report) + report.notes
    dataset.meta["note"] = "; ".join(report.notes)

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

    if report.attached or report.observations:
        notes.append(
            f"{report.attached} giaciture puntuali agganciate entro "
            f"{mapping.attach_within:.0f} m; {report.observations} tenute come "
            f"`observation` con scritto perché"
        )

    if report.stopped:
        # First among what the DEM has to say, because it is what the rest of it
        # has to be read against: a file whose fitting was given up on carries no
        # fit at all, and saying so is the difference between that and a sheet
        # the topography refused.
        notes.append(
            "fit dal DTM interrotto: nessun `fit` è stato scritto, perché "
            "scriverne una parte renderebbe indistinguibile «rifiutato» da "
            "«non raggiunto»"
        )
    elif mapping.dem_path:
        notes.append(
            f"{report.fits} `fit` con from={FROM_DEM} su {report.read} strutture, "
            f"uno per tratto in cui la finestra tiene"
        )
        # The three counts separately, and in the file rather than only in the
        # dialog: whoever reads this later has to be able to tell a sheet the
        # DEM does not cover from one the topography refused.
        notes.append(
            f"senza `fit`: {report.silent} che non determinano niente, "
            f"{report.too_short} più corte della finestra, "
            f"{report.unreached} fuori dal DTM"
        )
        notes.append(
            f"{report.swept} strutture scelgono la propria finestra, "
            f"{report.shortened} sono state lette sotto il ripiego di "
            f"{mapping.fit_fallback:.0f} m perché più corte, il resto al ripiego"
        )

    coined = []

    if report.minted:
        coined.append(f"{report.minted} su linee ({MINTED}0001...)")

    if report.minted_points:
        coined.append(f"{report.minted_points} su punti ({MINTED_POINT}0001...)")

    # Both counted, because a minted ident is a handle this made up and the
    # reader has no other way to know: on the AOI a station layer with no code
    # column gives every one of 11933 points a name nobody chose, and
    # `observation P0042` otherwise reads like a surveyor's own.
    if coined:
        notes.append("ident coniati qui: " + ", ".join(coined))

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


# -- the measured points -------------------------------------------------

# The ident an observation gets when the point layer has no code column. Kept
# apart from `MINTED` so that a name in the file says which layer it came out
# of: `L0003` is a line nobody named and `P0003` is a station nobody named, and
# with one prefix for both they would collide in a file that has both.
MINTED_POINT = "P"


def _attach(gstruct, dataset, frame_crs, mapping, report):
    """
    Measured points onto the traces they belong to; the rest kept as observations.

    **Nearest among the structures written, not among the features read.** A
    multipart feature has already become several structures by the time this
    runs, and that is the right thing for it to see: a station belongs to the
    fragment it stands on, and matching against the feature would attach it to
    whichever part of the fault happened to come first in the file.

    **The threshold decides a kind, not whether to keep it.** Past it the point
    is still written -- as an `observation`, with the nearest structure, the
    distance and the threshold that refused it -- which is rule 3 and the reason
    the number can be argued with afterwards instead of having to be guessed at.
    A point with no readable plane goes the same way for a different reason, and
    the reason is in the line.

    The layer is reprojected where it declares a different CRS, silently,
    because that is not a decision: a station has one position on the ground and
    two ways of writing it, and refusing would only mean asking somebody to run
    `ogr2ogr` to say the same thing. A layer that declares *none* is refused --
    there the coordinates are of unknown meaning and any distance computed from
    them is unknown too.
    """

    import geopandas as gpd
    from shapely import STRtree
    from shapely.geometry import LineString

    frame = (
        gpd.read_file(mapping.points_path, layer=mapping.points_layer)
        if mapping.points_layer
        else gpd.read_file(mapping.points_path)
    )

    if frame.crs is None:
        raise ValueError(
            f"{mapping.points_path} declares no projection, so how far a point "
            f"is from a trace cannot be computed. An attachment is a distance "
            f"in metres and nothing else."
        )

    if frame_crs is not None and not frame.crs.equals(frame_crs):
        # Read out before the call and not after it, which is the whole of why
        # this is two statements: `frame.crs` is the target once `to_crs` has
        # returned, and the note would have said the layer was reprojected from
        # the system it was reprojected to.
        was = frame.crs.to_string()

        frame = frame.to_crs(frame_crs)

        report.notes.append(
            f"punti riproiettati da {was} a {frame_crs.to_string()}"
        )

    usable = frame.geometry.notna() & ~frame.geometry.is_empty
    without = int((~usable).sum())

    if without:
        report.dropped["points with no geometry"] = without

    frame = frame[usable]

    if frame.empty or not dataset.structures:
        return

    azimuth = numeric(frame[mapping.points_dip_dir_field])
    dip = numeric(frame[mapping.points_dip_field])

    keep, dropped = admissible(azimuth, dip)

    for reason, count in dropped.items():
        report.dropped[f"point plane: {reason}"] = count

    azimuth = normalised_azimuth(azimuth, dip)

    paths = [LineString(structure.path) for structure in dataset.structures]
    tree = STRtree(paths)

    minted = 0

    for ndx in range(len(frame)):
        point = frame.geometry.iloc[ndx]
        anchor = (float(point.x), float(point.y))

        said = _text(frame, mapping.points_ident_field, ndx)

        if not said:
            minted += 1
            said = f"{MINTED_POINT}{minted:04d}"
            report.minted_points += 1

        found, distance = tree.query_nearest(point, return_distance=True)

        if len(found) == 0:
            continue

        near = dataset.structures[int(found[0])]
        away = float(distance[0])

        # `station=` and not a bare ident, because that is the key
        # `attitude_at` reports the provenance by: a measurement it finds within
        # `max_gap` comes back as `misurata:<station>`, and with the key missing
        # it comes back as `misurata:obs` on every one of them.
        attrs = {"station": said, "src": "points"}

        if bool(keep[ndx]):
            # Rule 1 again, and on the same value as on a line: the two numbers
            # as the table wrote them, beside the dip direction they became.
            attrs["raw"] = (
                f"{mapping.points_dip_dir_field}={azimuth[ndx]:.0f} "
                f"{mapping.points_dip_field}={dip[ndx]:.0f}"
            )

            # The same conversion the lines get, through the same two calls.
            # Rewriting it here as `+90` would be a second implementation of one
            # convention, and the two would agree until one of them was fixed.
            from geogst.core.geology.orientations import Plane

            plane = gstruct_plane(Plane(
                float(azimuth[ndx]), float(dip[ndx]),
                is_rhr_strike=mapping.points_is_rhr_strike,
            ))
        else:
            plane = None
            attrs["no_attitude"] = NO_ATTITUDE

        if plane is not None and away <= mapping.attach_within:
            attitude = gstruct.Attitude(anchor=anchor, plane=plane, attrs=attrs)
            attitude.resolve(near.path)

            # Written because it is the one number that says how much of an
            # attachment this was: `s` is derived and looks exact whatever the
            # point's distance from the trace, and a station 80 m off a fault is
            # a different claim from one standing on it.
            attitude.attrs["off"] = f"{attitude.offset:.1f}"

            near.attitudes.append(attitude)
            report.attached += 1
            continue

        if plane is not None:
            attrs["unattached"] = UNATTACHED
            attrs["threshold"] = f"{mapping.attach_within:.0f}"

        attrs["nearest"] = near.ident
        attrs["distance"] = f"{away:.1f}"

        dataset.observations.append(
            gstruct.Observation(ident=said, anchor=anchor, plane=plane, attrs=attrs)
        )
        report.observations += 1


# -- the planes off the topography ---------------------------------------


def _dem_refusal(dem, frame_crs):
    """
    Why this DEM cannot be sampled for this layer, or None.

    One CRS or nothing, and the refusal is not fussiness. The trace would be
    sampled in the DEM's grid and the fit's anchors written in the layer's, so a
    mismatch puts the plane and the place it holds over in two different
    coordinate systems -- and the dip direction that came out would be measured
    from the DEM's north, which is not the north the file declares. Transforming
    the trace on the way in would fix the sampling and not that.
    """

    code = dem.crs.to_epsg() if dem.crs is not None else None
    wanted = frame_crs.to_epsg() if frame_crs is not None else None

    if code is not None and wanted is not None and code == wanted:
        return None

    return (
        f"the DEM is in {dem.crs.to_string() if dem.crs else 'no declared CRS'} "
        f"and the layer in "
        f"{frame_crs.to_string() if frame_crs else 'no declared CRS'}. A plane "
        f"read off the topography is a dip direction measured from the DEM's "
        f"north and written against the layer's: reproject one of them first, "
        f"because nothing here can make that mean the same thing."
    )


# How near an end of the path counts as being at it: a centimetre, which is the
# precision anchors are written to anyway.
AT_THE_END = 0.01


def _anchors(gstruct, path, start, end, span):
    """
    A stretch as a pair of anchors, with `*` wherever it reaches an end of the path.

    `*` is the format's own word for an end of the path, and writing it is not
    tidiness: on a **closed** trace `path[-1]` is `path[0]`, so a fit reaching
    both ends writes one coordinate twice and reads back covering nothing at all.
    Eight traces of `elementi_tettonici` are closed rings, and on every one a fit
    that had held over eighteen windows came back with an extent of zero -- a
    plane in the file, covering no part of the trace it had been read off.

    Both ends in one call, because on a ring the coordinate cannot say which end
    it is and only the caller knows.
    """

    return (
        None if start <= AT_THE_END else gstruct.point_at(path, start),
        None if end >= span - AT_THE_END else gstruct.point_at(path, end),
    )


def _reach(runs, index, length, ends):
    """
    How far along the trace a held run's plane is written as holding.

    **Half a window beyond the centres that held, but never into ground another
    verdict already claims.** Both halves of that are load-bearing, and the
    first was missing: `TraceSpans.runs` reports the interval the window
    *centres* covered, which for a run of one position is a single step. On 1200
    traces of `elementi_tettonici` that made every one of 244 fits claim 25 m of
    trace while having been read over 150 to 600 m of it -- so `attitude_at`
    answered `assente` over almost all of a trace whose plane the file had just
    determined. The centres are where the verdict was taken; the window is where
    the evidence was.

    Widening stops at a neighbouring run because that run is a verdict, and on
    these traces it is usually `line`: the gate, asked about that stretch, said
    the plane was not determined there, and answering with the neighbour's plane
    would be overruling the gate using the gate's own data. The ends of a part
    are different in kind -- no window could be centred there, so nothing was
    ever asked -- and that is the ground this may take.

    `traces` deliberately does not widen, and is right not to: there the runs
    are summed into metres per verdict, and widened ones would overlap and total
    past the trace. Nothing sums these.
    """

    _, s0, s1 = runs[index]

    before = runs[index - 1][2] if index else ends[0]
    after = runs[index + 1][1] if index + 1 < len(runs) else ends[1]

    return max(s0 - length / 2.0, before), min(s1 + length / 2.0, after)


def _fit_along(gstruct, structure, dem, mapping, gate, report):
    """
    Every stretch of one path whose plane the topography determines, as `fit`s.

    One path at a time and never the parts of a multipart together, which the
    split has already seen to: `trace_points` runs its progressive *on* across
    parts, so a fit computed over two fragments would be anchored by a
    progressive measured on their concatenation and land somewhere else when the
    file was read back.

    **A run that is only a line becomes nothing.** That is the whole of why this
    goes through the gate rather than writing the plane at every window: a plane
    through a straight trace is arbitrary and not merely imprecise, and a file
    saying `fit plane 90/47` where the trace never turned would be a number
    nobody could tell from a measurement.

    The window length is swept per trace and not chosen once, for the reason
    `traces.fit_records` gives -- where lengthening stops helping is a property
    of the trace. Most traces have no such turn and take the fallback, and the
    report counts which is which rather than quoting the tally that looks better.

    **And where the fallback does not fit, the longest window that does.** A part
    shorter than the window is not fitted at all -- `trace_spans` is deliberate
    about that, since reading one fragment at a different scale from the rest
    makes the verdicts along a trace incomparable -- but the *fallback* is a
    default and not a scale somebody chose. On `elementi_tettonici` the median
    trace is 172 m, so keeping to 250 m would refuse two thirds of the sheet on
    the strength of a number nobody typed. Which window was used is in the file
    as `window=`, so the scale a plane was read at is never in doubt.
    """

    from .traces import holding_length, trace_points, trace_spans, window_sweep

    sampled = min(10.0, mapping.fit_step)

    parts = trace_points([structure.path], dem, step=sampled)

    if not parts:
        report.unreached += 1
        return []

    sweep = window_sweep(
        parts, mapping.lengths, step=mapping.fit_step, gate=gate
    )
    length = holding_length(sweep)

    if length is not None:
        report.swept += 1
        spans = sweep[length]
    else:
        length = mapping.fit_fallback
        spans = (
            sweep[length] if length in sweep
            else trace_spans(parts, length, step=mapping.fit_step, gate=gate)
        )

        if len(spans) == 0:
            shorter = [
                other for other in sorted(sweep, reverse=True)
                if other < length and len(sweep[other])
            ]

            if not shorter:
                report.too_short += 1
                return []

            length = shorter[0]
            spans = sweep[length]
            report.shortened += 1

    out = []

    runs = spans.runs()

    # The ends of a part carry no verdict at all: a window has to fit, so the
    # first centre sits half a window in and the ground outside that was never
    # classified. Which is what `_reach` is allowed to claim and a neighbouring
    # run is not.
    progressive = parts[0][1]
    ends = (float(progressive[0]), float(progressive[-1]))
    span = gstruct.path_length(structure.path)

    for index, (verdict, s0, s1) in enumerate(runs):
        if verdict != mapping.fit_keep:
            continue

        # On the unwidened bounds, always: these are the centres whose windows
        # held, and averaging over the reach instead would pull in the windows
        # that failed.
        attitude = spans.mean_attitude(s0, s1)

        if attitude is None:
            continue

        covered = int(((spans.progressive >= s0) & (spans.progressive <= s1)).sum())

        start, end = _reach(runs, index, length, ends)

        # Anchors and not the progressives, which is the format's rule and not a
        # preference: a reader projects them onto whatever path it has, so the
        # same stretch survives the trace being redigitised, where a stored `s`
        # would migrate.
        start, end = _anchors(gstruct, structure.path, start, end, span)

        out.append(gstruct.Fit(
            plane=gstruct.Plane(attitude[0], attitude[1]),
            start=start,
            end=end,
            attrs={
                "from": FROM_DEM,
                "src": "gsurf",
                # The two FORMAT.md names for this producer's diagnostic, and
                # no third: `nvert` counts digitised vertices and this counts
                # DEM samples, so it is not a number this has.
                "window": f"{length:.0f}",
                "span_verdict": verdict,
                "windows": str(covered),
                "step": f"{spans.step:.0f}",
                "sampled": f"{sampled:.0f}m",
                "dem": dem.path.name,
            },
        ))

    return out


def _fit_every(gstruct, dataset, frame_crs, mapping, report, progress=None):
    """
    The DEM read along every path, with the fits attached only if it finished.

    **Nothing is attached until the whole sweep has run**, and that is the
    answer to a cancel rather than a convenience. Fits written on the first
    three thousand structures of a sheet and not the rest would leave the file
    unable to say which is which: a structure with no fit would mean either that
    the topography refused it or that nobody got there, and those are opposite
    facts. So a run given up on writes no fit at all and says so in the header,
    which is recoverable -- the same import runs again.

    **The lever floor is measured off the layer**, which is the way round
    `TraceGate.from_traces` exists for: a departure from straightness smaller
    than the wander of the pen that drew the line is not evidence of a turn, and
    how much the pen wandered is a property of the sheet rather than a constant.
    Where the measurement does not come back -- a self-affine layer, which is
    what a mapped contact usually is -- the class default stands and the note
    says so, rather than a number being extrapolated and made to look measured.
    """

    from .dem import Dem
    from .traces import TraceGate, digitising_jitter

    dem = Dem(mapping.dem_path)

    try:
        said = _dem_refusal(dem, frame_crs)

        if said:
            raise ValueError(said)

        paths = [structure.path for structure in dataset.structures]

        jitter = digitising_jitter(paths)
        gate = TraceGate.from_traces(paths)

        report.notes.append(
            f"gate: lever {gate.min_lever:.1f} m "
            + (
                f"(3x il jitter misurato, {jitter['sigma']:.1f} m)"
                if jitter is not None and jitter["sigma"] is not None
                else "(default: il jitter non si misura su questo layer)"
            )
        )

        found, total = {}, len(dataset.structures)

        for index, structure in enumerate(dataset.structures):
            if progress is not None and progress(index, total) is False:
                report.stopped = True
                break

            fits = _fit_along(gstruct, structure, dem, mapping, gate, report)

            if fits:
                found[index] = fits
            else:
                report.silent += 1

        if report.stopped:
            report.swept = report.shortened = 0
            report.silent = report.too_short = report.unreached = 0
            return

        for index, fits in found.items():
            structure = dataset.structures[index]

            # First, ahead of a plane read off the attribute table. `attitude_at`
            # returns the *first* fit that covers a progressive where `span_at`
            # returns the *last* span, so the more specific statement goes first
            # among fits and last among spans -- and a stretch read off the
            # ground is more specific than a column speaking for the whole trace.
            structure.fits[:0] = fits

            report.fits += len(fits)
            report.read += 1
    finally:
        dem.close()


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
        self.keep_list.setMaximumHeight(90)

        # -- the measured points, and the DEM ------------------------------

        self._points = None

        self.points_label = QtWidgets.QLabel("no point layer")
        self.points_label.setStyleSheet("color: gray; font-size: 10px;")

        points_browse = QtWidgets.QPushButton("Browse...")
        points_browse.clicked.connect(self._browse_points)

        points_clear = QtWidgets.QPushButton("Clear")
        points_clear.clicked.connect(self._clear_points)

        self.points_layer_combo = QtWidgets.QComboBox()
        self.points_layer_combo.setEnabled(False)
        self.points_layer_combo.currentTextChanged.connect(self._on_points_layer)

        self.station_combo = QtWidgets.QComboBox()
        self.points_dip_dir_combo = QtWidgets.QComboBox()
        self.points_dip_combo = QtWidgets.QComboBox()

        self.points_convention_combo = QtWidgets.QComboBox()
        self.points_convention_combo.addItems([label for label, _ in CONVENTIONS])

        self.attach_spin = QtWidgets.QDoubleSpinBox()
        self.attach_spin.setRange(1.0, 10000.0)
        self.attach_spin.setValue(ATTACH_M)
        self.attach_spin.setSuffix(" m")
        self.attach_spin.setDecimals(0)

        self._dem = None

        self.dem_label = QtWidgets.QLabel("no DEM")
        self.dem_label.setStyleSheet("color: gray; font-size: 10px;")

        dem_browse = QtWidgets.QPushButton("Browse...")
        dem_browse.clicked.connect(self._browse_dem)

        dem_clear = QtWidgets.QPushButton("Clear")
        dem_clear.clicked.connect(self._clear_dem)

        self.step_spin = QtWidgets.QDoubleSpinBox()
        self.step_spin.setRange(5.0, 500.0)
        self.step_spin.setValue(25.0)
        self.step_spin.setSuffix(" m step")
        self.step_spin.setDecimals(0)

        self.fallback_spin = QtWidgets.QDoubleSpinBox()
        self.fallback_spin.setRange(50.0, 5000.0)
        self.fallback_spin.setValue(250.0)
        self.fallback_spin.setSuffix(" m window")
        self.fallback_spin.setDecimals(0)

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

        points = QtWidgets.QGroupBox("Measured points, attached to the traces")
        at = QtWidgets.QGridLayout(points)
        at.addWidget(points_browse, 0, 0)
        at.addWidget(self.points_layer_combo, 0, 1, 1, 2)
        at.addWidget(points_clear, 0, 3)
        at.addWidget(self.points_label, 1, 0, 1, 4)
        at.addWidget(QtWidgets.QLabel("station"), 2, 0)
        at.addWidget(self.station_combo, 2, 1)
        at.addWidget(QtWidgets.QLabel("within"), 2, 2)
        at.addWidget(self.attach_spin, 2, 3)
        at.addWidget(QtWidgets.QLabel("azimuth"), 3, 0)
        at.addWidget(self.points_dip_dir_combo, 3, 1)
        at.addWidget(QtWidgets.QLabel("dip"), 3, 2)
        at.addWidget(self.points_dip_combo, 3, 3)
        at.addWidget(QtWidgets.QLabel("read as"), 4, 0)
        at.addWidget(self.points_convention_combo, 4, 1, 1, 3)
        at.setColumnStretch(1, 1)
        at.setColumnStretch(3, 1)

        dem = QtWidgets.QGroupBox("Planes off a DEM, one per stretch that holds")
        off = QtWidgets.QGridLayout(dem)
        off.addWidget(dem_browse, 0, 0)
        off.addWidget(self.step_spin, 0, 1)
        off.addWidget(self.fallback_spin, 0, 2)
        off.addWidget(dem_clear, 0, 3)
        off.addWidget(self.dem_label, 1, 0, 1, 4)
        off.setColumnStretch(1, 1)
        off.setColumnStretch(2, 1)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.report_label, 1)
        buttons.addWidget(self.write_button)
        buttons.addWidget(close)

        # Scrolled rather than tall. Five group boxes is more than a laptop's
        # working height, and a dialog whose Write button is off the bottom of
        # the screen is a dialog with no way to finish.
        inner = QtWidgets.QWidget()
        stacked = QtWidgets.QVBoxLayout(inner)
        stacked.setContentsMargins(0, 0, 0, 0)
        stacked.addWidget(source)
        stacked.addWidget(names)
        stacked.addWidget(keep)
        stacked.addWidget(points)
        stacked.addWidget(dem)
        stacked.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(scroll, 1)
        layout.addLayout(buttons)

        self.setMinimumWidth(600)
        self.resize(600, 700)

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

        # The DEM changes the order of magnitude of the wait, not the size of the
        # file, so it is said here rather than left to be discovered: the rate is
        # the one measured on the AOI sheets, and the number is deliberately a
        # rough one because its job is to stop a click, not to be right.
        #
        # And only where there is a wait to warn about. On a handful of traces
        # the estimate rounds to nothing, and "about 0 s" is a sentence that
        # teaches the reader to stop reading this line.
        if self._dem and count * DEM_SECONDS_EACH >= 5.0:
            said += f"; with a DEM, about {count * DEM_SECONDS_EACH / 60.0:.0f} min to fit" \
                if count * DEM_SECONDS_EACH >= 120.0 \
                else f"; with a DEM, about {count * DEM_SECONDS_EACH:.0f} s to fit"

        self.report_label.setText(said)

    # -- the measured points, and the DEM ----------------------------------

    def _browse_points(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose the layer of measured points", "",
            "Vector (*.gpkg *.shp *.geojson *.json *.gml *.kml *.sqlite *.fgb);;"
            "All files (*)",
        )

        if path:
            self.set_points(path)

    def set_points(self, path, layer=None):
        """Lists the point layers in the file, and guesses their angle columns."""

        try:
            layers = VectorSource.candidate_layers(path, "points")
        except Exception as err:
            QtWidgets.QMessageBox.warning(
                self, "Unreadable", f"{path}\n\n{type(err).__name__}: {err}"
            )
            return False

        if not layers:
            QtWidgets.QMessageBox.information(
                self, "No points",
                f"{path}\n\nno layer of points in this file. What attaches to a "
                f"trace is a measurement at a place: a line has no one place to "
                f"be, which is what makes it a `fit` instead.",
            )
            return False

        self._points = path
        self.points_label.setText(str(path))

        with QtCore.QSignalBlocker(self.points_layer_combo):
            self.points_layer_combo.clear()
            self.points_layer_combo.addItems(layers)
            self.points_layer_combo.setCurrentText(
                layer if layer in layers else layers[0]
            )

        self.points_layer_combo.setEnabled(True)

        self._on_points_layer(self.points_layer_combo.currentText())

        return True

    def _on_points_layer(self, layer):
        if not self._points or not layer:
            return

        every, numbers = fields_of(self._points, layer)
        proposed = guess(self._points, layer)

        station = first_match(STATION_FIELDS, every) or proposed.ident_field

        for combo, fields, chosen in (
            (self.station_combo, every, station),
            (self.points_dip_dir_combo, numbers, proposed.dip_dir_field),
            (self.points_dip_combo, numbers, proposed.dip_field),
        ):
            with QtCore.QSignalBlocker(combo):
                combo.clear()
                combo.addItem(self.NO_FIELD)
                combo.addItems(fields)
                combo.setCurrentText(chosen or self.NO_FIELD)

        if (self.points_dip_dir_combo.currentText() or "").lower().startswith("strike"):
            self.points_convention_combo.setCurrentIndex(1)

    def _clear_points(self):
        """Backing out of the points, which has to be possible in one click."""

        self._points = None
        self.points_label.setText("no point layer")

        with QtCore.QSignalBlocker(self.points_layer_combo):
            self.points_layer_combo.clear()

        self.points_layer_combo.setEnabled(False)

        for combo in (self.station_combo, self.points_dip_dir_combo,
                      self.points_dip_combo):
            with QtCore.QSignalBlocker(combo):
                combo.clear()

    def _browse_dem(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose the DEM", "",
            "Raster (*.tif *.tiff *.vrt *.asc *.img *.dem);;All files (*)",
        )

        if path:
            self._dem = path
            self.dem_label.setText(str(path))
            self._describe(self.layer_combo.currentText())

    def _clear_dem(self):
        self._dem = None
        self.dem_label.setText("no DEM")
        self._describe(self.layer_combo.currentText())

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
            points_path=self._points,
            points_layer=(self.points_layer_combo.currentText() or None)
            if self._points else None,
            points_ident_field=self._named(self.station_combo),
            points_dip_dir_field=self._named(self.points_dip_dir_combo),
            points_dip_field=self._named(self.points_dip_combo),
            points_is_rhr_strike=CONVENTIONS[
                self.points_convention_combo.currentIndex()
            ][1],
            attach_within=float(self.attach_spin.value()),
            dem_path=self._dem,
            fit_step=float(self.step_spin.value()),
            fit_fallback=float(self.fallback_spin.value()),
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

        if mapping.points_path and not mapping.has_points:
            return (
                "A point layer is chosen and its two angle columns are not both "
                "named. A point with no plane on it is not a measurement to "
                "attach -- it would go into the file as an `observation` saying "
                "the source had no attitude, which is true of every point in the "
                "layer and is not worth a file. Name both, or clear the layer."
            )

        if mapping.fit_step > mapping.fit_fallback / 2.0:
            return (
                f"A step of {mapping.fit_step:.0f} m with a window of "
                f"{mapping.fit_fallback:.0f} m leaves neighbouring windows "
                f"barely overlapping, so a stretch that holds is one window wide "
                f"and its extent is the step rather than the evidence. Keep the "
                f"step at a fraction of the window."
            )

        return None

    # -- writing -----------------------------------------------------------

    def _bar(self, mapping):
        """
        The progress dialog and the callback, as a pair, or `(None, None)`.

        Only where a DEM was chosen, because that is the only phase that takes
        long enough to be worth a window: transcribing 24717 lines is seconds and
        reading a plane off each of them is minutes. A bar over the fast phase
        would be a bar that flashes.

        `Stop` here does not stop the import -- it stops the fitting, and then no
        `fit` is written at all. Which is why the label says so: a run given up
        on part way through is the one outcome the file could not describe.
        """

        if not mapping.dem_path:
            return None, None

        dialog = QtWidgets.QProgressDialog(
            "Reading a plane off the DEM along each trace...\n"
            "Stopping leaves the traces and writes no fit.",
            "Stop", 0, 100, self,
        )
        dialog.setWindowTitle("gSurf - fitting from the DEM")
        dialog.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        dialog.setMinimumDuration(0)
        dialog.setValue(0)

        # The same stride as the sections tool, for the same reason: pumping the
        # event loop once per structure would be 24717 repaints to move a bar by
        # a hundredth of its width.
        state = {"stride": 1}

        def tick(done, total):
            if total and dialog.maximum() != total:
                dialog.setMaximum(total)
                state["stride"] = max(1, total // 200)

            if done % state["stride"] == 0:
                dialog.setValue(done)
                QtWidgets.QApplication.processEvents()

            return not dialog.wasCanceled()

        return dialog, tick

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

        progress, tick = self._bar(mapping)

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)

        try:
            text, report = transcript_of(self._path, mapping, progress=tick)
        except Exception as err:
            QtWidgets.QMessageBox.critical(
                self, "Not imported", f"{type(err).__name__}: {err}"
            )
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

            if progress is not None:
                progress.close()

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
