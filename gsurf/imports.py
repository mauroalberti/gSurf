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

**A station is a place and a measure is a row, and a survey keeps them apart.**
A point layer carrying one dip direction and one dip can hold one measurement
per place, which is not what a fault station is: the surface may be curved and
sampled three times, and a single surface may carry two striations. So the
measures live in a table, keyed to the station -- `fault_attitudes` in
`geology.gpkg`, 24 rows against 27 points -- and the join is by that key rather
than by distance. `export_geology.py` already did this join, once, hardcoded on
one survey's column names; here it is the same `merge` with the names asked for
instead of assumed, which is the whole difference between a script and an
importer.

**All of the N travel, and the key says which one was used.** A station with
three measured surfaces gives three attitudes at one anchor, hence three at one
`s`, and `attitude_at` takes `min` by distance along the trace -- so a tie is
broken by file order and the file cannot say it was a tie. Keeping one and
dropping two would have been that silence made permanent; writing three where
each is called `S4/p2` after the surface it was measured on makes the drawn plane
name itself, and `siblings=` on each one says how many others stood at the same
spot. What this does *not* do is rank them: the table's row order is the order
they are written in, because a rule for which of three measured surfaces is the
real one is a decision for whoever measured them.

**And a measure whose station does not exist is still a measure.** The join is
outer for exactly that reason: `how="left"` -- which is what the one existing
join does -- turns a typo in a station code into a row that was never read, and
rule 3 does not have an exception for the source's own bookkeeping. It comes out
as an `observation` with no anchor, which the format writes as `*` and reads
back as nothing, carrying the key that matched nothing.

**The projection is asked for, because here it is not only a ruler.** The point
layer is reprojected silently -- a station has one position and two ways of
writing it -- but the trace layer's CRS *becomes the file's*, and that decides
what every anchor in it means, what the `crs` line declares, and whether the
file can be measured along at all: four of the five layers of `geology.gpkg`
that carry geometry are in degrees, where an anchor written to two decimals is
a kilometre wide and every threshold in this module is metres. So the target
is a field with the layer's own projection proposed in it, or a UTM zone worked
out from the layer's own extent where the layer is in degrees and has none to
propose.
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

# What separates a station's code from the measure taken there, where there is
# more than one. Not `PART`, because the two say different things and the file
# is read by people: `F003.2` is the second piece of one trace, `S4/2` is the
# second surface measured at one station, and the first is an accident of
# digitising where the second is what the surveyor found.
MEASURE = "/"

# What a station's column is renamed to where the measures table has one by the
# same name. The table wins the bare name because that is where the angles are
# read from, and the station's is kept rather than dropped -- `comments` on a
# station and `comments` on a measure are both worth carrying, and a merge that
# quietly kept one of them would be indistinguishable from a survey that only
# wrote one.
STATION_SUFFIX = ".station"

# Why a measure is in the file with no position. Rule 3 again, on the one case
# the join can produce that distance cannot: a key in the measures table that
# matches no station. The row is real -- somebody wrote it down -- and what is
# missing is the station, so that is what the reason says.
ORPHAN = "stazione-assente"

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
    # What the join found, which is four different facts. A station is a place
    # and a measure is a row, and reporting only the rows would hide both ends
    # of the interesting case: a station whose surfaces were measured three
    # times, and a measure whose station is not in the layer.
    sites: int = 0           # stations the join read
    measures: int = 0        # rows of the measures table that found a station
    several: int = 0         # stations carrying more than one plane at one anchor
    orphans: int = 0         # measures whose key matched no station
    lineations: int = 0      # striations written as `lineation`
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

        if self.measures:
            text += (
                f"; {self.measures} measure(s) joined to {self.sites} station(s)"
            )

        if self.attached:
            text += f"; {self.attached} measured point(s) attached"

        if self.several:
            text += f", {self.several} station(s) with more than one plane"

        if self.lineations:
            text += f", {self.lineations} lineation(s)"

        if self.observations:
            text += f", {self.observations} kept as observations"

        if self.orphans:
            text += f" ({self.orphans} with no station)"

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

    # The measures, where they are not on the points. A table keyed to the
    # station rather than a layer with a position, which is the shape a survey
    # takes as soon as one station carries two measurements: the angle columns
    # above are then read off *this* table, because that is where they are.
    measures_path: str = None
    measures_layer: str = None
    measures_join_field: str = None      # the key on the measures
    points_join_field: str = None        # the key on the stations
    # Which surface the measure was taken on, and the only column here that
    # cannot be guessed from a name: it is what tells two planes measured at one
    # station apart, and with no such column they are told apart by their row
    # order and nothing else.
    surface_field: str = None
    # The striation, trend and plunge. `lineation` is the format's own record
    # for it, and until now nothing wrote one: a fault surface with a slip
    # direction on it is two facts, and only the first had a way in.
    trend_field: str = None
    plunge_field: str = None
    # The columns of the station and of the measure carried through as `raw.*`,
    # the same question `keep_fields` asks of the line layer. Asked separately
    # because they are a different table: `operator` and `survey_date` are on
    # the station, a note on the striation is on the measure, and a rake this
    # does not interpret is preserved here or nowhere.
    points_keep_fields: tuple = ()

    # The projection to write the file in, as an EPSG string, or None for the
    # layer's own. Not a preference: it is what every anchor in the file is
    # written in and what the `crs` line declares, so a layer in degrees becomes
    # a file that can be measured along only by answering this.
    target_crs: str = None

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
    def has_measures(self):
        """A table to join, and both halves of the key it joins on."""

        return bool(self.points_path) and bool(self.measures_path) \
            and bool(self.measures_join_field) and bool(self.points_join_field)

    @property
    def has_lineations(self):
        """Both halves of a striation: a trend with no plunge is not a line."""

        return bool(self.trend_field) and bool(self.plunge_field)

    @property
    def measures_source(self):
        """The table as `(path, layer)`, which is where the angles are read."""

        if not self.has_measures:
            return self.points_path, self.points_layer

        return self.measures_path, self.measures_layer

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

# What the column pointing back at a station is called on the measures table.
# The station's own names first, because a table that names its key after what it
# points at is the ordinary case -- `station_code` on both sides of
# `geology.gpkg` -- and then the plain words for a code. Deliberately without
# `fid` and `objectid`, which `IDENT_FIELDS` ends with and which are row numbers
# the file format assigned: a join on one of those matches nothing, and a guess
# that has to be undone before anything works is worse than no guess.
JOIN_FIELDS = STATION_FIELDS + ("codice", "code", "ident", "id_stazione")

# The same two lists the sources dialog guesses from, and deliberately the same:
# it is the same question asked of the same attribute tables.
DIP_DIR_FIELDS = (
    "immersione", "dipdir", "dip_dir", "dipdirection", "dip_direction",
    "azimuth", "azimut", "dir", "strike",
)

DIP_FIELDS = ("inclinazione", "dip", "dipangle", "dip_angle", "angolo", "incl")

# The striation, as a trend and a plunge -- a bearing and a dip, measured in the
# world. Neither `rake` nor `pitch` is guessed at, and they are the two names a
# survey is most likely to have used: both are angles measured *in the plane of
# the fault*, so either one becomes a trend only through the plane, which is a
# computation and not a reading. They travel as `raw.*` and stay numbers somebody
# can convert on purpose. The Monte Alpi notebook has "pitch 1 30 gradi, p. 2 80
# gradi" in a single `comments` cell, which is two striations on one surface
# written where no column could reach them.
TREND_FIELDS = ("trend", "az_stria", "stria_trend", "lineation_trend", "trend_stria")

PLUNGE_FIELDS = ("plunge", "incl_stria", "stria_plunge", "plunge_stria")

# Which surface at the station, for telling two measurements at one place apart.
# No survey convention to guess from, so these are the plain words: what matters
# is that the column exists at all, and only the surveyor knows its name.
SURFACE_FIELDS = (
    "surface", "superficie", "piano", "plane", "plane_id", "surface_id",
    "faccia", "measure_id", "misura_n",
)


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


def crs_refusal(said):
    """
    Why this cannot be the projection to write the file in, or None.

    Three ways of being wrong and one of them is not an error anywhere else: a
    geographic CRS is a perfectly good projection to hold a layer in and a
    hopeless one to measure a trace along, and `degrees_not_metres` says so --
    but it says so from inside `transcript_of`, after the traces have been read
    and written. Said here it is said where it can still be changed, and to
    somebody who is looking at the field they typed it into.
    """

    said = (said or "").strip()

    if not said:
        return None

    from pyproj import CRS

    try:
        crs = CRS.from_user_input(said)
    except Exception as err:
        return (
            f"`{said}` is not a projection anything here can read: "
            f"{type(err).__name__}: {err}. An EPSG code -- `EPSG:25833`, or "
            f"just `25833` -- is what this asks for."
        )

    if crs.to_epsg() is None:
        return (
            f"`{said}` has no EPSG code ({crs.name}), and the `crs` line of the "
            f"file holds one word: written as WKT it would be read back as the "
            f"first word of its own."
        )

    if crs.is_geographic:
        return (
            f"`{said}` is geographic ({crs.name}), so its coordinates are "
            f"degrees. Everything measured along a trace here is metres -- the "
            f"attachment threshold, the fitting window, `attitude_at`'s reach -- "
            f"and an anchor is written to two decimals, which is a centimetre in "
            f"a projected CRS and about a kilometre in this one."
        )

    return None


def projected_for(path, layer=None):
    """
    The projection to propose writing in, as `(epsg, why)`.

    The layer's own wherever the layer has one worth keeping, and that is the
    answer that changes nothing: proposing something else for a layer already in
    metres would move every coordinate in the file to no purpose.

    Where the layer is in degrees there is nothing of its own to propose, so the
    UTM zone is worked out from its own extent -- which is a fact about the data
    and not a preference, and stays visible in the dialog to be overridden. The
    zone arithmetic rather than `query_utm_crs_info`, which returns nothing for
    this area: its datum is the WGS 84 *ensemble*, a name the EPSG database has
    no UTM zone registered against.

    ETRS89 in, ETRS89 out, where the zone has one: a survey in EPSG:4258 taken
    to WGS 84 / UTM would carry a datum shift of about half a metre that nobody
    asked for, and this area's own files are written in EPSG:25833.
    """

    import pyogrio
    from pyproj import CRS

    info = pyogrio.read_info(path, layer=layer) if layer else pyogrio.read_info(path)

    declared = info.get("crs")

    if not declared:
        return None, "the layer declares no projection, so there is none to convert from"

    crs = CRS.from_user_input(declared)

    if not crs.is_geographic:
        code = crs.to_epsg()

        return (f"EPSG:{code}" if code else None), (
            None if code
            else f"the layer's projection ({crs.name}) has no EPSG code to write"
        )

    bounds = info.get("total_bounds")

    if bounds is None or any(value != value for value in bounds):
        return None, f"the layer is in {crs.name} and has no extent to place a zone by"

    lon = (float(bounds[0]) + float(bounds[2])) / 2.0
    lat = (float(bounds[1]) + float(bounds[3])) / 2.0

    zone = int((lon + 180.0) // 6.0) + 1

    # 28N to 38N is the whole of ETRS89 / UTM's own range, and outside it the
    # code would be an EPSG number that means something else entirely.
    if "ETRS89" in crs.name.upper() and 28 <= zone <= 38 and lat >= 0.0:
        code = 25800 + zone
    else:
        code = (32600 if lat >= 0.0 else 32700) + zone

    return f"EPSG:{code}", (
        f"the layer is in degrees ({crs.name}); UTM zone {zone}, worked out "
        f"from its own extent around lon {lon:.2f}"
    )


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

    report = Imported()

    # Before `crs_of`, because this is what `crs_of` then reads. The traces are
    # reprojected and everything else follows them: the point layer is brought to
    # `frame.crs` a few lines down and the DEM is checked against it, so
    # answering this once puts the whole file in one projection.
    if mapping.target_crs and frame.crs is not None:
        said = crs_refusal(mapping.target_crs)

        if said:
            raise ValueError(said)

        from pyproj import CRS

        target = CRS.from_user_input(mapping.target_crs)

        if not frame.crs.equals(target):
            was = frame.crs.to_string()

            frame = frame.to_crs(target)

            report.notes.append(
                f"tracce riproiettate da {was} a {mapping.target_crs}: è la "
                f"proiezione in cui il file è scritto, e in cui valgono tutte "
                f"le sue ancore"
            )

    crs, refusal = crs_of(frame)

    if refusal is not None:
        raise ValueError(refusal)

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

    # The table too, and with the key it was joined on: a station layer and the
    # measures keyed to it are two files, and which column married them is the
    # one thing about the join that cannot be worked out again from the result.
    if mapping.has_measures:
        read.append(
            f"measures={mapping.measures_path}"
            f"{' layer=' + mapping.measures_layer if mapping.measures_layer else ''}"
            f" on={mapping.points_join_field}={mapping.measures_join_field}"
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

    if mapping.has_measures:
        notes.append(
            f"misure da tabella unite alle stazioni su "
            f"{mapping.points_join_field}={mapping.measures_join_field}: "
            f"{report.measures} righe su {report.sites} stazioni"
        )

    if report.several:
        # The one thing a reader of this file cannot work out for themselves, and
        # the reason all N are here: `attitude_at` answers with one plane, takes
        # the nearest by `s`, and at one anchor they are all equally near -- so
        # the answer is the first written and the function has no way to say it
        # was a choice. The key says which, `siblings=` says how many.
        notes.append(
            f"{report.several} stazioni portano più di un piano sulla stessa "
            f"ancora: le giaciture escono tutte, distinte per "
            f"`station=<codice>{MEASURE}<superficie>` e contate da `siblings=`. "
            f"A parità di `s` `attitude_at` risponde con la prima scritta, che "
            f"è l'ordine della tabella e non una graduatoria"
        )

    if report.lineations:
        notes.append(
            f"{report.lineations} `lineation` con trend/plunge, ancorate come le "
            f"giaciture; la stria e il piano di una stessa misura condividono "
            f"`station=`, che è l'unico legame che il formato ha fra i due"
        )

    if report.orphans:
        notes.append(
            f"{report.orphans} misure senza stazione ({ORPHAN}): scritte come "
            f"`observation` con ancora `*`, con la chiave che non ha trovato "
            f"riscontro"
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

    if not name:
        return ""

    if name not in frame.columns:
        # After a join the station's own column may have been moved out of the
        # way of the measures table's -- `comments` on a site beside `comments`
        # on a surface -- and the mapping was answered against the layer, where
        # it is still called what the layer calls it. Looked for under both
        # names rather than read as a column that is not there.
        name = f"{name}{STATION_SUFFIX}"

        if name not in frame.columns:
            return ""

    value = frame[name].iloc[ndx]

    if value is None or value != value:
        return ""

    said = str(value).strip()

    return "" if said.lower() in ("", "nan", "none", "<null>") else said


def _station_text(frame, name, ndx):
    """
    One cell of the *station's* own side of a join, which `_text` cannot find.

    The two sides can hold a column of the same name, and the measures table wins
    the bare one -- so `station_code` after a join is the code the measure points
    at, which is blank on exactly the stations that had no measure. Read through
    `_text` a station with nothing measured at it would come out nameless, minted
    a handle, and be indistinguishable in the file from a survey with no codes at
    all: the name that has to be looked for first is the suffixed one.
    """

    if name and f"{name}{STATION_SUFFIX}" in frame.columns:
        name = f"{name}{STATION_SUFFIX}"

    return _text(frame, name, ndx)


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

# And the one a measure with no station gets, which is a third kind of nameless:
# a row of the measures table whose key is blank, so there is not even a code
# that failed to match. Kept apart from both the others for the same reason they
# are kept apart from each other -- the name says which table it came out of.
MINTED_MEASURE = "M"


def _stations(mapping, frame_crs, report):
    """
    The station layer, in the traces' projection.

    The layer is reprojected where it declares a different CRS, silently,
    because that is not a decision: a station has one position on the ground and
    two ways of writing it, and refusing would only mean asking somebody to run
    `ogr2ogr` to say the same thing. A layer that declares *none* is refused --
    there the coordinates are of unknown meaning and any distance computed from
    them is unknown too.
    """

    import geopandas as gpd

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

    return frame


def _joined(stations, mapping, report):
    """
    The stations with their measures, as `(rows, orphans)`.

    One row per measure, each carrying its station's geometry, which is the
    shape the rest of this already reads: everything below the join works a row
    at a time and does not care that four rows share a position.

    **Outer, and that is the whole point of doing it here.** A left join -- what
    `export_geology.py` does -- answers "what was measured at each station" and
    silently loses the other question: a measure whose station code matches
    nothing is a row somebody wrote down, and rule 3 has no exception for the
    source's own bookkeeping. They come back separately because they have no
    position and so cannot be attached to anything, not even wrongly.

    **The table wins the bare column name.** `comments` exists on both sides of
    `geology.gpkg` -- a note about the site and a note about the surface -- and
    pandas would suffix both into `comments_x` and `comments_y`, names that say
    nothing about which is which and that no mapping could have been written
    against beforehand. So the station's is suffixed with `STATION_SUFFIX` and
    the measure keeps its own name, which is also where the angle columns are.
    """

    import uuid

    import pyogrio

    path, layer = mapping.measures_path, mapping.measures_layer

    table = (
        pyogrio.read_dataframe(path, layer=layer, read_geometry=False)
        if layer
        else pyogrio.read_dataframe(path, read_geometry=False)
    )

    left, right = mapping.points_join_field, mapping.measures_join_field

    for frame, name, whose in ((stations, left, "point layer"),
                               (table, right, "measures table")):
        if name not in frame.columns:
            raise ValueError(
                f"`{name}` is not a column of the {whose}, so the two cannot be "
                f"joined on it. A join is one column on each side and nothing "
                f"else."
            )

    # A key nothing can match, for the rows whose key is blank. pandas matches
    # null against null in a merge, so a station with no code would collect
    # every measure that also has none -- and one of `geology.gpkg`'s 27
    # stations has no code. A key that was never written down joins to nothing,
    # and a value from `uuid` is the one way of saying that which cannot
    # accidentally be a station somebody named.
    blank = uuid.uuid4().hex

    stations = stations.copy()
    table = table.copy()

    # Both sides as strings, and stripped. A station code is a label whatever
    # the column's declared type is, and the two sides of one survey are
    # routinely not the same type: a table written from a spreadsheet comes back
    # with integer codes that would match the layer's text ones nowhere at all,
    # which looks exactly like a survey where nothing was measured.
    stations["_key"] = [
        _key(value) or f"{blank}.station.{ndx}"
        for ndx, value in enumerate(stations[left])
    ]
    table["_key"] = [
        _key(value) or f"{blank}.measure.{ndx}"
        for ndx, value in enumerate(table[right])
    ]

    doubled = int(stations["_key"].duplicated().sum())

    if doubled:
        # Not refused, because it is the source saying something about itself and
        # the result is still every measure: two stations sharing a code and three
        # measures against it give six rows, which is six claims and not one
        # dropped. Said out loud because the count in the summary is then not the
        # count in the table.
        report.notes.append(
            f"attenzione: {doubled} codici stazione ripetuti nel layer puntuale; "
            f"il join li moltiplica, e ogni misura esce su ciascuna stazione che "
            f"porta quel codice"
        )

    merged = stations.merge(
        table, on="_key", how="outer", suffixes=(STATION_SUFFIX, ""),
        indicator="_side",
    )

    orphans = merged[merged["_side"] == "right_only"]
    rows = merged[merged["_side"] != "right_only"]

    report.sites = len(stations)
    report.measures = int((merged["_side"] == "both").sum())
    report.orphans = len(orphans)

    return rows.reset_index(drop=True), orphans.reset_index(drop=True)


def _key(value):
    """A join key as the string it is, with a null and a blank coming back None."""

    if value is None or value != value:
        return None

    said = str(value).strip()

    return said or None


def _measure_names(frame, mapping, report):
    """
    The name each row's measurement is written under, and how many share a station.

    A station code is not a name for a measurement as soon as the station carries
    two: three surfaces at S4 are three attitudes at one anchor, so they are at
    one `s`, and the only thing left distinguishing them in `attitude_at`'s
    answer -- `misurata:S4` -- is nothing. So the surface column is appended to
    the code where there is one and the row's ordinal where there is not, and a
    profile that draws one of the three says which of the three it drew.

    The ordinal is the source's own row order, not a ranking: which of three
    measured surfaces is the one the trace follows is a question for whoever
    measured them, and inventing an answer here is the silent decision this
    whole module exists to refuse.
    """

    names, siblings, ordinals, used, minted = [], {}, {}, set(), 0

    sites = [
        _station_text(frame, mapping.points_ident_field, ndx)
        for ndx in range(len(frame))
    ]

    counted = {}

    for said in sites:
        counted[said] = counted.get(said, 0) + 1

    report.several = sum(1 for said, n in counted.items() if said and n > 1)

    for ndx, said in enumerate(sites):
        if not said:
            # Minted per row and not per station: with no code column there is
            # nothing saying two rows are one place, and calling them one would
            # be an assertion about the survey rather than a handle.
            minted += 1
            said = f"{MINTED_POINT}{minted:04d}"
            report.minted_points += 1
            names.append(said)
            siblings[ndx] = 0
            continue

        if counted[said] == 1:
            names.append(said)
            siblings[ndx] = 0
            continue

        surface = _text(frame, mapping.surface_field, ndx)

        ordinals[said] = ordinals.get(said, 0) + 1

        name = f"{said}{MEASURE}{surface or ordinals[said]}"

        if name in used:
            # The surface column naming one surface twice at one station, which
            # is what a repeated trace ident is and gets the same answer: the row
            # it was, appended. Two attitudes under one key would be precisely
            # the tie this function exists to break, arrived at the long way
            # round.
            name = f"{name}{PART}{ordinals[said]}"

        used.add(name)

        names.append(name)
        siblings[ndx] = counted[said] - 1

    return names, siblings


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

    **One row is up to three records, and they carry the same name.** A fault
    station is a plane, or a striation, or both, and the format has a record for
    each: the plane is an `attitude` and the striation a `lineation`, at the same
    anchor, under the same `station=`. That key is the only thing tying them
    together -- the format has no field for "the lineation on *that* plane" --
    and it is enough as long as it is unique, which is what `_measure_names` is
    for.
    """

    frame = _stations(mapping, frame_crs, report)

    orphans = None

    if mapping.has_measures:
        frame, orphans = _joined(frame, mapping, report)

    usable = frame.geometry.notna() & ~frame.geometry.is_empty
    without = int((~usable).sum())

    if without:
        # A station with no position, which after a join is also a measure with
        # no position: the reason it is not in the file as an observation is that
        # an observation with neither an anchor nor a structure to name is not a
        # record of anything.
        report.dropped["points with no geometry"] = without

    frame = frame[usable].reset_index(drop=True)

    if not dataset.structures:
        return

    if not frame.empty:
        _measured(gstruct, dataset, frame, mapping, report)

    if orphans is not None and not orphans.empty:
        _unstationed(gstruct, dataset, orphans, mapping, report)


def _column(frame, name, whose):
    """The column, or a sentence saying which table it was expected in."""

    if name in frame.columns:
        return frame[name]

    if f"{name}{STATION_SUFFIX}" in frame.columns:
        return frame[f"{name}{STATION_SUFFIX}"]

    raise ValueError(
        f"`{name}` is not a column of the {whose}. Where the measures are in a "
        f"table, the angles are columns of the table and not of the point layer: "
        f"that is what the join is for."
    )


def _angles(frame, mapping, report, whose="point"):
    """The row's two angles, normalised, with a mask of the readable ones."""

    azimuth = numeric(_column(frame, mapping.points_dip_dir_field, "measures"))
    dip = numeric(_column(frame, mapping.points_dip_field, "measures"))

    keep, dropped = admissible(azimuth, dip)

    for reason, count in dropped.items():
        report.dropped[f"{whose} plane: {reason}"] = count

    return normalised_azimuth(azimuth, dip), dip, keep


def _striation(frame, mapping, report, whose="point"):
    """
    The row's striation, or `(None, None, None)` where the columns are not named.

    **Not read through `admissible`, and the difference is the whole reason this
    is its own function.** That one exempts a zero dip from needing an azimuth,
    because a horizontal bed has no dip direction -- and a horizontal striation
    has a perfectly good trend, so the same exemption passes a row whose trend is
    blank and `nan` goes into the file as a number. A lineation needs both
    numbers, both in range, and neither of them stands in for the other.

    A row with neither is not a striation that failed to travel: it is a
    measurement of a plane, which is most of them. Only a row that gives one of
    the two is counted as something lost, because only that row was trying to say
    something this could not write.
    """

    if not mapping.has_lineations:
        return None, None, None

    import numpy as np

    trend = numeric(_column(frame, mapping.trend_field, "measures"))
    plunge = numeric(_column(frame, mapping.plunge_field, "measures"))

    known = np.isfinite(trend) & np.isfinite(plunge)
    ranged = (trend >= 0.0) & (trend <= 360.0) & (plunge >= 0.0) & (plunge <= 90.0)

    half = int(((np.isfinite(trend) | np.isfinite(plunge)) & ~known).sum())
    outside = int((known & ~ranged).sum())

    if half:
        report.dropped[f"{whose} lineation: one angle of the two"] = half

    if outside:
        report.dropped[f"{whose} lineation: outside 0-360/0-90"] = outside

    # 360 is how north gets written down, here as much as in a dip direction, and
    # `_a`'s two decimals are not the only place a number is rounded: a trend of
    # 360 written as 360 reads back as a bearing the parser is perfectly happy
    # with and that no compass has.
    return np.where(np.isfinite(trend), trend % 360.0, trend), plunge, known & ranged


def _raw_of(frame, mapping, ndx):
    """The row's carried columns, as `raw.*`, in the order they were asked for."""

    out = {}

    for name in mapping.points_keep_fields:
        said = _text(frame, name, ndx)

        if said:
            out[f"raw.{name}"] = said

    return out


def _measured(gstruct, dataset, frame, mapping, report):
    """Every row that has a station: an attitude, a lineation, or an observation."""

    from geogst.core.geology.orientations import Plane
    from shapely import STRtree
    from shapely.geometry import LineString

    azimuth, dip, keep = _angles(frame, mapping, report)
    trend, plunge, striated = _striation(frame, mapping, report)

    names, siblings = _measure_names(frame, mapping, report)

    paths = [LineString(structure.path) for structure in dataset.structures]
    tree = STRtree(paths)

    for ndx in range(len(frame)):
        point = frame.geometry.iloc[ndx]
        anchor = (float(point.x), float(point.y))

        said = names[ndx]

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

        # How many others were measured at this station, written on each of them.
        # `attitude_at` answers with one plane and cannot say it chose: this is
        # the number that says there was a choice, on the record that won it.
        if siblings[ndx]:
            attrs["siblings"] = str(siblings[ndx])

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
            plane = gstruct_plane(Plane(
                float(azimuth[ndx]), float(dip[ndx]),
                is_rhr_strike=mapping.points_is_rhr_strike,
            ))
        else:
            plane = None
            attrs["no_attitude"] = NO_ATTITUDE

        has_stria = striated is not None and bool(striated[ndx])
        within = away <= mapping.attach_within

        carried = _raw_of(frame, mapping, ndx)

        attached = plane is not None and within

        if attached:
            attitude = gstruct.Attitude(anchor=anchor, plane=plane, attrs=attrs)
            attitude.resolve(near.path)

            # Written because it is the one number that says how much of an
            # attachment this was: `s` is derived and looks exact whatever the
            # point's distance from the trace, and a station 80 m off a fault is
            # a different claim from one standing on it.
            attitude.attrs["off"] = f"{attitude.offset:.1f}"

            # The carried columns on the attitude, which is the row's first
            # record: a note about the measurement belongs on the measurement,
            # and repeating it on the lineation would put one sentence in the
            # file twice for one thing that was written down once.
            attitude.attrs.update(carried)

            near.attitudes.append(attitude)
            report.attached += 1

        # A striation attaches on its own terms: the station is inside the
        # threshold or it is not, and whether the plane was readable has nothing
        # to do with it. A surface too weathered to give a reliable dip can still
        # carry a slip direction, and that is a row with a lineation and no
        # attitude -- which the format can hold, `lineation` being its own record
        # and not a field of `attitude`.
        striation = None

        if has_stria and within:
            striation = gstruct.Lineation(
                anchor=anchor,
                trend=float(trend[ndx]), plunge=float(plunge[ndx]),
                attrs={
                    "station": said, "src": "points",
                    "raw": (
                        f"{mapping.trend_field}={trend[ndx]:.0f} "
                        f"{mapping.plunge_field}={plunge[ndx]:.0f}"
                    ),
                },
            )
            striation.resolve(near.path)
            striation.attrs["off"] = f"{striation.offset:.1f}"

            # The row's carried columns go on the first record it produced, and
            # once: a note somebody wrote against one measurement is one note,
            # and on two records it reads as two.
            if not attached:
                striation.attrs.update(carried)

            near.lineations.append(striation)
            report.lineations += 1

        if attached:
            continue

        if plane is not None:
            attrs["unattached"] = UNATTACHED
            attrs["threshold"] = f"{mapping.attach_within:.0f}"

        attrs["nearest"] = near.ident
        attrs["distance"] = f"{away:.1f}"

        # The striation on the observation as one value, where there was no
        # `lineation` to write it as: that record lives on a structure, and a
        # point past the threshold reached none. Written normalised rather than
        # left in `raw` alone, so that whoever disagrees with the threshold has
        # the measurement and not only the source's spelling of it.
        if has_stria and striation is None:
            attrs["lineation"] = f"{trend[ndx]:.0f}/{plunge[ndx]:.0f}"

        if striation is None:
            attrs.update(carried)

        dataset.observations.append(
            gstruct.Observation(ident=said, anchor=anchor, plane=plane, attrs=attrs)
        )
        report.observations += 1


def _unstationed(gstruct, dataset, orphans, mapping, report):
    """
    The measures whose key matched no station, as observations with no anchor.

    `*` where the anchor goes, which the format writes and reads back as nothing
    -- so the measurement is in the file, with the key that matched nothing, and
    the one thing it cannot claim is a place. Rule 3 with no position to argue
    about: the plane is still a plane, and the station it was taken at is the
    part that is missing.
    """

    from geogst.core.geology.orientations import Plane

    azimuth, dip, keep = _angles(orphans, mapping, report, whose="unstationed")
    trend, plunge, striated = _striation(orphans, mapping, report, whose="unstationed")

    for ndx in range(len(orphans)):
        said = _text(orphans, mapping.measures_join_field, ndx)

        attrs = {
            "src": "measures",
            "orphan": ORPHAN,
            "key": said or "(vuoto)",
        }

        plane = None

        if bool(keep[ndx]):
            attrs["raw"] = (
                f"{mapping.points_dip_dir_field}={azimuth[ndx]:.0f} "
                f"{mapping.points_dip_field}={dip[ndx]:.0f}"
            )

            plane = gstruct_plane(Plane(
                float(azimuth[ndx]), float(dip[ndx]),
                is_rhr_strike=mapping.points_is_rhr_strike,
            ))
        else:
            attrs["no_attitude"] = NO_ATTITUDE

        if striated is not None and bool(striated[ndx]):
            attrs["lineation"] = f"{trend[ndx]:.0f}/{plunge[ndx]:.0f}"

        attrs.update(_raw_of(orphans, mapping, ndx))

        dataset.observations.append(
            gstruct.Observation(
                ident=said or f"{MINTED_MEASURE}{ndx + 1:04d}",
                anchor=None, plane=plane, attrs=attrs,
            )
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

        # Typed rather than chosen from a list, because the list is the whole EPSG
        # database and the answer is one code somebody already knows. Prefilled
        # from the layer, which is the answer in every case where the layer has
        # one: see `projected_for`.
        self.crs_edit = QtWidgets.QLineEdit()
        self.crs_edit.setPlaceholderText("EPSG:25833 - the layer's own where it has one")

        self.crs_note = QtWidgets.QLabel()
        self.crs_note.setWordWrap(True)
        self.crs_note.setStyleSheet("color: gray; font-size: 10px;")

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

        # -- the measures, where they are in a table of their own -----------

        self._measures = None

        self.measures_label = QtWidgets.QLabel("measures on the points themselves")
        self.measures_label.setStyleSheet("color: gray; font-size: 10px;")

        measures_browse = QtWidgets.QPushButton("Table...")
        measures_browse.clicked.connect(self._browse_measures)

        measures_clear = QtWidgets.QPushButton("Clear")
        measures_clear.clicked.connect(self._clear_measures)

        self.measures_layer_combo = QtWidgets.QComboBox()
        self.measures_layer_combo.setEnabled(False)
        self.measures_layer_combo.currentTextChanged.connect(self._on_measures_layer)

        self.points_join_combo = QtWidgets.QComboBox()
        self.measures_join_combo = QtWidgets.QComboBox()

        self.surface_combo = QtWidgets.QComboBox()
        self.trend_combo = QtWidgets.QComboBox()
        self.plunge_combo = QtWidgets.QComboBox()

        self.points_keep_list = QtWidgets.QListWidget()
        self.points_keep_list.setMaximumHeight(80)

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
        grid.addWidget(QtWidgets.QLabel("write in"), 2, 0)
        grid.addWidget(self.crs_edit, 2, 1)
        grid.addWidget(self.crs_note, 3, 0, 1, 2)
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
        # The table and its key between the station and the angles, because that
        # is the order the questions depend on each other in: which layer, which
        # table, which column marries them, and only then what the columns of the
        # married thing mean.
        at.addWidget(measures_browse, 3, 0)
        at.addWidget(self.measures_layer_combo, 3, 1, 1, 2)
        at.addWidget(measures_clear, 3, 3)
        at.addWidget(self.measures_label, 4, 0, 1, 4)
        at.addWidget(QtWidgets.QLabel("join on"), 5, 0)
        at.addWidget(self.points_join_combo, 5, 1)
        at.addWidget(QtWidgets.QLabel("="), 5, 2)
        at.addWidget(self.measures_join_combo, 5, 3)
        at.addWidget(QtWidgets.QLabel("azimuth"), 6, 0)
        at.addWidget(self.points_dip_dir_combo, 6, 1)
        at.addWidget(QtWidgets.QLabel("dip"), 6, 2)
        at.addWidget(self.points_dip_combo, 6, 3)
        at.addWidget(QtWidgets.QLabel("read as"), 7, 0)
        at.addWidget(self.points_convention_combo, 7, 1, 1, 3)
        at.addWidget(QtWidgets.QLabel("trend"), 8, 0)
        at.addWidget(self.trend_combo, 8, 1)
        at.addWidget(QtWidgets.QLabel("plunge"), 8, 2)
        at.addWidget(self.plunge_combo, 8, 3)
        at.addWidget(QtWidgets.QLabel("surface"), 9, 0)
        at.addWidget(self.surface_combo, 9, 1, 1, 3)
        at.addWidget(QtWidgets.QLabel("as raw.*"), 10, 0)
        at.addWidget(self.points_keep_list, 10, 1, 1, 3)
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

        self._propose_crs(layer)

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

    def _propose_crs(self, layer):
        """
        The projection to write in, filled in from the layer, and why.

        Filled rather than merely offered, and overwritten on every change of
        layer: a code left over from the previous layer is the one answer here
        that would be wrong without looking wrong, since a file in EPSG:25833
        holding coordinates transformed from somewhere else reads as perfectly
        ordinary.
        """

        try:
            epsg, why = projected_for(self._path, layer)
        except Exception as err:
            epsg, why = None, f"{type(err).__name__}: {err}"

        self.crs_edit.setText(epsg or "")
        self.crs_note.setText(why or "")

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

        every, _ = fields_of(self._points, layer)
        proposed = guess(self._points, layer)

        station = first_match(STATION_FIELDS, every) or proposed.ident_field

        # The station's code and the key it is joined by are proposed as the same
        # column, because on every survey that keeps its measures apart they are:
        # `station_code` is what the station is called and what the measure points
        # at. Two fields and not one, because nothing guarantees it.
        for combo, fields, chosen in (
            (self.station_combo, every, station),
            (self.points_join_combo, every, station),
        ):
            with QtCore.QSignalBlocker(combo):
                combo.clear()
                combo.addItem(self.NO_FIELD)
                combo.addItems(fields)
                combo.setCurrentText(chosen or self.NO_FIELD)

        self._on_measure_columns()

    def _measures_side(self):
        """Which table the angles are read off, as `(path, layer)`."""

        if self._measures:
            return self._measures, self.measures_layer_combo.currentText() or None

        return self._points, self.points_layer_combo.currentText() or None

    def _on_measure_columns(self):
        """
        What the measures' own columns mean, off whichever table holds them.

        One method for both shapes, because it is one question: the angles are
        columns of the thing the angles are in. On a point layer carrying its own
        dip direction that is the point layer; with a table joined to it, it is
        the table, and this is the only place in the dialog that has to know
        which.
        """

        path, layer = self._measures_side()

        if not path:
            return

        every, numbers = fields_of(path, layer)
        proposed = guess(path, layer)

        for combo, fields, chosen in (
            (self.points_dip_dir_combo, numbers, proposed.dip_dir_field),
            (self.points_dip_combo, numbers, proposed.dip_field),
            (self.trend_combo, numbers, first_match(TREND_FIELDS, numbers)),
            (self.plunge_combo, numbers, first_match(PLUNGE_FIELDS, numbers)),
            (self.surface_combo, every, first_match(SURFACE_FIELDS, every)),
            (self.measures_join_combo, every,
             first_match(JOIN_FIELDS, every) if self._measures else None),
        ):
            with QtCore.QSignalBlocker(combo):
                combo.clear()
                combo.addItem(self.NO_FIELD)
                combo.addItems(fields)
                combo.setCurrentText(chosen or self.NO_FIELD)

        if (self.points_dip_dir_combo.currentText() or "").lower().startswith("strike"):
            self.points_convention_combo.setCurrentIndex(1)

        self._fill_points_keep()

    def _fill_points_keep(self):
        """
        The columns of both tables on offer to be carried, under the names the
        join will leave them with.

        Unchecked where the column is already in the file under its own name --
        the angles, the striation, the key, the code, the surface. `raw.dip_dir=165`
        beside `raw="dip_dir=165 dip=65"` is the same number written twice, and a
        list where everything is ticked is a list nobody reads.
        """

        path, layer = self._measures_side()

        measured, _ = fields_of(path, layer)

        offered = list(measured)

        if self._measures:
            station_side, _ = fields_of(
                self._points, self.points_layer_combo.currentText() or None
            )

            # The same rule the merge uses, and it has to be the same rule: what
            # is offered here is the name the column will be called by then.
            offered += [
                f"{name}{STATION_SUFFIX}" if name in measured else name
                for name in station_side
            ]

        spoken = {
            self._named(combo) for combo in (
                self.points_dip_dir_combo, self.points_dip_combo,
                self.trend_combo, self.plunge_combo, self.surface_combo,
                self.station_combo, self.points_join_combo,
                self.measures_join_combo,
            )
        }
        spoken.discard(None)
        spoken.add("fid")

        self.points_keep_list.clear()

        for name in offered:
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                QtCore.Qt.CheckState.Unchecked
                if name in spoken or name.rsplit(STATION_SUFFIX, 1)[0] in spoken
                else QtCore.Qt.CheckState.Checked
            )
            self.points_keep_list.addItem(item)

    def _browse_measures(self):
        # Opened where the stations are, which is where the table usually is:
        # `geology.gpkg` holds the point layer and the measures both.
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose the table of structural measures", self._points or "",
            "Table or vector (*.gpkg *.sqlite *.csv *.dbf *.xlsx *.ods);;All files (*)",
        )

        if path:
            self.set_measures(path)

    def set_measures(self, path, layer=None):
        """Lists the tables in the file, and guesses the columns off one."""

        if not self._points:
            QtWidgets.QMessageBox.information(
                self, "No stations",
                "a table of measures has no position of its own: what gives one "
                "to a measurement is the station it was taken at. Choose the "
                "point layer of stations first.",
            )
            return False

        try:
            layers = VectorSource.candidate_layers(path, VectorSource.TABLE)
        except Exception as err:
            QtWidgets.QMessageBox.warning(
                self, "Unreadable", f"{path}\n\n{type(err).__name__}: {err}"
            )
            return False

        if not layers:
            QtWidgets.QMessageBox.information(
                self, "No tables",
                f"{path}\n\nno layer without geometry in this file. What belongs "
                f"here is the table the measurements are in -- rows keyed to a "
                f"station code. A layer with a geometry of its own is a layer of "
                f"points, and goes in the slot above.",
            )
            return False

        self._measures = path
        self.measures_label.setText(str(path))

        with QtCore.QSignalBlocker(self.measures_layer_combo):
            self.measures_layer_combo.clear()
            self.measures_layer_combo.addItems(layers)
            self.measures_layer_combo.setCurrentText(
                layer if layer in layers else layers[0]
            )

        self.measures_layer_combo.setEnabled(True)

        self._on_measure_columns()

        return True

    def _on_measures_layer(self, layer):
        if not self._measures or not layer:
            return

        self._on_measure_columns()

    def _clear_measures(self):
        """Back to the measures being on the points, in one click."""

        self._measures = None
        self.measures_label.setText("measures on the points themselves")

        with QtCore.QSignalBlocker(self.measures_layer_combo):
            self.measures_layer_combo.clear()

        self.measures_layer_combo.setEnabled(False)

        with QtCore.QSignalBlocker(self.measures_join_combo):
            self.measures_join_combo.clear()

        if self._points:
            self._on_measure_columns()

    def _clear_points(self):
        """Backing out of the points, which has to be possible in one click."""

        self._points = None
        self.points_label.setText("no point layer")

        with QtCore.QSignalBlocker(self.points_layer_combo):
            self.points_layer_combo.clear()

        self.points_layer_combo.setEnabled(False)

        # The table goes with them: it is keyed to stations, and with no station
        # layer left there is nothing for its rows to be measurements on.
        self._clear_measures()

        for combo in (self.station_combo, self.points_dip_dir_combo,
                      self.points_dip_combo, self.points_join_combo,
                      self.trend_combo, self.plunge_combo, self.surface_combo):
            with QtCore.QSignalBlocker(combo):
                combo.clear()

        self.points_keep_list.clear()

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

        carried = tuple(
            self.points_keep_list.item(row).text()
            for row in range(self.points_keep_list.count())
            if self.points_keep_list.item(row).checkState()
            == QtCore.Qt.CheckState.Checked
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
            measures_path=self._measures,
            measures_layer=(self.measures_layer_combo.currentText() or None)
            if self._measures else None,
            measures_join_field=self._named(self.measures_join_combo)
            if self._measures else None,
            points_join_field=self._named(self.points_join_combo)
            if self._measures else None,
            surface_field=self._named(self.surface_combo),
            trend_field=self._named(self.trend_combo),
            plunge_field=self._named(self.plunge_combo),
            points_keep_fields=carried if self._points else (),
            target_crs=self.crs_edit.text().strip() or None,
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

        said = crs_refusal(mapping.target_crs)

        if said:
            return said

        if mapping.measures_path and not mapping.points_path:
            return (
                "A table of measures is chosen and no layer of stations. A row in "
                "that table says what was measured and nothing about where: the "
                "position comes from the station it is keyed to, and with no "
                "station layer there is nothing to key it to."
            )

        if mapping.measures_path and not mapping.has_measures:
            return (
                "A table of measures is chosen and the two halves of the key are "
                "not both named. A join is one column on each side -- the code the "
                "station is called by, and the column in the table that points at "
                "it -- and guessing either half would attach measurements to "
                "whatever happened to sort first."
            )

        if mapping.points_path and not mapping.has_points:
            return (
                "A point layer is chosen and its two angle columns are not both "
                "named. A point with no plane on it is not a measurement to "
                "attach -- it would go into the file as an `observation` saying "
                "the source had no attitude, which is true of every point in the "
                "layer and is not worth a file. Name both, or clear the layer."
                + (
                    " With a table joined, those two columns are the table's: the "
                    "angles are where the angles are."
                    if mapping.measures_path else ""
                )
            )

        if bool(mapping.trend_field) != bool(mapping.plunge_field):
            return (
                "One half of a striation is named and the other is not. A trend "
                "with no plunge is a bearing and not a line, and a plunge with no "
                "trend is a number: `lineation` takes both, so name both or "
                "neither."
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
