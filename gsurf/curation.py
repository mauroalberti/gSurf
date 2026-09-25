"""
The boundary with gstruct: a plane changes library here, and a file becomes records.

`gstruct` is the text format the curation is written in and read back from --
one fact per line, anchored to coordinates rather than to progressives, and
readable without GDAL, which is the property it was built for. It lives in its
own repository and gSurf imports it; this module is the only place that does,
so that everything the two projects have to agree about is agreed in one file.

Two jobs, and the second is the reason for the first.

**A source.** `records_of` turns a dataset into `TraceRecord`s, so a `.gstruct`
can be opened in the traces slot exactly as a GeoPackage layer is. The records
are the same records; what differs is where the anchor comes from. A GeoPackage
cannot hold `@x,y`, so `export_gsurf.py` flattens every anchor to a progressive
in an `anchor_s` column -- and a progressive means nothing without the line it
was measured along, which is the one thing FORMAT.md refuses to store. Read
here, the anchor arrives as coordinates and `Structure.resolve` projects it onto
the path in hand. The measured cost of the difference is in `test_roundtrip.py`:
redigitise the trace and an anchored event moves 1.6 m, the same event held as
`s` moves 49.

**An overlay.** `apply_to` takes a file that carries no geometry at all -- five
structures named by ident, an assertion each -- and lays it over records already
open. That is what a curation is: `curation.gstruct` says so in its own header,
*si applica sopra merid_faults.gstruct*. The source layer is never written to,
which is the bargain the panel already strikes and the reason the file exists.

What the two libraries have to agree about, first of all, is which way a normal
points.

Both libraries name a plane the same way -- dip direction and dip angle, in
degrees -- so the conversion carries two numbers across and invents nothing.
The agreement that matters is one level down. FORMAT.md is explicit about it:
the normal points *upward*, and its horizontal component points *toward* the
dip, not against it. So does geogst, through `norm_direct_up`; a plane dipping
30 to the east has the normal (0.5, 0, 0.866) in both, east-north-up.

The trap is that `norm_direct_up` is not the method gSurf reaches for. What it
uses everywhere else is `Plane.normal_axis`, which points *down*, and which is
right where it is used: the orientation tensor is axial, a bed and the same bed
overturned are one pole, and a sign there would be a distinction without a
difference. Measured over eight planes, `norm_direct_up` agrees with gstruct
eight times and `normal_axis` once -- the once being the vertical plane, where
up and down are both horizontal and the question does not arise.

Which is exactly why the convention is pinned in `checks/check_curation.py`
rather than trusted. An error of 180 degrees in a normal does not show up when
two planes are compared, because the angle between them is taken through the
absolute value of the dot product and the sign cancels; it shows up much later,
somewhere a plane is compared against a normal, as a result that is wrong by a
hemisphere and looks like a result.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SUFFIX = ".gstruct"

# The verdict a fit carries when the trace it was read off is straight. FORMAT.md
# is explicit that the number is there and the measurement is not: a straight
# trace does not constrain the dip, the answer comes out near vertical by
# default rather than by evidence, and `attitude_at` skips it. A record carrying
# it arrives switched off -- present in the panel, out of the section, and with
# the reason in its own `verdict` where it can be argued with.
UNCONSTRAINED = "traccia-rettilinea"

# The axes the format defines. Anything else in a file is somebody's proposal,
# and is reported rather than acted on: `value_at` would read an invented axis
# perfectly well, which is exactly why acting on one silently would let it
# become real without anybody deciding it should.
AXES = ("certainty", "exposure", "use")

# `use` is the one of the three this tool acts on rather than carries. It is the
# curator's decision about a stretch -- `rejected` means nothing there holds, not
# the fit and not the measurement -- and it went into gstruct 0.2 because the
# panel needed to write a refusal down and had nowhere to put one. This module
# used to read it off a structure attribute called `in_section`, which was a name
# invented here; an axis says it in the format's own grammar, and says it about a
# stretch rather than about a whole fault.
USE = "use"
ACCEPTED, REJECTED, UNSAID = "accepted", "rejected", "unknown"

# The version of gstruct that has the axis in it.
NEEDS = (0, 2)

# Where a record's identity is looked for, in order. `ident` is what gstruct
# calls it; `code` is what `export_gsurf.py` names the column it flattens the
# ident into, so records read from that GeoPackage answer to the same file.
IDENT_KEYS = ("ident", "code")


def gstruct_plane(plane):
    """
    A geogst plane as a gstruct one.

    `Plane` carries its azimuth as a dip direction whatever it was built from:
    `is_rhr_strike` is a way of answering the constructor, not a state the
    object keeps, so there is nothing to ask here and nothing to convert.
    """

    import gstruct

    return gstruct.Plane(float(plane.dipazim), float(plane.dipang))


def geogst_plane(plane):
    """
    A gstruct plane as a geogst one.

    `is_rhr_strike=False` and not the source's setting, because what is coming
    in is a dip direction: the format writes `140/31` and says in its own
    grammar that the first number is the dip direction. A layer read as RHR
    strike was converted on the way in, and converting again on the way back
    would turn a right angle into a fact about nothing.
    """

    from geogst.core.geology.orientations import Plane

    return Plane(float(plane.dip_dir), float(plane.dip), is_rhr_strike=False)


# -- the file ------------------------------------------------------------


def is_gstruct(path):
    """Whether this is a file to read here rather than through GDAL."""

    return path is not None and Path(path).suffix.lower() == SUFFIX


def module():
    """
    The gstruct library, or an ImportError written to be read in a dialog.

    The one dependency this project cannot name in its `pyproject.toml`: the
    name on PyPI belongs to something else, so gstruct is installed from its own
    repository or not at all. Every caller of this shows what comes out of here
    to somebody, so it says what to do rather than where it broke.
    """

    try:
        import gstruct
    except ImportError as err:
        raise ImportError(
            "gstruct is not installed. It is on no package index -- the name "
            "there belongs to an unrelated project -- so it is installed from "
            "its own repository:  pip install -e <gstruct repo>"
        ) from err

    # The same refusal gstruct makes about a file from the future, in the other
    # direction: 0.2 is where the `use` axis and `span_at` arrived, and an older
    # library would read a refusal without acting on it -- a stretch drawn into
    # a section that somebody had rejected, which is a wrong answer that looks
    # like an answer.
    if tuple(int(n) for n in gstruct.VERSION.split(".")) < NEEDS:
        raise ImportError(
            f"gstruct {gstruct.VERSION} is installed and this needs "
            f"{'.'.join(str(n) for n in NEEDS)} or later: the `use` axis is "
            f"read there, and an older library would ignore a refusal instead "
            f"of refusing to read it. Update the gstruct repository."
        )

    return gstruct


def read(path):
    """
    The file as a dataset, with every anchor projected onto its own path.

    `load` resolves on the way in, so `s` is already derived by the time this
    returns -- for the structures that carry a path. A curation carries none and
    its intervals stay unresolved, which is the same rule and not an exception
    to it: no path, no projection. `apply_to` is what resolves those, against
    the geometry the records brought with them.
    """

    return module().load(str(path))


def reproject(dataset, source, target):
    """
    Every coordinate moved onto another projection, and the anchors re-derived.

    The order is the whole of the care needed here. An anchor's `s` was computed
    by projecting it onto the path in the source projection, and the map between
    two projected systems does not preserve distance, so a progressive carried
    across would be a measurement of the wrong ruler. Moving the coordinates and
    resolving again costs one pass and leaves `s` meaning what it says: metres
    along the path that is in hand.
    """

    from pyproj import Transformer

    if source is None or target is None or source.equals(target):
        return dataset

    transformer = Transformer.from_crs(source, target, always_xy=True)

    def moved(xy):
        if xy is None:
            return None
        x, y = transformer.transform(xy[0], xy[1])
        return (float(x), float(y))

    for structure in dataset.structures:
        if structure.path:
            xs, ys = transformer.transform(*zip(*structure.path))
            structure.path = [(float(x), float(y)) for x, y in zip(xs, ys)]

        for span in (*structure.spans, *structure.fits):
            span.start, span.end = moved(span.start), moved(span.end)

        for event in (*structure.attitudes, *structure.lineations):
            event.anchor = moved(event.anchor)

    for observation in dataset.observations:
        observation.anchor = moved(observation.anchor)

    return dataset.resolve()


def frame_of(path):
    """
    The projection and the extent a `.gstruct` frames a session with.

    The counterpart of `pyogrio.read_info` for a file pyogrio cannot open. It is
    not as cheap as that one -- there is no header to read, so the paths are
    parsed to be measured -- but the file is text and the alternative is a
    session that cannot be opened on it at all.
    """

    from pyproj import CRS

    dataset = read(path)

    # The observations count towards the frame as much as the traces do. They
    # are measurements that attached to nothing, not measurements from nowhere,
    # and a frame drawn without them would put the session's own edge between
    # the survey and part of itself -- which `records_of` would then read as a
    # reason to drop them.
    places = [
        observation.anchor for observation in dataset.observations
        if observation.anchor is not None
    ]

    xs = [x for structure in dataset.structures for x, _ in structure.path]
    ys = [y for structure in dataset.structures for _, y in structure.path]

    xs += [x for x, _ in places]
    ys += [y for _, y in places]

    if not xs or not dataset.crs:
        return None, None

    return CRS.from_user_input(dataset.crs), (min(xs), min(ys), max(xs), max(ys))


def degrees_not_metres(dataset, fallback=None):
    """
    Why this file cannot be measured along, or None if it can.

    Everything read along a trace is a progressive in the file's own units --
    `attitude_at`'s reach, a fit's window, `DEFAULT_MAX_GAP` -- and everything
    written back is a coordinate to two decimals, which is a centimetre in a
    projected CRS and 0.01 degrees in a geographic one. Measured on the first
    vertex of `merid_faults.gstruct` taken to EPSG:4326: `@16.27,39.92`, which
    is 472 m from the point that was picked. The reason an anchor is snapped to
    the trace at all is that fifty metres of error would say something false
    about where somebody stood, so writing four hundred is not a rounding.

    The CRS is the one declared, or the caller's where the file declares none:
    a file with no `crs` line is read at face value in the session's, so that is
    the ruler either way. An unreadable one is refused here rather than left to
    raise from inside a transformer.
    """

    declared = dataset.crs or fallback

    if declared is None:
        return None

    from pyproj import CRS

    # Which of the two it is matters to the sentence: a file that declares a
    # projection is wrong about itself, and one that declares none is being read
    # in the session's, which is somebody's answer in a dialog and a different
    # thing to go and fix.
    whose = (
        f"this file is written in {declared}"
        if dataset.crs
        else f"this file declares no projection and is read as the session's, {declared}"
    )

    try:
        crs = CRS.from_user_input(declared)
    except Exception as err:
        return f"{whose}, which cannot be read: {type(err).__name__}: {err}"

    if not crs.is_geographic:
        return None

    said = (
        f"{whose}, which is geographic: its "
        f"coordinates are degrees, and the ruler everything along a trace is "
        f"measured with -- a progressive, a reach, the window a fit was read "
        f"on -- is metres. An anchor is written to two decimals, a centimetre "
        f"in a projected CRS and about a kilometre in this one."
    )

    slip = _anchor_slip(crs, dataset)

    if slip is not None:
        said += (
            f" The first vertex in it would be written {slip:,.0f} m from "
            f"where it is."
        )

    return said + (
        " Reproject it to the projection the survey was mapped in -- "
        "`export_gsurf.py` writes the layer's own -- and open it again."
    )


def _anchor_slip(crs, dataset):
    """How far `@x,y` to two decimals lands from the first vertex, in metres."""

    point = next(
        (vertex for structure in dataset.structures for vertex in structure.path[:1]),
        None,
    )
    geod = crs.get_geod()

    if point is None or geod is None:
        return None

    lon, lat = point

    # The same expression `insert_anchor` writes with, so this is the error the
    # editor would make and not an estimate of it.
    return geod.inv(lon, lat, float(f"{lon:.2f}"), float(f"{lat:.2f}"))[2]


# -- the file as text, and the blocks it is made of ------------------------

# What `attitude_at` answers with, as the word before the colon. The five are
# gstruct's own and not a tool's: they fall out of the precedence rule, which is
# where the format decides what holds at a place. Named here, at the boundary,
# rather than in whatever happens to draw them.
PROVENANCE = ("rifiutata", "misurata", "fit", "misurata-lontana", "assente")

# How far a measurement still answers for, in metres. The same number and the
# same judgement as `attitudes.DEFAULT_HALF_SPAN` -- how much of a fault one
# compass reading speaks for -- approached from the two ends: there it builds
# the interval around an anchor, here it is how far from the anchor the answer
# is still that reading's. Written out rather than imported because `attitudes`
# imports this module, and it is a default either way: the caller sets it.
DEFAULT_MAX_GAP = 250.0


def _content(line):
    """Whether a line says anything: not blank, and not a comment of its own."""

    stripped = line.strip()

    return bool(stripped) and not stripped.startswith("#")


def _opens(line):
    """
    The keyword of a line that starts a top-level record, or None.

    Indentation is how the format says a line continues the record above it, so
    a line starting in column one is the only kind that can begin a new one.
    The cut at `#` and the split on a single space are the parser's own, kept
    the same deliberately: a scanner that disagreed with `loads` about where a
    record starts would slice the file somewhere `loads` does not.
    """

    if not line[:1].strip():
        return None

    word = line.split("#", 1)[0].strip().partition(" ")[0]

    return word if word in ("structure", "observation") else None


def _split_keeping_ends(source):
    """
    The content lines, and the terminator each one came with.

    `splitlines()` decides *where* the lines are, because that is what `loads`
    uses and a scanner that disagreed with it would slice the file somewhere it
    does not -- but it throws the terminators away, and rejoining with `"\\n"`
    then rewrites every line of a CRLF file in order to replace one block of it.
    Measured on `curation.gstruct` converted to CRLF: thirty carriage returns
    in, none out, for an edit to a block of two lines, and in git a one-block
    change that arrives as a whole-file diff is a change nobody reads.

    The join was only the second half of that, and the smaller one. The first
    was that `read_text` is text mode, so the carriage returns were already gone
    before any of this saw them -- which is why `Document` opens with
    `newline=""` and this is handed a source that still has its own endings in
    it. Both halves had to go for either to be worth fixing.

    So the terminator is kept beside its line. The last one is `""` where the
    file ends without a newline, which is preserved rather than tidied: what the
    editor promises is the bytes it did not touch.
    """

    lines, ends = [], []

    for kept in source.splitlines(keepends=True):
        # One line in, one line out: `kept` is what `splitlines` calls a line,
        # so splitting it again can only give that line back, and the rest of
        # it is the separator -- whichever of the seven `splitlines` honours.
        (bare,) = kept.splitlines()

        lines.append(bare)
        ends.append(kept[len(bare):])

    return lines, ends


def _dominant(ends):
    """The terminator a line written from scratch takes: the file's own."""

    counted = Counter(end for end in ends if end)

    return counted.most_common(1)[0][0] if counted else "\n"


@dataclass
class Block:
    """The lines one structure occupies: `start` inclusive, `end` exclusive."""

    start: int
    end: int


def _blocks(lines):
    """
    Where each structure's own lines are, in file order.

    A block runs from its `structure` line to its last line that says anything.
    The blanks and comments between it and the next record belong to neither and
    stay where they are: the comment block above `structure F0058` in
    `curation.gstruct` explains F0058, and handing it to the structure *before*
    it would move somebody's reasoning onto a different fault the first time
    either was edited.
    """

    out = []
    opens = [i for i, line in enumerate(lines) if _opens(line) is not None]

    for n, start in enumerate(opens):
        if _opens(lines[start]) != "structure":
            continue

        stop = opens[n + 1] if n + 1 < len(opens) else len(lines)

        while stop > start + 1 and not _content(lines[stop - 1]):
            stop -= 1

        out.append(Block(start, stop))

    return out


class Document:
    """
    A `.gstruct` held as the text it is, with the model parsed beside it.

    The editor writes to the file it opened, which is a thing nothing else in
    this project does, and the reason it can is that it never rewrites the file
    -- it replaces the lines of the one structure that was edited and leaves
    every other byte alone.

    **That is not fastidiousness, it is what `dumps` costs.** Run
    `curation.gstruct` through load and dump and the ten lines of comment in it
    are gone: the block saying *why* those five thrusts are `exposed` --
    that they were walked, that the facets grown from the DTM agree within three
    to nine degrees -- is gone, because comments are not in the model and a
    writer can only write what it has. A tool whose Save deletes the geologist's
    reasoning is not an editor. Splicing one block also means a plane typed as
    `140.5/31` stays `140.5/31`, where a round trip would round it to `140/31`:
    `Plane.__str__` writes whole degrees, and text that is never re-serialised
    cannot lose anything at all.

    The model is parsed beside the text and not from it, because the text is
    what the file says and the model is what it means. Editing goes through
    `loads` -- the block is re-read under the file's own header -- so a block
    that would not parse is refused with the parser's own words instead of
    being written and discovered tomorrow.
    """

    def __init__(self, path):
        self.path = Path(path)

        # `newline=""` and not `read_text`, which is where the terminators were
        # being lost before they got anywhere near the writing: text mode
        # translates CRLF to LF on the way in, so a file's own line endings are
        # not something a later join can put back -- they were never read. The
        # same argument holds in the other direction and `save` says so there.
        with self.path.open(encoding="utf-8", newline="") as handle:
            source = handle.read()

        self.lines, self.ends = _split_keeping_ends(source)
        self.newline = _dominant(self.ends)
        self.dataset = module().loads(source)
        self.blocks = _blocks(self.lines)
        self.dirty = False

        if len(self.blocks) != len(self.dataset.structures):
            # Not reachable through any file the parser accepts, and checked
            # anyway: everything here indexes the model and the text by the same
            # number, so the two disagreeing about how many structures there are
            # is the one failure that would edit the wrong fault silently.
            raise ValueError(
                f"{len(self.blocks)} structure block(s) in the text and "
                f"{len(self.dataset.structures)} in the model: refusing to "
                f"index one by the other"
            )

    # -- one block --------------------------------------------------------

    def text_of(self, index):
        """The lines of one structure, exactly as they are in the file."""

        block = self.blocks[index]

        return "\n".join(self.lines[block.start:block.end])

    def replace(self, index, text):
        """
        One structure's block as this text, once it parses. Raises ValueError.

        Nothing is touched until everything has agreed: the text is parsed, the
        result is counted, the splice is made on a copy and the copy is scanned
        again. A block refused here leaves the document exactly as it was, which
        is what lets the panel show the error and keep the text on screen for
        the mistake to be fixed in.
        """

        gstruct = module()

        first = next((line for line in text.splitlines() if _content(line)), None)

        if first is None:
            # There is no delete here, and that is the format's answer rather
            # than a missing button: a contact that does not hold is said not to
            # hold -- `span use * * rejected`, with the reason on it -- which
            # leaves the geometry and the grounds in the file. Removing the lines
            # would leave neither, and tomorrow nobody could tell a fault that
            # was rejected from one that was never mapped.
            raise ValueError(
                "a block cannot be emptied: a structure that does not hold is "
                "said so with `span use * * rejected reason=...`, which keeps "
                "both the geometry and the grounds"
            )

        if not first[:1].strip():
            raise ValueError(
                "a block starts in column one: indented, its first line reads as "
                "a continuation of the record above it"
            )

        if _opens(first) != "structure":
            raise ValueError(
                f"a block starts with a `structure` line, and this one starts "
                f"with {first.strip().partition(' ')[0]!r}"
            )

        try:
            parsed = gstruct.loads(self._header() + text)
        except Exception as err:
            raise ValueError(f"{type(err).__name__}: {err}") from err

        if len(parsed.structures) != 1:
            raise ValueError(
                f"one block declares one structure, and this declares "
                f"{len(parsed.structures)}"
            )

        if parsed.observations:
            raise ValueError(
                "an `observation` belongs to the file rather than to a "
                "structure -- it attaches to no path, which is what it is for -- "
                "so it cannot be written inside a block"
            )

        block = self.blocks[index]
        fresh = text.splitlines()
        candidate = list(self.lines)
        candidate[block.start:block.end] = fresh
        blocks = _blocks(candidate)

        if len(blocks) != len(self.blocks):
            raise ValueError(
                f"the edited text reads as {len(blocks) - len(self.blocks) + 1} "
                f"block(s) where the file expects one"
            )

        # The lines that went in take this file's terminator, and the file's
        # last line keeps the one it had: whether a file ends with a newline is
        # a property of the file, not of whichever block happens to be last in
        # it. Editing any other block does not reach the tail, so the
        # restoration is only doing anything in the case where it has to.
        ends = list(self.ends)
        tail = ends[-1] if ends else ""
        ends[block.start:block.end] = [self.newline] * len(fresh)

        if ends:
            ends[-1] = tail

        self.lines = candidate
        self.ends = ends
        self.blocks = blocks
        self.dataset.structures[index] = parsed.structures[0]
        self.dirty = True

        return parsed.structures[0]

    def _header(self):
        """
        The two file-level lines put back in front of a block, so that `loads`
        reads a file and not a fragment: the version the file declares, and its
        CRS.

        Neither of them gates what is in the block. `_version` refuses a file
        *ahead* of the library and nothing else, and a file that has already
        loaded cannot be ahead of it; no line is read differently for the
        version written over it. So the `span use` the `+ span` button writes --
        which is 0.2 -- goes into a file that says `gstruct 0.1` without a word,
        and that is how either curation in the AOI can be edited at all, since
        both of them declare 0.1. The upgrade is quiet in the direction nobody
        would guess: the file's contents move and its declaration does not, and
        a reader that really is 0.1 would then draw a stretch the curator had
        rejected -- which is the thing the version was raised to prevent.

        The declaration travels anyway because it is the file's and not the
        library's, and a block belongs to the file it came out of: if a
        construct is ever gated by version, it has to be gated on what this
        file says. The CRS travels for the same reason -- a block carries
        coordinates and nothing else in it says which projection they are in --
        though today nothing in the parse asks.
        """

        head = [f"gstruct {self.dataset.meta.get('version', module().VERSION)}"]

        if self.dataset.crs:
            head.append(f"crs {self.dataset.crs}")

        return "\n".join(head) + "\n\n"

    # -- the whole of it --------------------------------------------------

    def text(self):
        """The file as it would be written out, terminators and all."""

        return "".join(line + end for line, end in zip(self.lines, self.ends))

    def save(self, path=None):
        """Writes it, and takes the path written to as its own from then on."""

        target = self.path if path is None else Path(path)

        # `newline=""` again, and this half of it has never been exercised:
        # text mode turns every `"\n"` into `os.linesep`, which is `"\n"` here
        # and `"\r\n"` on Windows. Written through `write_text` there, a file
        # read as LF would be saved as CRLF throughout -- the same whole-file
        # rewrite as before, in the other direction and on the other platform.
        with target.open("w", encoding="utf-8", newline="") as handle:
            handle.write(self.text())

        self.path = target
        self.dirty = False

        return target


def nearest_structure(dataset, x, y, within=None):
    """
    The structure passing closest to a point, as `(index, s, distance)`.

    What a click on the map means. The box is tested before the geometry and the
    limit shrinks as better candidates turn up, so most traces are four
    comparisons rather than a loop over their segments -- on 393 faults the whole
    thing is under a millisecond, and the arrangement is what would keep a click
    usable on a sheet with twenty-four thousand contacts on it.

    The progressive comes back with the index because the click is worth more
    than the selection: where along the trace it landed is what a picked anchor
    is written from, and it has already been computed here.
    """

    gstruct = module()

    limit = float("inf") if within is None else float(within)
    best = None

    for index, structure in enumerate(dataset.structures):
        path = structure.path

        if len(path) < 2:
            continue

        xs = [px for px, _ in path]
        ys = [py for _, py in path]

        if (
            x < min(xs) - limit or x > max(xs) + limit
            or y < min(ys) - limit or y > max(ys) + limit
        ):
            continue

        s, distance, _ = gstruct.project(path, (float(x), float(y)))

        if distance <= limit:
            best, limit = (index, s, distance), distance

    return best


def place_on(path, x, y):
    """
    A point against one path, as `(progressive, distance from it)`.

    `nearest_structure` for the case where which structure is not in question --
    an anchor being picked for a block that is already open, where the nearest
    trace is not the one the anchor is about to be written into.
    """

    s, distance, _ = module().project(path, (float(x), float(y)))

    return s, distance


def point_on(path, s):
    """
    The ground at a progressive, snapped to the path.

    What an anchor picked off the map is written from. Snapped rather than taken
    where the mouse was, because that is what the anchor means: `resolve`
    projects it onto the path to get `s` back, so a point written a hundred
    metres off the line would read back as the same progressive while saying
    something false about where anybody stood.
    """

    return module().point_at(path, float(s))


def stretch(path, s0, s1):
    """
    The ground between two progressives, as the points that draw it.

    The path's own vertices between the ends, not a resampling of them: a
    stretch drawn through sampled points would round off exactly the bends that
    made somebody reject it.
    """

    gstruct = module()

    out = [gstruct.point_at(path, float(s0))]
    walked = 0.0

    for length, vertex in zip(gstruct.seg_lengths(path), path[1:]):
        walked += length

        if s0 < walked < s1:
            out.append(vertex)

    out.append(gstruct.point_at(path, float(s1)))

    return out


def provenance_of(structure, samples=400, max_gap=DEFAULT_MAX_GAP):
    """
    What holds along a whole trace, sampled: `(s, plane, said, kind)` each.

    `attitude_at` answers at one place, and this asks it everywhere -- which is
    the one thing reading the file cannot tell you. Precedence is a computation
    over several lines at once: a refusal beats a measurement, a measurement
    within `max_gap` beats a fit, a fit beats a measurement further off. Which
    of those lines is winning at a given metre, and where along the trace the
    winner changes, is not something you can see by looking at them.

    `kind` is `said` up to the colon, which is one of `PROVENANCE`. The rest of
    `said` is the detail -- which station, how far, on what grounds it was
    refused -- and it belongs to the one place rather than to the stretch.
    """

    length = structure.length

    if length <= 0.0 or samples < 2:
        return []

    out = []

    for n in range(samples):
        s = length * n / (samples - 1)
        plane, said = structure.attitude_at(s, max_gap)

        out.append((s, plane, said, said.split(":")[0]))

    return out


# -- a dataset as records -------------------------------------------------


@dataclass
class LooseAttitude:
    """
    A field measurement that never found a trace to belong to.

    The format keeps these deliberately -- rule three, a datum that does not
    attach is not thrown away -- and they arrive here with nowhere to go, since
    a `TraceRecord` is a plane *and its outcrop trace*. Carried rather than
    dropped, and counted out loud, so that the twenty-three measurements of a
    survey do not become twenty-two without anybody saying so.
    """

    ident: str
    x: float
    y: float
    plane: object
    attrs: dict = field(default_factory=dict)


@dataclass
class Reading:
    """What a dataset turned into, and what did not survive the turning."""

    records: list = field(default_factory=list)
    loose: list = field(default_factory=list)
    dropped: dict = field(default_factory=dict)
    outside: int = 0

    # Geometries, not records. One structure carrying three planes is three
    # records on one trace, and counting the trace three times would report a
    # fault layer half as large again as the map it was drawn from.
    traces: int = 0


def records_of(dataset, bounds=None):
    """
    A dataset as `TraceRecord`s: one per plane, on the trace it was read along.

    The same shape `export_gsurf.py` writes into a GeoPackage, built here from
    the file itself. One structure gives as many records as it carries planes,
    which is what the panel already expects -- a trace crossed by the section
    twice, carrying two attitudes, is two rows and two ticks.

    Three decisions worth stating, all of them taken from the format rather than
    invented:

    `kind` is the category, not `ident`. A category is what a legend entry is
    made of, and 393 structures under their own names would be 393 entries for
    one map. The ident goes into the attributes, where `apply_to` looks for it.

    An **attitude** gets no span. It is a measurement at a point, and how far
    along the fault it speaks for is a judgement about that fault -- which is
    the judgement the reach column exists to make, and writing a default here
    would answer it before anybody was asked. A **fit** does get one: it was
    computed over an interval and it holds over that interval, which is a fact
    and not an opinion.

    A structure carrying no plane at all still becomes a record. That is a
    mapped contact whose attitude was never written down, and it is precisely
    what `fit_records` reads a plane off; dropping it would lose 350 of the 393
    traces of a fault layer on the grounds that nobody had measured them yet.
    """

    from geogst.core.geometries.shapes.lines import Ln

    from .attitudes import TraceRecord

    reading = Reading()
    dropped = defaultdict(int)

    for structure in dataset.structures:
        coords = np.asarray(structure.path, dtype=float)

        if coords.ndim != 2 or len(coords) < 2:
            dropped["with no path"] += 1
            continue

        if bounds is not None and not _overlaps(coords, bounds):
            reading.outside += 1
            continue

        lines = [Ln(coords[:, :2])]
        length = float(lines[0].length_2d())
        reading.traces += 1

        # The structure's own attributes first, so that anything the event says
        # about itself wins: `raw` on an attitude is the line it was read from,
        # and `raw.type` on the structure is a different claim about a different
        # thing. The ident and the kind go on last because they are the identity
        # rather than a note about it.
        common = dict(structure.attrs)
        common.update(ident=structure.ident, kind=structure.kind)

        if structure.label:
            common["label"] = structure.label

        planes = 0

        for attitude in structure.attitudes:
            if attitude.plane is None:
                dropped["measurements with no plane"] += 1
                continue

            planes += 1
            place = length / 2.0 if attitude.s is None else float(attitude.s)
            axes = _axes_at(structure, place)

            reading.records.append(
                TraceRecord(
                    category=structure.kind or "unknown",
                    plane=geogst_plane(attitude.plane),
                    lines=lines,
                    length=length,
                    anchor=None if attitude.s is None else float(attitude.s),
                    span=None,
                    enabled=_accepted(axes),
                    attrs={
                        **common,
                        **axes,
                        **dict(attitude.attrs),
                        "src": attitude.attrs.get("src", "field"),
                        **({} if attitude.offset is None
                           else {"off_m": round(float(attitude.offset), 1)}),
                    },
                )
            )

        for fit in structure.fits:
            if fit.plane is None:
                dropped["fits with no plane"] += 1
                continue

            planes += 1
            span = (
                None if fit.s0 is None or fit.s1 is None
                else (float(fit.s0), float(fit.s1))
            )
            verdict = fit.attrs.get("verdict", "")
            place = length / 2.0 if span is None else (span[0] + span[1]) / 2.0
            axes = _axes_at(structure, place)

            reading.records.append(
                TraceRecord(
                    category=structure.kind or "unknown",
                    plane=geogst_plane(fit.plane),
                    lines=lines,
                    length=length,
                    anchor=None,
                    span=span,
                    # Off where the trace never constrained the dip, and left on
                    # where it did: the rule is `attitude_at`'s own, and it
                    # travels with the record instead of being reapplied here.
                    # A curator's `use` overrules it either way -- the verdict is
                    # a default, and somebody who has been to the outcrop knows
                    # something the singular values do not.
                    enabled=_accepted(axes, otherwise=verdict != UNCONSTRAINED),
                    attrs={
                        **common,
                        **axes,
                        **dict(fit.attrs),
                        "src": fit.attrs.get("from", "fit"),
                        "fitted": True,
                    },
                )
            )

        if planes == 0:
            axes = _axes_at(structure, length / 2.0)

            reading.records.append(
                TraceRecord(
                    category=structure.kind or "unknown",
                    plane=None,
                    lines=lines,
                    length=length,
                    enabled=_accepted(axes),
                    attrs={**common, **axes},
                )
            )

    for observation in dataset.observations:
        if observation.plane is None or observation.anchor is None:
            dropped["observations with no plane or no place"] += 1
            continue

        x, y = observation.anchor

        if bounds is not None and not _inside(x, y, bounds):
            dropped["observations outside the map"] += 1
            continue

        reading.loose.append(
            LooseAttitude(
                ident=observation.ident,
                x=float(x),
                y=float(y),
                plane=geogst_plane(observation.plane),
                attrs=dict(observation.attrs),
            )
        )

    reading.dropped = dict(dropped)

    return reading


def _axes_at(structure, s):
    """
    What the format's axes say at one place along the trace.

    `span_at` and not the spans directly, because the rule that the last span
    covering a progressive wins is how a local correction is written -- by
    adding a line, never by editing one -- and reading the list any other way
    would quietly prefer the general statement to the correction over it.

    The span rather than its value, for `use` alone, so that `reason=` travels
    with the refusal. A row that arrives switched off with no way to ask why is
    a row somebody will switch back on.
    """

    out = {}

    for axis in AXES:
        span = structure.span_at(axis, s)
        out[axis] = UNSAID if span is None else span.value

        if axis == USE and span is not None and span.attrs.get("reason"):
            out["use.reason"] = span.attrs["reason"]

    return out


def _accepted(axes, otherwise=True):
    """
    Whether a record arrives in the section, once the `use` axis has spoken.

    Where it says nothing -- which is the normal case, and is not a refusal --
    the automatic rule decides, and that is what `otherwise` carries: for a fit
    it is its own verdict, for a measurement it is yes. Where it does speak it
    wins outright, both ways: a curator who writes `accepted` over a stretch a
    straight trace had switched off is overruling the gate on purpose, and a
    gate that could not be overruled would be a rule rather than a default.
    """

    said = axes.get(USE, UNSAID)

    if said == REJECTED:
        return False

    if said == ACCEPTED:
        return True

    return otherwise


def _overlaps(coords, bounds):
    """
    Whether a trace's box meets the frame's box.

    The box and not the geometry, which is the one place this reader is looser
    than the GeoPackage one, where shapely answers exactly. A trace that bends
    around a corner of the frame without entering it is offered here anyway. The
    slack is on the side of keeping, and both readers keep a trace whole or drop
    it whole, so what is at stake is whether a contact at the edge appears in
    the panel at all -- and appearing loses nothing.
    """

    left, bottom, right, top = bounds

    return (
        coords[:, 0].min() <= right and coords[:, 0].max() >= left
        and coords[:, 1].min() <= top and coords[:, 1].max() >= bottom
    )


def _inside(x, y, bounds):
    left, bottom, right, top = bounds

    return left <= x <= right and bottom <= y <= top


# -- a curation over records already open ---------------------------------


@dataclass
class Applied:
    """What a curation turned out to be about, once laid over the records."""

    named: int = 0          # structures the file speaks of
    matched: int = 0        # of those, found among the records
    assertions: int = 0     # claims that landed on something
    added: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    nowhere: int = 0        # claims about a structure that is here, landing off it
    ignored: dict = field(default_factory=dict)

    def summary(self):
        text = (
            f"{self.assertions} assertion(s) over {self.matched} of {self.named} "
            f"structure(s)"
        )

        if self.added:
            text += f", {len(self.added)} new record(s)"

        if self.missing:
            shown = ", ".join(self.missing[:5])
            more = "" if len(self.missing) <= 5 else f" and {len(self.missing) - 5} more"
            text += f"; {len(self.missing)} not among the records ({shown}{more})"

        if self.nowhere:
            text += f"; {self.nowhere} covering no record of their own structure"

        if self.ignored:
            detail = ", ".join(f"{axis} x{count}" for axis, count in self.ignored.items())
            text += f"; axes this tool does not act on: {detail}"

        return text


def ident_of(record):
    """The name a curation would have to call this record by, or None."""

    for key in IDENT_KEYS:
        said = record.attrs.get(key)

        if said is not None and str(said).strip():
            return str(said)

    return None


def apply_to(records, dataset):
    """
    A curation laid over records already open, as `(records, Applied)`.

    The returned list is the one that went in plus whatever the file added to
    it. Nothing is written back to the layer the records came from -- that is
    the bargain the panel already strikes, and this is the other half of it:
    what was saved as an assertion comes back as an assertion.

    **Matched by ident**, which is why `records_of` puts it in the attributes,
    and why `export_gsurf.py`'s `code` column is looked at too. A file naming
    structures nothing here has heard of applies cleanly to nothing, and says
    so; it is the silent version of that which would be the bug.

    **A plane in a curation adds a record, it does not overwrite one.** That is
    the format's own answer -- `export_geology.py` merges a curation by
    extending the lists, and `attitude_at` sorts out precedence afterwards by
    distance -- and it is also the honest one: a measurement somebody made at an
    outcrop does not delete the one the survey recorded, it stands beside it.

    **A span applies where it covers the record's place.** A record occupies one
    point along the trace and an attribute carries one value, so the rule is
    `value_at`'s: the interval that covers the place decides. A claim that
    covers no record of its own structure is counted as landing nowhere rather
    than dropped, because a curation whose intervals miss is a file that looks
    applied and is not.

    **`use` is the one axis acted on rather than carried.** `rejected` takes a
    record out of the section and `accepted` puts it back, both of them against
    whatever the automatic rule had decided -- a curator overruling a gate is
    the point of a curation, not an accident to be guarded against. It reaches a
    plane the same file adds, too, since an axis is about a stretch of ground
    and not about the rows that happened to be open when it was read.
    """

    from .attitudes import TraceRecord

    by_ident = defaultdict(list)

    for record in records:
        ident = ident_of(record)

        if ident is not None:
            by_ident[ident].append(record)

    applied = Applied(named=len(dataset.structures))
    ignored = defaultdict(int)

    for structure in dataset.structures:
        found = by_ident.get(structure.ident)

        if not found:
            applied.missing.append(structure.ident)
            continue

        applied.matched += 1

        lines, length = found[0].lines, found[0].length
        category = found[0].category

        # What is true of the structure rather than of one plane read on it.
        # A record added below inherits this and nothing else: `station`,
        # `verdict` and the fit's diagnostics belong to the measurement they
        # came with, and copying them onto a different measurement would be
        # handing it a provenance it has not got.
        identity = {
            key: value for key, value in found[0].attrs.items()
            if key in IDENT_KEYS or key in AXES or key in ("kind", "label")
        }

        # Resolved once, because a plane the file *adds* sits somewhere too and
        # has to answer to a refusal covering that somewhere. An axis is about a
        # stretch of ground, not about the rows that happened to be open when it
        # was read.
        refusals = [
            (*_interval(lines, length, span), span.value)
            for span in structure.spans if span.axis == USE
        ]

        for span in structure.spans:
            if span.axis not in AXES:
                ignored[span.axis] += 1
                continue

            s0, s1 = _interval(lines, length, span)
            landed = 0

            for record in found:
                if s0 <= _place(record) <= s1:
                    record.attrs[span.axis] = span.value
                    landed += 1

                    if span.axis == USE:
                        record.enabled = _accepted(
                            {USE: span.value}, otherwise=record.enabled
                        )

                        if span.attrs.get("reason"):
                            record.attrs["use.reason"] = span.attrs["reason"]

            if landed:
                applied.assertions += 1
            else:
                applied.nowhere += 1

        for attitude in structure.attitudes:
            if attitude.plane is None:
                continue

            at = None if attitude.anchor is None else _project(lines, attitude.anchor)
            said = _said_at(refusals, length / 2.0 if at is None else at[0])

            applied.added.append(
                TraceRecord(
                    category=category,
                    plane=geogst_plane(attitude.plane),
                    lines=lines,
                    length=length,
                    anchor=None if at is None else at[0],
                    span=None,
                    enabled=_accepted({USE: said}, otherwise=_accepted(identity)),
                    attrs={
                        **identity,
                        **({} if said == UNSAID else {USE: said}),
                        **dict(attitude.attrs),
                        "src": attitude.attrs.get("src", "field"),
                        **({} if at is None else {"off_m": round(at[1], 1)}),
                    },
                )
            )
            applied.assertions += 1

        for fit in structure.fits:
            if fit.plane is None:
                continue

            s0, s1 = _interval(lines, length, fit)
            verdict = fit.attrs.get("verdict", "")
            said = _said_at(refusals, (s0 + s1) / 2.0)

            applied.added.append(
                TraceRecord(
                    category=category,
                    plane=geogst_plane(fit.plane),
                    lines=lines,
                    length=length,
                    anchor=None,
                    span=(s0, s1),
                    enabled=_accepted(
                        {USE: said},
                        otherwise=verdict != UNCONSTRAINED and _accepted(identity),
                    ),
                    attrs={
                        **identity,
                        **({} if said == UNSAID else {USE: said}),
                        **dict(fit.attrs),
                        "src": fit.attrs.get("from", "fit"),
                        "fitted": True,
                    },
                )
            )
            applied.assertions += 1

    applied.ignored = dict(ignored)

    return list(records) + applied.added, applied


# -- records as a curation ------------------------------------------------

# What a fit says about itself, carried out in the order FORMAT.md lists it and
# skipped where the record has not got it. A fit read in from a `.gstruct` goes
# back out with its own diagnostics intact; one computed here reports `window`
# and `span_verdict`, which are the two numbers it actually has. Nothing on this
# list is invented for the occasion, and `nvert` in particular is never written
# by gSurf: the format reserves it for digitised vertices, and this tool samples
# the DEM, which is a different count of a different thing.
FIT_KEYS = (
    "window", "span_verdict", "verdict", "nvert", "dof", "s2", "s3", "eps",
    "snr", "flat", "jack", "plan", "drape", "seed", "licence", "span", "dem",
    "sampled", "station",
)


def curation_of(records, half_span, crs=None, source=None):
    """
    The records as a curation, as `(text, report)`: what was decided, in gstruct.

    Written with the format's own writer rather than as text, which is the whole
    reason this lives here and not in the panel. A fragment built by hand has to
    get the quoting, the ordering and the grammar right on its own and can be
    wrong in ways that only show up when somebody tries to read it back; a
    `Dataset` handed to `dumps` cannot be unparseable, because `dumps` is the
    other half of the parser.

    **What comes out is two kinds of line, and the file says which is which.**
    A `span use ... rejected` is a human decision. A `fit` is a derivative, and
    carries `from=` saying what derived it -- a window swept over a trace, or a
    reach somebody set by hand. The file this replaces declared every line in
    itself a human assertion, and so had to drop every fit on the floor to stay
    honest; declaring the two apart is what lets the fits be written at all.

    **Only what was decided or computed here.** A refusal the record's own
    attributes already account for is not restated, and neither is a fit that
    came in with the file. Both would be true, and both would be wrong to write:
    a curation restating its source is not laid over it, it is a second copy of
    it, and applied back it would double every fit it had just read.

    **A record is named by its ident, not by its category.** A category is a
    legend entry and a curation has to name one structure, so a record the layer
    gives no ident to cannot be spoken about: those are counted in the report,
    not guessed at. Records sharing an ident share one `structure` block.

    **The interval on a refusal is a locator.** The rule elsewhere is that a
    reach nobody set is not written down -- writing the tool's default out as an
    assertion would put words in the geologist's mouth. A refusal is different:
    the claim is `rejected` and the interval only says where, so the record's own
    extent is used, default half-span and all. The granularity is the format's:
    rejecting a stretch rejects what sits on it, and where two planes sit at one
    place it cannot tell them apart.
    """

    gstruct = module()

    dataset = gstruct.Dataset(
        crs=crs or "",
        meta={
            "version": gstruct.VERSION,
            "project": "Curatela da gSurf: portata e giaciture decise in sezione",
            "note": "Si applica sopra il dataset sorgente, che non viene toccato. "
                    "Le righe `span use` sono decisioni umane; i `fit` sono "
                    "derivati e portano in `from=` da dove vengono.",
        },
    )

    if source:
        dataset.meta["source"] = str(source)

    report = dict(structures=0, refusals=0, fits=0, unnamed=0, planeless=0)
    structures = {}

    for record in records:
        ends = record.reach_endpoints(half_span)
        start, end = (None, None) if ends is None else ends
        lines = []

        if record.enabled != _accounted(record):
            lines.append(gstruct.Span(
                axis=USE, value=ACCEPTED if record.enabled else REJECTED,
                start=start, end=end, attrs={"src": "gsurf"},
            ))
            report["refusals"] += 1

        fit = _fit_of(gstruct, record, start, end)

        if fit is not None:
            lines.append(fit)
            report["fits"] += 1
        elif record.span is not None and record.plane is None:
            # A reach on a trace nobody has read an attitude off. There is no
            # fit to write, because a fit is a plane over an interval and there
            # is no plane; and the record draws no tick in a section either, so
            # what was decided has nothing yet to be a decision about.
            report["planeless"] += 1

        if not lines:
            continue

        # Looked up here and not at the top: a record with nothing to say about
        # it is not a record that could not be named, and counting it as one
        # would report a whole unnamed layer as a drawer full of lost decisions.
        ident = ident_of(record)

        if ident is None:
            report["unnamed"] += len(lines)
            continue

        if ident not in structures:
            # No `kind`: a curation names structures, it does not describe them
            # again. The kind is the source's fact and is already in the file
            # this one is laid over.
            structures[ident] = gstruct.Structure(ident=ident)
            dataset.structures.append(structures[ident])
            report["structures"] += 1

        for line in lines:
            if isinstance(line, gstruct.Fit):
                structures[ident].fits.append(line)
            else:
                structures[ident].spans.append(line)

    return gstruct.dumps(dataset), report


def _accounted(record):
    """
    Whether a record would be in the section on what it already carries.

    The writer's half of `records_of`, and the reason `use` has two values
    rather than one. A record is only written about where the panel and its own
    attributes disagree: switched off with nothing to account for it is a
    refusal somebody made here, and switched *on* against a `rejected` axis or a
    straight-trace verdict is somebody overruling, which is just as much a
    decision and would be lost if only refusals were written.

    Without this the file would restate every refusal it had just read, and the
    ones the gate made on its own would come back signed by a geologist who
    never made them.
    """

    verdict = record.attrs.get("verdict", "")

    return _accepted(record.attrs, otherwise=verdict != UNCONSTRAINED)


def _fit_of(gstruct, record, start, end):
    """
    A record as a `fit`, or None where there is nothing derived to write.

    Two ways a plane comes to hold over an interval rather than at a point, and
    both of them are derivations, which is what `fit` is for. One is computed --
    a window swept along the trace until it holds -- and carries the diagnostics
    of that computation. The other is a reach somebody set in the table: a
    measurement taken at one outcrop, extended along the fault by judgement.
    `from=` is what tells them apart afterwards, and it is not decoration --
    `attitude_at` gives a measurement precedence over a fit near the outcrop and
    falls back to the fit further along, which is exactly what a reach means.

    A reach still on the tool's default is not one: nobody decided 250 m, so
    nobody says so. Neither is a fit that arrived in the file this curation will
    be laid over: the header says *si applica sopra il dataset sorgente*, and a
    file that restates its source is not laid over it -- applied back, it would
    add every one of those fits a second time, which is our own round trip
    corrupting the thing it was supposed to preserve. `window` is what tells
    them apart, and it is a fact about the fit rather than a flag set for this
    purpose: only gSurf's own fitter reads a plane over a swept window, so only
    its fits carry the metres they were read over.

    What this does not carry is a plane *edited* in the table on a record that
    came with one. That is neither a computation nor a reach, and there is no
    way here to tell an edited value from the one that was read -- it is the
    editor's to write, and it is named here so that it is a gap somebody chose
    rather than one nobody noticed.
    """

    if record.plane is None:
        return None

    ours = record.attrs.get("window") is not None

    if not ours and (record.span is None or record.attrs.get("fitted")):
        return None

    attrs = {"from": str(record.attrs.get("src") or "gsurf") if ours else "reach"}

    if not ours:
        attrs["src"] = "gsurf"

    for key in FIT_KEYS:
        said = record.attrs.get(key)

        if said is not None and str(said) != "":
            attrs[key] = str(said)

    return gstruct.Fit(
        plane=gstruct_plane(record.plane), start=start, end=end, attrs=attrs
    )


def _place(record):
    """
    Where along its trace a record sits, in metres.

    `anchor_point` in progressives, and deliberately the same three cases: the
    anchor where there is one, the middle of the stretch a fit held on where
    there is not, and the middle of the trace for a record that speaks for all
    of it.
    """

    if record.anchor is not None:
        return float(record.anchor)

    if record.span is not None:
        return (record.span[0] + record.span[1]) / 2.0

    return record.length / 2.0


def _said_at(intervals, place):
    """
    The last of a set of `(s0, s1, value)` covering a place, or `unknown`.

    `Structure.value_at`'s rule, applied to intervals already resolved against
    the records' own trace. The last one wins, which is how a correction is
    written in this format: by adding a line, never by editing one.
    """

    said = UNSAID

    for s0, s1, value in intervals:
        if s0 <= place <= s1:
            said = value

    return said


def _interval(lines, length, span):
    """
    A curation's interval in metres along the records' own trace.

    From the anchors every time, never from `s0`/`s1`. A curation carries no
    path, so those are unresolved; and where a file does carry one they were
    resolved against *that* geometry, which is not the geometry the section is
    being drawn on. `*` still means the end of the path -- of this path.
    """

    s0 = 0.0 if span.start is None else _project(lines, span.start)[0]
    s1 = length if span.end is None else _project(lines, span.end)[0]

    return (s0, s1) if s0 <= s1 else (s1, s0)


def _project(lines, point):
    """
    A point against a record's trace, as (progressive, distance from it).

    gstruct's own projection, part by part, with the progressive running on
    across the parts -- which is how `clip_to_span` and `point_at` read one, and
    a trace interrupted by cover has to be measured the same way by all three.
    Run over the parts joined end to end instead, a point near the gap could
    project onto the join, which is a segment that does not exist on the ground.
    """

    import gstruct

    best, walked = None, 0.0

    for line in lines:
        coords = line.coords

        if coords is None or len(coords) < 2:
            continue

        path = [(float(x), float(y)) for x, y in coords[:, :2]]
        s, offset, _ = gstruct.project(path, (float(point[0]), float(point[1])))

        if best is None or offset < best[1]:
            best = (walked + s, offset)

        walked += gstruct.path_length(path)

    return best if best is not None else (0.0, float("inf"))
