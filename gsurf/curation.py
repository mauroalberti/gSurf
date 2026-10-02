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

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple

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

    def reading_of(self, text):
        """
        What one block would mean if it were applied, as a `Structure`, or None.

        **`replace` without the replacing**, for the windows that have to show
        the consequence of a press before the press. The consequence of moving a
        reading is not where the dot goes, it is what `attitude_at` answers
        along the trace afterwards -- which is a computation over several lines
        at once, and the one thing reading a file cannot tell you.

        Through `loads` under this file's own header rather than by mutating the
        model, and the difference is worth the parse. A moved anchor has to be
        re-projected to be worth anything, and `resolve` is what `loads` ends
        with; a copy of a `Structure` with one field poked would be this tool's
        idea of what the parser does, shown to a curator as the file's answer.
        Here the thing measured is the text that is about to be written.

        None where it would not parse, the refusal itself belonging to the press:
        a window asking what a candidate means while somebody is still typing in
        it gets a half-written line more often than not, and a refusal shown for
        every keystroke is a refusal nobody reads.
        """

        gstruct = module()

        try:
            parsed = gstruct.loads(self._header() + text)
        except Exception:
            return None

        if len(parsed.structures) != 1:
            return None

        return parsed.structures[0]

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


# Where a `span` and a `fit` keep their two ends: `span <axis> <start> <end>
# <value>` and `fit <kind> <start> <end> <plane>` put them in the same pair of
# places counted from the keyword, which is why one reader serves both.
ENDS_AT = slice(2, 4)

# And where each of the three keeps the anchors a click can fill. `ENDS_AT` for
# the two that claim ground, the single anchor for the one that claims a place.
ANCHORS_AT = {"span": ENDS_AT, "fit": ENDS_AT, "attitude": slice(1, 2)}


def anchors_written(line):
    """
    Whether every anchor slot of a line already holds a picked coordinate.

    **Not `interval_of(line) is not None`**, and the difference is the whole of
    what this is for: `*` reads as an end of the path, so a template straight
    off its button has an interval -- the whole trace -- and would be
    indistinguishable here from two anchors enclosing a stretch somebody chose.
    True only of `@x,y` in every slot, which is to say of a claim that has been
    decided.

    Two questions turn on that one fact. Whether a written plane can be applied
    as it stands, and whether another picked anchor has anywhere to go: a click
    landing on a line whose slots are full writes its coordinate at the caret,
    where `loads` finds a token it has no slot for and drops it -- so the line
    goes on claiming what it claimed and the click leaves no trace anywhere.

    False for anything that is not one of the three, a blank line included.
    """

    tokens = line.split()
    slots = ANCHORS_AT.get(tokens[0] if tokens else "")

    if slots is None or len(tokens) < slots.stop:
        return False

    return all(token.startswith("@") for token in tokens[slots])


def interval_of(line, path):
    """
    The progressives a `span` or `fit` line claims, as written, or None.

    What the line in the box is about to say, before it is applied: an anchor is
    a coordinate and a stretch is a piece of ground, and the box can only show
    the first. Two clicks put two coordinates on a line and nothing anywhere
    says what they enclose.

    **Split on whitespace, which is the wrong way to read this format** --
    values are quoted and a quoted value holds spaces. It is right in these two
    slots and nowhere else, because of the only two things they can hold: `@x,y`
    or `*`, neither of which can be quoted or contain a space. So the tokens up
    to the fourth are the ones `loads` would find, and a slot holding anything
    else is not an end, which comes back None rather than guessed at.

    `*` is read as the format reads it -- the end of the path -- including the
    one a template arrives with. That is not a compromise over an unfilled
    placeholder: a template applied as it stands *does* claim the whole trace,
    and a curator who can see that has been told something true.

    Returned as written and never sorted, because `covers` is `s0 <= s <= s1`
    and a pair the wrong way round covers nothing at all. Sorting them here
    would draw a stretch the file would not honour, which is the one kind of
    wrong a picture can be.
    """

    tokens = line.split()

    if len(tokens) < 4 or tokens[0] not in ("span", "fit") or len(path) < 2:
        return None

    gstruct = module()
    ends = []

    for token, otherwise in zip(tokens[ENDS_AT], (0.0, gstruct.path_length(path))):
        if token == "*":
            ends.append(otherwise)
            continue

        if not token.startswith("@"):
            return None

        try:
            x, y = token[1:].split(",")
            ends.append(place_on(path, float(x), float(y))[0])
        except ValueError:
            return None

    return tuple(ends)


def with_ends_in_order(line, path):
    """
    The same line with its two ends swapped, or None if they need no swapping.

    `interval_of` above is right to hand back what is written and never sort
    it, and this is the other half of that rule rather than a retraction of
    it. Sorting on the way *out* would draw a stretch the file does not
    honour; putting the pair right on the way *in* leaves the picture and the
    file saying the same thing, which is what the refusal to sort was
    protecting.

    So it belongs to the gesture that makes a pair without being asked which
    way round it goes. **Two shift-clicks are that gesture**: the curator is
    aiming at ground, and the direction the trace was digitised in is not on
    the map, is not a property of the fault, and is not something anybody
    should have to hold in their head to click two ends of an outcrop. Picking
    the far one first is not a decision that was made. A pair typed out by hand
    is left exactly as typed -- there the tokens are the curator's own text and
    `claim_said` already says what it covers -- because a box that rewrote what
    was being typed into it would be a different and worse tool.

    A `*` cannot be on the wrong side of an anchor: the format reads the first
    as 0 and the second as the whole length, so either pairing with one is
    ordered by construction and a reversed pair is always two picked anchors.
    This therefore only ever moves what a click wrote.

    A splice and not a rebuild, for `with_plane`'s reason: every byte outside
    the two slots stays where it was, the indent that puts a line inside its
    structure included.
    """

    interval = interval_of(line, path)

    if interval is None or interval[0] <= interval[1]:
        return None

    (first, first_at, first_to), (second, second_at, second_to) = (
        _tokens_of(line)[ENDS_AT]
    )

    return (
        line[:first_at] + second
        + line[first_to:second_at] + first
        + line[second_to:]
    )


# Where a line keeps its plane, counted from the keyword: `fit <kind> <start>
# <end> <plane>` and `attitude <anchor> plane <plane>`. Two keywords and two
# different places, which is why this is a table and `ENDS_AT` is a constant --
# the ends happen to coincide and the planes do not, and writing the second as
# though it were the first is how a plane would land in an anchor's slot.
#
# `span` is deliberately absent. Its fourth slot holds a vocabulary word --
# `rejected`, `inferred`, `exposed` -- and a plane written there would parse:
# `value_at` reads any string, so the file would carry `use 140.5/31` and mean
# nothing by it, silently.
PLANE_AT = {"fit": 4, "attitude": 3}

# What the format's own two keywords are between the anchor and the plane, and
# the reason the table above is not enough on its own: a `fit` that does not say
# `plane` is a fit of something else, and `attitude @x,y 140/31` is a line the
# parser reads as having no plane at all (`pos[1] == "plane"` is its test).
PLANE_KIND_AT = {"fit": 1, "attitude": 2}

# How a plane is written back into a line. One decimal, which is `fits`' own
# choice and for its reason: a number that came off a calculation should not be
# rounded as it is written, or the only thing separating one answer from the
# next is thrown away. Here the calculation is a hand on a dial, and the dial
# steps by a tenth.
PLANE_DECIMALS = 1


def _tokens_of(line):
    """The line's whitespace-separated tokens, each with where it sits."""

    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", line)]


def plane_of(line):
    """
    The plane a `fit` or an `attitude` line carries, as `(dip dir, dip)`, or None.

    For a control that has to show what is already written before it offers to
    change it. Reading a line this way is the same trade `interval_of` makes and
    is sound for the same reason: the slot can hold `140.5/31` or a template's
    `000/00`, neither of which can be quoted or hold a space, so the tokens up
    to the fourth are the ones `loads` would find.

    A slot holding something that is not a plane comes back None rather than
    guessed at -- which includes the half-typed `140/`, since a line being
    written passes through every prefix of itself.
    """

    tokens = [token for token, _, _ in _tokens_of(line)]
    at = PLANE_AT.get(tokens[0] if tokens else None)

    if at is None or len(tokens) <= at:
        return None

    if tokens[PLANE_KIND_AT[tokens[0]]] != "plane":
        return None

    try:
        dip_dir, dip = tokens[at].split("/")

        return float(dip_dir), float(dip)
    except ValueError:
        return None


# The keywords whose first positional token is a single anchor rather than a
# pair: `attitude @x,y ...` and `lineation @x,y ...`. A place and not a stretch,
# which is why `interval_of` answers None for both and this exists beside it.
ANCHOR_AT = {"attitude": 1, "lineation": 1}


def anchor_of(line):
    """
    The coordinate an `attitude` or a `lineation` line is pinned at, or None.

    `interval_of`'s other half, for the lines that claim a place instead of a
    stretch. Same reading and same refusals: `*` is not a coordinate here but
    the format's word for an end of the path, and a line carrying one is a
    record whose anchor nobody has picked yet -- so it comes back None rather
    than as the start of the trace, which would be a place somebody would have
    to be told was not chosen.
    """

    tokens = [token for token, _, _ in _tokens_of(line)]
    at = ANCHOR_AT.get(tokens[0] if tokens else None)

    if at is None or len(tokens) <= at or not tokens[at].startswith("@"):
        return None

    try:
        x, y = tokens[at][1:].split(",")

        return float(x), float(y)
    except ValueError:
        return None


# The keywords a block keeps that are a claim about the trace: something with a
# place on it and a value at that place. `path` and its vertices are the trace
# itself, `structure` is the heading, and a blank or a comment is neither -- all
# of them stay in the block and are written back untouched, they are just not
# rows.
ROW_WORDS = ("span", "attitude", "fit", "lineation")

# Which positional token names the sort of claim: the axis of a `span`
# (`certainty`, `exposure`, `use`), and the `plane` of the other two, which is
# the format's own way of saying what kind of thing is being stated. Separate
# from `PLANE_KIND_AT` although the numbers agree for two of the four: that one
# is a test with a wrong answer to give, this one is a label.
SORT_AT = {"span": 1, "fit": 1, "attitude": 2, "lineation": 2}

# Where a `span` keeps the word it is asserting. `fit` and `attitude` hold a
# plane in the same slot, which `plane_of` reads; `span`'s is a vocabulary word
# and stays a string, because the vocabulary is the file's and not this tool's.
VALUE_AT = {"span": 4}


class Row(NamedTuple):
    """
    One line of a block, read as the claim it makes rather than as text.

    Built from the **raw line** and not from the parsed model, which is the same
    choice `interval_of` makes and for the same reason: what is on screen has to
    be what is in the file, including a line half-written and a line the model
    would not accept. `at` is the line's place in the block, which is what a
    splice needs -- every edit downstream of here rewrites one line by index and
    leaves every other byte where it was.
    """

    at: int
    line: str
    word: str
    sort: Optional[str]
    ends: Optional[Tuple[float, float]]
    place: Optional[float]
    plane: Optional[Tuple[float, float]]
    value: Optional[str]
    attrs: Dict[str, str]
    note: str


def rows_of(text, path):
    """
    The claims a block makes, in the order it makes them, one per line.

    Order is the block's own and is never sorted, because in this format order
    is meaning: `attitude_at` takes the **first** fit that covers a metre, so
    two rows swapped are two different files. Anything that shows these has to
    show them like this -- which is a thing to say out loud, since a table that
    sorts on a header click is the default a toolkit hands you.

    Attributes are split with gstruct's own `_split`, not with whitespace: a
    value can be quoted and hold spaces (`raw.comments.station="Possibly within
    CSC or CTC on CSC"` is in the files), and a reader that broke that would
    show the curator a shorter sentence than the one they wrote. The comment is
    cut the way `loads` cuts it, at the first `#` and without regard for quotes,
    so what the table reads and what the parser reads are the same string --
    a reader kinder than the parser would hide the one difference worth seeing.
    """

    gstruct = module()
    rows = []

    for at, line in enumerate(text.splitlines()):
        if line.lstrip().startswith("#"):
            continue

        bare, hashed, note = line.partition("#")
        tokens = bare.split()

        if not tokens or tokens[0] not in ROW_WORDS:
            continue

        word = tokens[0]

        try:
            pos, attrs = gstruct._split(bare)
        except Exception:
            pos, attrs = tokens, {}

        sort_at = SORT_AT.get(word)
        value_at = VALUE_AT.get(word)

        rows.append(Row(
            at=at,
            line=line,
            word=word,
            sort=pos[sort_at] if sort_at is not None and len(pos) > sort_at else None,
            ends=interval_of(bare, path),
            place=_place_of(bare, path),
            plane=plane_of(bare),
            value=(
                pos[value_at]
                if value_at is not None and len(pos) > value_at else None
            ),
            attrs=attrs,
            note=note.strip() if hashed else "",
        ))

    return rows


# How far a line has to be indented before `loads` reads it as more of the
# record above it rather than as a record of its own. Four, which is the
# parser's own number and not a guess: `loads` tests `indent >= 4` and merges
# that line's `key=value` pairs into the previous record's attributes.
CONTINUES_AT = 4


def continued_at(text, at):
    """
    The lines after `at` that `loads` would fold into the record written on it.

    **A hole in `rows_of`, named here so the windows above can refuse instead of
    discover.** That function reads one physical line at a time, so a `Row`'s
    `attrs` is everything the record carries only while the record is on one
    line. The parser does not work that way: a line indented four or more is
    read as a continuation and its attributes are merged into the record before
    it, and nothing in a `Row` can tell you that happened.

    The direction the loss runs in is what makes it worth a function. A line
    rewritten from its row's attributes does not *drop* the continuations --
    they stay where they were, untouched, which is this tool's whole rule -- and
    `loads` then applies them **over** the rewrite. So an amended `note=` would
    read back as the old one, with the press having visibly happened and
    changed nothing: the failure mode this codebase keeps meeting, a gesture
    that produces a valid line meaning something nobody chose.

    Neither file in the AOI has one. Every claim in `merid_faults.gstruct` and
    in `montealpi_01.gstruct` is on a single line, and the only lines indented
    four or more in either are path vertices. FORMAT.md's own example is
    wrapped, though, which is why this is a check rather than an assumption.

    Blank lines and comments do not end the run, because `loads` skips them
    without letting go of the record they follow -- so a comment between a line
    and its continuation hides nothing. A vertex of a `path` cannot be mistaken
    for one of these: `path` itself is indented two, which ends the run before
    the first vertex is reached.
    """

    lines = text.splitlines()
    out = []

    for n in range(at + 1, len(lines)):
        line = lines[n]

        if not line.strip() or line.lstrip().startswith("#"):
            continue

        if len(line) - len(line.lstrip()) < CONTINUES_AT:
            break

        out.append(n)

    return tuple(out)


def _place_of(line, path):
    """Where along the path a single-anchor line sits, or None."""

    anchor = anchor_of(line)

    if anchor is None or len(path) < 2:
        return None

    return place_on(path, *anchor)[0]


def fits_in(text, path):
    """
    The `fit` lines a block makes, in the order it makes them.

    `rows_of` with one word kept, and it is a function rather than a
    comprehension at each call site because *which* lines are fits is a fact
    about the format: `attitude` and `lineation` also carry a plane and are not
    these -- they are something somebody measured, where a fit is something a
    computation returned. A table that mixed them would be offering one gesture
    over two kinds of claim with different grounds behind them.

    Order is the block's, unsorted, for `rows_of`'s reason: in this format the
    first fit covering a progressive is the one that answers there, so position
    in this list is meaning and not presentation.
    """

    return [row for row in rows_of(text, path) if row.word == "fit"]


# The complement of `fit` among the lines that state an orientation, named for
# what they have in common rather than for the keyword: somebody measured these.
# `lineation` is in here although no file in this project holds one, and that is
# deliberate -- a window that listed only `attitude` would show "1 reading" over
# a block holding two, and a curator cannot be asked to know that the tool reads
# one keyword and not the other. What cannot be read of a lineation comes out
# empty, which is the honest answer and not a hidden row.
READING_WORDS = ("attitude", "lineation")


def readings_in(text, path):
    """
    The lines of a block that state something somebody measured, in order.

    `fits_in`'s sibling, and the division between them is the one that function
    argues for: a fit is what a computation returned and a reading is what a
    compass was pointed at, and one table over both would offer a single gesture
    across two kinds of claim with different grounds behind them. So there are
    two functions, and above them there will be two windows.

    Order is the block's, unsorted, for `rows_of`'s reason -- though unlike fits
    the order of readings does not decide who answers where: `attitude_at` takes
    the *nearest* reading, not the first. Kept unsorted anyway, because the index
    into this list is the index a splice needs, and a table sorted on a header
    click is how that correspondence gets quietly broken.
    """

    return [row for row in rows_of(text, path) if row.word in READING_WORDS]


def reading_said(row):
    """
    One reading in a few words, for a sentence rather than for a table.

    The plane and who measured it, which is what distinguishes one row of a
    readings table from the next -- and `station=` first among the attributes
    because in these files that is the name a geologist knows the measurement
    by. Falls back to the keyword where there is no station, so that a line
    typed by hand and not yet named still says what it is.
    """

    attrs = row.attrs or {}
    named = attrs.get("station") or row.word
    plane = row.plane

    return (
        f"{named} {plane[0]:.0f}/{plane[1]:.0f}"
        if plane is not None
        else f"{named} (no plane this tool can read)"
    )


def from_a_file(row):
    """
    Whether a reading carries the source string an import would write again.

    The whole of the difference between taking a reading off a trace and losing
    it. `raw=` is the format's first rule -- the source string is always kept
    beside the normalised value -- so a line that has one came out of a file that
    still holds it, and detaching it undoes an importer's guess at which
    structure the point belongs to. A line without one was typed here, is the
    only copy there is, and nothing but this file remembers it.
    """

    return any(
        key == "raw" or key.startswith("raw.") for key in (row.attrs or {})
    )


def reading_line(x, y, dip_dir, dip, attrs=None, indent="  "):
    """
    One measurement as the line a file holds, built the format's own way.

    `fits.as_line`'s counterpart, and it borrows that function's argument whole:
    the anchor and the attributes are spelled with `gstruct`'s own `_a` and `_kw`
    rather than with new ones here, because two spellings of a quoted value are
    two spellings that drift.

    **Whole degrees, where a fit gets one decimal**, and the difference is not
    cosmetic. A fit's decimal exists because the number written is a true azimuth
    computed in grid and corrected by `converg=`, and at `.0f` a correction under
    half a degree vanishes while claiming to have happened. A reading typed from
    field notes has no correction to lose: a compass corrected for declination
    already reads in true azimuth, which is the azimuth this format writes, so
    there is nothing for a decimal to carry. Every imported `attitude` in the AOI
    is written this way, and a hand-made one that looked different would look
    like it came from somewhere else.

    Nothing about north goes on the line for the same reason. `north=`/`converg=`
    are what a producer says when it computed a plane from projected coordinates;
    a measurement was not computed, and FORMAT.md's rule -- the number in the
    file is always a true azimuth -- already covers it.
    """

    gstruct = module()

    return (
        f"{indent}attitude {gstruct._a((float(x), float(y)))} plane "
        f"{float(dip_dir):.0f}/{float(dip):.0f}"
        f"{gstruct._kw(attrs or {})}"
    )


def span_line(axis, value, start, end, attrs=None, indent="  "):
    """
    One `span` as the line a file holds, with `*` where an end is the path's own.

    `reading_line`'s sibling, built out of `gstruct._a` and `gstruct._kw` for that
    function's reason: two spellings of a quoted value are two spellings that
    drift, and a `reason=` written by a curator is the attribute in these files
    most likely to hold a space.

    `None` for an end is the format's `*`, which `_a` already writes -- and the
    difference between `* *` and two anchors at the path's two ends is not
    cosmetic. `*` goes on reading as *the end of the path*, so a trace
    redigitised past its old end keeps the span over the whole of it, where two
    computed anchors would stop the claim where the trace used to stop.

    Nothing is validated here. The vocabularies are `gstruct.CERTAINTY`,
    `EXPOSURE` and `USE`, the parser checks none of them, and the place to hold a
    value to one of them is the window that offers the choice -- a writer that
    refused would refuse a file this format allows.
    """

    gstruct = module()

    return (
        f"{indent}span {axis} {gstruct._a(start)} {gstruct._a(end)} {value}"
        f"{gstruct._kw(attrs or {})}"
    )


def detachment_note(row, why, today, indent="  "):
    """
    The comment lines that stand where a detached reading stood.

    Chosen over a new keyword in the format, and the trade is stated rather than
    hidden: a `#` costs nothing in `FORMAT.md`, in the parser or in the
    precedence, survives a Save because `Document` replaces lines and never
    rewrites a file, and is read by the only reader that needs it -- the next
    person to open the block. What it buys is exactly what deleting the line
    silently would throw away: `attitude_at` already answers *assente* where
    nobody measured, and without this a reading somebody decided against reads
    the same as ground nobody ever walked.

    What `attitude_at` will not do is see it. That is the cost of the choice and
    not an oversight: a detachment this file should compute with is a 0.3
    conversation about vocabulary, and this is the line that makes the decision
    legible in the meantime.

    The whole original line goes in, not a summary of it. A note saying `S26
    140/35` would be a reading of the row by this tool, and the row is what is
    being removed -- `off=`, `raw.comments.station=` and the rest are the only
    record that survives of what the importer put there and why it was wrong.
    """

    said = reading_said(row)
    stripped = row.line.strip()
    lines = [
        f"{indent}# {today}: detached {said} -- {why}".rstrip(" -"),
        f"{indent}#   was: {stripped}",
    ]

    if not from_a_file(row):
        # Said on the line itself and not only in the window that asked, because
        # the window closes. A reading with no `raw=` was typed here, and these
        # two comment lines are then the only place its numbers still exist.
        lines.append(
            f"{indent}#   typed here, with no `raw=`: this comment is the only "
            f"copy left of it"
        )

    return "\n".join(lines)


# How the source string spells a plane, inside the `raw=` of a reading. Matched
# with a boundary in front of `dip`, or `dip_dir=140` would answer the second
# pattern as well as the first and every reading would read as dipping 140.
RAW_DIP_DIR = re.compile(r"(?:^|[\s,;])dip_dir\s*=\s*(-?[\d.]+)")
RAW_DIP = re.compile(r"(?:^|[\s,;])dip\s*=\s*(-?[\d.]+)")


def raw_plane_of(row):
    """
    The plane the row's own `raw` attributes state, as `(dip dir, dip)`, or None.

    **Whether the line can still say what the source said**, which is the one
    question `owed_record` turns on. FORMAT.md's first rule keeps the source
    string beside the normalised value, so on a reading that came out of a layer
    the old plane is not lost by rewriting the slot: it is sitting a few
    characters further along the same line. On a reading typed into this tool
    there is no such string, and the slot is the only copy there is.

    `from_a_file` is not this test and is not a substitute for it. That one asks
    whether *any* `raw.` key is present, which is the right question for a
    detachment -- a line about to vanish takes every `raw.` with it -- and the
    wrong one here: `raw.comments.station="Possibly within CSC or CTC on CSC"`
    is a source string about a note and holds no plane at all, so a line
    carrying only that would pass `from_a_file` while losing its numbers.

    Read out of whichever `raw` key holds both halves, in the order the line
    wrote them. All 44 readings in the AOI keep it in `raw=` and all 44 agree
    with their own slot, which is the baseline worth stating: today nothing in
    these files is a reading whose source string has already been overtaken.
    """

    for key, value in (row.attrs or {}).items():
        if key != "raw" and not key.startswith("raw."):
            continue

        dip_dir = RAW_DIP_DIR.search(f" {value}")
        dip = RAW_DIP.search(f" {value}")

        if dip_dir is None or dip is None:
            continue

        try:
            return float(dip_dir.group(1)), float(dip.group(1))
        except ValueError:
            continue

    return None


def owed_record(row, anchor, plane):
    """
    Why an amendment has to leave a comment behind, or None if it owes none.

    **The rule the format writes for us**, and the whole of why amending a
    reading is not one operation but two with different costs. Rewriting the
    plane slot of an imported reading loses nothing: `raw="dip_dir=140 dip=35"`
    goes on standing beside it, which is rule 1 of FORMAT.md working exactly as
    intended -- the source string next to the normalised value, and the
    difference between them *is* the curation, legible on one line. Rewriting
    the anchor slot loses the only copy of where the source put the point, every
    time, because the format keeps no `raw.` of a placing.

    So a move is always recorded and a replanning usually is not, and the
    exceptions are measured rather than assumed: a reading with no source string
    for its plane (typed here, by `ReadingsHere`) and a reading whose source
    string has already been overtaken by an earlier amendment both owe the
    comment, because in neither case is the number about to be written over
    held anywhere else.

    `anchor` and `plane` are what the press would write, `None` meaning
    unchanged. Returns the sentence to put in the comment's first line, so that
    what the file says and what the window said before the press come out of one
    place and cannot drift.
    """

    was_anchor = anchor_of(row.line)
    moved = (
        anchor is not None
        and was_anchor is not None
        and (abs(anchor[0] - was_anchor[0]) > 0.005
             or abs(anchor[1] - was_anchor[1]) > 0.005)
    )

    replanned = (
        plane is not None
        and row.plane is not None
        and (abs(plane[0] - row.plane[0]) > 0.05 or abs(plane[1] - row.plane[1]) > 0.05)
    )

    if moved:
        # First and unconditionally, because it is the only one of the two that
        # the file cannot state another way -- and it is said as a loss rather
        # than as a policy, so that a curator reading the comment tomorrow knows
        # why this one is here and the plane ones are not.
        return (
            "the anchor it came in on is in no `raw.` anywhere, so it is here"
        )

    if not replanned:
        return None

    held = raw_plane_of(row)

    if held is None:
        return "typed here, with no `raw=` to stand beside the new number"

    if abs(held[0] - row.plane[0]) > 0.05 or abs(held[1] - row.plane[1]) > 0.05:
        return (
            f"its `raw=` already reads {held[0]:.0f}/{held[1]:.0f} and the slot "
            f"reads {row.plane[0]:.0f}/{row.plane[1]:.0f}, so the slot is the "
            f"only copy of the second"
        )

    return None


def amend_note(row, why, today, owed, indent="  "):
    """
    The comment that records a reading as it stood, where nothing else can.

    `detachment_note`'s sibling, the same `#` for the same reason -- it costs
    nothing in FORMAT.md, in the parser or in the precedence, it survives a Save
    because `Document` replaces lines instead of dumping the file, and the one
    reader who needs it is the next person to open the block.

    The narrowing is the difference, and it is written on the comment itself.
    A detachment always leaves this, the line having gone. An amendment leaves
    it only where the amended line could not otherwise say what the old one
    said, and `owed` is that reason, decided once in `owed_record` and spent
    here -- so a comment in the file always carries the argument for its own
    existence, and a reader who finds three amendments with comments and a
    fourth without can tell which rule each of them fell under.

    The whole original line goes in, as in a detachment and for its reason: a
    summary would be this tool's reading of the row, and `off=`, `raw=` and the
    station codes are the only record of what the importer put there.
    """

    lines = [
        f"{indent}# {today}: amended {reading_said(row)} -- {why}".rstrip(" -"),
        f"{indent}#   was: {row.line.strip()}",
        f"{indent}#   kept because {owed}",
    ]

    return "\n".join(lines)


def covered_metres(ends, earlier):
    """
    How many metres of a stretch are already claimed by the fits before it.

    The arithmetic behind one sentence, which is the one thing a curator cannot
    get out of a file by reading it: `attitude_at` takes the **first** fit
    covering a progressive, and a fit written now is appended after the ones the
    block already had. So every metre an earlier fit covers is a metre the new
    line will never be asked about -- legal, parsing, and inert.

    Which is how `montealpi_01.gstruct` came to carry three fits over
    `2887.500..2937.503 m` of `L0071`: two presses of a button whose window
    never said the first one had happened. The third is not a worse line than
    the first, it is the same line, and nothing anywhere distinguished them.

    Returns metres, so that the caller can say *how much* rather than only
    whether -- a fit half covered answers over the other half and is worth
    keeping, and a fit covered to the last metre is not. None where the stretch
    cannot be read; 0.0 for a stretch written backwards, which covers no ground
    at all and so has none to lose.

    `earlier` holds intervals as the file wrote them, including None and
    including reversed pairs, both of which claim nothing and are dropped here:
    a line that holds over no ground cannot shadow one that does.

    **The same arithmetic serves a `span`, with the list turned round**, and that
    is worth saying here because the two rules are opposite and only one of them
    is written in FORMAT.md. `attitude_at` takes the **first** fit covering a
    progressive; `span_at` takes the **last** span covering one, documented with
    its reason -- a local correction is written by adding a line over the general
    one. So a fit is shadowed by what came before it and a span by what comes
    after it, and a caller asking how much of a span is in force passes the spans
    *below* it in the block. What this function knows is only how much of one
    interval a set of others covers; which set that is, is the rule.
    """

    if ends is None:
        return None

    s0, s1 = ends

    if s1 <= s0:
        return 0.0

    total = 0.0
    reach = s0

    for a, b in sorted(
        one for one in earlier if one is not None and one[1] > one[0]
    ):
        a, b = max(a, reach), min(b, s1)

        if b > a:
            total += b - a
            reach = b

        if reach >= s1:
            break

    return total


def with_plane(line, dip_dir, dip, decimals=PLANE_DECIMALS):
    """
    The same line with its plane slot rewritten, or None if it has no plane slot.

    **A splice and not a rebuild**, which is `Document`'s own rule one line
    further down: joining the tokens back together would write the line in this
    function's spacing rather than in the file's, so a line somebody had lined
    up under its neighbours would come back single-spaced, and a `fit` written
    by `as_line` would lose the two-space indent that puts it inside its
    structure. Every byte outside the slot is left where it was.

    The line is not checked for being complete. A template with its anchors
    still `*` takes a plane perfectly well -- that is the normal way round here,
    the number being the thing chosen last -- and a line that would not parse is
    refused by Apply, with the parser's words, which is the one place in this
    tool that refuses anything.
    """

    tokens = _tokens_of(line)
    word = tokens[0][0] if tokens else None
    at = PLANE_AT.get(word)

    if at is None or len(tokens) <= at:
        return None

    if tokens[PLANE_KIND_AT[word]][0] != "plane":
        return None

    _, start, end = tokens[at]

    return (
        line[:start]
        + f"{float(dip_dir) % 360.0:.{decimals}f}/{float(dip):.{decimals}f}"
        + line[end:]
    )


def with_anchor(line, x, y):
    """
    The same line with its single-anchor slot rewritten, or None if it has none.

    `with_plane`'s sibling for the other half of what an `attitude` claims, and
    a splice for that function's reason: every byte outside the slot stays where
    it was, so the attributes keep the spacing and the order the file gave them.

    **This is the one slot in the format with no `raw.` beside it**, and that
    asymmetry decides what the window above has to do. FORMAT.md's first rule
    keeps the source string next to the normalised value, and on all 44 readings
    of the AOI that string is the plane -- `raw="dip_dir=140 dip=35"`, agreeing
    with the slot on every one of them. The anchor has no such copy, because the
    source geometry *is* the anchor: a point out of a layer, written down. So a
    plane rewritten here leaves the file still able to say what the source said,
    and an anchor rewritten here leaves nothing anywhere -- see `owed_record`,
    where that difference is the rule and not a remark.

    A `*` in the slot is written over like a coordinate, because for a *place* it
    is not an end of the path but nothing at all: `anchor_of` answers None for it
    and `Anchored.resolve` leaves `s` unset. Anything else in the slot comes back
    None rather than overwritten -- `attitude plane 140/35`, a line missing its
    anchor, would otherwise be "repaired" into `attitude @x,y 140/35`, which
    parses, claims a plane nobody wrote, and reads as deliberate.
    """

    tokens = _tokens_of(line)
    word = tokens[0][0] if tokens else None
    at = ANCHOR_AT.get(word)

    if at is None or len(tokens) <= at:
        return None

    held, start, end = tokens[at]

    if not held.startswith("@") and held != "*":
        return None

    gstruct = module()

    return line[:start] + gstruct._a((float(x), float(y))) + line[end:]


def with_attrs(line, attrs):
    """
    The same line with `key=value` filled in where it is missing or empty.

    **Missing or empty, and never over a value somebody wrote.** The one thing
    this is for is saying where a number came from -- `from=`, and what north it
    was measured against -- and the templates arrive carrying `from=` with
    nothing after it, which is the format's way of leaving a slot open rather
    than a value of its own (`_kw` does not write an empty one at all). So an
    empty value is an invitation and a filled one is a decision, and the
    difference matters enough to be the rule.

    Appended in the order given, at the end of the line, which is where
    attributes go: the grammar is positional tokens first, then `key=value`, and
    a pair inserted in the middle would be read as positional by anything
    counting from the keyword -- `interval_of` and `with_plane` above, for two.

    Quoting is gstruct's own `_q` and not a rule reimplemented here. The DEM's
    file name is the value that needs it -- `dem=Monte Alpi.tif` reads back as a
    `dem` of `Monte` and a stray token the parser has no reason to refuse, so
    the file would not say what it appears to say. `fits.as_line` borrows the
    same two helpers for the same reason.
    """

    gstruct = module()
    tokens = _tokens_of(line)

    held = {
        token.partition("=")[0] for token, _, _ in tokens
        if "=" in token and token.partition("=")[2] not in ("", '""')
    }

    wanted = {
        key: value for key, value in attrs.items()
        if key not in held and value not in (None, "")
    }

    if not wanted:
        return line

    # The empty ones come out rather than being left beside the filled ones:
    # `from= from=plane-dem` on one line is two tokens the parser reads in
    # order, so the second wins and the first sits there looking like a
    # contradiction of it.
    #
    # Cut out by where they sit, back to front, rather than by rebuilding the
    # line from its tokens -- which is `with_plane`'s rule, and it bites harder
    # here: joining on single spaces would also close up the runs of spaces
    # inside a quoted value, and `reason="la traccia qui ricalca"` is four
    # tokens to anything splitting on whitespace.
    out = line

    for at in reversed(range(len(tokens))):
        token, start, end = tokens[at]

        if "=" not in token or token.partition("=")[0] not in wanted:
            continue

        # From the end of the token before it, so the space that separated the
        # two goes with it. Cutting the token alone would leave the two spaces
        # either side of it as one gap of two, which the parser does not mind
        # and a reader does.
        out = out[: tokens[at - 1][2] if at else start] + out[end:]

    return out.rstrip() + gstruct._kw(wanted)


def with_values(line, values):
    """
    The same line with `key=value` set where it sits, appended, or taken off.

    **`with_attrs`'s opposite number, and the two are not one function with a
    flag.** That one fills a slot a template left open and *refuses* to write
    over a value somebody wrote -- the refusal is its whole argument, and
    weakening it with a keyword would weaken it everywhere it is called from.
    This one exists to carry out a decision about a value that is already there,
    which is a different act and needs a different thing standing behind it: not
    a rule in the writer, but a window that shows the old value beside the new
    one and a press that happens after somebody has looked at both.

    **Replaced where it sits**, rather than cut and re-appended. On every reading
    in these files `station=` comes first and `raw=` near the end, and a curator
    who fixed a station name would otherwise find it had moved to the end of the
    line: a diff full of motion nobody asked for, in a file people read. The
    same splice rule as `with_plane` and for one more reason here -- rebuilding
    from tokens would close up the runs of spaces inside a quoted value, and
    `site_note="Possibly within CSC or CTC on CSC"` is six tokens to anything
    splitting on whitespace.

    `None` takes a key off. A key the line has not got is appended through
    `gstruct._kw`, which is the one place the quoting is decided -- `_q` quotes a
    value holding a space, a quote, a backslash or an `=`, and a `note=` written
    by a curator is the attribute in these files most likely to hold all four.

    **Where the token ends is asked of the parser's own lexer**, `gstruct._TOK`,
    and not of whitespace. The first draft of this used `_tokens_of` and the
    first test of it broke a file: `site_note="Possibly within CSC or CTC on
    CSC"` is **six** whitespace tokens, so taking that key off cut the first of
    them and left `within CSC or CTC on CSC"` standing on the line as positional
    tokens -- a line that still parses, with a stray quote in it, claiming
    nothing anybody wrote. `rows_of` already refuses to split this way and says
    why; a writer had to be held to the same rule, and the way to be held to it
    is to use the same lexer rather than a second reading of the grammar.
    """

    gstruct = module()
    found = list(gstruct._TOK.finditer(line))
    wanted = dict(values)
    out = line

    for at in reversed(range(len(found))):
        token = found[at]
        key = token.group(1)

        if not key or key not in wanted:
            continue

        value = wanted.pop(key)
        start, end = token.span()

        if value is None:
            # From the end of the token before it, so the space that separated
            # the two goes with it -- `with_attrs`' rule, and the same reason:
            # cutting the token alone leaves a gap of two spaces, which the
            # parser does not mind and a reader does.
            out = out[: found[at - 1].end() if at else start] + out[end:]
        else:
            out = out[:start] + f"{key}={gstruct._q(value)}" + out[end:]

    # What is left never appeared on the line, so it goes where attributes go:
    # at the end. A pair inserted among the positional tokens would be counted
    # as positional by everything that counts from the keyword -- `anchor_of`,
    # `plane_of` and `interval_of`, for three.
    fresh = {
        key: value for key, value in wanted.items() if value not in (None, "")
    }

    if not fresh:
        return out

    return out.rstrip() + gstruct._kw(fresh)


# How `attitude_at` writes the distance it answered from, at the end of its
# provenance string: `misurata:S22@12m` and `misurata-lontana:3422m`, the one
# with an `@` and the other with a bare colon.
ANSWERED_FROM = re.compile(r"[:@]-?[\d.]+m$")


def answering(said):
    """
    Which line is answering, out of a provenance string, with the distance off.

    **For comparing two provenances to each other**, which is a different job
    from showing one. `attitude_at` ends `misurata` and `misurata-lontana` with
    how far away the reading it used is, and that distance changes at every
    metre of the trace -- so two sampled provenances compared string by string
    differ everywhere, and a reading nudged a hundred metres reads as having
    changed the whole fault. Measured: 3284 m of `F0055`'s 3531, where the
    answer is the same reading throughout and 109 m of it changed tier.

    What is left is the identity of the line that answered, as far as the string
    carries it: the station for `misurata`, the verdict for `fit`, the reason
    for `rifiutata`, `assente` for nothing. **`misurata-lontana` carries none**,
    the far tier naming only its distance, so two different distant readings
    come back equal here -- which is why a caller that needs to tell them apart
    compares the plane as well, rather than reading this string a second way.
    """

    return ANSWERED_FROM.sub("", said.split("@")[0])


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
