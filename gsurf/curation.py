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

from collections import defaultdict
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
