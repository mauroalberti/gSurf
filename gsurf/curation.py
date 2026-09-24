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
AXES = ("certainty", "exposure")

# What `in_section` has to say for a record to arrive held out of the section.
# The name is the one gSurf already reads off a column (`ENABLED_FIELD`), so the
# same claim travels under the same name whether it comes in a GeoPackage field
# or as a structure attribute.
OUT_OF_SECTION = ("no", "false", "0", "off", "excluded")

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


def read(path):
    """
    The file as a dataset, with every anchor projected onto its own path.

    `load` resolves on the way in, so `s` is already derived by the time this
    returns -- for the structures that carry a path. A curation carries none and
    its intervals stay unresolved, which is the same rule and not an exception
    to it: no path, no projection. `apply_to` is what resolves those, against
    the geometry the records brought with them.
    """

    try:
        import gstruct
    except ImportError as err:
        # The one dependency this project cannot name in its `pyproject.toml`:
        # the name on PyPI belongs to something else, so gstruct is installed
        # from its own repository or not at all. Both callers of this show what
        # comes out of here in a dialog, so it is written to be read there.
        raise ImportError(
            "gstruct is not installed. It is on no package index -- the name "
            "there belongs to an unrelated project -- so it is installed from "
            "its own repository:  pip install -e <gstruct repo>"
        ) from err

    return gstruct.load(str(path))


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

        enabled = _enabled(structure.attrs)
        planes = 0

        for attitude in structure.attitudes:
            if attitude.plane is None:
                dropped["measurements with no plane"] += 1
                continue

            planes += 1
            place = length / 2.0 if attitude.s is None else float(attitude.s)

            reading.records.append(
                TraceRecord(
                    category=structure.kind or "unknown",
                    plane=geogst_plane(attitude.plane),
                    lines=lines,
                    length=length,
                    anchor=None if attitude.s is None else float(attitude.s),
                    span=None,
                    enabled=enabled,
                    attrs={
                        **common,
                        **_axes_at(structure, place),
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
                    enabled=enabled and verdict != UNCONSTRAINED,
                    attrs={
                        **common,
                        **_axes_at(structure, place),
                        **dict(fit.attrs),
                        "src": fit.attrs.get("from", "fit"),
                        "fitted": True,
                    },
                )
            )

        if planes == 0:
            reading.records.append(
                TraceRecord(
                    category=structure.kind or "unknown",
                    plane=None,
                    lines=lines,
                    length=length,
                    enabled=enabled,
                    attrs={**common, **_axes_at(structure, length / 2.0)},
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
    What the format's two axes say at one place along the trace.

    `value_at` and not the spans directly, because the rule that the last span
    covering a progressive wins is how a local correction is written -- by
    adding a line, never by editing one -- and reading the list any other way
    would quietly prefer the general statement to the correction over it.
    """

    return {axis: structure.value_at(axis, s) for axis in AXES}


def _enabled(attrs):
    """Whether a structure arrives in the section or held out of it."""

    said = attrs.get("in_section")

    if said is None:
        return True

    return str(said).strip().lower() not in OUT_OF_SECTION


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

        if "in_section" in structure.attrs:
            for record in found:
                record.enabled = _enabled(structure.attrs)

            applied.assertions += 1

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

            if landed:
                applied.assertions += 1
            else:
                applied.nowhere += 1

        for attitude in structure.attitudes:
            if attitude.plane is None:
                continue

            at = None if attitude.anchor is None else _project(lines, attitude.anchor)

            applied.added.append(
                TraceRecord(
                    category=category,
                    plane=geogst_plane(attitude.plane),
                    lines=lines,
                    length=length,
                    anchor=None if at is None else at[0],
                    span=None,
                    attrs={
                        **identity,
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

            applied.added.append(
                TraceRecord(
                    category=category,
                    plane=geogst_plane(fit.plane),
                    lines=lines,
                    length=length,
                    anchor=None,
                    span=(s0, s1),
                    enabled=verdict != UNCONSTRAINED,
                    attrs={
                        **identity,
                        **dict(fit.attrs),
                        "src": fit.attrs.get("from", "fit"),
                        "fitted": True,
                    },
                )
            )
            applied.assertions += 1

    applied.ignored = dict(ignored)

    return list(records) + applied.added, applied


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
