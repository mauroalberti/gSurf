"""
Located attitudes: the measurements a structural tool reads.

This is not the backdrop layer of `vectors.py`, which is drawn and no more. Here
the points *are* the data: each carries an orientation, and what a tool does
with them depends on that orientation being right. So the reading is strict
where it has to be and forgiving where the geology says it should be, and it
says out loud what it would not take.

Two shapes carry an attitude and both are here. A point carries the attitude of
the plane measured at it; a line carries the attitude of the plane it is the
outcrop trace of, which is what a profile crosses. What they share is the
reading of the two angles, and that is deliberately one piece of code: the
rules in `admissible` are geology -- 999 for a horizontal bed, 360 for north,
99 for contorted bedding -- and two copies of them would drift.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# What the azimuth field means, shared by everything that reads one. Dip
# direction is what an Italian survey writes down (`Immersione`); the
# right-hand-rule strike is what an English-language one usually does.
CONVENTIONS = (
    ("dip direction", False),
    ("strike, right-hand rule", True),
)


def numeric(column):
    """The column as floats, with whatever will not convert becoming NaN."""

    import pandas as pd

    return pd.to_numeric(column, errors="coerce").to_numpy(dtype=float)


def admissible(azimuth, dip):
    """
    Which rows carry an attitude, and a tally of why the others do not.

    The dip decides first, because it decides whether the azimuth means
    anything. Out of 0-90 there is no plane: the CARG sheets write 99 for
    contorted bedding measured as a mean, and taken at face value that is a
    plane overturned past the vertical. Only once the dip is a real one, and
    not zero, does the azimuth have to be a bearing.

    :return: the boolean mask of the rows to keep, and the reasons the others
        were dropped with a count each -- reasons that did not occur are left
        out rather than reported as zero.
    """

    dropped = {}

    dip_known = np.isfinite(dip)
    dip_sane = dip_known & (dip >= 0.0) & (dip <= 90.0)

    dropped["dip missing"] = int((~dip_known).sum())
    dropped["dip outside 0-90"] = int((dip_known & ~dip_sane).sum())

    # Zero dip is exempt: the azimuth is not read, so it cannot be wrong.
    needs_azimuth = dip_sane & (dip > 0.0)
    azimuth_known = np.isfinite(azimuth)

    # 360 inclusive, and not the half-open range a normalised azimuth lives in.
    # 360 is how north gets written down: on the Marsico Nuovo sheet ten
    # attitudes carry it and not one carries 0, so a half-open rule threw away
    # every north-dipping bed on the map and called them errors. It is
    # normalised away by the caller, not refused.
    azimuth_sane = azimuth_known & (azimuth >= 0.0) & (azimuth <= 360.0)

    dropped["dip direction missing"] = int((needs_azimuth & ~azimuth_known).sum())
    dropped["dip direction outside 0-360"] = int(
        (needs_azimuth & azimuth_known & ~azimuth_sane).sum()
    )

    keep = dip_sane & (~needs_azimuth | azimuth_sane)

    return keep, {reason: count for reason, count in dropped.items() if count}


def numeric_fields(path, layer=None):
    """The numeric fields of the layer, which are the ones worth offering."""

    import pyogrio

    info = pyogrio.read_info(path, layer=layer) if layer else pyogrio.read_info(path)

    return [
        str(field)
        for field, dtype in zip(info["fields"], info["dtypes"])
        if str(dtype) != "object"
    ]


def normalised_azimuth(azimuth, dip):
    """
    The azimuth as a bearing, with the horizontal beds neutralised.

    A horizontal bed has no dip direction: whatever the field holds -- 999 in
    the CARG sheets, a blank, last measurement's leftover -- it is not a
    bearing, and the pole is vertical whichever way it is read.
    """

    return np.where(dip == 0.0, 0.0, azimuth % 360.0)


class AttitudeSource:
    """
    A point layer read as planar attitudes, with the poles taken once.

    The poles are built at load and kept: they are what every window is a
    subset of, and rebuilding a few dozen of them per frame would be paying
    for the same arithmetic hundreds of times a second.
    """

    CONVENTIONS = CONVENTIONS

    def __init__(
        self,
        path,
        crs,
        layer=None,
        dip_dir_field=None,
        dip_field=None,
        is_rhr_strike=False,
        bounds=None,
    ):
        import geopandas as gpd

        self.path = Path(path)
        self.layer = layer
        self.dip_dir_field = dip_dir_field
        self.dip_field = dip_field
        self.is_rhr_strike = bool(is_rhr_strike)
        self.problem = None
        self.dropped = {}
        self.outside = 0

        self.xy = np.empty((0, 2))
        self.azimuths = np.empty(0)
        self.dips = np.empty(0)
        self.poles = []

        try:
            frame = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
        except Exception as err:
            self.problem = str(err).split("\n")[0]
            return

        if frame.crs is None:
            self.problem = "no CRS"
            return

        for field in (dip_dir_field, dip_field):
            if field not in frame.columns:
                self.problem = f"no field '{field}'"
                return

        frame = frame.to_crs(crs)
        frame = frame[frame.geometry.notna() & (frame.geometry.geom_type == "Point")]

        if frame.empty:
            self.problem = "no point geometry"
            return

        azimuth = numeric(frame[dip_dir_field])
        dip = numeric(frame[dip_field])

        keep, self.dropped = admissible(azimuth, dip)

        if not keep.any():
            self.problem = "no readable attitude"
            return

        frame = frame[keep]
        azimuth, dip = normalised_azimuth(azimuth[keep], dip[keep]), dip[keep]

        self.frame = frame
        self.xy = np.c_[frame.geometry.x.to_numpy(), frame.geometry.y.to_numpy()]
        self.azimuths = azimuth
        self.dips = dip
        self.poles = self._pole_axes(azimuth, dip)

        if bounds is not None:
            left, bottom, right, top = bounds
            inside = (
                (self.xy[:, 0] >= left)
                & (self.xy[:, 0] <= right)
                & (self.xy[:, 1] >= bottom)
                & (self.xy[:, 1] <= top)
            )
            self.outside = int((~inside).sum())

    # -- reading -----------------------------------------------------------

    def _pole_axes(self, azimuth, dip):
        """
        The poles as geogst axes, downward-pointing and sign-blind.

        An axis and not a direction: the orientation tensor is the axial
        counterpart of a mean, and a bed dipping 30 to the east and the same bed
        overturned are the same pole. That is also why overturned bedding needs
        no special case here -- it does for polarity, which is not this
        question.
        """

        from geogst.core.geology.orientations import Plane

        return [
            Plane(float(a), float(d), is_rhr_strike=self.is_rhr_strike).normal_axis()
            for a, d in zip(azimuth, dip)
        ]

    # -- windows -----------------------------------------------------------

    @property
    def is_loaded(self):
        return len(self.poles) > 0

    def __len__(self):
        return len(self.poles)

    def within(self, x, y, radius):
        """
        The stations inside a circle, as indices.

        A circle and not a cell: a cell would make the answer depend on which
        way the grid happens to be turned, and there is no north in the question
        being asked.

        Scanned outright rather than through a spatial index. One window on 1757
        stations costs 0.05 ms against 0.013 with a KD-tree, and neither is what
        a frame is spent on. This once read that a tree would earn its place at
        ten thousand windows; measured, it does not -- a 16289-cell grid spends
        0.89 s searching against 0.046 with a tree, out of 6.9 s in total, so the
        tree would buy 13% of a field in exchange for a dependency. It would earn
        its place if the tensor stopped dominating, which is the opposite of what
        happened.
        """

        if not self.is_loaded:
            return np.empty(0, dtype=int)

        offsets = self.xy - np.array([x, y])

        return np.flatnonzero((offsets * offsets).sum(axis=1) <= radius * radius)

    def poles_at(self, indices):
        return [self.poles[i] for i in indices]

    def dip_directions(self):
        """The azimuths as dip directions, whichever way they were written."""

        if not self.is_rhr_strike:
            return self.azimuths

        return (self.azimuths + 90.0) % 360.0

    def summary(self):
        if self.problem:
            return f"{self.layer or self.path.stem}: {self.problem}"

        text = f"{self.layer or self.path.stem}: {len(self)} attitudes"

        if self.dropped:
            detail = ", ".join(f"{count} {reason}" for reason, count in self.dropped.items())
            text += f" ({detail})"

        if self.outside:
            text += f"; {self.outside} outside the map"

        return text


# The columns a layer may carry to say where a measurement was taken and how
# far it is held to reach. Read when present, ignored when not: a layer of
# plain traces has none of them and behaves exactly as it did before they
# existed. The names are the ones `gstruct/export_gsurf.py` writes.
ANCHOR_FIELD = "anchor_s"
SPAN_FIELDS = ("span_s0", "span_s1")
ENABLED_FIELD = "in_section"

# How far a measurement made at one point governs along the trace it was made
# on, when nothing else says. A field measurement is a point, and extending it
# is an interpretation -- one that belongs here rather than in whatever wrote
# the layer, because here it can be seen against the section and changed. The
# value is where the gstruct model puts its own default; it is a starting
# position to argue with, which is why it is a parameter and not a constant.
DEFAULT_HALF_SPAN = 250.0


def clip_to_span(lines, s0, s1):
    """
    The part of a record's trace between two progressives, in metres.

    Progressive runs from the start of the first line and carries on across the
    others, so a trace interrupted by cover keeps one running measure -- the
    same one an anchor read off the table was computed against.

    The cut ends land inside the segment they fall in rather than on the
    nearest vertex: on a trace digitised every 40 m a 250 m span would
    otherwise be anything between 200 and 300, and the number the tool shows
    would not be the number it used.
    """

    from geogst.core.geometries.shapes.lines import Ln

    kept, walked = [], 0.0

    for line in lines:
        coords = line.coords

        if coords is None or len(coords) < 2:
            continue

        steps = np.hypot(*np.diff(coords[:, :2], axis=0).T)
        progressive = walked + np.concatenate(([0.0], np.cumsum(steps)))
        walked = progressive[-1]

        piece = _between(coords, progressive, s0, s1)

        if piece is not None:
            kept.append(Ln(piece))

    return kept


def point_at(lines, s):
    """
    The point on a record's trace at a progressive, or None where it runs off.

    The same walk and the same interpolation as `clip_to_span`, and for the
    same reason: a progressive means nothing except against the measure it was
    computed with. Read back against the nearest vertex instead, an anchor
    would land up to half a segment away -- 20 m on a trace digitised every
    40 -- which is the size of the thing being located.

    Whatever the coordinates carry comes back, two values or three. The line's
    own third value is not an elevation anybody measured, though: where the
    height matters it is the DEM that has it, and the fit read it from there.
    """

    if s < 0.0:
        return None

    walked = 0.0

    for line in lines:
        coords = line.coords

        if coords is None or len(coords) < 2:
            continue

        steps = np.hypot(*np.diff(coords[:, :2], axis=0).T)
        progressive = walked + np.concatenate(([0.0], np.cumsum(steps)))
        walked = progressive[-1]

        if s <= progressive[-1]:
            return tuple(float(v) for v in _at(coords, progressive, s)[0])

    return None


def within(records, area):
    """
    The records whose trace enters `area`, each kept whole or dropped whole.

    The same rule as the `bounds` a source is read with, for the same reason: a
    trace is cut by the profile it is intersected with, and not by the ground
    somebody has decided to work on. Trimming it here would invent an endpoint
    where the selection ends -- and the fit would then read a plane off a bend
    that is the edge of a choice rather than of a contact. What narrows is
    which contacts are in hand, never what any one of them says.

    The bounding box first and the geometry only for what survives it. On a
    CARG sheet this is twenty-two thousand records against one polygon, all but
    a couple of hundred of them decided by four comparisons.
    """

    from shapely.geometry import LineString

    left, bottom, right, top = area.bounds
    kept = []

    for record in records:
        traced = [
            line.coords[:, :2] for line in record.lines
            if line.coords is not None and len(line.coords) >= 2
        ]

        near = [
            xy for xy in traced
            if xy[:, 0].min() <= right and xy[:, 0].max() >= left
            and xy[:, 1].min() <= top and xy[:, 1].max() >= bottom
        ]

        if near and any(area.intersects(LineString(xy)) for xy in near):
            kept.append(record)

    return kept


def _finite(value):
    """A number, or nothing where the cell was empty."""

    return None if value is None or value != value else float(value)


def _between(coords, progressive, s0, s1):
    """One line's coordinates between two progressives, ends interpolated."""

    low, high = max(s0, progressive[0]), min(s1, progressive[-1])

    if high <= low:
        return None

    inside = (progressive > low) & (progressive < high)

    return np.vstack(
        [_at(coords, progressive, low), coords[inside], _at(coords, progressive, high)]
    )


def _at(coords, progressive, s):
    """The point at a progressive, interpolated within the segment holding it."""

    j = int(np.searchsorted(progressive, s, side="right")) - 1
    j = min(max(j, 0), len(coords) - 2)

    step = progressive[j + 1] - progressive[j]
    t = 0.0 if step == 0.0 else (s - progressive[j]) / step

    return (coords[j] + t * (coords[j + 1] - coords[j]))[None, :]


@dataclass
class TraceRecord:
    """
    One plane, the trace it crops out on, and how far along it it is held to go.

    The reach is the part that is not in the data. A plane fitted to a trace
    owns that trace -- the trace is the evidence, and it reaches as far as it
    is drawn. A compass reading taken at one outcrop owns a point, and how much
    of the fault it speaks for is a judgement about that fault: metres, where
    it is a local break; the whole kilometre, where the surface was walked and
    validated. So `anchor` and `span` are kept apart. An anchor is a fact off
    the table; a span is a decision, and `None` means nobody has made one yet.
    """

    category: str
    plane: object
    lines: list
    length: float
    anchor: float = None
    span: tuple = None
    enabled: bool = True
    attrs: dict = field(default_factory=dict)

    def extent(self, half_span):
        """The interval actually used, in metres along the trace."""

        if self.span is not None:
            return self.span

        # No anchor, no question to answer: the trace is the datum.
        if self.anchor is None or half_span is None:
            return 0.0, self.length

        return self.anchor - half_span, self.anchor + half_span

    def reach_endpoints(self, half_span):
        """
        Where the reach starts and ends on the ground, or None if it is whole.

        Coordinates rather than progressives, because that is what a span is
        anchored by outside this process: a progressive means nothing without
        the line it was measured along, and the line can be redigitised.
        """

        if self.is_whole(half_span):
            return None

        s0, s1 = self.extent(half_span)
        clipped = clip_to_span(self.lines, s0, s1)

        if not clipped:
            return None

        first, last = clipped[0].coords, clipped[-1].coords

        return tuple(first[0][:2]), tuple(last[-1][:2])

    def anchor_point(self):
        """
        The one point the attitude can be filed under, or None off the trace.

        A measurement has to land somewhere to be mapped, and the three cases
        answer to different things. An anchor is a fact off the table and wins
        outright. A fitted record has no anchor from anywhere else, and the
        middle of the stretch its window held on is where the plane was read.
        A record with neither speaks for the whole trace, so the middle of that
        is the least wrong place to put it -- and the span written out beside
        it says so, rather than the point pretending to a precision it has not
        got.

        Deliberately not the midpoint of `extent`, which answers a different
        question: given a table anchor and no half-span, `extent` widens to the
        whole trace, and the middle of the trace is not where the reading was
        taken.
        """

        if self.anchor is not None:
            s = self.anchor
        elif self.span is not None:
            s = (self.span[0] + self.span[1]) / 2.0
        else:
            s = self.length / 2.0

        return point_at(self.lines, s)

    def is_whole(self, half_span):
        """
        Whether the reach covers the trace, to within a metre of either end.

        Not an exact comparison, because a span read off a table has been
        rounded on the way in and a length is summed from the geometry: a fit
        that covers its whole trace comes out three centimetres short of it,
        and taken literally that sends every one of them through a clip which
        removes nothing. A metre is far inside the digitising precision of a
        trace mapped at 1:10.000, where half a millimetre on the sheet is five.
        """

        s0, s1 = self.extent(half_span)

        return s0 <= 1.0 and s1 >= self.length - 1.0


class TraceAttitudeSource:
    """
    A line layer read as outcrop traces, each carrying the plane it traces.

    Where a profile crosses one of these lines, the plane it belongs to cuts
    the section at a computable apparent dip: that is the tick a geological
    section carries, and it is what `Profilers.intersect_lines_with_attitudes`
    is given. The traces are the data, not the backdrop -- a fault drawn from
    `vectors.py` is a blue line and nothing else.

    **Records, and why they are grouped.** A category is a name, and under it
    sit one or more records, each a plane and the lines that are its outcrop.
    Lines of one category that carry the *same* attitude are pooled into a
    single record, because that is what they are: one plane, digitised in
    pieces. On Timpa San Lorenzo six features share 67.3/39 and are the six
    fragments of one fault's trace. Lines that carry different attitudes stay
    separate records, and all of them are kept -- the library used to keep only
    the last of them, which is why the grouping is worth stating rather than
    assuming.

    **Reach.** A record may say where along its trace the attitude was taken
    (`anchor_s`) and over what interval it holds (`span_s0`, `span_s1`). A
    layer carrying neither behaves as one plane per whole trace, which is what
    a digitised fault system is. Where there is an anchor and no span, the
    reach comes from `half_span` and can be changed afterwards: that number is
    the tool's to offer and the geologist's to set, and it is deliberately not
    baked into whatever wrote the file.

    **Geometry.** Held as geogst `Ln` in the session's projection, flattened to
    two dimensions on the way in: the intersection is computed in a horizontal
    plane and the elevation of every crossing comes from the topography, not
    from the trace. A MultiLineString becomes several `Ln` under one record,
    which is what a trace interrupted by cover or by a sheet edge looks like.
    """

    CONVENTIONS = CONVENTIONS

    def __init__(
        self,
        path,
        crs,
        layer=None,
        category_field=None,
        dip_dir_field=None,
        dip_field=None,
        is_rhr_strike=False,
        bounds=None,
        anchor_field=ANCHOR_FIELD,
        span_fields=SPAN_FIELDS,
        enabled_field=ENABLED_FIELD,
        half_span=DEFAULT_HALF_SPAN,
    ):
        import geopandas as gpd

        self.path = Path(path)
        self.layer = layer
        self.crs = crs
        self.category_field = category_field
        self.dip_dir_field = dip_dir_field
        self.dip_field = dip_field
        self.is_rhr_strike = bool(is_rhr_strike)

        # Both fields, or neither. Neither is a layer of mapped contacts whose
        # attitudes were never written down -- the plane is in the line and the
        # ground under it, and `traces.fit_records` is what reads it out. One
        # of the two is an unfinished answer, and picking which half was meant
        # is worse than refusing.
        self.has_attitudes = bool(dip_dir_field) and bool(dip_field)

        self.anchor_field = anchor_field
        self.span_fields = span_fields
        self.enabled_field = enabled_field
        self.half_span = half_span
        self.problem = None
        self.dropped = {}
        self.outside = 0

        self.traces = []
        self.num_lines = 0
        self._records = None

        try:
            frame = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
        except Exception as err:
            self.problem = str(err).split("\n")[0]
            return

        if frame.crs is None:
            # Said plainly because it happens to real data and the cure is
            # elsewhere: the traces of Timpa San Lorenzo ship with an empty
            # .prj, while the profile and the faults beside them declare one.
            self.problem = "no CRS declared by the layer"
            return

        if bool(dip_dir_field) != bool(dip_field):
            self.problem = "one angle field named and the other not"
            return

        if self.has_attitudes:
            for name in (dip_dir_field, dip_field):
                if name not in frame.columns:
                    self.problem = f"no field '{name}'"
                    return

        frame = frame.to_crs(crs)

        # What can hold a line, which is not only what is named after one. A
        # GeometryCollection is how a mapped fault comes back from a cleaning
        # pass that left a node behind, and the line inside it is still the
        # fault: `_lines` takes the line parts and counts the rest. Admitting
        # it here rather than filtering it out is the difference between a
        # trace used and a trace that disappeared without a word.
        usable = frame.geometry.notna() & frame.geometry.geom_type.isin(
            ("LineString", "MultiLineString", "GeometryCollection")
        )

        no_line_geometry = int((~usable).sum())
        frame = frame[usable]

        if frame.empty:
            self.problem = "no line geometry"
            return

        if self.has_attitudes:
            azimuth = numeric(frame[dip_dir_field])
            dip = numeric(frame[dip_field])

            keep, self.dropped = admissible(azimuth, dip)

            if no_line_geometry:
                self.dropped["with no line geometry"] = no_line_geometry

            if not keep.any():
                self.problem = "no readable attitude"
                return

            frame = frame[keep]
            azimuth, dip = normalised_azimuth(azimuth[keep], dip[keep]), dip[keep]
        else:
            # Nothing to test for admissibility, and so nothing dropped for it:
            # a line with no angles on it cannot be unreadable, only unread.
            azimuth = dip = None

            if no_line_geometry:
                self.dropped["with no line geometry"] = no_line_geometry

        if bounds is not None:
            # Kept whole or dropped whole. A trace is clipped by the profile it
            # is intersected with, not by the map window, and cutting it here
            # would invent an endpoint where the sheet ends.
            from shapely.geometry import box

            left, bottom, right, top = bounds
            inside = frame.intersects(box(left, bottom, right, top)).to_numpy()

            self.outside = int((~inside).sum())

            frame = frame[inside]

            if self.has_attitudes:
                azimuth, dip = azimuth[inside], dip[inside]

            if frame.empty:
                self.problem = "no trace in the area"
                return

        self.frame = frame
        self.traces = (
            self._group(frame, azimuth, dip) if self.has_attitudes else self._bare(frame)
        )
        self.num_lines = sum(len(record.lines) for record in self.traces)

    # -- reading -----------------------------------------------------------

    def _categories(self, frame):
        """
        The name each row goes under, or one name for the whole layer.

        With no field chosen everything is one system, named after the layer:
        a profile still gets its ticks, and the alternative -- one category per
        feature -- would put a legend entry on every fragment of one fault.
        """

        if not self.category_field or self.category_field not in frame.columns:
            return [self.layer or self.path.stem] * len(frame)

        return frame[self.category_field].astype("string").fillna("n/a").astype(str).tolist()

    def _group(self, frame, azimuth, dip):
        """
        The records, pooled on (category, attitude, anchor) as read off the table.

        The angles are compared as they came rather than rounded: two fragments
        of one plane carry the identical value, having been computed from it,
        while two measurements that differ differ for a reason and stay apart.

        The anchor joins the key because two readings can agree and still be
        two readings, taken at two places on the same fault -- pooling them
        would leave one of the two places with nothing at it.
        """

        from geogst.core.geology.orientations import Plane

        anchors = self._optional(frame, self.anchor_field)
        starts = self._optional(frame, self.span_fields[0])
        ends = self._optional(frame, self.span_fields[1])
        enabled = self._flags(frame, self.enabled_field)
        columns = self._attr_columns(frame)

        pooled, first = defaultdict(list), {}
        not_lines = 0

        for ndx, (category, a, d, geometry) in enumerate(
            zip(self._categories(frame), azimuth, dip, frame.geometry)
        ):
            anchor = None if anchors is None else _finite(anchors[ndx])
            key = (category, float(a), float(d), anchor)

            lines, skipped = self._lines(geometry)

            not_lines += skipped
            pooled[key].extend(lines)
            first.setdefault(key, ndx)

        if not_lines:
            self.dropped["not a line"] = not_lines

        records = []

        for (category, a, d, anchor), lines in pooled.items():
            # A collection that held no line at all leaves a key behind with
            # nothing under it. The parts are already counted; what must not
            # happen is a row in the panel, and an entry in the legend, for a
            # measurement with no trace to put it on.
            if not lines:
                continue

            ndx = first[(category, a, d, anchor)]

            span = None
            if starts is not None and ends is not None:
                s0, s1 = _finite(starts[ndx]), _finite(ends[ndx])
                if s0 is not None and s1 is not None:
                    span = (s0, s1)

            records.append(
                TraceRecord(
                    category=category,
                    plane=Plane(a, d, is_rhr_strike=self.is_rhr_strike),
                    lines=lines,
                    length=float(sum(line.length_2d() for line in lines)),
                    anchor=anchor,
                    span=span,
                    enabled=True if enabled is None else bool(enabled[ndx]),
                    attrs=self._attrs_at(columns, ndx),
                )
            )

        return records

    def _bare(self, frame):
        """
        One record per feature, none of them carrying a plane yet.

        Nothing is pooled here, and that is the difference from `_group` rather
        than an omission in it. Two fragments are pooled there because they
        carry the identical attitude, and carrying it is the evidence that they
        are one plane digitised in pieces. With no attitude there is no such
        evidence -- only a shared name, and a category on a CARG sheet is
        `contatto stratigrafico e/o litologico` sixteen thousand times over.
        Pooling on that would hand the fit one curve made of every contact in
        the sheet.

        No anchor and no span either. An anchor is where a measurement was
        taken and there is no measurement; a span is how far one reaches.
        `enabled` is read, because holding a contact out of the section is a
        decision that still means something with no plane on it.
        """

        enabled = self._flags(frame, self.enabled_field)
        columns = self._attr_columns(frame)

        records, not_lines = [], 0

        for ndx, (category, geometry) in enumerate(
            zip(self._categories(frame), frame.geometry)
        ):
            lines, skipped = self._lines(geometry)

            not_lines += skipped

            # Same reason as in `_group`: the parts are counted, and what must
            # not happen is a row in the panel for a trace that is not there.
            if not lines:
                continue

            records.append(
                TraceRecord(
                    category=category,
                    plane=None,
                    lines=lines,
                    length=float(sum(line.length_2d() for line in lines)),
                    enabled=True if enabled is None else bool(enabled[ndx]),
                    attrs=self._attrs_at(columns, ndx),
                )
            )

        if not_lines:
            self.dropped["not a line"] = not_lines

        return records

    def _attr_columns(self, frame):
        """
        The columns that are neither geometry nor already a field of a record,
        lifted out of the loop.

        Read row by row this is `frame.iloc[ndx]`, which builds a Series per
        call: unnoticeable over the fifty records a curated fault layer makes,
        and eight and a half seconds over the twenty-four thousand a bare CARG
        sheet makes. The lists are in the frame's own order, which is what the
        indices here count in.
        """

        used = {
            self.category_field,
            self.dip_dir_field,
            self.dip_field,
            self.anchor_field,
            self.enabled_field,
            *self.span_fields,
            frame.geometry.name,
        }

        return {
            str(name): frame[name].tolist()
            for name in frame.columns
            if name not in used
        }

    @staticmethod
    def _attrs_at(columns, ndx):
        """One record's share of them, nulls left out."""

        return {
            name: values[ndx]
            for name, values in columns.items()
            if values[ndx] is not None and values[ndx] == values[ndx]
        }

    @staticmethod
    def _optional(frame, name):
        """A numeric column if the layer has one, else nothing to read."""

        return numeric(frame[name]) if name and name in frame.columns else None

    @staticmethod
    def _flags(frame, name):
        """A truth column, with anything unreadable counting as on."""

        if not name or name not in frame.columns:
            return None

        return frame[name].fillna(True).astype(bool).to_numpy()

    @staticmethod
    def _lines(geometry):
        """
        One geometry as geogst lines, flat, and how many parts were not lines.

        The coordinates go across as an array rather than a list of points:
        `Ln` takes one, and the traces of a mapped fault run to hundreds of
        vertices each.

        The containers are unwrapped by `single_parts` rather than by a test on
        the outer type, because a GeometryCollection does not announce itself
        as one: it is the only one whose name does not begin with 'Multi', and
        a fault that has been through a cleaning pass comes back as exactly
        that -- its line plus whatever node the pass left behind.
        """

        from geogst.core.geometries.shapes.lines import Ln

        from .vectors import single_parts

        parts, skipped = single_parts(geometry, "LineString")

        return [Ln(np.asarray(part.coords)[:, :2]) for part in parts], skipped

    @staticmethod
    def candidate_layers(path):
        """The line layers in the file, from the metadata alone."""

        from .vectors import VectorSource

        return VectorSource.candidate_layers(path, "lines")

    # -- what was read -----------------------------------------------------

    @property
    def records(self):
        """
        What `intersect_lines_with_attitudes` is given: category -> [(plane, lines)].

        Derived rather than stored, because a span is meant to be argued with
        while the section it feeds is on screen, and a dictionary that outlived
        a change to one would be a section disagreeing with its own legend.
        Held until something moves it.

        A record with no plane contributes nothing, because the tick a section
        carries is an apparent dip and there is none to compute. A bare trace
        layer is therefore inert here until it has been fitted, which is the
        honest answer: drawing it anyway would put a contact in the section as
        though it had been read, when all that is known is where it crops out.
        """

        if self._records is None:
            grouped = defaultdict(list)

            for record in self.traces:
                if not record.enabled or record.plane is None:
                    continue

                lines = record.lines

                if not record.is_whole(self.half_span):
                    s0, s1 = record.extent(self.half_span)
                    lines = clip_to_span(lines, s0, s1)

                if lines:
                    grouped[record.category].append((record.plane, lines))

            self._records = dict(grouped)

        return self._records

    def set_half_span(self, metres):
        """How far an anchored measurement reaches when it says nothing itself."""

        self.half_span = metres
        self._records = None

    def set_traces(self, records):
        """
        Put a different set of records in front of the layer's own.

        For `traces.fit_records`, which reads an attitude off each trace and the
        topography instead of off the columns, and gives back several records
        where one trace determines several planes. The layer is not touched and
        nothing here is written back to it: `read()` would give the survey
        again, which is the same bargain the curation file strikes.
        """

        self.traces = list(records)
        self.num_lines = sum(len(record.lines) for record in self.traces)
        self._records = None

    def set_span(self, record, s0, s1):
        """
        Fix one record's interval, or hand it back to the default with None.

        Set, not merged: this is the one record's own answer, and the caller
        writing it down elsewhere as an assertion is what keeps the source
        layer untouched.
        """

        record.span = None if s0 is None or s1 is None else (float(s0), float(s1))
        self._records = None

    def set_enabled(self, record, flag):
        record.enabled = bool(flag)
        self._records = None

    def set_plane(self, record, azimuth, dip):
        """
        Correct one record's attitude.

        A reading is transcribed before it is used and a transcription can be
        wrong; and a fit is a calculation, which can be overruled by somebody
        who stood on the outcrop. Either way what changes is this record, and
        the file it was read from is not touched -- the correction is written
        down as an assertion elsewhere, or it is not written down at all.
        """

        from geogst.core.geology.orientations import Plane

        record.plane = Plane(
            float(azimuth) % 360.0, float(dip), is_rhr_strike=self.is_rhr_strike
        )
        self._records = None

    @property
    def is_loaded(self):
        return bool(self.traces)

    def __len__(self):
        """How many records: planes, not fragments."""

        return len(self.traces)

    def attitudes(self):
        """Every record that has one, as (category, dip direction, dip)."""

        return [
            (record.category, record.plane.dipazim, record.plane.dipang)
            for record in self.traces
            if record.plane is not None
        ]

    @property
    def unread(self):
        """How many records are still waiting for a plane."""

        return sum(1 for record in self.traces if record.plane is None)

    def summary(self):
        if self.problem:
            return f"{self.layer or self.path.stem}: {self.problem}"

        categories = {record.category for record in self.traces}

        # Called planes where they are planes and traces where they are not:
        # a bare layer holds no plane at all yet, and counting its contacts as
        # planes would announce a section's worth of data that is not there.
        unread = self.unread

        if unread == len(self):
            # Nothing has been read off this layer, so there is nothing to
            # count but the contacts themselves -- and the parts only where a
            # multipart geometry makes them differ from the records.
            text = f"{self.layer or self.path.stem}: {len(self)} traces with no attitude read"

            if self.num_lines != len(self):
                text += f" in {self.num_lines} parts"

            text += f", {len(categories)} categories -- fit them off the trace"
        else:
            # Planes, not records: with some of them read by hand and the rest
            # not, `len(self)` is the two together and would overstate what the
            # section has to draw by exactly the number still waiting.
            text = (
                f"{self.layer or self.path.stem}: {len(self) - unread} planes in "
                f"{len(categories)} categories, {self.num_lines} traces"
            )

            if unread:
                text += f"; {unread} with no attitude read yet"

        anchored = sum(1 for record in self.traces if record.anchor is not None)
        off = sum(1 for record in self.traces if not record.enabled)

        if anchored:
            text += f"; {anchored} anchored at a point"

        if off:
            text += f", {off} held out of the section"

        if self.dropped:
            detail = ", ".join(f"{count} {reason}" for reason, count in self.dropped.items())
            text += f" ({detail})"

        if self.outside:
            text += f"; {self.outside} outside the map"

        return text
