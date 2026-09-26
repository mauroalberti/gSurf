"""
A section as a thing that can be put down and picked up again.

The tool already remembered one: the trace, the framing, the bundle and the
reach were written on the way out and taken back up on the way in, so a morning
did not start by dragging yesterday's line again. But that was one slot,
overwritten by the next section drawn -- which makes it a way of continuing and
not a way of keeping. A section arrived at over an afternoon is a result: it
belongs in the directory the data is in, under a name, beside the figure it
produced, in whatever the rest of the work is versioned with.

So the same payload goes out to a file as well. Nothing here is a second format
for it: `payload` is `ProfilesWindow.current_state` plus what a file needs that a
store does not -- which projection its metres are in, and what wrote it.

**The file is the two ends and the numbers, not the section.** A section is
recomputed from a trace, a count, a spacing and a DEM, in a quarter of a second;
storing the profiles themselves would be storing the slow, large, derived half
of that, and it would be wrong the moment the DEM under it was improved. What is
here is what was *decided* -- where the line goes and how wide the bundle is --
and the topography is read again.

**Opened, it is fitted to the session rather than asserted onto it.** The
implicit slot can be strict because it is silent: `applicable` drops a trace
from another source and comes up in the middle of the DEM, and nobody is owed an
explanation for a state they never asked to be restored. A file is asked for by
name, and the trace is the whole reason it was opened -- so here the ground is
met halfway. Metres in another projection are reprojected onto this one. A
framing that cannot be carried is replaced by one made from the trace, because a
section opened outside the view is a file that appears to have done nothing. And
what cannot be met at all -- a trace that is not on this DEM -- is refused
whole, with the numbers, rather than half-applied.

Both doors read the same numbers through the same checks, which is why those
live here and not in the tool: the conf file and the section file can each be
edited by hand, and a number that `Profilers` raises on must not get in through
either. Nothing in this module imports Qt, so a saved section can be opened in a
script -- and `checks/check_section_files.py` does exactly that, against ground
whose answers are known by construction.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# What the controls can hold, and so what either kind of stored section may say.
# Here rather than beside the spin boxes because a stored number is checked
# against the same bounds the box would clamp to: a number the box would clamp
# and a section computed from the number before clamping are two different
# things, and the gap between them is a box that disagrees with the map. The
# count is odd throughout -- a central bundle has a middle, and `Profilers`
# raises without one.
BUNDLE_RANGE = (1, 41)
OFFSET_RANGE = (10.0, 20000.0)
REACH_RANGE = (0.0, 50000.0)

# The first key of the file, and the reason a mistyped name is refused instead
# of half-read. Any JSON at all parses; this says the parsed thing was meant to
# be a section.
MARKER = "gsurf_section"
FORMAT = 1
SUFFIX = ".json"

# Room around the trace when a framing has to be made rather than carried. A
# quarter of the length either way: enough to see which way the line runs out of
# the view, not so much that the section is a speck in the middle of a mosaic.
FRAME_MARGIN = 0.25

# Below this the reprojected middle is not worth a sentence. A straight line in
# one projection is not straight in another, so the two ends can be carried over
# exactly and the ground the line crosses between them still moves; over a
# section's length between neighbouring zones that is metres, which is under the
# cell of any DEM this is used on. Said when it is more.
DEVIATION_WORTH_SAYING_M = 5.0


class SectionError(ValueError):
    """
    What is wrong with a file, in words fit to put in a box.

    Raised as itself for the one refusal that cannot be sorted into either of
    the two below: a trace with no length on this ground is either a press with
    no drag or coordinates this projection makes nothing of, and nothing here
    can tell which.
    """


class NotASection(SectionError):
    """Not this format: unparseable, unmarked, or from a later version of it."""


class SectionElsewhere(SectionError):
    """A section of somewhere that is not the ground this session is open on."""


# -- the payload -----------------------------------------------------------


def payload(state, crs=None, when=None):
    """
    The state as a file writes it: the same numbers, plus where they were measured.

    The projection goes in as WKT and as an EPSG code. The code is what a person
    reads and what the quick comparison uses; the WKT is what a transformer can
    always be built from, including for the projections that have no code --
    `Session` takes its CRS from whatever was opened, and a DEM written with a
    bare projection string has none to give. Neither is derivable from the
    other here, so both are written.

    `source` is carried for the same reason it is in the stored state, and it is
    provenance rather than a key: the file says which DEM the line was dragged
    over, and `open_onto` will still put it on another one covering the same
    ground in the same projection. A trace is metres in a projection; it is not
    a property of a raster.
    """

    written = (when or datetime.now(timezone.utc)).replace(microsecond=0)

    out = {
        MARKER: FORMAT,
        "written": written.isoformat(),
        "source": state.get("source") or None,
        "epsg": state.get("epsg"),
        "crs": None if crs is None else crs.to_wkt(),
    }

    # In the order the tool thinks about them: where the line is, how wide the
    # bundle is, how far a measurement reaches, and then the view -- which is
    # the one thing in here that is about looking rather than about the section.
    for key in ("trace", "profiles", "offset", "reach", "extent", "legend"):
        if key in state:
            out[key] = state[key]

    return out


def save_text(state, crs=None, when=None):
    """The file's whole text, newline-terminated."""

    # Indented and so diffable. A section is a handful of numbers that lands in
    # the same directory as the data and, often, in the same repository: a
    # one-line JSON would turn moving the trace two hundred metres into a
    # changed file with nothing readable in the change.
    return json.dumps(payload(state, crs=crs, when=when), indent=2) + "\n"


def write(path, state, crs=None, when=None):
    """
    The text to a file, with the line endings it was built with.

    `newline=""` because text mode translates `"\\n"` to `os.linesep` on the way
    out, which on Windows would write CRLF into a file every other tool in this
    project reads and writes LF. The rule is the trace editor's, arrived at the
    hard way over a file it had promised not to change; this one is written
    whole every time, so it costs nothing to keep.
    """

    path = Path(path)
    path.write_text(save_text(state, crs=crs, when=when), encoding="utf8", newline="")

    return path


def read(path):
    """
    A file as its payload, or a refusal that says which of the three it was.

    The marker is checked before anything is read out of it. Any JSON parses,
    and a dictionary that happens to have a `trace` in it could come from
    anywhere; a file picked in a dialog by mistake has to be refused as the
    wrong file rather than reported as a section with things missing.
    """

    path = Path(path)

    try:
        content = json.loads(path.read_text(encoding="utf8"))
    except (OSError, UnicodeDecodeError) as err:
        raise NotASection(f"{path.name} could not be read: {err}") from err
    except ValueError as err:
        raise NotASection(
            f"{path.name} is not JSON, so it is not a saved section: {err}"
        ) from err

    if not isinstance(content, dict) or MARKER not in content:
        raise NotASection(
            f"{path.name} is JSON but not a saved section: it has no "
            f'"{MARKER}" in it.'
        )

    version = content[MARKER]

    # Refused forward and not backward, the rule `gstruct` uses for its own
    # header: a file from a later version may be using something this reader
    # would skip, and a section quietly opened without the half it did not
    # understand is worse than one that would not open.
    if not isinstance(version, int) or version > FORMAT:
        raise NotASection(
            f"{path.name} is a section written to format {version}, and this "
            f"reads up to {FORMAT}. Update gSurf."
        )

    return content


# -- opening one onto a session --------------------------------------------


class Opened:
    """
    What a file turned into here, and everything that had to be done to it.

    The notes are the point of this being an object rather than a dictionary.
    Reprojecting a trace, dropping a framing, ignoring a spacing no box could
    hold -- each of those is the file being disagreed with, and a tool that
    disagreed silently would be one you could not trust to have opened what you
    saved.
    """

    def __init__(self, state, notes=(), moved=None):
        self.state = state
        self.notes = list(notes)
        self.moved = moved          # metres the middle of the line shifted, or None

    def summary(self):
        """The lines to put in the box that says it opened."""

        (x0, y0), (x1, y1) = self.state["trace"]
        length = float(np.hypot(x1 - x0, y1 - y0))

        said = [
            f"{length / 1000.0:.2f} km, "
            f"{self.state.get('profiles', '?')} profile(s) at "
            f"{self.state.get('offset', 0.0):.0f} m."
        ]

        return "\n\n".join(said + self.notes)


def open_onto(content, session):
    """
    A payload as a state this session can be set to, or a refusal.

    The habits come across as they do through `applicable` and for the same
    reason -- a count and a spacing are ways of working, and the ranges are
    what the boxes hold. The places are where this differs: they are adapted
    rather than dropped, because they are what was asked for.
    """

    notes = []
    kept = _habits(content)

    for key, what in (
        ("profiles", "The count"),
        ("offset", "The spacing"),
        ("reach", "The reach"),
    ):
        if key in content and key not in kept:
            notes.append(
                f"{what} in the file ({content[key]!r}) is not one the tool can "
                f"be set to, so it was left as it is."
            )

    if "legend" in content and isinstance(content["legend"], bool):
        kept["legend"] = content["legend"]

    ends = _two_ends(content.get("trace"))

    if ends is None:
        raise NotASection(
            "The file carries no usable trace: a section is two ends, each a "
            "pair of numbers."
        )

    source_crs = _crs_of(content)
    trace, moved = _onto_crs(ends, source_crs, session.crs)

    # After the reprojection and not before it, which is a thing this got wrong
    # first time round: a metre is a length in the session's projection and the
    # file's numbers need not be metres at all. A section written in degrees is
    # a tenth of a degree long -- ten kilometres of ground -- and a floor of one
    # applied to the file as it came would have refused it as a section with no
    # length in it.
    # The base class and neither of the two below it, deliberately: which of
    # them this is cannot be told apart here, and the message says both.
    if not _has_length(trace):
        raise SectionError(
            "The two ends of that section are "
            f"{_length(trace):.2f} m apart on this ground, which is not a "
            "section. Either the file holds a press with no drag after it, or "
            "its coordinates are not what this projection makes of them."
        )

    if moved is not None and moved > DEVIATION_WORTH_SAYING_M:
        notes.append(
            f"The trace was written in {_crs_label(content)} and reprojected "
            f"onto EPSG:{session.epsg}. The two ends are where they were; the "
            f"ground the straight line between them crosses has moved by up to "
            f"{moved:.0f} m in the middle, a straight line in one projection "
            f"not being one in another."
        )
    elif source_crs is None:
        # Not a refusal: a file with no projection in it is most likely one
        # written by hand, and the numbers may well be right. But nothing
        # checked them, so it is said.
        notes.append(
            "The file says nothing about which projection its coordinates are "
            f"in. They were taken as EPSG:{session.epsg}, unchanged."
        )

    _must_be_on(trace, session)

    kept["trace"] = trace
    extent = _as_extent(content.get("extent")) if _same_ground(content, session) else None

    if extent is not None:
        kept["extent"] = extent
    else:
        # Always something, and this is why it is here rather than left to the
        # tool: a view is not part of the section, but a section opened outside
        # the view is indistinguishable from a file that did nothing.
        kept["extent"] = framing_for(trace)

        if content.get("extent") is not None:
            notes.append(
                "The framing was not carried -- it is a rectangle in the "
                "projection it was written in -- so the map was put on the "
                "trace instead."
            )

    return Opened(kept, notes=notes, moved=moved)


def framing_for(trace, margin=FRAME_MARGIN):
    """A view with the trace across the middle of it, as `extent` is ordered."""

    (x0, y0), (x1, y1) = trace

    # Off the length and not off each side: a north-south section has no width
    # to take a margin from, and a fraction of nothing is nothing.
    room = margin * float(np.hypot(x1 - x0, y1 - y0))

    return [
        min(x0, x1) - room, max(x0, x1) + room,
        min(y0, y1) - room, max(y0, y1) + room,
    ]


def _habits(state):
    """The numbers that travel anywhere, each checked against what a box holds."""

    kept = {}
    count, offset = state.get("profiles"), state.get("offset")

    if isinstance(count, int) and count % 2 == 1 and _within(count, BUNDLE_RANGE):
        kept["profiles"] = int(count)

    if isinstance(offset, (int, float)) and _within(offset, OFFSET_RANGE):
        kept["offset"] = float(offset)

    # None is a value here and the one the box calls "whole trace", so it is
    # kept as it comes; a zero is not, the box reading that as the whole trace
    # too and `_on_reach_changed` never writing one.
    reach = state.get("reach")

    if reach is None and "reach" in state:
        kept["reach"] = None
    elif isinstance(reach, (int, float)) and REACH_RANGE[0] < reach <= REACH_RANGE[1]:
        kept["reach"] = float(reach)

    return kept


def _crs_of(content):
    """The projection the file's metres are in, or None if it does not say."""

    from pyproj import CRS

    wkt = content.get("crs")

    if isinstance(wkt, str) and wkt.strip():
        try:
            return CRS.from_user_input(wkt)
        except Exception:
            # Falls through to the code. A WKT this cannot parse and an EPSG
            # beside it is a file worth opening on the code -- which is the
            # case `checks/` has a fixture for, a truncated WKT that pyproj
            # refuses while the code next to it is perfectly good.
            pass

    epsg = content.get("epsg")

    if isinstance(epsg, int):
        try:
            return CRS.from_epsg(epsg)
        except Exception as err:
            raise NotASection(f"EPSG:{epsg} in the file is not a projection: {err}")

    return None


def _crs_label(content):
    epsg = content.get("epsg")

    return f"EPSG:{epsg}" if isinstance(epsg, int) else "another projection"


def _onto_crs(ends, source_crs, target_crs):
    """
    The two ends in the session's projection, and how far the middle moved.

    The deviation is measured rather than bounded by a formula: the file's own
    midpoint is reprojected and compared with the midpoint of the reprojected
    chord. That is the whole of what is lost by carrying a section across
    projections, in metres, on this line -- and it is reported instead of being
    assumed small, because whether it is depends on the length and on how far
    apart the two projections are.
    """

    if source_crs is None or target_crs is None or source_crs.equals(target_crs):
        return [tuple(float(v) for v in end) for end in ends], None

    from pyproj import Transformer

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    (x0, y0), (x1, y1) = ends
    xs, ys = transformer.transform([x0, x1, (x0 + x1) / 2.0], [y0, y1, (y0 + y1) / 2.0])

    trace = [(float(xs[0]), float(ys[0])), (float(xs[1]), float(ys[1]))]
    middle = ((trace[0][0] + trace[1][0]) / 2.0, (trace[0][1] + trace[1][1]) / 2.0)

    moved = float(np.hypot(xs[2] - middle[0], ys[2] - middle[1]))

    return trace, moved


def _must_be_on(trace, session):
    """Raises unless both ends are on the ground the session covers."""

    left, bottom, right, top = session.bounds
    away = [
        (x, y) for x, y in trace
        if not (left <= x <= right and bottom <= y <= top)
    ]

    if not away:
        return

    # With the numbers, and in kilometres, because this is the message that has
    # to be actionable without the file being opened in an editor: the usual
    # cause is the right section over the wrong DEM, and the two extents beside
    # each other say so at a glance.
    raise SectionElsewhere(
        "This section is not on the ground now open. "
        + ", ".join(f"one end at {x:.0f} {y:.0f}" for x, y in away)
        + f", and the session covers {left:.0f} {bottom:.0f} to {right:.0f} "
        f"{top:.0f}. Open the DEM it was drawn on, or one that covers it."
    )


def _same_ground(content, session):
    """Whether a rectangle in the file is a rectangle here."""

    return content.get("epsg") is not None and content.get("epsg") == session.epsg


# -- the stored state, which is the same numbers through the same checks ----


# Where the tool keeps the section it was last left on. One JSON string rather
# than a key per number, for the reason `recent.py` gives about its own lists:
# QSettings has no faithful round trip for a nested list, and reads a
# one-element one back as a scalar.
STATE_KEY = "state/last"


def read_state(settings):
    """What the last run wrote about the section, or nothing."""

    if settings is None:
        return {}

    raw = settings.value(STATE_KEY)

    if not isinstance(raw, str):
        return {}

    try:
        state = json.loads(raw)
    except ValueError:
        return {}

    return state if isinstance(state, dict) else {}


def applicable(session, state):
    """
    The part of a remembered state that belongs to the session being opened.

    Two kinds of thing are in there and they travel differently.

    How many profiles at what spacing, and how far a point measurement
    reaches, are ways of working. They are not tied to anywhere and they come
    back whatever is open -- somebody who works in bundles of thirteen at
    250 m works that way in the next area too.

    Where the trace was and how the map was framed are *places*: metres in a
    projection. The same pair of numbers is somewhere else under a different
    projection and nowhere at all under a DEM of another region, so those come
    back only over the source they were written on, and are dropped rather
    than guessed at -- a section restored off the DEM would come up empty with
    nothing to say why, which is worse than coming up in the middle.

    Which is the whole difference between this and `open_onto`, and it is a
    difference in what was asked for rather than in how careful either is: this
    runs on every open, with no one to tell, so it prefers the middle of the DEM
    to a guess. A file was named, so it is adapted and reported on.
    """

    kept = _habits(state)

    # Whether the legend is up is a habit like the rest of them: somebody
    # reading sections wants the names, somebody preparing a figure does not,
    # and neither has anything to do with which DEM is open.
    if isinstance(state.get("legend"), bool):
        kept["legend"] = state["legend"]

    if state.get("source") != source_key(session) or state.get("epsg") != session.epsg:
        return kept

    trace = _as_trace(state.get("trace"), session.bounds)
    if trace is not None:
        kept["trace"] = trace

    extent = _as_extent(state.get("extent"))
    if extent is not None:
        kept["extent"] = extent

    return kept


def _within(value, limits):
    return limits[0] <= value <= limits[1]


def source_key(session):
    """What the coordinates in a state were measured on."""

    return str(session.base_path) if session.base_path is not None else ""


def _two_ends(value):
    """Two pairs of numbers, in whatever units they were written in."""

    try:
        (x0, y0), (x1, y1) = ((float(x), float(y)) for x, y in value)
    except (TypeError, ValueError):
        return None

    return [(x0, y0), (x1, y1)]


def _length(trace):
    (x0, y0), (x1, y1) = trace

    return float(np.hypot(x1 - x0, y1 - y0))


def _has_length(trace, least=1.0):
    """
    Whether there is a section here at all, in the units the trace is now in.

    A section with no length is what the tool refuses to compute anyway, and a
    press with no drag after it leaves exactly that behind. Only ever asked of a
    trace already on the session's ground -- see `open_onto`.
    """

    return _length(trace) >= least


def _as_trace(value, bounds):
    """Two ends on this DEM, with a length between them, or None."""

    ends = _two_ends(value)

    if ends is None or not _has_length(ends):
        return None

    left, bottom, right, top = bounds

    for x, y in ends:
        if not (left <= x <= right and bottom <= y <= top):
            return None

    return ends


def _as_extent(value):
    """A framing that can be set on an axis, or None."""

    try:
        left, right, bottom, top = (float(v) for v in value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite([left, right, bottom, top]).all():
        return None

    if not (left < right and bottom < top):
        return None

    return [left, right, bottom, top]
