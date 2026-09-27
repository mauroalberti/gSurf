"""
The topography read along one trace, as `fit`s: the producer, away from its one caller.

`imports` wrote this and is no longer the only thing that wants it. What it does
is per structure -- one path in, zero or many anchored intervals out -- and none
of that belongs to transcribing a layer: the trace editor has a path, a DEM, and
a curator looking at what already holds along it, which is exactly the place
where a plane read off the ground is worth having one trace at a time.

So this is `_fit_along` and the four things it needed, with the import dialog's
`Mapping` taken out of it. Two changes came with the move, and both are about
what a second caller needs that the first did not:

**The four sweep numbers travel as a `Sweep`** rather than as attributes of an
object that also knows which column holds an ident. They were always chosen
together -- a step half a window apart is a different sampling, and a fallback is
only a fallback relative to the lengths swept -- so they are one thing with one
set of defaults, in one place, instead of defaults in a dataclass and defaults in
a function signature.

**The outcome comes back as a word, not as a tally.** Per trace it is exactly one
of five, and the import's `Imported` counts them because a report over twelve
thousand traces is a count. A caller with one trace and a status line needs the
word; deriving it back out of four counters that each went up by one would be
reading a histogram to find a scalar.

What has not changed is the arithmetic, the gate, or anything written into a
`fit`: `checks/check_imports.py` asserts the planes, the widening, the ring, and
the straight trace that carries nothing, and it is the net under this move.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# What a fit off the topography says made it. FORMAT.md's own name for this
# producer, and the thing `attitude_at` has no opinion about -- `from=` is read
# by people and by anything asking which producer to believe.
FROM_DEM = "trace-dem"

# How near an end of the path counts as being at it: a centimetre, which is the
# precision anchors are written to anyway.
AT_THE_END = 0.01

# How many decimals a fitted plane is written to, where it is written as a line
# rather than through `dumps`. Not a constraint -- `dumps` writes whole degrees
# and the parser reads floats either way -- so it is a choice, and the choice is
# that a plane computed over a swept window is a computed number: rounding it as
# it is written throws away the only thing separating one window's answer from
# the next one's, and the diagnostics beside it already say how well determined
# it is. The value does not also have to pretend.
PLANE_DECIMALS = 1


@dataclass(frozen=True)
class Sweep:
    """
    How the window is swept along a trace: four numbers, which are chosen together.

    Not independently, and that is the whole reason they are one object.
    Lengthening the window accumulates more turn and so clears the gate's floor,
    but it also averages away a real change of attitude -- so the floor and the
    length trade against each other, and a step is a sampling of whatever length
    was picked rather than a number with a meaning of its own.

    `lengths` empty means `traces.DEFAULT_SWEEP`, which is the set those costs
    were measured on. Left as a default here rather than filled in, so that a
    caller passing nothing and a caller passing the default are the same call.
    """

    lengths: tuple = ()
    step: float = 25.0
    fallback: float = 250.0
    keep: str = "held"


# What reading one trace came to. Exactly one of these, always, and they are
# apart because a single "carries no attitude" count reports the wrong one:
# measured on `elementi_tettonici`, 6371 of 12718 traces are off the 5 m DEM
# entirely and the median trace is shorter than the window, so lumping the three
# would say the topography refused a sheet it was never asked about.
UNREACHED = "unreached"     # no elevation under the trace at all
TOO_SHORT = "too-short"     # shorter than the shortest window that was swept
SWEPT = "swept"             # the trace chose its own window length
SHORTENED = "shortened"     # read at less than the fallback, that being all that fit
FALLBACK = "fallback"       # read at the fallback, nothing having picked a length


@dataclass
class Reading:
    """
    What the topography said along one trace, and how it came to say it.

    `spans` is the whole `TraceSpans` and not a summary of it, because the useful
    thing to show when nothing came out is the metres under each verdict -- and
    which of those is worth quoting depends on who is asking. The section panel
    wants a share of the layer; a curator looking at one fault wants to know
    whether the gate said `loose` or said `line`, since the first is an argument
    for a longer window and the second is not.
    """

    fits: list = field(default_factory=list)
    outcome: str = UNREACHED
    length: float = None        # the window the fits were read over, where there was one
    spans: object = None        # the `TraceSpans` they came out of, or None

    @property
    def silent(self):
        """
        Sampled, long enough, and nothing held: a verdict rather than a failure.

        The import counts this separately from the two ways of never being asked,
        and the distinction is the same one here: a trace the gate refused has
        been read and answered about, and offering to read it again at the same
        settings would be offering to get the same answer.
        """

        return not self.fits and self.outcome not in (UNREACHED, TOO_SHORT)

    def describe(self):
        """One line of what happened, for somebody who asked about one trace."""

        if self.outcome == UNREACHED:
            return "no elevation anywhere under this trace: it is off the DEM"

        if self.outcome == TOO_SHORT:
            return "shorter than the shortest window swept: nothing was read"

        how = {
            SWEPT: "the length it holds at",
            SHORTENED: "all that fits on it",
            FALLBACK: "the fallback: no length held better than another",
        }[self.outcome]

        over = f"read over {self.length:.0f} m ({how})"

        if self.fits:
            return f"{len(self.fits)} fit(s), {over}"

        metres = self.spans.metres() if self.spans is not None else {}
        said = ", ".join(
            f"{metres[verdict]:.0f} m {verdict}"
            for verdict in ("held", "loose", "line")
            if metres.get(verdict)
        )

        return f"nothing held, {over}" + (f": {said}" if said else "")


def dem_refusal(dem, frame_crs):
    """
    Why this DEM cannot be sampled for traces in this projection, or None.

    One CRS or nothing, and the refusal is not fussiness. The trace would be
    sampled in the DEM's grid and the fit's anchors written in the file's, so a
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
        f"and the traces in "
        f"{frame_crs.to_string() if frame_crs else 'no declared CRS'}. A plane "
        f"read off the topography is a dip direction measured from the DEM's "
        f"north and written against the traces': reproject one of them first, "
        f"because nothing here can make that mean the same thing."
    )


def gate_for(paths):
    """
    The gate to read a set of traces through, as `(gate, the sigma it measured)`.

    **The lever floor is measured off the traces themselves**, which is what
    `TraceGate.from_traces` exists for: a departure from straightness smaller
    than the wander of the pen that drew the line is not evidence of a turn, and
    how much the pen wandered is a property of the drawing rather than a
    constant. Where the measurement does not come back -- a self-affine layer,
    which is what a mapped contact usually is -- the class default stands, and
    the sigma comes back `None` so that the caller can say so rather than quote
    a number that was extrapolated.

    The wording is the caller's and not this function's, because the two callers
    write for different readers: the import puts its sentence in the file's
    header, in the language the header is in, and the editor puts its own in a
    status bar. One sentence serving both would be in the wrong language once.

    Off every path together, never one: a floor measured on the trace being
    fitted would move from trace to trace, and then the verdicts along a sheet
    would not be comparable with each other -- which is the one thing a verdict
    has to be.

    Measured once and both answers taken off that one measurement, which is what
    `from_jitter` is for: the sigma reported here is the sigma the gate was built
    on, by construction rather than by the two agreeing.
    """

    from .traces import TraceGate, digitising_jitter, measured_pen

    jitter = digitising_jitter(paths)

    return TraceGate.from_jitter(jitter), measured_pen(jitter)


def anchors_of(gstruct, path, start, end, span):
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


def reach_of(runs, index, length, ends):
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


def fits_along(structure, dem, gate, sweep=None, gstruct=None):
    """
    Every stretch of one path whose plane the topography determines, as a `Reading`.

    One path at a time and never the parts of a multipart together, which the
    import's split has already seen to: `trace_points` runs its progressive *on*
    across parts, so a fit computed over two fragments would be anchored by a
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
    `Reading` says which happened rather than leaving the caller to quote
    whichever reads better.

    **And where the fallback does not fit, the longest window that does.** A part
    shorter than the window is not fitted at all -- `trace_spans` is deliberate
    about that, since reading one fragment at a different scale from the rest
    makes the verdicts along a trace incomparable -- but the *fallback* is a
    default and not a scale somebody chose. On `elementi_tettonici` the median
    trace is 172 m, so keeping to 250 m would refuse two thirds of the sheet on
    the strength of a number nobody typed. Which window was used is in the file
    as `window=`, so the scale a plane was read at is never in doubt.
    """

    from .traces import (
        DEFAULT_SWEEP,
        holding_length,
        trace_points,
        trace_spans,
        window_sweep,
    )

    if gstruct is None:
        from .curation import module

        gstruct = module()

    sweep = sweep or Sweep()
    sampled = min(10.0, sweep.step)

    parts = trace_points([structure.path], dem, step=sampled)

    if not parts:
        return Reading(outcome=UNREACHED)

    swept = window_sweep(
        parts, sweep.lengths or DEFAULT_SWEEP, step=sweep.step, gate=gate
    )
    length = holding_length(swept)

    if length is not None:
        outcome = SWEPT
        spans = swept[length]
    else:
        outcome = FALLBACK
        length = sweep.fallback
        spans = (
            swept[length] if length in swept
            else trace_spans(parts, length, step=sweep.step, gate=gate)
        )

        if len(spans) == 0:
            shorter = [
                other for other in sorted(swept, reverse=True)
                if other < length and len(swept[other])
            ]

            if not shorter:
                return Reading(outcome=TOO_SHORT)

            length = shorter[0]
            spans = swept[length]
            outcome = SHORTENED

    out = []

    runs = spans.runs()

    # The ends of a part carry no verdict at all: a window has to fit, so the
    # first centre sits half a window in and the ground outside that was never
    # classified. Which is what `reach_of` is allowed to claim and a neighbouring
    # run is not.
    progressive = parts[0][1]
    ends = (float(progressive[0]), float(progressive[-1]))
    span = gstruct.path_length(structure.path)

    for index, (verdict, s0, s1) in enumerate(runs):
        if verdict != sweep.keep:
            continue

        # On the unwidened bounds, always: these are the centres whose windows
        # held, and averaging over the reach instead would pull in the windows
        # that failed.
        attitude = spans.mean_attitude(s0, s1)

        if attitude is None:
            continue

        covered = int(((spans.progressive >= s0) & (spans.progressive <= s1)).sum())

        start, end = reach_of(runs, index, length, ends)

        # Anchors and not the progressives, which is the format's rule and not a
        # preference: a reader projects them onto whatever path it has, so the
        # same stretch survives the trace being redigitised, where a stored `s`
        # would migrate.
        start, end = anchors_of(gstruct, structure.path, start, end, span)

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

    return Reading(
        fits=out, outcome=outcome, length=float(length), spans=spans
    )


def as_line(fit, indent="  ", decimals=PLANE_DECIMALS, gstruct=None):
    """
    One `fit` as the line a file holds, for a caller that writes lines and not files.

    `dumps` is the writer everywhere else and cannot be the writer here: the
    trace editor splices lines into a file it otherwise leaves untouched, because
    the format's own writer deletes the comments -- ten of them on
    `curation.gstruct`, which are the argument for why five thrusts are
    `exposed`. See `curation.Document`.

    So the line is built here, out of the format's own spellings of an anchor and
    of an attribute rather than out of new ones. `_q` quotes a value with a space
    in it, and exactly one attribute written here can have one: the DEM's file
    name. `dem=Monte Alpi.tif` would read back as a `dem` of `Monte` and a stray
    token on the line, which the parser has no reason to refuse -- so the file
    would not say what it appears to say, which is the failure this format's
    quoting exists to prevent. Reimplementing those two rules here is how the
    two spellings would drift apart, and the check reads the line back to prove
    they have not.
    """

    if gstruct is None:
        from .curation import module

        gstruct = module()

    return (
        f"{indent}fit plane {gstruct._a(fit.start)} {gstruct._a(fit.end)} "
        f"{fit.plane.dip_dir:.{decimals}f}/{fit.plane.dip:.{decimals}f}"
        f"{gstruct._kw(fit.attrs)}"
    )
