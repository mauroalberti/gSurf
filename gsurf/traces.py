"""
The attitude along an outcrop trace, and where along it there is one to read.

The arithmetic is misah's `best_fit_planes`, which is `BestFitGeoplanes` of
geoSurfDEM ported. What is added here is the part that is not arithmetic: that a
plane fitted to a straight line is arbitrary and not merely imprecise, and that
an attitude read where a section crosses a contact is a fact about that stretch
of trace rather than about the contact.

geoSurfDEM answered this on a square grid, one plane per cell, and called the
result a *local* best-fit attitude for that reason. A cell is the right
parameterisation for a cloud of points off a mesh and the wrong one for a single
mapped line: it gives a long stretch of trace where the line crosses it
diagonally and three points where the line clips its corner. Along one line the
line's own arc length is the parameterisation, so the window slides on the
progressive -- the same coordinate `attitudes.TraceRecord` anchors and spans on,
which is what lets a result here be handed straight back as a record.

Measured on the fourteen lines of the Monte Alpi section, 19.5 km of mapped
trace resampled at 10 m, one core: a 250 m window every 25 m is 648 fits in
0.04 s, and a sweep over five lengths from 150 to 900 m is 2771 fits in 0.22 s.
Fitted across those lengths, 0.04 ms per position and 1.0 us per point in it,
so the points dominate at every window worth using and a longer window costs
more than a denser step does. All of it is far inside a frame, which is the
answer that matters: this can be recomputed while a control is being dragged,
and does not need to be a job with a progress bar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# misah's own `SINGULAR_VALUE_FLOOR`, and deliberately the same value: under it
# a singular value is the arithmetic's own noise, and every ratio built on it is
# noise over noise.
SINGULAR_FLOOR = 1.0e-6

# The grid is the thing being replaced, so it is made to have one cell. Any
# window, at any length this module would be asked for, falls inside it whole.
ONE_CELL = 1.0e7


def collinearity(s1, s2):
    """
    misah's own diagnostic, `-log10(s2/s1)`, large where the points are a line.

    Recomputed here rather than carried out of the kernel so that a gate can be
    re-asked of a stored field without the points that produced it, which is
    what `TraceSpans.regate` is for.
    """

    if s1 <= SINGULAR_FLOOR:
        return float("inf")

    return -math.log10(max(s2, SINGULAR_FLOOR) / s1)


def transverse_spread(n, s2):
    """
    How far the trace departs from a straight line, in metres, as an RMS.

    `s2` itself is not that, and taking it for a distance is the mistake this
    function exists to stop. A singular value is a sum of squares over the
    points, so it grows as the square root of how many there are: the same V
    resampled every 5 m instead of every 10 has an `s2` larger by root two,
    and a floor in metres laid on `s2` would let the resampling step decide
    which traces carry an attitude. Dividing by root n takes it back to a
    distance, which is then comparable with the width of a pen.

    Measured on a synthetic V of 10 m amplitude in plan: 2.9 m at a dip of 10
    degrees and 4.0 m at 75. So it is not the plan amplitude either -- a V on a
    steep plane implies relief, and the relief is part of how far the trace
    stands off its own straight line in three dimensions.
    """

    return s2 / math.sqrt(n) if n > 0 else 0.0


# Median of |X| for X normal is 0.6745 of its sigma; and the offset of a vertex
# from the chord of its two neighbours has a sigma of root 1.5 times the
# vertex's own, since the chord carries half the noise of each end. The two
# together turn a median sagitta into a per-vertex error.
SAGITTA_TO_SIGMA = 1.0 / (0.6745 * math.sqrt(1.5))

# How fast the roughness may grow with the baseline and still be called a pen.
# Independent jitter gives zero and a smooth arc gives two; the line has to sit
# near the bottom of that for a shortest-baseline sagitta to be the hand rather
# than a slice of a roughness that never stops.
MAX_SCALED_EXPONENT = 0.5


def _roughness(lines, stride):
    """Median sagitta and median spacing with every `stride`-th vertex kept."""

    spacings, sagittas = [], []

    for line in lines:
        xy = np.asarray(getattr(line, "coords", line), dtype=float)

        if len(xy) < 2 * stride + 1:
            continue

        xy = xy[::stride, :2]
        spacings.append(np.hypot(*np.diff(xy, axis=0).T))

        a, b, c = xy[:-2], xy[1:-1], xy[2:]
        chord = c - a
        span = np.hypot(chord[:, 0], chord[:, 1])
        usable = span > 0.0

        if not usable.any():
            continue

        # Twice the triangle's area over its base is the height of the middle
        # vertex above the chord.
        area = np.abs(
            chord[usable, 0] * (b[usable, 1] - a[usable, 1])
            - chord[usable, 1] * (b[usable, 0] - a[usable, 0])
        )
        sagittas.append(area / span[usable])

    if not sagittas:
        return None

    sagitta = np.concatenate(sagittas)
    spacing = np.concatenate(spacings)

    if len(sagitta) == 0:
        return None

    return float(np.median(spacing)), float(np.median(sagitta)), int(len(sagitta))


def digitising_jitter(lines, strides=(1, 2, 3, 4, 6, 8)):
    """
    How much the pen wandered, in metres, measured off the lines themselves.

    The first stage asks whether a departure from straightness is real or is
    noise, so what it needs is the *roughness* of the digitised line and not its
    positional accuracy. The two are different and only one of them matters
    here: a trace drawn fifty metres off where the contact really is does not
    thereby acquire any transverse spread, while one drawn with a shaky hand
    does. So the floor is calibrated against roughness.

    Roughness is the offset of a vertex from the chord joining its neighbours,
    the sagitta. But a single number off that is not the jitter, and taking it
    for one is the mistake this function is shaped around: a sagitta holds the
    hand *and* the geology, and the two separate by how they scale with the
    baseline they are measured over. Independent jitter does not care how far
    apart the vertices are; real curvature is a sagitta of h squared over twice
    the radius, so it falls away as the baseline shortens.

    So the vertices are decimated -- every second, every third, every eighth --
    and `median^2 = jitter^2 + c * h^4` is fitted across the strides. The
    intercept is the roughness left at zero baseline, which is the pen. What
    forced this: on the Lauria sheet the fourteen lines crossing the Monte Alpi
    section report 5.1 m of sigma at a 39.5 m vertex spacing and the whole layer
    reports 1.5 m at 19.8 m, a factor of 3.3 on the same map with the same pen.
    The ratio of the spacings squared is 4.0, which says almost all of it was
    geology.

    `exponent` is the slope of log sagitta against log baseline, and it decides
    whether there is anything to report at all. Near zero the roughness has a
    scale, the shortest baselines are all pen, and `sigma` means what it says.
    Near two the line is smooth and the pen is finer than its vertices can show.
    **In between, `sigma` is None and the refusal is the answer**, because a
    roughness that keeps growing with the baseline has no scale to be measured
    at -- the line is self-affine, and asking for its pen is asking how long a
    coastline is.

    That middle case is not a corner. It is what the Lauria sheet does: over the
    fourteen lines crossing the Monte Alpi section the sagitta goes 4.2, 12.2,
    18.9, 24.5, 24.6, 30.4 m as the baseline goes 40 to 341, an exponent of
    +0.90. A mapped geological contact is rough at every scale one looks at it,
    so there is no pen width hiding under the geology, and `for_scale` -- a
    statement about the drawing rather than a measurement of it -- is what is
    left to calibrate against. The first version of this returned 20.8 m for
    that layer, extrapolated from a shortest-baseline sagitta of 4.2, and it
    looked like a measurement.

    **Off the original vertices, never off a resampled trace.** Resampling
    interpolates along the segments, so its points sit exactly on the chords by
    construction: the short strides find no roughness and the long ones find the
    original vertices again. That climb is the same signature as a self-affine
    line and draws the same refusal, so the mistake comes back as None rather
    than as a number -- but by luck of the guard rather than by design, which is
    why it is still worth saying that the vertices are what to measure.
    """

    curve = [
        got for got in (_roughness(lines, stride) for stride in strides)
        if got is not None
    ]

    if not curve:
        return None

    h = np.array([point[0] for point in curve])
    m = np.array([point[1] for point in curve])

    exponent = float("nan")
    if len(curve) >= 2 and np.all(h > 0.0) and np.all(m > 0.0):
        exponent = float(np.polyfit(np.log(h), np.log(m), 1)[0])

    sigma = None

    # Only where the roughness has a scale. The fit below would return an
    # intercept for any curve at all, and on a self-affine line that intercept
    # is an artefact of forcing an h^4 model onto an h^0.9 measurement -- it
    # came out five times the shortest sagitta actually measured.
    if len(curve) >= 2 and np.isfinite(exponent) and exponent < MAX_SCALED_EXPONENT:
        # Least squares on (1, h^4) -- the two terms as they add, rather than a
        # slope on logs, so that a curve which is part pen and part curvature
        # still gives up its intercept.
        design = np.column_stack([np.ones_like(h), h ** 4])
        (intercept, _), *_ = np.linalg.lstsq(design, m ** 2, rcond=None)

        # Explicitly, and not through max(): an intercept that came back NaN
        # would pass a comparison against zero the wrong way round and be
        # reported as a measured jitter of nothing.
        if np.isfinite(intercept) and intercept > 0.0:
            sigma = float(np.sqrt(intercept)) * SAGITTA_TO_SIGMA

    return dict(
        sigma=sigma,
        exponent=exponent,
        curve=[(round(point[0], 2), round(point[1], 3)) for point in curve],
        median=float(m[0]),
        spacing=float(h[0]),
        vertices=int(curve[0][2]),
    )


# -- what a window has to clear ------------------------------------------


@dataclass(frozen=True)
class TraceGate:
    """
    The two stages a window passes before its plane is worth reading, in order.

    They are sequential and not alternative, which is the whole of this class.
    `s2` is the lever arm across the trace, and what it measures is whether the
    trace ever *turned*. Relief alone will not do it -- a contact climbing
    straight up an escarpment crosses 150 m of it and still has no lever arm.
    Where `s2` is noise every plane through that line fits equally well, so the
    attitude is arbitrary; and `s3/s2`, computed there, is noise over noise.
    `s3/s2` is the precision, and it means nothing until `s2` is established.

    The lever arm has to clear two floors that fail for different reasons, which
    is why they are two fields. `max_collinearity` is misah's `-log10(s2/s1)`
    and catches the arithmetically degenerate line. `min_lever` catches the
    cartographic one: a departure from straightness smaller than the wander of
    the pen that drew the line is not evidence of a turn, whatever the ratio
    says. It is read against `transverse_spread` and not against `s2`, so that
    it stays a distance and does not move with the resampling step.

    **Where the default comes from.** A trace whose vertices each wander by
    sigma comes back with a transverse spread of about sigma even when it is
    dead straight, so the floor has to stand clear of sigma rather than sit on
    it. Three times is the bar here: the spread carries signal and noise in
    quadrature, so a spread of three sigma is a real departure of 2.8 sigma,
    which is not something noise produces. `digitising_jitter` measures sigma
    off the lines themselves and `from_traces` puts the two together, which is
    the right way round -- 15 m is the answer for the sheet this was built on
    and not a constant of nature.

    `min_points` is three because three is where a plane stops being defined.
    That floor is the arithmetic's and not a policy, so it is not somewhere a
    threshold should be tuned.
    """

    max_collinearity: float = 3.0       # -log10(s2/s1); misah's own default
    min_lever: float = 15.0             # metres of RMS spread; see `from_traces`
    min_precision: float = 4.0          # s2/s3, once the plane exists at all
    min_points: int = 3

    @classmethod
    def from_traces(cls, lines, factor=3.0, **rest):
        """
        The lever floor measured off the layer, which is the way round to prefer.

        Falls back to the class default where the measurement does not come
        back -- a layer of two-point segments has no roughness to report, and a
        layer smooth to the limit of its own vertices has a pen finer than it
        can show -- rather than refusing to build a gate, because a tool that
        cannot open a file is worse than one whose threshold came from
        somewhere else. Call `digitising_jitter` directly to say which happened,
        and to see `exponent` before trusting the number.
        """

        jitter = digitising_jitter(lines)

        if jitter is None or jitter["sigma"] is None:
            return cls(**rest)

        return cls(min_lever=factor * jitter["sigma"], **rest)

    @classmethod
    def for_scale(cls, denominator, pen=3.0e-4, **rest):
        """
        The lever floor off the scale of the sheet, for when there is no layer yet.

        Second best, and kept because the scale is known before the geometry is
        read and a panel has to show a number from the start. Three tenths of a
        millimetre at 1:50.000 is 15 m on the ground and at 1:10.000 is three.

        Three tenths of a millimetre is not the width of the line, which is
        about half and is the figure usually quoted as the precision of a 50k
        sheet. The two get confused, and they differ by enough to decide whether
        a trace is read or refused. It is set here at the value that reproduces
        what the Lauria sheet's own roughness gives -- so this is calibrated
        against a measurement rather than against a rule of thumb, and on a
        sheet drawn by another hand the measurement is what to trust.
        """

        return cls(min_lever=float(denominator) * pen, **rest)

    def determines(self, n, s1, s2, s3):
        """Stage one: whether there is a plane here at all."""

        return (
            n >= self.min_points
            and transverse_spread(n, s2) >= self.min_lever
            and collinearity(s1, s2) <= self.max_collinearity
        )

    def verdict_for(self, n, s1, s2, s3):
        """
        "held", "loose" or "line" -- three states, because two would lie.

        "line" is not a worse "loose". A loosely held plane says roughly this
        way, and drawing it dashed says so honestly. A line says only where the
        contact crops out, which on a straight trace is the one thing the map
        established, and drawing that as a dashed plane would be inventing the
        rest.
        """

        if not self.determines(n, s1, s2, s3):
            return "line"

        return "held" if s2 / max(s3, SINGULAR_FLOOR) >= self.min_precision else "loose"

    def refusal_for(self, n, s1, s2, s3):
        """
        Why this window is not a held attitude, or "" when it is.

        The tally in a field is kept on the verdict and not by reading these
        strings back, which is what `folds.Gate` has to do. There the states are
        admitted or not and the reason has to be recovered; here the reason and
        the state are the same two stages, so the state carries it already.
        """

        if n < self.min_points:
            return f"{n} points, fewer than {self.min_points}"

        spread = transverse_spread(n, s2)
        if spread < self.min_lever:
            return (
                f"{spread:.1f} m off straight: the trace does not turn enough "
                f"to fix a plane"
            )

        value = collinearity(s1, s2)
        if value > self.max_collinearity:
            return f"collinearity {value:.1f}: the points lie along a line"

        precision = s2 / max(s3, SINGULAR_FLOOR)
        if precision < self.min_precision:
            wobble = math.degrees(math.atan2(s3, max(s2, SINGULAR_FLOOR)))
            return f"s2/s3 = {precision:.1f}: the plane may tilt {wobble:.0f} deg"

        return ""


# -- one window ----------------------------------------------------------


@dataclass(frozen=True)
class TraceWindow:
    """One stretch of trace inverted: a plane, and what it is worth."""

    centre: float                   # metres along the trace
    length: float                   # of the window, not of the trace
    n: int
    dip_direction: float
    dip: float
    centroid: tuple                 # a point the plane passes through
    singular_values: tuple          # s1 >= s2 >= s3
    rms: float

    @property
    def collinearity(self):
        """-log10(s2/s1), as misah reports it."""

        return collinearity(self.singular_values[0], self.singular_values[1])

    @property
    def spread(self):
        """Metres the trace stands off its own straight line, as an RMS."""

        return transverse_spread(self.n, self.singular_values[1])

    @property
    def wobble(self):
        """
        How far the plane may tilt about the trace and stay inside the scatter.

        `atan(s3/s2)` in degrees -- an angle, so that it is comparable with the
        15 degrees by which neighbouring compass readings disagree, and not a
        bare ratio that has to be taken on trust.
        """

        s2, s3 = self.singular_values[1], self.singular_values[2]

        return math.degrees(math.atan2(s3, max(s2, SINGULAR_FLOOR)))


def window_attitude(points, centre=0.0, length=0.0, coincidence=1.0):
    """
    The best-fit plane of one window, with the degenerate cells kept.

    Called with `max_collinearity=inf`, which is the trap this module exists to
    contain. Left at its default the kernel sets the degenerate cell aside
    itself, and then the window does not come back at all -- and the windows
    that would vanish are exactly the ones this module is here to count. A
    stretch of trace too straight to carry an attitude is a result, not an
    absence, and it has to arrive as one.
    """

    from misah.kernels import best_fit_planes

    points = np.ascontiguousarray(points, dtype=float)

    if len(points) < 3:
        return None

    got = best_fit_planes(
        points,
        ONE_CELL,
        coincidence_distance=coincidence,
        max_collinearity=float("inf"),
    )

    if got is None:
        return None

    found, _ = got
    counts = np.asarray(found["point_counts"])

    if len(counts) == 0:
        return None

    # One cell by construction, but the kernel is entitled to return the grid it
    # chose rather than the one that was asked for, so the fullest cell is taken
    # instead of the first.
    index = int(np.argmax(counts))
    attitude = np.asarray(found["attitudes"])[index]

    return TraceWindow(
        centre=float(centre),
        length=float(length),
        n=int(counts[index]),
        dip_direction=float(attitude[0]),
        dip=float(attitude[1]),
        centroid=tuple(float(v) for v in np.asarray(found["centroids"])[index]),
        singular_values=tuple(
            float(v) for v in np.asarray(found["singular_values"])[index]
        ),
        rms=float(np.asarray(found["rms_distance"])[index]),
    )


# -- the trace, hung on the topography -----------------------------------


def _resample(coords, step):
    """Vertices every `step` along a polyline, with their progressives."""

    segments = np.hypot(*np.diff(coords, axis=0).T)
    walked = np.concatenate(([0.0], np.cumsum(segments)))

    if walked[-1] <= 0.0:
        return None

    count = max(2, int(walked[-1] / step) + 1)
    s = np.linspace(0.0, walked[-1], count)

    return (
        np.column_stack([
            np.interp(s, walked, coords[:, 0]),
            np.interp(s, walked, coords[:, 1]),
        ]),
        s,
    )


def _elevations(window, xs, ys):
    """Nearest-cell elevations out of an already-read window."""

    origin_x, pixel_w, _, origin_y, _, pixel_h = window.geotransform

    cols = np.floor((xs - origin_x) / pixel_w).astype(int)
    rows = np.floor((ys - origin_y) / pixel_h).astype(int)

    height, width = window.data.shape
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)

    z = np.full(len(xs), np.nan)
    z[inside] = window.data[rows[inside], cols[inside]]

    return z


def trace_points(lines, dem, step=10.0, margin=50.0):
    """
    A mapped line hung on the DEM: `[(coords (N, 3), progressive (N,))]`, per part.

    One entry per part and not one concatenated array, because a
    MultiLineString is a trace digitised in pieces -- interrupted by cover, or
    cut at the edge of a sheet -- and a window straddling the gap would invert a
    plane through two outcrops that never touched.

    The progressive, though, runs *on* across the parts rather than restarting,
    because that is the measure `attitudes.clip_to_span` and every anchor in
    `TraceRecord` are stated against. A span computed here would otherwise clip
    the wrong piece of the trace it came from, and clip it silently.

    The DEM is read once over the whole trace rather than point by point:
    `Dem.elevation_at` is a rasterio read per call, and a trace is thousands of
    points asked for again at every window length of a sweep.
    """

    parts = []
    for line in lines:
        coords = np.asarray(getattr(line, "coords", line), dtype=float)
        if len(coords) >= 2:
            parts.append(coords[:, :2])

    if not parts:
        return []

    stacked = np.vstack(parts)
    box = (
        float(stacked[:, 0].min()),
        float(stacked[:, 1].min()),
        float(stacked[:, 0].max()),
        float(stacked[:, 1].max()),
    )

    window = dem.window_over(box, margin=margin)

    if window is None:
        return []

    out, walked = [], 0.0

    for coords in parts:
        resampled = _resample(coords, step)

        if resampled is None:
            continue

        xy, s = resampled
        z = _elevations(window, xy[:, 0], xy[:, 1])

        # The progressive advances over the whole part, including the stretches
        # that fell on nodata: it measures the trace, not the samples kept off
        # it. Dropping the gap instead would shorten every span downstream of a
        # hole in the DEM.
        kept = np.isfinite(z)
        walked_next = walked + s[-1]

        if kept.sum() >= 3:
            out.append((
                np.column_stack([xy[kept, 0], xy[kept, 1], z[kept]]),
                walked + s[kept],
            ))

        walked = walked_next

    return out


# -- one trace, cut into stretches ---------------------------------------


@dataclass
class TraceSpans:
    """
    One trace read at one window length, as arrays over window positions.

    Aligned arrays and not a list of objects, for the reason `FoldAxisField`
    gives: a result of this shape is read column-wise -- every dip, every s2 --
    far more often than position by position.

    Two boolean columns and not one three-valued one, because the two are the
    two stages of the gate and are asked separately: `determined` is stage one,
    is there a plane, and `held` is both. "loose" is the difference between
    them and is derived rather than stored, so that no code can put the three
    states into a combination the gate would never produce.
    """

    progressive: np.ndarray         # (M,) window centres, metres along the trace
    part: np.ndarray                # (M,) which fragment of the trace
    counts: np.ndarray              # (M,)
    dip_directions: np.ndarray      # (M,) NaN where no plane came back
    dips: np.ndarray                # (M,)
    centroids: np.ndarray           # (M, 3)
    singular_values: np.ndarray     # (M, 3) s1 >= s2 >= s3
    rms: np.ndarray                 # (M,)
    determined: np.ndarray          # (M,) bool: a plane exists
    held: np.ndarray                # (M,) bool: and it is tightly held
    length: float                   # the window length these were read at
    step: float
    trace_length: float
    gate: TraceGate
    refusals: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.progressive)

    @property
    def walked(self):
        """
        Metres of trace that carry a verdict, which is less than the trace.

        A window has to fit, so the first centre sits half a window in from each
        end and the ends of every part are never classified. Quoting a share of
        the trace when it is a share of the walked part is the easiest way for
        this to flatter itself.
        """

        return float(len(self) * self.step)

    def verdicts(self):
        """The three-state column, derived from the two boolean ones."""

        return np.where(self.held, "held", np.where(self.determined, "loose", "line"))

    def runs(self):
        """
        Consecutive positions of one verdict, as `(verdict, s0, s1)`.

        Broken at a change of part as well as at a change of verdict: two
        fragments of one trace separated by cover are not a continuous stretch,
        and a run spanning the gap would claim ground nobody mapped.

        The ends are half a step beyond the first and last centre, so that a
        single position is a step of trace rather than nothing at all and the
        runs of a part add back up to what was walked. The window that produced
        each verdict reaches half its length further either way; the run is
        deliberately not widened to that, because then neighbouring runs would
        overlap and their metres would sum past the trace.
        """

        out, verdicts = [], self.verdicts()

        for index in range(len(self)):
            centre = float(self.progressive[index])
            same = (
                out
                and out[-1][0] == verdicts[index]
                and out[-1][3] == self.part[index]
            )

            if same:
                out[-1][2] = centre + self.step / 2.0
            else:
                out.append([
                    str(verdicts[index]),
                    centre - self.step / 2.0,
                    centre + self.step / 2.0,
                    self.part[index],
                ])

        return [(verdict, s0, s1) for verdict, s0, s1, _ in out]

    def metres(self):
        """Metres of trace under each verdict."""

        totals = {"held": 0.0, "loose": 0.0, "line": 0.0}

        for verdict, s0, s1 in self.runs():
            totals[verdict] += s1 - s0

        return totals

    def _index_at(self, s):
        """
        The position whose window covers a progressive, or None.

        The nearest centre whose window still reaches `s`, rather than the
        nearest centre outright: within half a window of either end there is no
        centre nearby, but there is a window that contains the point, and it is
        the one that was fitted there.

        One rule with two callers, so that `at` and `run_at` cannot disagree
        about whether a crossing was covered. They did: `at` reached half a
        window and `run_at` half a step, so the first hundred metres of every
        trace came back with an attitude and no extent to put it over -- which
        is exactly the pairing this module exists to keep together.
        """

        if len(self) == 0:
            return None

        index = int(np.argmin(np.abs(self.progressive - s)))

        if abs(float(self.progressive[index]) - s) > self.length / 2.0:
            return None

        return index

    def at(self, s):
        """
        The window covering a progressive, or None where the trace was not walked.

        This is what a section asks. A crossing does not take its attitude from
        its contact, it takes it from the stretch it happened to land on -- and
        on Monte Alpi the best-determined fault of the whole section holds on a
        fraction of its own trace, so which stretch that was had been luck until
        there was something to ask.
        """

        index = self._index_at(s)

        if index is None:
            return None

        return TraceWindow(
            centre=float(self.progressive[index]),
            length=self.length,
            n=int(self.counts[index]),
            dip_direction=float(self.dip_directions[index]),
            dip=float(self.dips[index]),
            centroid=tuple(float(v) for v in self.centroids[index]),
            singular_values=tuple(float(v) for v in self.singular_values[index]),
            rms=float(self.rms[index]),
        )

    def run_at(self, s):
        """
        The run a progressive falls in, as `(verdict, s0, s1)`, or None.

        The companion of `at`, and the more honest of the two to quote: an
        attitude without the extent it was read over invites being drawn across
        the whole contact, which is the thing this module exists to stop.

        Found through the covering window rather than by testing the runs, so
        that a crossing which `at` answers is one `run_at` answers too. Near the
        ends of a trace the covering window is not centred on the crossing, and
        a test on the run's own bounds would refuse what `at` had just returned.
        """

        index = self._index_at(s)

        if index is None:
            return None

        centre = float(self.progressive[index])

        for verdict, s0, s1 in self.runs():
            if s0 <= centre <= s1:
                return verdict, s0, s1

        return None

    def mean_attitude(self, s0, s1):
        """
        The mean plane over a stretch, or None where no position in it has one.

        Averaged as unit normals and not as dip directions, which do not
        average: two readings at 350 and 10 degrees are 20 apart and their
        arithmetic mean is 180 out.

        The upward normal, so that there is no sign to resolve first. A plane
        dipping between 0 and 90 has its upward normal in the upper hemisphere
        whatever way it faces, so the normals can be summed as they are -- which
        is what separates this from averaging fold axes, where the two ends of a
        line are the same line and the signs have to be settled before anything
        is added.
        """

        inside = (
            (self.progressive >= s0)
            & (self.progressive <= s1)
            & np.isfinite(self.dip_directions)
        )

        if not inside.any():
            return None

        azimuth = np.radians(self.dip_directions[inside])
        dip = np.radians(self.dips[inside])

        normal = np.column_stack([
            np.sin(azimuth) * np.sin(dip),
            np.cos(azimuth) * np.sin(dip),
            np.cos(dip),
        ]).sum(axis=0)

        norm = np.linalg.norm(normal)

        if norm <= SINGULAR_FLOOR:
            return None

        normal = normal / norm

        return (
            float(np.degrees(np.arctan2(normal[0], normal[1])) % 360.0),
            float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0)))),
        )

    def regate(self, gate):
        """
        Decides the whole trace again against a different gate, in place.

        Nothing is re-inverted: the singular values and the count are all the
        gate reads, and they are already here. That is the payoff of keeping
        this as arrays -- a threshold is a question asked of a result, not part
        of computing it, and sweeping one over a field already computed is the
        only way to see how much of the answer depends on where it was put.
        """

        self.gate = gate
        refusals = {}

        for index in range(len(self)):
            s1, s2, s3 = (float(v) for v in self.singular_values[index])
            n = int(self.counts[index])

            verdict = gate.verdict_for(n, s1, s2, s3)
            self.determined[index] = verdict != "line"
            self.held[index] = verdict == "held"

            if verdict != "held":
                refusals[verdict] = refusals.get(verdict, 0) + 1

        self.refusals = refusals

        return self

    def summary(self):
        totals = self.metres()
        walked = self.walked

        shares = ", ".join(
            f"{totals[verdict] / 1000.0:.2f} km {verdict}"
            f" ({100.0 * totals[verdict] / walked:.0f}%)"
            for verdict in ("held", "loose", "line")
        ) if walked else "nothing walked"

        return (
            f"{len(self)} windows of {self.length:.0f} m every {self.step:.0f} m "
            f"over {self.trace_length / 1000.0:.2f} km of trace: {shares}; "
            f"{describe_sampling(self.length, self.step, walked)}"
        )


def trace_spans(parts, length, step=25.0, gate=None, progress=None):
    """
    A fit at every window position along a trace.

    `parts` is what `trace_points` returns. `progress` is called with the
    number of positions done and the total and may return False to give up,
    for the reason `folds.fold_axis_field` gives: the window length that makes
    a sweep too slow is one keystroke away from the one that does not.
    """

    gate = gate or TraceGate()

    positions = []
    for index, (_, progressive) in enumerate(parts):
        span = progressive[-1] - progressive[0]

        # A part shorter than the window is not fitted at a shorter length
        # instead. The window length is the scale the answer is stated at, and
        # quietly reading one fragment at a different scale from the rest would
        # make the verdicts along a trace incomparable with each other.
        if span < length:
            continue

        first = progressive[0] + length / 2.0
        last = progressive[-1] - length / 2.0

        for centre in np.arange(first, last + 1e-9, step):
            positions.append((index, float(centre)))

    total = len(positions)
    trace_length = float(parts[-1][1][-1]) if parts else 0.0

    counts = np.zeros(total, dtype=int)
    part = np.zeros(total, dtype=int)
    centres = np.zeros(total)
    dip_directions = np.full(total, np.nan)
    dips = np.full(total, np.nan)
    centroids = np.full((total, 3), np.nan)
    singular_values = np.zeros((total, 3))
    rms = np.full(total, np.nan)
    determined = np.zeros(total, dtype=bool)
    held = np.zeros(total, dtype=bool)
    refusals = {}

    stride = max(1, min(256, total // 200))

    for index, (which, centre) in enumerate(positions):
        if progress is not None and index % stride == 0:
            if progress(index, total) is False:
                break

        coords, progressive = parts[which]
        inside = np.abs(progressive - centre) <= length / 2.0

        part[index], centres[index] = which, centre

        window = window_attitude(coords[inside], centre=centre, length=length)

        if window is None:
            # No plane at all, not even an arbitrary one: fewer than three
            # points survived. Recorded as a position rather than skipped, so
            # that a gap in the DEM shows up as unreadable trace instead of
            # quietly joining the runs on either side of it.
            refusals["line"] = refusals.get("line", 0) + 1
            continue

        counts[index] = window.n
        dip_directions[index] = window.dip_direction
        dips[index] = window.dip
        centroids[index] = window.centroid
        singular_values[index] = window.singular_values
        rms[index] = window.rms

        s1, s2, s3 = window.singular_values
        verdict = gate.verdict_for(window.n, s1, s2, s3)

        determined[index] = verdict != "line"
        held[index] = verdict == "held"

        if verdict != "held":
            refusals[verdict] = refusals.get(verdict, 0) + 1

    return TraceSpans(
        progressive=centres,
        part=part,
        counts=counts,
        dip_directions=dip_directions,
        dips=dips,
        centroids=centroids,
        singular_values=singular_values,
        rms=rms,
        determined=determined,
        held=held,
        length=float(length),
        step=float(step),
        trace_length=trace_length,
        gate=gate,
        refusals=refusals,
    )


# -- the window length is not a parameter --------------------------------


def window_sweep(parts, lengths, step=25.0, gate=None, progress=None):
    """
    The same trace at several window lengths: `{length: TraceSpans}`.

    Lengthening the window first helps and then hurts, and the turn is the
    measurement. It helps because a longer stretch of trace is a longer lever
    arm. It hurts because past some length the window starts to take in
    curvature that is real, and real curvature lands in `s3` and fails the
    second stage.

    So the peak is not a tuning artefact to be calibrated away. It estimates the
    length over which this trace holds a single orientation, which is what a
    question about variation in attitude along a fault is actually asking. On
    the Monte Alpi lines it lands at 150 m for one trace and 900 m for another,
    and never for a third: there is no one good length, which is why this is a
    function and not an argument.
    """

    out = {}
    lengths = list(lengths)

    for index, length in enumerate(lengths):
        def relay(done, total, index=index):
            if progress is None:
                return True
            return progress(index * total + done, len(lengths) * total)

        out[float(length)] = trace_spans(
            parts, length, step=step, gate=gate, progress=relay
        )

    return out


def holding_length(sweep):
    """
    The window length at the peak of the held share, or None where there is none.

    None in three situations, all of which have to be refused rather than
    reported. A trace held at no length says only that it is unreadable. A trace
    held at every length says only that it is planar, and the sweep resolved
    nothing. And a peak sitting at either end of the lengths swept is not a
    peak -- the turn, if there is one, is outside what was asked, and the
    answer would be the edge of the sweep rather than a property of the trace.
    """

    if len(sweep) < 3:
        return None

    lengths = sorted(sweep)
    shares = [
        sweep[length].metres()["held"] / sweep[length].walked
        if sweep[length].walked else 0.0
        for length in lengths
    ]

    if max(shares) <= 0.0 or max(shares) - min(shares) < 0.05:
        return None

    peak = int(np.argmax(shares))

    if peak in (0, len(shares) - 1):
        return None

    return float(lengths[peak])


# -- honesty about the sampling, and cost before the work ----------------


def sampling(length, step, walked):
    """
    How much the windows overlap, which no control on screen shows.

    At 250 m every 25 m each point of the trace falls in ten windows and
    neighbours share nine tenths of their data: the spans are smoothed, and a
    count of them is not a count of independent observations. The parallel with
    `folds.sampling` is exact and one-dimensional -- and the same warning
    applies at the end, that windows which merely fail to overlap are still not
    independent, because a structure does not stop at the edge of a window.
    """

    return dict(
        per_point=length / step if step > 0 else float("inf"),
        tiling=walked / length if length > 0 else 0.0,
        overlapping=step < length,
    )


def describe_sampling(length, step, walked):
    """The overlap in one line, for a panel or a status bar."""

    counts = sampling(length, step, walked)

    if not counts["overlapping"]:
        return "windows do not overlap"

    return (
        f"each point in ~{counts['per_point']:.0f} windows; "
        f"~{counts['tiling']:.0f} would tile the trace"
    )


def sweep_cost(parts, lengths, step):
    """
    What a sweep will cost, before it is asked for.

    Coarse on purpose, as `folds.field_cost` is: it exists so that a step typed
    by hand cannot start work of unknown length without saying how long, not to
    be right to the millisecond.

    Fitted on the fourteen Monte Alpi lines over window lengths from 150 to
    900 m, then rounded up rather than onto the measurements: 0.06 ms per
    position against a measured 0.04, and 1.2 us per point against 1.0. A wait
    that turns out shorter than promised costs nothing and one that turns out
    longer is the reason for putting a number here at all.

    The points dominate at every window worth using, which is the shape worth
    remembering: lengthening the window costs more than tightening the step.
    """

    positions, points = 0, 0

    for length in lengths:
        for coords, progressive in parts:
            span = progressive[-1] - progressive[0]

            if span < length:
                continue

            count = int((span - length) / step) + 1
            positions += count

            # The points in a window, off the resampling actually used rather
            # than off the nominal step: a trace crossing a gap in the DEM has
            # fewer, and it is the one whose cost is most likely to be asked.
            density = len(coords) / span if span > 0 else 0.0
            points += int(count * density * length)

    return dict(
        positions=positions,
        points=points,
        seconds=positions * 0.06e-3 + points * 1.2e-6,
    )


# -- the bridge: spans are records ---------------------------------------


def records_from_spans(spans, lines, category, keep="held", is_rhr_strike=False,
                       attrs=None):
    """
    Held stretches as `attitudes.TraceRecord`, one record per stretch.

    This is why the module sits here and not inside the profiles tool. A record
    is a plane, the lines that are its outcrop, and a span in metres along them
    -- which is precisely a run, and precisely what `TraceAttitudeSource`
    already hands to `Profilers.intersect_lines_with_attitudes`. A section needs
    no new way to consume this: it needs the records it already draws, with
    their span set from the evidence instead of from a half-span typed into a
    table.

    Which also settles the question `TraceRecord` deliberately left open. An
    anchor is a fact and a span is a decision, it says, and `None` means nobody
    has made one. A fitted plane owns its trace -- but not all of it. It owns
    the stretch the fit held on, and here that stretch has become a measurement.

    The whole trace goes into every record and the span does the cutting, rather
    than the lines being clipped on the way in: `TraceRecord.reach_endpoints`
    clips against its own lines, so a record given pre-cut lines *and* a span
    stated in the original progressive would cut the wrong piece of itself.
    """

    from geogst.core.geology.orientations import Plane

    from .attitudes import TraceRecord

    lines = list(lines)
    length = float(sum(line.length_2d() for line in lines))

    records = []

    for verdict, s0, s1 in spans.runs():
        if verdict != keep:
            continue

        attitude = spans.mean_attitude(s0, s1)

        if attitude is None:
            continue

        records.append(
            TraceRecord(
                category=category,
                plane=Plane(attitude[0], attitude[1], is_rhr_strike=is_rhr_strike),
                lines=lines,
                length=length,
                anchor=(s0 + s1) / 2.0,
                span=(s0, s1),
                attrs=dict(attrs or {}, span_verdict=verdict, fitted=True),
            )
        )

    return records


DEFAULT_SWEEP = (150.0, 250.0, 400.0, 600.0, 900.0)


def fit_records(records, dem, lengths=DEFAULT_SWEEP, step=25.0, gate=None,
                keep="held", fallback=250.0, progress=None):
    """
    Every record re-read off its own trace, as `(records, report)`.

    One record in, none or several out: none where the trace never determines a
    plane, several where it determines different ones along its length. That
    asymmetry is the point of the whole module, and it is why this returns a
    report as well -- a call that turns fourteen records into nine has thrown
    five contacts out of the section, and doing that without saying so would be
    the worst version of this feature.

    **The window is swept per trace and not chosen once.** Lengthening a window
    helps until it starts covering real curvature, and where that turn falls is
    a property of the trace: on the Monte Alpi lines it is 250 m for one and
    600 m for another, and for several there is no turn at all. So each trace
    gets its own `holding_length`, and `fallback` is for the ones that have
    none -- which is most of them, and is stated in the report rather than
    hidden by picking a number that makes the tally look better.

    The category is carried through unchanged, so several stretches of one
    contact stay one entry in a legend and one system in a section. What
    distinguishes them is the span, which is what they differ by.
    """

    gate = gate or TraceGate()

    fitted, report = [], dict(
        traces=len(records), fitted=0, silent=0, runs=0,
        swept=0, walked=0.0, held=0.0, lengths={}, stopped=False,
    )

    for index, record in enumerate(records):
        # Said in the report rather than left to be inferred from the counts.
        # A run stopped part way through has read the first N traces in file
        # order, which is a corner of a sheet and not a sample of it, and a
        # caller that applied it would be building a section out of that.
        if progress is not None and progress(index, len(records)) is False:
            report["stopped"] = True
            break

        parts = trace_points(record.lines, dem, step=min(10.0, step))

        if not parts:
            report["silent"] += 1
            continue

        sweep = window_sweep(parts, lengths, step=step, gate=gate)
        length = holding_length(sweep)

        if length is None:
            length = fallback
            spans = trace_spans(parts, length, step=step, gate=gate)
        else:
            report["swept"] += 1
            spans = sweep[length]

        report["lengths"][length] = report["lengths"].get(length, 0) + 1
        report["walked"] += spans.walked
        report["held"] += spans.metres()["held"]

        got = records_from_spans(
            spans, record.lines, record.category, keep=keep,
            is_rhr_strike=getattr(record.plane, "is_rhr_strike", False),
            attrs=dict(record.attrs, src=f"fit {length:.0f} m"),
        )

        if got:
            report["fitted"] += 1
            report["runs"] += len(got)
            fitted.extend(got)
        else:
            report["silent"] += 1

    return fitted, report


def describe_fit(report):
    """The report in one paragraph, for a message box or a status bar."""

    if not report["traces"]:
        return "no traces to fit"

    if report.get("stopped"):
        return (
            f"Stopped after {report['fitted'] + report['silent']} of "
            f"{report['traces']} traces, and nothing was applied. What had been "
            f"read by then is the head of the file rather than a sample of the "
            f"sheet, which is not something to put a section on."
        )

    share = (
        100.0 * report["held"] / report["walked"] if report["walked"] else 0.0
    )
    lengths = ", ".join(
        f"{count} at {length:.0f} m"
        for length, count in sorted(report["lengths"].items())
    )

    return (
        f"{report['traces']} traces gave {report['runs']} stretches on "
        f"{report['fitted']} of them; {report['silent']} determine no attitude "
        f"anywhere and are gone from the section.\n\n"
        f"{share:.0f}% of the trace walked is readable. "
        f"{report['swept']} traces chose their own window ({lengths}); the rest "
        f"hold at every length or at none, and took the fallback."
    )
