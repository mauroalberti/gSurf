"""
A plane laid on the topography, for a caller that draws it: the producer, away
from its one window.

`tools/intersection.py` is the tool this comes out of, and it is still the tool:
a DEM, a point, a dial, and the marching-squares chords redrawn while the hand
moves. What it does *not* have is anything to aim at. The point is put wherever
you click and the answer is read out loud -- you look at the curves, you decide
they run along the fault, and the number goes into a notebook.

The trace editor is the other half of that sentence. There the fault is on
screen, the stretch being decided is drawn under it, and the file that wants the
number is open beside it. So this is the calculation with the window's controls
taken off it, and two things in their place:

**The window is sized from the ground being asked about**, not fixed at a
thousand cells. In the tool the window is a cost dial -- the plane is unbounded
and the point is anywhere, so the only sensible rule is a constant number of
cells per frame. Here there is a stretch of trace, and the question is whether
the plane runs along *that*: a window is big enough when it shows the stretch
and a little beyond, and bigger than that is cells scanned to draw curves
outside the question. `side_for` is that rule, with a ceiling, because a claim
can be a trace twenty kilometres long and a frame still has to land.

**Convergence is carried, not applied and forgotten.** `Laid` comes back holding
the grid azimuth it actually ran on and the convergence it came from, because
the number is about to be written into a file: see `curation.with_attrs` and
what the editor puts beside it. The tool could leave this in a label under the
dial. A file cannot.

What has not changed is the kernel or the arithmetic around it, which is
`misah.kernels.intersect_plane_grid` either way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# What a plane steered by hand against the topography says made it. A producer
# name of its own, and it has to be: FORMAT.md's rule for a `fit` is that
# whoever writes one says which producer made it and does not fill in the
# diagnostics of another. This one has none to fill in -- there is no gate here,
# no `snr`, no `jack`, no window swept. What it has is a person who looked at
# two lines and judged them to run together, and the honest way to write that is
# a `from=` nobody can mistake for a computation.
FROM_STEERED = "plane-dem"

# How much ground beyond the claimed stretch the window shows, as a fraction of
# the stretch. The judgement being made is whether the intersection runs *along*
# a piece of trace, and a curve that leaves the picture exactly where the claim
# ends cannot be seen to diverge -- half again on each side is enough to see it
# go.
MARGIN = 0.5

# And the window's bounds, in cells. The floor is there because a claim can be a
# few metres of trace and a kernel on nine cells answers nothing; the ceiling is
# a frame budget -- 1200 square is about 32 ms against the 22 ms the tool
# measures at 1000, and past that steering stops being steering. Where the
# ceiling bites, the window is centred on the point and the caller says so: a
# picture smaller than the question is honest only if it is labelled.
MIN_CELLS = 120
MAX_CELLS = 1200

# How far the cut is from the trace, drawn as a multiple of the DEM's own cell.
# Full weight at one cell and gone at ten, and the scale is the data's rather
# than a number of metres somebody picked: the cut is marching squares over
# cells, so it cannot be located better than one of them, and a band that
# claimed to distinguish two metres on a five-metre grid would be drawing
# precision the kernel does not have. One decade, so the ramp is linear in
# log10 and its middle is about three cells.
GAP_FULL_CELLS = 1.0
GAP_FADE_CELLS = 10.0

# Below this slope difference there is no gap to report, and the number is not
# picked: it falls out of the two above.
#
# The elevation along the trace is read at the nearest cell, so the vertical
# mismatch is known to about half a cell whatever else is true. The gap is that
# mismatch divided by `|grad f|`, so its own uncertainty is `(cell/2) / tilt` --
# and once that reaches the distance the band fades out at, every value the band
# could draw is inside its own error bar. Setting the two equal,
# `(cell/2) / tilt = GAP_FADE_CELLS * cell`, and the cell cancels: what is left
# is a pure number, half over ten.
#
# It has to be a refusal and not a warning, which is what the first version got
# wrong. Lay a plane at the DEM's own attitude and the arithmetic does not
# divide by zero -- it divides 4.6e-13 by 2.0e-14 and hands back twenty-two
# metres, a number with no information in it at all, and the band drew it at
# full colour over 580 m of trace.
FLAT = 0.5 / GAP_FADE_CELLS

# And the most segments the band is drawn with. A bound on the frame rather
# than on the ground: at the window's own ceiling of 1200 cells this is a
# sample every metre and a half of screen, which is finer than the cut can be
# drawn, and it keeps a trace that wanders twenty kilometres through the window
# from putting twenty thousand segments into a collection redrawn by hand.
SAMPLE_CAP = 800


@dataclass
class Laid:
    """
    What one plane on one window came to, with the reference it ran in.

    `points` and `segments` are the kernel's own answer -- vertices and chord
    indices -- and not a path ready to draw, because between the two sits a
    coordinate transform the caller may or may not need. `broken_path` is the
    rest of the way, and it is separate for exactly that reason.
    """

    at: tuple = None                # the point the plane was laid on, x y z
    dip_dir: float = None           # as steered: azimuth from true north
    dip: float = None
    grid_dip_dir: float = None      # what the kernel was actually given
    convergence: float = 0.0
    points: object = None
    segments: object = None
    cells: tuple = None             # the window's shape, rows by columns
    refused: str = None             # why there is nothing, where there is nothing

    @property
    def chords(self):
        return 0 if self.segments is None else len(self.segments)

    def describe(self):
        """One line of what happened, for a status bar."""

        if self.refused:
            return self.refused

        rows, cols = self.cells

        return (
            f"{self.dip_dir:.1f}/{self.dip:.1f} on the DEM: {self.chords} chord(s) "
            f"over {cols}x{rows} cells; grid {self.grid_dip_dir:.1f} "
            f"(convergence {self.convergence:+.2f})"
        )


def side_for(span, res, margin=MARGIN, floor=MIN_CELLS, ceiling=MAX_CELLS):
    """
    The window's side in cells, for a stretch of ground `span` metres across.

    Square, and centred on the point the plane is laid on, which is the middle
    of the stretch: a rectangle fitted to the claim would be the tighter answer
    and the wrong shape to judge in, because the intersection leaves the trace
    sideways and the room to see it go is room across the strike.
    """

    if not res:
        return floor

    wanted = int(round(span * (1.0 + 2.0 * margin) / res))

    return max(floor, min(ceiling, wanted))


def laid_on(window, at, dip_dir, dip, nodata=None, convergence=0.0):
    """
    Where the plane through `at` cuts the topography in this window.

    `at` is the whole point, elevation and all, and the caller is the one that
    decides what that elevation is: on the ground under the trace for a contact
    somebody walked, off it for a horizon projected above today's topography.
    Nothing here reads the DEM for it, so nothing here can quietly put the plane
    somewhere other than where it was asked for.

    The convergence comes off before the kernel is called, and is kept. The dial
    is a true azimuth because that is what a compass reads and what the rest of
    this project's interfaces say; the DEM is on the projection's grid, and the
    two norths are 0.41 to 1.04 degrees apart in the southern Apennines. Over
    five kilometres of trace that is up to 91 metres of where the curve runs,
    which is the whole width of the judgement being made here.
    """

    from misah.kernels import intersect_plane_grid

    grid_dip_dir = (float(dip_dir) - float(convergence)) % 360.0

    points, segments = intersect_plane_grid(
        window.data,
        window.geotransform,
        [float(v) for v in at],
        grid_dip_dir,
        float(dip),
        nodata,
    )

    return Laid(
        at=tuple(float(v) for v in at),
        dip_dir=float(dip_dir),
        dip=float(dip),
        grid_dip_dir=grid_dip_dir,
        convergence=float(convergence),
        points=points,
        segments=segments,
        cells=tuple(window.shape),
    )


def broken_path(xs, ys, segments):
    """
    The kernel's loose chords as one path, with NaN between one and the next.

    A single `Line2D` and not a `LineCollection`, which is the tool's own
    measurement and the reason this shape exists: the chords are thousands of
    two-vertex segments, and for matplotlib one broken path costs 4-5 times less
    than as many separate paths -- 2.0 ms against 9.1 on a 1000x1000 window.

    `xs` and `ys` are taken apart from the kernel's `points` so that whatever
    has to happen to the coordinates can happen before the NaNs go in. A
    reprojection is the case: pyproj turns a NaN into an infinity, so a path
    already broken cannot be transformed, and a path transformed first can be
    broken.
    """

    if segments is None or not len(segments):
        return [], []

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    pairs = np.asarray(segments)

    path = np.full((len(pairs) * 3, 2), np.nan)
    path[0::3, 0] = xs[pairs[:, 0]]
    path[0::3, 1] = ys[pairs[:, 0]]
    path[1::3, 0] = xs[pairs[:, 1]]
    path[1::3, 1] = ys[pairs[:, 1]]

    return path[:, 0], path[:, 1]


# -- how far the cut runs from the trace, metre by metre ---------------------
#
# The eye can see that a cut runs along a trace when the two are on top of each
# other at the zoom being looked at. It cannot see thirty metres at 1:100000, it
# cannot tell which of a dozen curves in the window is the one near the trace,
# and it cannot say where along the trace the agreement stops -- which is the
# one number this whole gesture exists to produce, because it is what gets
# written as the ends of a `fit`.
#
# So the quantity is computed rather than looked at, and **not by measuring to
# the drawn chords**. The cut is the zero set of `f = z_plane - z_dem`, a field
# defined at every point of the trace whether a chord came near it or not, and
# the distance from a point to that zero set is `|f| / |grad f|` -- exact where
# `f` is linear, which it is for the plane's half of it. That buys three things
# the chords could not: a number everywhere instead of only near a curve, a cost
# that does not grow with how finely the kernel chopped the window, and the
# divisor itself.
#
# The divisor is the point. `grad f` is the difference between the plane's slope
# and the ground's, so it goes to zero exactly when the plane lies down on the
# hillside -- and that is when the gap explodes, which is the 1/sin measured
# across 154 attitudes and written up in the README. A method that measured to
# the chords would report a beautiful agreement there and say nothing about why
# it was worthless. This one divides by the thing that is failing, and can name
# it.


def walked(path, box=None, step=1.0, cap=SAMPLE_CAP):
    """
    Progressives and ground coordinates along a path, evenly, inside a box.

    Evenly and not at the path's own vertices, which is the opposite of
    `curation.stretch` and for the opposite reason: that draws a piece of trace,
    where a resampling would round off the bends somebody rejected it for, and
    this reads a quantity *as a function of s*, where the vertices are wherever
    the digitising happened to put them and would weight a densified stretch
    over a sparse one.

    `np.interp` over the cumulative chord length, which is the same definition
    of progressive the format uses, vectorised -- a call per sample into gstruct
    would be eight hundred of them inside a frame.

    The cap is applied last, by stride, so that it thins what survived the box
    rather than the whole path: a trace twenty kilometres long crossing a
    six-kilometre window should be sampled at the window's resolution, not at a
    twentieth of it.
    """

    points = np.asarray([(float(p[0]), float(p[1])) for p in path], dtype=float)

    if len(points) < 2:
        return np.zeros(0), np.zeros((0, 2))

    steps = np.hypot(*np.diff(points, axis=0).T)
    along = np.concatenate(([0.0], np.cumsum(steps)))
    length = float(along[-1])

    if length <= 0.0:
        return np.zeros(0), np.zeros((0, 2))

    s = np.arange(0.0, length + float(step), float(step))
    s = s[s <= length]

    xy = np.column_stack((
        np.interp(s, along, points[:, 0]),
        np.interp(s, along, points[:, 1]),
    ))

    if box is not None:
        left, bottom, right, top = box
        inside = (
            (xy[:, 0] >= left) & (xy[:, 0] <= right)
            & (xy[:, 1] >= bottom) & (xy[:, 1] <= top)
        )
        s, xy = s[inside], xy[inside]

    if cap and len(s) > cap:
        stride = int(np.ceil(len(s) / float(cap)))
        s, xy = s[::stride], xy[::stride]

    return s, xy


@dataclass
class Ground:
    """
    The topography read once along a trace: what does not change as a dial turns.

    The split is the whole reason the band can be drawn inside a frame. Reading
    the DEM and differencing it costs a gather over the window; the plane costs
    six multiplications. Steering moves the plane and nothing else, so the DEM
    half is done when the window or the trace changes -- once per pan, not
    ninety times per turn of a dial.

    `slope` is the ground's own gradient by central differences, which is why
    the edge cells of the window are dropped rather than one-sided: a sample
    against the window's rim is outside the question anyway, and a one-sided
    difference there would be a different estimator quietly mixed into the same
    column.
    """

    s: object = None                # progressives, metres along the trace
    xy: object = None               # where that is on the ground
    z: object = None                # the DEM there
    slope: object = None            # its gradient, (n, 2), metres per metre
    step: float = 0.0               # the spacing the samples came out at
    cell: float = 0.0               # the DEM's, which is what the gap is read in
    whole: float = 0.0              # how long the trace is, sampled or not

    def __len__(self):
        return 0 if self.s is None else len(self.s)

    @property
    def metres(self):
        """How much trace this is, which is the denominator of every share."""

        return len(self) * self.step


def ground_on(window, s, xy, nodata=None, cell=None, whole=0.0):
    """
    `Ground` for the samples that fall inside this window, with the rest dropped.

    Nearest cell, as `Dem.elevation_at` is, and deliberately the same: the plane
    is hung at the elevation of the cell the pin is in, so reading the trace by
    interpolation would make the pin's own sample come out a metre off zero for
    no reason but the two readings disagreeing.

    `window_at` does not turn nodata into NaN -- the kernel is given the value
    and wants it -- so it is turned into a dropped sample here. Both, in fact:
    NaN arrives from `window_over`, the sentinel from `window_at`, and a band
    drawn over a hole would be the one place it said something confident about
    ground nobody has measured.
    """

    x0, dx, _, y0, _, dy = [float(v) for v in window.geotransform]
    data = window.data
    rows, cols = data.shape

    if not len(s):
        return Ground(np.zeros(0), np.zeros((0, 2)), np.zeros(0),
                      np.zeros((0, 2)), 0.0, cell or abs(dx), whole)

    col = np.floor((xy[:, 0] - x0) / dx).astype(int)
    row = np.floor((xy[:, 1] - y0) / dy).astype(int)

    inside = (col >= 1) & (col <= cols - 2) & (row >= 1) & (row <= rows - 2)

    s, xy, row, col = s[inside], xy[inside], row[inside], col[inside]

    if not len(s):
        return Ground(s, xy, np.zeros(0), np.zeros((0, 2)), 0.0, cell or abs(dx), whole)

    z = data[row, col].astype(float)

    east = (data[row, col + 1].astype(float) - data[row, col - 1].astype(float))
    # `row - 1` is the northern neighbour: `dy` is negative, so a larger row is
    # a smaller northing, and dividing by `2 * dy` instead would hand back a
    # gradient pointing south while everything else here counts northwards.
    north = (data[row - 1, col].astype(float) - data[row + 1, col].astype(float))

    slope = np.column_stack((east / (2.0 * dx), north / (2.0 * abs(dy))))

    good = np.isfinite(z) & np.isfinite(slope).all(axis=1)

    if nodata is not None:
        good &= z != nodata
        good &= (
            (data[row, col + 1] != nodata) & (data[row, col - 1] != nodata)
            & (data[row - 1, col] != nodata) & (data[row + 1, col] != nodata)
        )

    s, xy, z, slope = s[good], xy[good], z[good], slope[good]

    spacing = float(np.median(np.diff(s))) if len(s) > 1 else 0.0

    return Ground(s, xy, z, slope, spacing, cell or abs(dx), whole)


@dataclass
class Gaps:
    """How far the cut runs from the trace, sample by sample, and how sure that is."""

    ground: Ground = None
    gap: object = None              # metres from the trace to the cut, unsigned
    rise: object = None             # the plane above the ground there, signed
    tilt: object = None             # |grad f|: what the gap was divided by

    @property
    def close(self):
        """
        Weight from one to nought over the decade above a cell, for drawing.

        Linear in log10, so the ramp spends as much of itself between one cell
        and three as between three and ten. A linear ramp would put nine tenths
        of its colour on distances the kernel cannot resolve apart.
        """

        cell = self.ground.cell or 1.0

        with np.errstate(divide="ignore", invalid="ignore"):
            decades = np.log10(np.maximum(self.gap, 1e-9) / (GAP_FULL_CELLS * cell))

        decades = np.where(np.isfinite(decades), decades, np.inf)
        span = np.log10(GAP_FADE_CELLS / GAP_FULL_CELLS)

        return np.clip(1.0 - decades / span, 0.0, 1.0)

    def within(self, metres):
        """Metres of the sampled trace the cut runs closer than that to."""

        return float(np.count_nonzero(self.gap <= metres)) * self.ground.step

    @property
    def flat(self):
        """
        The share of the trace where the plane lies too near the slope to answer.

        Reported and not only acted on. A band gone pale can mean the cut is far
        away or that there is no cut to be far, and those are opposite findings:
        the first says this attitude is wrong, the second says the ground here
        cannot tell any attitude from another.
        """

        if not len(self.ground):
            return 0.0

        return float(np.count_nonzero(self.tilt < FLAT)) / len(self.ground)

    def describe(self):
        """
        One line, and it says where there was no question as well as the answer.

        Never the agreement alone. A cut that hugs the trace because the plane
        is parallel to the slope hugs every trace on that slope, and reporting
        only the metres would be the tool agreeing with whatever it was handed.
        """

        ground = self.ground

        if not len(ground):
            return "no trace in the window to measure against"

        cell = ground.cell or 1.0
        near, far = GAP_FULL_CELLS * cell, GAP_FADE_CELLS * cell

        said = (
            f"within {near:.0f} m of {self.within(near):.0f} m of trace and "
            f"within {far:.0f} m of {self.within(far):.0f} m, "
            f"of {ground.metres:.0f} sampled"
        )

        # Where the band stops because the window did, and not because the
        # agreement did. The two look identical on the map -- a band that fades
        # out and a band that is cut off are both a band ending -- and the
        # difference is the whole reading: one says the plane stops following
        # the trace there, the other says nobody has asked yet.
        if ground.whole > ground.metres + ground.step:
            said += (
                f" of {ground.whole:.0f} on the trace: the band stops where the "
                f"window does"
            )

        if self.flat:
            said += (
                f" -- and over {self.flat * 100:.0f}% of it the plane runs "
                f"within {FLAT * 100:.0f}% slope of the ground, where the cut "
                f"is wherever it was put: no answer rather than a good one"
            )

        return said


def gaps_on(ground, at, dip_dir, dip, convergence=0.0):
    """
    Where the plane's cut runs, relative to the trace `ground` was read along.

    The plane is in true azimuth as everything the hand touches is, and the
    ground is on the grid, so the convergence comes off here for the same reason
    and by the same rule as in `laid_on` -- and through the same subtraction, so
    the two pictures on the map cannot come from two different planes.

    Where the gradients cancel there is no answer and the gap is infinite, which
    is right and is not a failure: two surfaces of the same slope either
    coincide or never meet, and neither of those is a distance. `np.errstate`
    rather than a guard, because the zero is a fact about the geometry and not
    about the arithmetic.
    """

    if not len(ground):
        empty = np.zeros(0)

        return Gaps(ground, empty, empty, empty)

    azimuth = np.radians((float(dip_dir) - float(convergence)) % 360.0)
    fall = np.tan(np.radians(float(dip)))

    # A plane dipping `dip` towards `azimuth` loses height at `tan(dip)` along
    # that bearing and none across it, so its gradient is that, pointed the
    # other way. Written out rather than taken off the kernel: the kernel
    # answers with chords, and a second reading of the same two numbers is
    # cheaper than asking it what it thinks the surface is.
    gx = -fall * np.sin(azimuth)
    gy = -fall * np.cos(azimuth)

    x0, y0, z0 = [float(v) for v in at]

    rise = (
        z0 + gx * (ground.xy[:, 0] - x0) + gy * (ground.xy[:, 1] - y0) - ground.z
    )

    tilt = np.hypot(gx - ground.slope[:, 0], gy - ground.slope[:, 1])

    with np.errstate(divide="ignore", invalid="ignore"):
        gap = np.abs(rise) / tilt

    # Not-a-number becomes infinity and never nought. `0/0` happens where the
    # two surfaces have the same slope *and* touch, and a NaN left in this
    # column would be swallowed by the first comparison that met it -- `within`
    # counts it out, but a clip would have handed it back as the closest sample
    # on the trace. Infinity is also what it means: no distance to a set the
    # plane lies inside of.
    #
    # And the same for a divisor under `FLAT`, which is the case that actually
    # happens: exact zero needs the two surfaces to be parallel to the last bit,
    # while a tenth of a degree apart divides two small numbers and answers with
    # a plausible one. See `FLAT` for where the floor comes from -- it is the
    # point past which the answer is inside the half cell the elevation was read
    # to.
    gap = np.where(np.isfinite(gap) & (tilt >= FLAT), gap, np.inf)

    return Gaps(ground, gap, rise, tilt)
