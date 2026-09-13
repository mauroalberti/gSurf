"""
The fold axis of a set of bedding poles, and whether there is one to report.

The arithmetic is geogst's. `bingham_axes` gives the eigenvalues and
eigenvectors of the orientation tensor with Woodcock's K and C, which is the
whole of the calculation; what is added here is the part that is not
arithmetic -- that an axis is only an axis when the poles lie on a girdle, and
that a number computed from eleven measurements is not the same claim as one
computed from a hundred.

Costs, measured on the Potenza-Irsina sheet: 0.39 ms for a window of 20 poles,
0.99 ms for 68, 4.2 ms for 313. At a frame that is affordable, which is why
this calls geogst rather than reimplementing it in arrays. A grid of ten
thousand windows is a different question, and the answer to it belongs upstream
in geogst as a vectorised orientation tensor -- not here as a second copy of
the same mathematics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Gate:
    """
    What a fold axis has to clear before it is worth reading.

    K below one is the whole of it geologically: above one the poles cluster,
    which is a homocline, and the minimum eigenvector of a cluster is the least
    determined direction in the data rather than a fold axis. Measured on the
    1757 CARG attitudes of Potenza-Irsina, three windows in four fail this --
    which is the reason the gate exists rather than an argument against it.

    C keeps out the other end, a spread so weak that all three axes are
    interchangeable, and the count keeps out the windows where a girdle is two
    measurements and an accident.
    """

    min_points: int = 10
    max_k: float = 1.0
    min_c: float = 1.0

    def admits(self, result):
        return not self.refusal(result)

    def refusal(self, result):
        """Why this result is not a fold axis, or an empty string if it is."""

        if result is None:
            return "no measurement"

        return self.refusal_for(result.n, result.k, result.c)

    def refusal_for(self, n, k, c):
        """
        The same rule, on the three numbers alone.

        A field keeps its cells as arrays and not as objects, and this is what
        lets the gate be re-applied to one without recomputing a single tensor.
        One rule with two callers, so that a threshold moved on screen cannot
        mean something different from the same threshold at a frame.
        """

        if n < self.min_points:
            return f"{n} attitudes, fewer than {self.min_points}"

        if math.isnan(k):
            return "no shape: the three axes are equal"

        if k > self.max_k:
            return f"K = {k:.2f}: a cluster, not a girdle"

        if c < self.min_c:
            return f"C = {c:.2f}: too weak to have a direction"

        return ""


@dataclass(frozen=True)
class FoldAxis:
    """The orientation tensor of a set of poles, read as a fold."""

    n: int
    eigenvalues: tuple          # S1 >= S2 >= S3, summing to one
    principal: tuple            # the matching axes, as (trend, plunge)
    k: float                    # Woodcock's shape
    c: float                    # Woodcock's strength

    @property
    def axis(self):
        """
        The fold axis: the minimum eigenvector.

        For poles spread on a girdle the plane they spread on is the plane
        normal to S3, and its normal -- S3 itself -- is the direction they all
        turn about. That is the axis, and it is beta and pi at once because the
        poles are of the folded surface itself.
        """

        return self.principal[2]

    @property
    def girdle(self):
        """The best-fit girdle as a plane, dip direction and dip angle."""

        trend, plunge = self.axis

        return (trend + 180.0) % 360.0, 90.0 - plunge

    @property
    def description(self):
        from geogst.core.geology.stats.directional import woodcock_description

        return woodcock_description(self.k, self.c)


def fold_axis(poles):
    """
    The fold axis of a set of poles, or None if there is not enough to say.

    Takes the poles as geogst axes, which is how `AttitudeSource` keeps them:
    building them per call would be paying for the same arithmetic on every
    frame, since a window is a subset of a set that does not change.
    """

    from geogst.core.geology.stats.directional import bingham_axes

    if len(poles) < 2:
        return None

    result, err = bingham_axes(list(poles))

    if err:
        return None

    principal, eigenvalues, k, c = result

    return FoldAxis(
        n=len(poles),
        eigenvalues=tuple(float(v) for v in eigenvalues),
        principal=tuple((float(axis.d[0]), float(axis.d[1])) for axis in principal),
        k=float(k),
        c=float(c),
    )


# -- the same question, everywhere at once --------------------------------


@dataclass
class FoldAxisField:
    """
    One window's answer per cell, as arrays that line up with each other.

    Aligned arrays and a tally rather than a list of objects, which is the shape
    misah's `best_fit_planes` settled on for the same kind of result: a field is
    read column-wise -- every trend, every K -- far more often than cell by cell,
    and a thousand small objects would be built to be immediately taken apart.

    `admitted` is the column that matters. A field where every cell carries an
    axis is a field that has not been read: on the Potenza-Irsina sheet three
    windows in four are clusters, and drawing those would be drawing noise with
    the confidence of a measurement.

    The whole eigenframe is kept and not only the axis. S1, S2 and S3 are an
    orthonormal triad, and sign-blind in each of the three, which is the
    symmetry of a double couple: a pair of them can be compared by the rotation
    that carries one onto the other, the way two focal mechanisms are. The axis
    alone cannot -- the rotations carrying one line onto another are a family,
    not a rotation. Dropping S1 and S2 here is what made that comparison need a
    second pass over the attitudes, so they are kept where they are computed.

    They are not the equal of S3, though. S1 and S2 are only separated by
    ln(S1/S2), and that vanishes on the good girdles: a swap of the two is a
    quarter turn about the axis, and no symmetry of the triad undoes it. The
    eigenvalues travel with the axes so that whoever compares two frames can
    first ask whether the frames are determined.
    """

    centres: np.ndarray         # (M, 2), the window centres in map coordinates
    counts: np.ndarray          # (M,) attitudes in each
    trends: np.ndarray          # (M,) axis trend, true azimuth; NaN where none
    plunges: np.ndarray         # (M,)
    s1: np.ndarray              # (M, 2) maximum eigenvector, (trend, plunge)
    s2: np.ndarray              # (M, 2) intermediate; S3 is trends/plunges above
    eigenvalues: np.ndarray     # (M, 3) S1 >= S2 >= S3, each row summing to one
    k: np.ndarray               # (M,) Woodcock shape; NaN where none
    c: np.ndarray               # (M,)
    admitted: np.ndarray        # (M,) bool: cleared the gate
    radius: float
    step: float
    gate: Gate
    refusals: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.centres)

    @property
    def occupied(self):
        """Cells with at least one attitude in them."""

        return self.counts > 0

    def regate(self, gate):
        """
        Decides the whole field again against a different gate, in place.

        No tensor is recomputed: K, C and the count are what the gate reads, and
        they are already here. That is the payoff of keeping a field as arrays
        rather than as a picture -- a threshold is a question asked of a result,
        not part of computing it, and sweeping one over an existing field is the
        only way to see how much of the answer depends on where it was put.
        """

        self.gate = gate
        refusals = {}

        for index in range(len(self)):
            if self.counts[index] == 0:
                self.admitted[index] = False
                continue

            refusal = gate.refusal_for(
                int(self.counts[index]), float(self.k[index]), float(self.c[index])
            )
            self.admitted[index] = not refusal

            if refusal:
                if "fewer than" in refusal:
                    kind = "too few attitudes"
                elif "cluster" in refusal:
                    kind = "cluster"
                else:
                    kind = "too weak"
                refusals[kind] = refusals.get(kind, 0) + 1

        self.refusals = refusals

        return self

    @property
    def bounds(self):
        left, bottom = self.centres.min(axis=0)
        right, top = self.centres.max(axis=0)

        return float(left), float(bottom), float(right), float(top)

    def summary(self):
        total, taken = len(self), int(self.admitted.sum())
        occupied = int(self.occupied.sum())

        text = (
            f"{total} cells at {self.step:.0f} m, r = {self.radius:.0f} m: "
            f"{occupied} with data, {taken} fold axes"
        )

        if self.refusals:
            detail = ", ".join(f"{count} {reason}" for reason, count in self.refusals.items())
            text += f" ({detail})"

        # Said again after the fact, and not only before. A count of axes is
        # the number that gets quoted, and it is the one that most needs the
        # overlap beside it.
        return f"{text}; {describe_sampling(self.bounds, self.radius, self.step)}"


def sampling(bounds, radius, step):
    """
    How much a grid overlaps itself, which is not a thing the controls show.

    The radius and the step are set in two different places and their ratio is
    never named, so a field can be read as though its cells were separate
    observations when they are mostly the same measurements counted again. A
    circular window of radius R laid down every S metres covers pi R^2 / S^2
    cells, so that is how many cells each attitude falls into: at r = 2000 and
    a step of 500, fifty. Neighbouring cells then share nine tenths of their
    data, and a map of a thousand axes carries perhaps fifty windows' worth of
    it.

    The two numbers multiply back to the cell count -- cells = per_attitude x
    tiling -- which is the whole point: a denser step buys resolution in the
    picture and no further information under it.

    `tiling` counts windows that do not overlap, which is geometry. It is not a
    count of independent observations: structures are continuous, and two
    windows that merely fail to touch can still be looking at the same fold.
    """

    left, bottom, right, top = bounds
    area = max(0.0, right - left) * max(0.0, top - bottom)
    window = math.pi * radius * radius

    return dict(
        per_attitude=window / (step * step) if step > 0 else float("inf"),
        tiling=area / window if window > 0 else 0.0,
        overlapping=step < 2.0 * radius,
    )


def describe_sampling(bounds, radius, step):
    """The overlap in one line, for a panel or a status bar."""

    counts = sampling(bounds, radius, step)

    if not counts["overlapping"]:
        return "cells do not overlap"

    return (
        f"each attitude in ~{counts['per_attitude']:.0f} cells; "
        f"~{counts['tiling']:.0f} windows would tile the area"
    )


def grid_centres(bounds, step):
    """
    The centres of a square grid over an area, as an (M, 2) array.

    The grid is anchored on the area's corner rather than on round coordinates:
    a field is read against the data it came from, and a step that shifted with
    the extent would make two runs on the same data incomparable.
    """

    left, bottom, right, top = bounds

    xs = np.arange(left, right + step, step, dtype=float)
    ys = np.arange(bottom, top + step, step, dtype=float)
    gx, gy = np.meshgrid(xs, ys)

    return np.c_[gx.ravel(), gy.ravel()]


def field_cost(attitudes, centres, radius, gate=None):
    """
    What a field will cost, before it is asked for.

    Measured on the Potenza-Irsina sheet, where the whole of it is the tensor:
    a 16289-cell grid at 250 m takes 6.9 s, of which 6.0 is geogst and 0.9 the
    search. The estimate is deliberately coarse -- it exists so that a step
    typed by hand cannot start a computation of unknown length without saying
    how long, not to be right to the millisecond.
    """

    gate = gate or Gate()

    # Sampled rather than counted: counting every cell is doing the search
    # twice, and the point is to answer before the work starts.
    sample = centres[:: max(1, len(centres) // 200)]
    found = [len(attitudes.within(x, y, radius)) for x, y in sample]

    # Two and not `gate.min_points`: a tensor is computed wherever one is
    # defined, so that the gate can be moved afterwards over its whole range.
    # Counting from the gate's floor is what made this estimate read 1.5 s for a
    # field that took 2.4.
    computed = [n for n in found if n >= 2]
    occupied_share = float(len(computed)) / len(found) if found else 0.0
    mean_n = float(np.mean(computed)) if computed else 0.0

    cells = len(centres)
    occupied = int(round(cells * occupied_share))

    # 17 microseconds per pole per window and a fixed 150 per window, fitted to
    # four grids on the Potenza-Irsina sheet spanning 312 to 16289 cells; the
    # search is about 55 nanoseconds per station per cell. Fitted to err high
    # rather than to sit on the measurements: a wait that turns out shorter than
    # promised costs nothing, and one that turns out longer is the reason for
    # putting a number here at all. Within a factor of two across that range,
    # which is what it claims and no more.
    seconds = occupied * (mean_n * 17e-6 + 150e-6) + cells * len(attitudes) * 55e-9

    return dict(cells=cells, occupied=occupied, mean_count=mean_n, seconds=seconds)


def fold_axis_field(attitudes, centres, radius, gate=None, progress=None):
    """
    A fold axis under every window of a grid.

    `progress` is called with the number of cells done and the total, and may
    return False to give up -- a field of any size has to be interruptible,
    because the step that makes it too slow is one keystroke away from the one
    that does not.
    """

    gate = gate or Gate()
    total = len(centres)

    counts = np.zeros(total, dtype=int)
    trends = np.full(total, np.nan)
    plunges = np.full(total, np.nan)
    s1 = np.full((total, 2), np.nan)
    s2 = np.full((total, 2), np.nan)
    eigenvalues = np.full((total, 3), np.nan)
    k = np.full(total, np.nan)
    c = np.full(total, np.nan)
    admitted = np.zeros(total, dtype=bool)
    refusals = {}

    # Report about two hundred times whatever the size, rather than every fixed
    # number of cells. A fixed stride of 256 left a grid of 121 cells with a
    # single report at zero -- a progress bar that never moves and a Stop button
    # that does nothing, on exactly the small fields where the wait is short
    # enough that both go unnoticed and untested.
    stride = max(1, min(256, total // 200))

    for index, (x, y) in enumerate(centres):
        if progress is not None and index % stride == 0:
            if progress(index, total) is False:
                break

        inside = attitudes.within(x, y, radius)
        counts[index] = len(inside)

        # Computed from two poles up, and not from the gate's minimum, even
        # though everything under it will be refused. The gate is a question
        # asked of the result afterwards, and `regate` can only re-ask it over
        # its whole range if the tensor is there to be asked about: stopping at
        # today's `min_points` would silently make lowering it impossible. Two
        # is where the tensor itself stops being defined, so that floor is the
        # arithmetic's and not a policy.
        if len(inside) < 2:
            if len(inside):
                refusals["too few attitudes"] = refusals.get("too few attitudes", 0) + 1
            continue

        result = fold_axis(attitudes.poles_at(inside))

        if result is None:
            continue

        trends[index], plunges[index] = result.axis
        s1[index] = result.principal[0]
        s2[index] = result.principal[1]
        eigenvalues[index] = result.eigenvalues
        k[index], c[index] = result.k, result.c

        refusal = gate.refusal(result)
        admitted[index] = not refusal

        if refusal:
            # The wording carries the numbers; the tally wants the kind.
            if "fewer than" in refusal:
                kind = "too few attitudes"
            elif "cluster" in refusal:
                kind = "cluster"
            else:
                kind = "too weak"
            refusals[kind] = refusals.get(kind, 0) + 1

    step = float(np.min(np.diff(np.unique(centres[:, 0])))) if len(centres) > 1 else 0.0

    return FoldAxisField(
        centres=centres,
        counts=counts,
        trends=trends,
        plunges=plunges,
        s1=s1,
        s2=s2,
        eigenvalues=eigenvalues,
        k=k,
        c=c,
        admitted=admitted,
        radius=float(radius),
        step=step,
        gate=gate,
        refusals=refusals,
    )
