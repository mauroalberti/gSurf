"""
How much a fold axis turns between one window and the next, and about what axis.

A field of fold axes says where each window's axis points. What it does not say
is how the pointing changes across the map, and that is the part with the
tectonics in it: a set of axes swinging through forty degrees over twenty
kilometres and a set meeting another set across a line are the same histogram
and different countries.

The angle alone is not the measurement. A fold axis is a line, the rotations
carrying one line onto another are a one-parameter family, and the one of least
angle has for its angle the angle between the two lines -- so calling it a
rotation adds a word and no content. What the rotation has that the angle does
not is an AXIS:

    steep plunge    rotation in plan: oroclinal flexure, transfer zones, block
                    rotation about the vertical
    shallow plunge  the axis changing its own plunge: tilting, refolding,
                    culminations and depressions

Three things here are not obvious and each one was paid for once.

Bearings must be grid north. Grid north is a single direction over the whole
projection plane, so a difference between two distant nodes is a real relative
rotation; true azimuths are each referred to their own meridian, and across a
map the convergence injects a systematic spurious rotation -- about a degree on
a hundred-kilometre span, which is the size of the signal being looked for.
`field_frame` writes `trend_grd` and `converg` for exactly this reason. The
counter-intuitive half: for a SPATIAL GRADIENT the grid is the right reference
and true north is the wrong one, which is the reverse of the usual advice.

Neighbours lie on a RING, not on the eight adjacent cells. The four diagonals of
a 3x3 neighbourhood are at step*sqrt(2) and not at step: dividing those by h as
well gets the gradient wrong by 41 per cent. A ring [h - step/2, h + step/2) has
one distance in it, and at large h holds many more nodes than eight. What the
ring gives back is an average over directions, so against a unidirectional ramp
it reads low by 2/pi -- a known factor, not an error, and `check_rotations`
pins it.

Rotations average as AXIS-ANGLE VECTORS, not as angles. With the axis in the
lower hemisphere and the angle signed, omega = theta * n is a true vector.
Averaging the magnitudes and averaging the vectors are two different questions,
and their ratio is the third: axis-sharing rotations give one, isotropic noise
gives zero.

A node with no valid neighbour comes out NaN, never zero. Zero is a rotation
that was measured and found to be nothing, and in a table that reaches a GIS the
difference between the two is invisible from downstream.

The naming of columns, the choice of baselines and what counts as a regional
mode are not here: they are what a study decides, and they change with the map.
This module returns arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Below this norm of the cross product the two axes are parallel and there is no
# rotation axis: angle zero, axis NaN. There is no axis about which one does not
# turn, and inventing one would put a direction into the average.
PARALLEL_TOL = 1.0e-9

# Below this sine of the plunge the rotation axis is horizontal and the lower
# hemisphere cannot tell its two ends apart. A thousandth of a degree, far below
# anything a measurement could mean.
HORIZONTAL_TOL = 1.0e-5


def versors(trend, plunge):
    """
    Unit vectors (East, North, Up) of downward directions given as trend/plunge.

    The convention geogst uses in `Direct.as_versor` and the checks use too: the
    third component points up, so a positive plunge makes it negative.
    """

    t = np.radians(np.asarray(trend, dtype=float))
    p = np.radians(np.asarray(plunge, dtype=float))

    return np.column_stack([np.sin(t) * np.cos(p), np.cos(t) * np.cos(p), -np.sin(p)])


def _clamped_to_one(values):
    """
    Pulls an overshoot of a few ulp back to one, and lets NaN through.

    A comparison chain and not `min`/`max`, which is a rule this project has had
    to learn in three languages: in Python `max(0.0, nan)` returns 0.0 and eats
    the NaN, and in gfortran the intrinsic returns whichever operand the
    compiler put second. Here `nan > 1.0` is False, so the NaN survives to
    downstream, where it means "not measured" instead of "measured zero".
    """

    return np.where(values > 1.0, 1.0, values)


def rotation_between_axes(a, b):
    """
    The least-angle rotation carrying axis `a` onto axis `b`.

    Takes two (N, 3) arrays of unit vectors and gives back (angle, trend,
    plunge) of the rotation axis, (N,) each.

    The angle is signed and lies in [-90, +90], positive being clockwise seen
    from above, with the rotation axis forced into the lower hemisphere. Without
    that convention the sign is not defined at all -- turning `a` by +theta about
    n and by -theta about -n are the same rotation -- and the handedness, which
    is what separates a right-stepping transfer zone from a left-stepping one,
    would be a coin toss.

    Where the two axes are parallel the angle is zero and the axis NaN.
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    product = np.sum(a * b, axis=1)

    # An axis is sign-blind: take whichever end of b lies on a's side, so the
    # angle falls in [0, 90] instead of [0, 180].
    b = b * np.where(product >= 0.0, 1.0, -1.0)[:, None]
    angle = np.degrees(np.arccos(_clamped_to_one(np.abs(product))))

    normal = np.cross(a, b)
    norm = np.linalg.norm(normal, axis=1)
    valid = norm > PARALLEL_TOL

    versor = np.full_like(normal, np.nan)
    np.divide(normal, norm[:, None], out=versor, where=valid[:, None])

    # Lower hemisphere, with the sign of the angle keeping the account.
    #
    # On a horizontal axis the hemisphere has no grip, and a fallback rule added
    # in OR does not fix it -- the primary rule has to be taken away. A rotation
    # of pure plunge has a horizontal axis by construction, but its third
    # component leaves the cross product as +1e-16 or -1e-16 depending on the
    # rounding, and `z > 0` on that number flips the axis for no reason that
    # exists. Without the case distinction two identical rotations -- 140/00 ->
    # 140/30 and 140/30 -> 140/60, both a plunging towards 140 -- come out with
    # opposite signs.
    #
    # Under tolerance the trend decides, brought into [0, 180): at zero plunge
    # the versor is (sin trend, cos trend, 0), so the trend is in [0, 180) when
    # the East component is not negative. The tie at East = 0 is between 000 and
    # 180 and is settled on North.
    horizontal = np.abs(versor[:, 2]) <= HORIZONTAL_TOL

    upward = ~horizontal & (versor[:, 2] > 0.0)
    upward |= horizontal & (versor[:, 0] < 0.0)
    upward |= horizontal & (np.abs(versor[:, 0]) <= HORIZONTAL_TOL) & (versor[:, 1] < 0.0)

    versor[upward] *= -1.0
    angle = np.where(upward, -angle, angle)
    angle = np.where(valid, angle, 0.0)

    # After the flip the third component is <= 0, so -z is the sine of the
    # plunge and already in [0, 1]: no abs, and no sign to put back.
    plunge = np.degrees(np.arcsin(_clamped_to_one(-versor[:, 2])))
    trend = np.degrees(np.arctan2(versor[:, 0], versor[:, 1])) % 360.0

    return angle, trend, plunge


def ring(step, h):
    """
    The integer grid offsets whose true distance falls in a ring about h.

    The ring is [h - step/2, h + step/2): one distance, to within half a step.
    Gives back the offsets (M, 2) and the true distances (M,), which are needed
    because a gradient divides by the distance of the individual neighbour and
    not by the nominal h.
    """

    limit = int(np.ceil((h + step) / step))
    di, dj = np.mgrid[-limit : limit + 1, -limit : limit + 1]
    distance = np.hypot(di, dj) * step

    kept = (distance >= h - step / 2.0) & (distance < h + step / 2.0)

    return np.column_stack([di[kept], dj[kept]]), distance[kept]


class Lattice:
    """
    Scattered grid nodes as integer indices, with a bordered lookup table.

    A fold axis field is computed on `grid_centres`, which is regular and
    anchored on the corner of the bounds, so every node sits on an integer
    lattice whatever subset of them survives a gate. A bordered array of row
    indices turns the search for a neighbour into a shift, and each offset into
    one vectorised pass instead of N lookups in a dict.

    The regularity is checked rather than assumed. Held in memory the field
    cannot be anything else, but the same nodes read back from a file can be:
    they have been through a projection, a driver and a rounding, and a table
    whose spacing is 499.97 m would otherwise be answered with silence and
    empty neighbourhoods.
    """

    def __init__(self, x, y, step, margin):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        i = np.round((x - x.min()) / step).astype(int)
        j = np.round((y - y.min()) / step).astype(int)

        drift = np.max(
            np.abs(np.column_stack([(x - x.min()) / step - i, (y - y.min()) / step - j]))
        )
        if drift > 0.01:
            raise ValueError(
                f"the nodes are not on a lattice of step {step:.0f} m "
                f"(worst drift {drift:.3f} cells): this is not a regular grid"
            )

        table = np.full((i.max() + 1 + 2 * margin, j.max() + 1 + 2 * margin), -1, dtype=int)
        table[i + margin, j + margin] = np.arange(len(i))

        self.i, self.j = i, j
        self.step = float(step)
        self.margin = int(margin)
        self.table = table

    @classmethod
    def for_baselines(cls, x, y, step, baselines):
        """
        Sized so that the longest baseline's ring still lands inside the border.
        """

        margin = int(np.ceil((max(baselines) + step) / step))

        return cls(x, y, step, margin)

    def __len__(self):
        return len(self.i)

    def shifted(self, di, dj):
        """
        For one offset: the nodes that have such a neighbour, and which it is.

        Two index arrays of equal length, `here` into the nodes and `there` into
        the same, so that a paired operation is a single fancy-indexed pass.
        """

        neighbour = self.table[self.i + self.margin + di, self.j + self.margin + dj]
        present = neighbour >= 0

        return np.flatnonzero(present), neighbour[present]


@dataclass
class BaselineRotations:
    """
    What the rotations at one baseline came to, one entry per node.

    `rotation` and `net` answer two different questions and their ratio is the
    third: mean magnitude against magnitude of the mean, which is one where the
    rotations share an axis and zero where they are isotropic noise.
    """

    baseline: float             # metres
    rotation: np.ndarray        # (N,) mean |angle|, degrees; NaN with no neighbour
    gradient: np.ndarray        # (N,) mean |angle| / distance, degrees per km
    net: np.ndarray             # (N,) magnitude of the mean axis-angle vector
    coherence: np.ndarray       # (N,) net / rotation
    axis_trend: np.ndarray      # (N,) the mean rotation axis
    axis_plunge: np.ndarray     # (N,)
    sense: np.ndarray           # (N,) net, signed: positive clockwise from above
    neighbours: np.ndarray      # (N,) how many contributed


def rotation_field(lattice, axes, baselines):
    """
    The rotation statistics at every baseline, for nodes that have an axis.

    `axes` are the (N, 3) versors of the nodes in `lattice`, in that order, and
    they must be on GRID north -- see the module docstring for why that is not a
    detail. Gives back one `BaselineRotations` per baseline, in the order asked.

    Nodes that failed the gate must not be in here at all. A refused window is
    not an axis, and letting one in as a neighbour measures the rotation towards
    a thing that was decided not to be a measurement.
    """

    axes = np.asarray(axes, dtype=float)
    n_nodes = len(lattice)
    out = []

    for h in baselines:
        offsets, distances = ring(lattice.step, h)

        magnitudes = np.zeros(n_nodes)
        gradients = np.zeros(n_nodes)
        omegas = np.zeros((n_nodes, 3))
        count = np.zeros(n_nodes, dtype=int)

        for (di, dj), distance in zip(offsets, distances):
            here, there = lattice.shifted(di, dj)

            if len(here) == 0:
                continue

            angle, trend, plunge = rotation_between_axes(axes[here], axes[there])
            omega = angle[:, None] * versors(trend, plunge)

            # Where the axes are parallel the rotation axis is NaN: that pair's
            # contribution to the vector is nil by construction, not missing.
            omega = np.where(np.isfinite(omega), omega, 0.0)

            magnitudes[here] += np.abs(angle)
            gradients[here] += np.abs(angle) / (distance / 1000.0)
            omegas[here] += omega
            count[here] += 1

        has = count > 0
        divisor = np.where(has, count, 1)

        rotation = np.where(has, magnitudes / divisor, np.nan)
        gradient = np.where(has, gradients / divisor, np.nan)

        mean_omega = np.where(has[:, None], omegas / divisor[:, None], np.nan)
        net = np.linalg.norm(mean_omega, axis=1)

        # The net rotation's own axis, put into the lower hemisphere with the
        # sign carrying the account, as for a single pair.
        with np.errstate(invalid="ignore", divide="ignore"):
            direction = mean_omega / net[:, None]
        sign = np.where(direction[:, 2] > 0.0, -1.0, 1.0)
        direction = direction * sign[:, None]

        out.append(BaselineRotations(
            baseline=float(h),
            rotation=rotation,
            gradient=gradient,
            net=net,
            coherence=np.where(rotation > 0, net / np.where(rotation > 0, rotation, 1), np.nan),
            axis_trend=np.degrees(np.arctan2(direction[:, 0], direction[:, 1])) % 360.0,
            axis_plunge=np.degrees(np.arcsin(_clamped_to_one(-direction[:, 2]))),
            sense=net * sign,
            neighbours=count,
        ))

    return out


def triad_kagan(lattice, s1, s2, eigenvalues, convergence, baselines, min_ln_e12):
    """
    The Kagan angle between the Bingham triads of a node and its surroundings.

    This is where an angle between orientations becomes a rotation with content.
    S1, S2 and S3 are orthonormal and sign-blind in each of the three, which is
    the 222 symmetry of a double couple, so `misah.kernels.kagan_angles` applies
    unchanged with S1 for P and S2 for T. The orthogonality of an exported triad
    holds to 1e-13 degrees against the one-degree tolerance of `PTBAxes`.

    S1 and S2 are given in true azimuth and are turned onto grid north here,
    because S3 is already there and a triad standing in two references is not a
    triad. Doing it at the edge of the module rather than at the call site is
    deliberate: it is the one mistake this computation can make that nothing
    downstream would show.

    Computed only where ln(e1/e2) clears `min_ln_e12` at BOTH nodes of a pair.
    Below that S1 and S2 can trade places, and a trade is a quarter turn about
    the axis that no symmetry of the triad undoes, so the angle would come out
    large for a reason that is arithmetic and not geological. The mask is
    returned so that a caller can say how much of the map was determined.

    Gives back a list of (mean angle, count) pairs, one per baseline, and the
    mask.
    """

    from misah.kernels import kagan_angles

    eigenvalues = np.asarray(eigenvalues, dtype=float)
    convergence = np.asarray(convergence, dtype=float)

    determined = np.log(eigenvalues[:, 0] / eigenvalues[:, 1]) >= min_ln_e12

    mechanisms = np.column_stack([
        (np.asarray(s1, dtype=float)[:, 0] - convergence) % 360.0,
        np.asarray(s1, dtype=float)[:, 1],
        (np.asarray(s2, dtype=float)[:, 0] - convergence) % 360.0,
        np.asarray(s2, dtype=float)[:, 1],
    ])

    n_nodes = len(lattice)
    out = []

    for h in baselines:
        offsets, _ = ring(lattice.step, h)

        total = np.zeros(n_nodes)
        count = np.zeros(n_nodes, dtype=int)

        for di, dj in offsets:
            neighbour = lattice.table[lattice.i + lattice.margin + di, lattice.j + lattice.margin + dj]
            present = (neighbour >= 0) & determined
            present[present] &= determined[neighbour[present]]

            if not present.any():
                continue

            here = np.flatnonzero(present)
            total[here] += kagan_angles(mechanisms[here], mechanisms[neighbour[here]])
            count[here] += 1

        has = count > 0
        out.append((np.where(has, total / np.where(has, count, 1), np.nan), count))

    return out, determined


def localisation_exponent(rotations, radius):
    """
    The slope of log(rotation) against log(distance), and its residual.

    A distributed rotation accumulates with distance: double the baseline,
    double the angle, slope one. A discontinuity saturates -- all of the
    rotation has already been crossed at the first step and the longer
    baselines find no more of it -- slope zero.

    The fit is on h >= 2R, where the windows share no attitudes. Below that two
    neighbouring nodes are largely the same measurements counted twice and the
    misorientation is suppressed by the kernel rather than by the field:
    including those points raises the slope for an instrumental reason.

    Gives back (slope, residual, baselines used), or (None, None, baselines)
    where fewer than three baselines are long enough -- from two points comes a
    slope with no residual, which is a number nothing is known about.

    Read it against synthetic anchors or not at all. The slope is not a property
    of the field: it is the rise of a signal ABOVE A NOISE FLOOR, and two
    identical fields with different floors give different exponents.
    """

    usable = [r for r in rotations if r.baseline >= 2.0 * radius]

    if len(usable) < 3:
        return None, None, [r.baseline for r in usable]

    stack = np.column_stack([r.rotation for r in usable])
    log_h = np.log(np.array([r.baseline for r in usable]) / 1000.0)

    slope = np.full(len(stack), np.nan)
    residual = np.full(len(stack), np.nan)

    complete = np.all(np.isfinite(stack) & (stack > 0), axis=1)

    if complete.any():
        log_r = np.log(stack[complete])
        centred = log_h - log_h.mean()
        coefficient = (log_r - log_r.mean(axis=1, keepdims=True)) @ centred / (centred @ centred)

        slope[complete] = coefficient
        expected = log_r.mean(axis=1, keepdims=True) + coefficient[:, None] * centred[None, :]
        residual[complete] = np.sqrt(np.mean((log_r - expected) ** 2, axis=1))

    return slope, residual, [r.baseline for r in usable]


def principal_axis(trend, plunge):
    """
    The mean axis of a set, as the leading eigenvector of their orientation tensor.

    An orientation tensor and not a circular mean: axes have a plunge, and a
    double-angle mean over the trend alone throws the third component away and
    then compares a three-dimensional rotation against what is left.

    Gives back (versor, trend, plunge), the versor in the lower hemisphere.
    """

    v = versors(trend, plunge)
    values, vectors = np.linalg.eigh(v.T @ v / len(v))
    axis = vectors[:, np.argmax(values)]

    if axis[2] > 0:
        axis = -axis

    return (
        axis,
        float(np.degrees(np.arctan2(axis[0], axis[1])) % 360.0),
        float(np.degrees(np.arcsin(min(1.0, abs(axis[2]))))),
    )
