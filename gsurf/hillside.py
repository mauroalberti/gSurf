"""
The plane of the hillside a stretch of trace lies on: FORMAT.md's `drape`, read
one stretch at a time.

**Why this is not a diagnostic of a fit.** `export_geology.py` computes `drape`
per fit, as an attribute of the fit, and uses it to classify: 17 of the 27 fits
in `merid_faults` came back `concorde-col-versante`, which is the number that
says what a trace fit mostly measures. gSurf has never had it, and the gap
showed the moment the editor had to offer a judgement about *exposure*. There
the question is not "is this fit independent of the hillside" but the one FORMAT
says only a geologist can answer: **is the ground here the fault surface?** --
and that question is asked about a stretch somebody picked, whether or not any
fit was ever computed over it.

So the measurement moves off the fit and onto the stretch. Same corridor, same
arithmetic, same definition of the angle; a different thing carries it.

**The corridor and not the trace.** The hillside is sampled at 30 and 60 m
either side of the trace and never on it, which is `export_geology`'s choice
kept deliberately: the trace's own points are what a trace fit was computed
from, so a hillside that included them would be partly the same sample, and an
angle between two overlapping samples understates how different they are. Off
the trace, the two are independent.

**What it refuses to say.** Nothing here returns a verdict. A small angle
between the hillside and a measured plane is consistent with a contact exhumed
as a dip slope -- and equally consistent with a trace somebody digitised along a
break of slope that the fault is not. FORMAT.md states that the distinction is
decided by the `exposure` axis, which is a datum the source does not hold; a
threshold here would be this module deciding it instead, off the one number that
cannot. `relief` is in the answer for the same reason: a plane fitted through a
corridor with four metres of relief in it is arithmetically fine and
geologically empty, and the residual alone does not say so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .planes import walked
from .traces import elevations

# How far across the trace the hillside is read, in metres. `export_geology.py`'s
# own four offsets, kept to the metre: the number this produces is meant to be
# read beside the `drape=` already written in `merid_faults.gstruct`, and a
# corridor of a different width would be a different measurement wearing the
# same name.
CORRIDOR = (-60.0, -30.0, 30.0, 60.0)

# How often along the stretch, in metres. Finer than the corridor is wide, so a
# stretch shorter than the corridor still gets samples down its length -- and
# coarse enough that a 3 km stretch is 1200 points rather than 12000, which the
# SVD does not need and a window redrawn on every pick would pay for.
ALONG = 10.0

# Below this the fit is not reported at all. `export_geology` asks for more than
# six corridor points, which is three above the three a plane needs; the same
# floor, said as the count rather than as the comparison.
MIN_SAMPLES = 8


@dataclass(frozen=True)
class Hillside:
    """
    One plane fitted to the ground beside a stretch, with what it was fitted to.

    `relief` and `rms` travel together and are read together: the first is how
    much height the corridor actually holds and the second is how far the plane
    misses it. Either alone is unreadable -- a 2 m residual is a plane over 200 m
    of relief and is noise over 3 m of it -- which is why neither is a threshold
    and both are shown.
    """

    dip_dir: float                  # true azimuth where the convergence is known
    dip: float
    n: int                          # corridor samples the plane was fitted to
    rms: float                      # metres, the residual of that fit
    relief: float                   # metres, the height the corridor spans
    centre: tuple                   # where the plane was fitted, for the record
    north: str                      # "true" or "grid", as FORMAT.md spells it
    converg: float                  # degrees, 0.0 where north is "grid"
    cell: float                     # the DEM's, which is what `rms` is read against
    along: float                    # the step the stretch was walked at
    across: tuple                   # the offsets the corridor was read at

    @property
    def plane(self):
        """The pair every other plane in this program travels as."""

        return self.dip_dir, self.dip

    @property
    def sampling_rms(self):
        """
        The residual the nearest-cell read alone puts in, on a slope this steep.

        `rms` has a floor and it is not small: a cell is read at its nearest
        centre, so a sample is up to half a cell off in plan, which on a slope is
        half a cell times the gradient in height. Uniform over that interval the
        root mean square is `cell * tan(dip) / (2*sqrt(3))` -- 0.8 m on a 30
        degree slope at 5 m, where the measured residual over an exactly planar
        DEM comes back 0.72. So this is the number `rms` is small *against*, and
        without it a planar hillside reads as three quarters of a metre of
        waviness that is not in the ground.
        """

        return self.cell * math.tan(math.radians(self.dip)) / (2.0 * math.sqrt(3.0))


def between(one, other):
    """
    The angle between two planes, in degrees, as the acute angle of their poles.

    Acute, which is the one thing to say about it: a pole is an axis and not a
    direction, so two planes four degrees apart have poles either four or a
    hundred and seventy-six degrees apart depending on which way each was
    written. `abs` on the dot product is what makes the answer about the planes.
    """

    def pole(plane):
        dip_dir, dip = float(plane[0]), float(plane[1])
        azimuth, tilt = math.radians(dip_dir), math.radians(dip)

        # The downward pole, in east-north-up. Which of the two it is does not
        # matter to the answer above; it matters that both are built the same
        # way, because a pair built by two rules differs by the error this
        # function exists to measure.
        return np.array([
            -math.sin(azimuth) * math.sin(tilt),
            -math.cos(azimuth) * math.sin(tilt),
            -math.cos(tilt),
        ])

    return math.degrees(
        math.acos(min(1.0, abs(float(pole(one) @ pole(other)))))
    )


def attitude_of(normal):
    """
    A normal in east-north-up as `(dip direction, dip)`, in grid azimuth.

    The upward normal's horizontal part points **down** dip -- a slope falling
    east has an upward normal leaning east -- so the dip direction is the azimuth
    of that part and needs no quarter-turn. Written out rather than taken from
    `geogst`, which is an optional extra here and cannot be a base module's
    dependency.
    """

    n = np.asarray(normal, dtype=float)
    length = float(np.linalg.norm(n))

    if not length:
        return None

    n = n / length

    if n[2] < 0.0:
        n = -n

    dip = math.degrees(math.acos(min(1.0, abs(n[2]))))
    dip_dir = math.degrees(math.atan2(n[0], n[1])) % 360.0

    return dip_dir, dip


def normal_of(dip_dir, dip):
    """
    `(dip direction, dip)` as the upward normal in east-north-up, a unit vector.

    `attitude_of` read backwards, and written here beside it so that the pair
    cannot drift: a round trip through the two is the identity, which is what
    `facets` leans on when it asks which cells of a DEM belong to a measured
    plane.
    """

    azimuth, tilt = math.radians(float(dip_dir)), math.radians(float(dip))

    return np.array([
        math.sin(azimuth) * math.sin(tilt),
        math.cos(azimuth) * math.sin(tilt),
        math.cos(tilt),
    ])


def corridor_of(points, step=ALONG, across=CORRIDOR):
    """
    Where the hillside is sampled beside a stretch: `(N, 2)` ground coordinates.

    The perpendicular is taken from the resampled stretch and not from the
    path's own vertices, so a densified trace and a sparsely digitised one put
    their samples in the same places. At each sample the tangent is a central
    difference, with the ends one-sided -- a corridor point is 30 m away, and the
    difference between the two estimators there is far below that.
    """

    _, xy = walked(points, step=float(step))

    if len(xy) < 2:
        return np.zeros((0, 2))

    tangent = np.gradient(xy, axis=0)
    norm = np.hypot(tangent[:, 0], tangent[:, 1])
    keep = norm > 0.0

    if not keep.any():
        return np.zeros((0, 2))

    xy, tangent, norm = xy[keep], tangent[keep], norm[keep]
    tangent = tangent / norm[:, None]

    # Left of the direction of travel, which is a choice with no consequence:
    # both signs of the offset are read, so swapping them relabels the same set.
    across_xy = np.column_stack((-tangent[:, 1], tangent[:, 0]))

    return np.vstack([
        xy + across_xy * float(offset) for offset in across
    ])


def hillside_on(points, dem, convergence=None, step=ALONG, across=CORRIDOR):
    """
    The plane of the ground beside a stretch of trace, or None.

    `points` is the stretch as a polyline -- `curation.stretch(path, s0, s1)` --
    and not a path with two progressives, because this module reads a DEM and
    does geometry and has no business knowing what a structure is.

    None where there is nothing to answer with: no DEM under the corridor, a
    corridor that fell on nodata, fewer samples than a plane can be fitted to.
    An absence and not a zero, for the reason the editor's columns give: a plane
    nobody could compute and a plane that came back flat are different answers.
    """

    if dem is None:
        return None

    corridor = corridor_of(points, step=step, across=across)

    if len(corridor) < MIN_SAMPLES:
        return None

    cell = max(float(dem.res_x), float(dem.res_y))

    box = (
        float(corridor[:, 0].min()),
        float(corridor[:, 1].min()),
        float(corridor[:, 0].max()),
        float(corridor[:, 1].max()),
    )

    # Enough margin for the nearest cell of a corridor point that sits exactly on
    # the box's edge, and no more: the window is read, so its size is paid for.
    window = dem.window_over(box, margin=2.0 * cell)

    if window is None:
        return None

    z = elevations(window, corridor[:, 0], corridor[:, 1])

    if dem.nodata is not None:
        z = np.where(z == dem.nodata, np.nan, z)

    good = np.isfinite(z)

    if int(good.sum()) < MIN_SAMPLES:
        return None

    cloud = np.column_stack((corridor[good, 0], corridor[good, 1], z[good]))
    centre = cloud.mean(axis=0)
    singular, rows = np.linalg.svd(cloud - centre, full_matrices=False)[1:]

    attitude = attitude_of(rows[2])

    if attitude is None:
        return None

    dip_dir, dip = attitude
    north, converged = "grid", 0.0

    if convergence is not None and getattr(convergence, "available", False):
        converged = convergence.at(float(centre[0]), float(centre[1]))
        dip_dir = convergence.to_true(dip_dir, float(centre[0]), float(centre[1]))
        north = "true"

    return Hillside(
        dip_dir=float(dip_dir),
        dip=float(dip),
        n=int(good.sum()),
        # The residual of the plane, in metres: the third singular value is the
        # root of the summed squared distances, so dividing by the root of the
        # count is the per-sample figure. `export_geology` writes `plan` the same
        # way, over two dimensions instead of three.
        rms=float(singular[2] / math.sqrt(len(cloud))),
        relief=float(np.nanmax(z[good]) - np.nanmin(z[good])),
        centre=(float(centre[0]), float(centre[1])),
        north=north,
        converg=float(converged),
        cell=cell,
        along=float(step),
        across=tuple(float(offset) for offset in across),
    )
