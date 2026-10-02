"""
The attitude of an exhumed surface: where the contact crops out, the DTM samples
the fault and not the hill.

**Why a region and not the trace.** A plane fitted along a trace is a
one-dimensional sample of a two-dimensional thing, and badly conditioned by
construction -- `traces.TraceGate` exists to refuse the cases where it is
hopeless, and on this AOI it refuses most of them. Where the surface crops out
as a dip slope there is no such problem: the facet is hectares of it. The method
is to grow a region from a measured site, taking cells whose own slope sits
within a tolerance of the measured plane, and fit a plane to those. On the AOI
the four largest come to 27-44 ha, 1-1.4 km across, with residuals of 7.5-11.5 m
-- half a degree of waviness over a kilometre.

**The precondition is a datum, not a number.** The surface has to *crop out*,
which is `exposure` in the format, which the source files leave entirely
`unknown` and only a geologist can fill. Without it a small `drape` is ambiguous
between "the fault is the hillside" and "the trace was drawn along a break of
slope the fault is not" -- see `tools/editor.ExposureHere`, which is where that
datum gets written, and `hillside`, which is where the ambiguity is measured.

**Ported from `gstruct/facets.py`, with three changes, all stated.**

*No scipy.* The reference calls `ndimage.uniform_filter` and `ndimage.label`.
The second of those labels the whole window to use one component, and the
component wanted is always the one holding the site -- so this grows that one
instead, which is less work and makes the "the site fell just outside" case
explicit rather than a `bincount` over a neighbourhood. The smoothing is a box
mean by summed-area table. Both are a dozen lines, against a base dependency
`pyproject.toml`'s own policy would not take for one window of one tool.

*North.* The reference compares the measured plane's normal, which the file
writes in **true** azimuth, against cell normals computed from projected
coordinates, which are in **grid**. The convergence is 0.6-1.0 deg here, far
under a 15 deg tolerance, so the mask barely moves -- but the fitted plane came
out grid and was written as though it were true. Here the measurement is turned
to grid to build the mask and the answer is turned back, which is what every
other computed plane in this program does.

*The stretch, which turns out not to be this module's to give.*
`export_geology.py` wrote the fit `* *`, over the whole trace, and the obvious
repair was to anchor it over the ground the facet covers. Measured, neither way
of doing that stands up: the facets of this AOI sit a median of 47 to 296 m from
their own trace and out to 714 m, because an exposed dip slope runs away *down
the dip* and the trace is its up-dip edge. Strict containment then gives twenty
metres of a twenty-seven hectare surface and nothing at all on two of the eight;
the nearest-point shadow gives the whole trace. So the stretch of a facet's
`fit` is the stretch somebody declared `exposure=exposed` -- the claim capped by
its own licence, decided by a person, and `offsets_on` is the diagnostic that
goes beside it rather than a guess that replaces it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .hillside import attitude_of, normal_of

# What a fit grown this way says made it. FORMAT.md's own name for the producer,
# and `export_geology.py` wrote the same word.
FROM_FACET = "exposed-facet"

# And what licenses it to run at all: one value of one axis, which is the whole
# argument of this module's docstring.
LICENCE = "exposure=exposed"

# How far from the site to look. Half a kilometre, which is the reference's own
# number: it bounds the window read, and a facet that reaches the rim is a facet
# this has not finished measuring -- which `Facet.at_the_rim` says out loud
# rather than leaving to be noticed.
RADIUS = 500.0

# How far a cell's own slope may sit from the measured plane and still count as
# the same surface. The assertion the method is made of, so it is the number the
# window lets a curator move.
TOL = 15.0

# The box mean the DEM is smoothed with before being differentiated, in cells. A
# gradient on raw 5 m lidar is mostly noise; five cells is 25 m, which is below
# anything these facets are and above the noise.
SMOOTH_CELLS = 5

# Under this a region is not a facet. One hectare is 400 cells at 5 m, which is
# 100 m by 100 m of hillside -- below that the "surface measured over hectares"
# argument is not being made any more.
MIN_AREA_HA = 1.0

# Past this share of the window with no data, there is nothing to grow through.
MAX_HOLES = 0.3


@dataclass(frozen=True)
class Facet:
    """
    One grown region and the plane fitted to it, with what grew it.

    `seed_plane` is kept because every number here depends on it: the region is
    the cells within `tol` of *that* plane, so the angle between the result and
    the seed is not a check of anything -- it is bounded by `tol` by
    construction. `export_geology.py` wrote that angle as `vs_field` and read it
    as corroboration, which is the one mistake this dataclass can prevent by
    making the dependency impossible to forget.
    """

    dip_dir: float                  # true azimuth where the convergence is known
    dip: float
    n: int                          # cells fitted
    area_ha: float
    rms: float                      # metres, the residual of the fit
    relief: float                   # metres, the height the region spans
    span: float                      # metres, the region's own diagonal in plan
    centre: tuple
    north: str
    converg: float
    cell: float
    radius: float
    tol: float
    smooth_m: float
    seed: tuple                     # where it was grown from
    seed_plane: tuple               # and the plane that grew it, as given
    mask: object                    # the region, as a boolean array
    geotransform: tuple             # which places that array on the ground
    at_the_rim: bool                # whether it reaches the window's edge

    @property
    def plane(self):
        return self.dip_dir, self.dip

    @property
    def waviness(self):
        """
        The residual as an angle over the region's own width, in degrees.

        **And not against `hillside.sampling_rms`**, which is the floor that
        belongs to a corridor and not to a facet. The difference is measured: on
        a synthetic DEM that is exactly a plane, a corridor read along a trace
        comes back with a 0.72 m residual against a 0.83 m floor, because its
        samples are trace positions snapped to the nearest cell and a snap on a
        slope is an error in height. A facet fits **the cells themselves**, whose
        centres are where the raster says they are, and on the same DEM its
        residual is 0.00 m. So a facet's residual has no sampling in it: all of
        it is the surface not being flat.

        Which is what makes this the number to read it as. 10.2 m over the
        1400 m of F0055 is four tenths of a degree -- the reference
        implementation's own claim that these surfaces are planar to half a
        degree over a kilometre, said as the quantity it was a claim about.
        """

        if self.span <= 0.0:
            return 0.0

        return math.degrees(math.atan(self.rms / self.span))


def _boxed(z, size):
    """
    A square box mean of `size` cells, edges replicated, by summed-area table.

    `scipy.ndimage.uniform_filter` without scipy. Its default edge handling is
    `reflect` and this is `nearest`, which differ over a two-cell rim of a
    two-hundred-cell window and nowhere else -- and a facet that reaches the rim
    is reported as such anyway, because a region still growing where the data
    stops has not been measured.
    """

    size = int(size)

    if size <= 1:
        return z

    pad = size // 2
    padded = np.pad(z, pad, mode="edge")
    table = np.pad(
        np.cumsum(np.cumsum(padded, axis=0), axis=1), ((1, 0), (1, 0))
    )
    rows, cols = z.shape

    total = (
        table[size:size + rows, size:size + cols]
        - table[0:rows, size:size + cols]
        - table[size:size + rows, 0:cols]
        + table[0:rows, 0:cols]
    )

    return total / float(size * size)


def _seed_cell(mask, row, col, reach=5):
    """
    The cell to grow from: the one the site is in, or the nearest that qualifies.

    The reference takes the commonest label in an 11x11 block when the site's own
    cell is not in the mask. This takes the **nearest** qualifying cell instead,
    which is the thing being approximated -- a site is a point somebody stood on
    and the cell it lands in can fail the tolerance for a metre of noise, while
    the surface starts one cell over. Commonest and nearest differ where two
    facets meet, and there the near one is the one under the boot.
    """

    rows, cols = mask.shape

    if not (0 <= row < rows and 0 <= col < cols):
        return None

    if mask[row, col]:
        return row, col

    top, left = max(0, row - reach), max(0, col - reach)
    block = mask[top:row + reach + 1, left:col + reach + 1]

    if not block.any():
        return None

    near = np.argwhere(block)
    away = np.hypot(near[:, 0] + top - row, near[:, 1] + left - col)
    at = near[int(np.argmin(away))]

    return int(at[0] + top), int(at[1] + left)


def _grown(mask, row, col):
    """
    The connected region of `mask` holding one cell, eight-connected.

    Dilation against the mask until it stops changing, which is a flood fill
    written as array operations: a window here is 200 cells across, so the
    longest a region can take to grow is a few hundred passes over 40 000
    booleans, and the alternative -- a queue in Python over the same cells --
    pays an interpreter step per neighbour.
    """

    here = np.zeros(mask.shape, dtype=bool)
    here[row, col] = True

    # A region cannot take more passes to grow than there are cells along a
    # staircase through the window. The cap is not expected to be reached: the
    # loop leaves the moment a pass adds nothing.
    for _ in range(sum(mask.shape)):
        grown = here.copy()

        grown[1:, :] |= here[:-1, :]
        grown[:-1, :] |= here[1:, :]
        grown[:, 1:] |= here[:, :-1]
        grown[:, :-1] |= here[:, 1:]
        grown[1:, 1:] |= here[:-1, :-1]
        grown[1:, :-1] |= here[:-1, 1:]
        grown[:-1, 1:] |= here[1:, :-1]
        grown[:-1, :-1] |= here[1:, 1:]

        grown &= mask

        if int(grown.sum()) == int(here.sum()):
            return grown

        here = grown

    return here


def facet_on(anchor, plane, dem, convergence=None, radius=RADIUS, tol=TOL,
             smooth=SMOOTH_CELLS, min_area=MIN_AREA_HA):
    """
    The facet grown from one measured site, and the plane fitted to it, or None.

    `anchor` is where the measurement was made and `plane` is `(dip direction,
    dip)` as the file holds it -- true azimuth, which is turned to grid here to
    be compared with cell normals and turned back on the way out.

    None, and not an empty facet, for every way this can decline: no window, a
    window mostly without data, a site off the raster, a site whose cell and
    whose neighbours all fail the tolerance, a region under `min_area`. Each of
    those is a different fact about the ground and the window that asks is the
    place to tell them apart; what they have in common is that no plane was
    measured, which is what None says.
    """

    if dem is None:
        return None

    x0, y0 = float(anchor[0]), float(anchor[1])
    window = dem.window_over(
        (x0 - radius, y0 - radius, x0 + radius, y0 + radius)
    )

    if window is None or window.data.size < 100:
        return None

    z = np.asarray(window.data, dtype=float)
    origin_x, pixel_w, _, origin_y, _, pixel_h = (
        float(v) for v in window.geotransform
    )

    if dem.nodata is not None:
        z = np.where(z == dem.nodata, np.nan, z)

    holes = np.isnan(z)

    if holes.mean() > MAX_HOLES:
        return None

    # Smoothed over the holes rather than through them: the mean of what is
    # there is the least committal fill, and the cells it fills are dropped from
    # the mask straight afterwards, so it only keeps a hole from poisoning the
    # gradient of its neighbours.
    filled = np.where(holes, np.nanmean(z), z)
    smoothed = _boxed(filled, smooth)

    cell = max(abs(pixel_w), abs(pixel_h))

    # `np.gradient` gives d/drow first, and a larger row is further **south**:
    # so dz/dy is minus that one. The upward normal is `(-dz/dx, -dz/dy, 1)`.
    down_row, down_col = np.gradient(smoothed, abs(pixel_h), abs(pixel_w))
    normals = np.dstack([-down_col, down_row, np.ones_like(smoothed)])
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)

    grid_dir = float(plane[0])
    converged = 0.0
    north = "grid"

    if convergence is not None and getattr(convergence, "available", False):
        converged = convergence.at(x0, y0)
        grid_dir = convergence.to_grid(grid_dir, x0, y0)
        north = "true"

    wanted = normal_of(grid_dir, float(plane[1]))
    away = np.degrees(np.arccos(np.clip(np.abs(normals @ wanted), 0.0, 1.0)))
    mask = (away < float(tol)) & ~holes

    col = int(math.floor((x0 - origin_x) / pixel_w))
    row = int(math.floor((y0 - origin_y) / pixel_h))
    seeded = _seed_cell(mask, row, col)

    if seeded is None:
        return None

    region = _grown(mask, *seeded)
    rows, cols = np.nonzero(region)

    if not len(rows):
        return None

    area_ha = len(rows) * cell ** 2 / 1.0e4

    if area_ha < float(min_area):
        return None

    # Cell centres, as `hillside` and `traces` both read them.
    cloud = np.column_stack((
        origin_x + (cols + 0.5) * pixel_w,
        origin_y + (rows + 0.5) * pixel_h,
        z[rows, cols],
    ))
    cloud = cloud[np.isfinite(cloud[:, 2])]

    if len(cloud) < 3:
        return None

    centre = cloud.mean(axis=0)
    singular, axes = np.linalg.svd(cloud - centre, full_matrices=False)[1:]
    attitude = attitude_of(axes[2])

    if attitude is None:
        return None

    dip_dir, dip = attitude

    if north == "true":
        dip_dir = convergence.to_true(dip_dir, float(centre[0]), float(centre[1]))

    height, width = region.shape
    at_the_rim = bool(
        region[0, :].any() or region[-1, :].any()
        or region[:, 0].any() or region[:, -1].any()
    ) if height > 1 and width > 1 else True

    return Facet(
        dip_dir=float(dip_dir),
        dip=float(dip),
        n=len(cloud),
        area_ha=float(area_ha),
        rms=float(singular[2] / math.sqrt(len(cloud))),
        relief=float(cloud[:, 2].max() - cloud[:, 2].min()),
        span=float(math.hypot(np.ptp(cloud[:, 0]), np.ptp(cloud[:, 1]))),
        centre=(float(centre[0]), float(centre[1])),
        north=north,
        converg=float(converged),
        cell=float(cell),
        radius=float(radius),
        tol=float(tol),
        smooth_m=float(smooth) * float(cell),
        seed=(x0, y0),
        seed_plane=(float(plane[0]), float(plane[1])),
        mask=region,
        geotransform=(origin_x, pixel_w, 0.0, origin_y, 0.0, pixel_h),
        at_the_rim=at_the_rim,
    )


def cells_of(facet, cap=600):
    """The facet's cell centres as ground coordinates, thinned to `cap` of them."""

    rows, cols = np.nonzero(facet.mask)
    origin_x, pixel_w, _, origin_y, _, pixel_h = facet.geotransform

    stride = max(1, len(rows) // int(cap))

    return np.column_stack((
        origin_x + (cols[::stride] + 0.5) * pixel_w,
        origin_y + (rows[::stride] + 0.5) * pixel_h,
    ))


def offsets_on(facet, path, cap=600):
    """
    How far the facet's cells lie from the trace, in metres, one per cell.

    **The number that decides whether a facet is the trace's surface at all**,
    and the reason this module does not hand back a stretch of trace. Measured
    over the eight facets of the AOI, the cells sit a median of 47 to 296 m from
    the trace and out to 714 m: a facet grown from a station beside a fault is
    not a band along the fault, it is the hillside running away down the dip
    from it, which is what an exposed dip slope *is*. The trace is its up-dip
    edge and touches it over twenty metres of its own length.

    So neither mapping from region to interval survives contact with the data.
    Strict containment gives twenty metres of a twenty-seven hectare surface --
    and nothing at all on two of the eight, where the region never crosses the
    line. The nearest-point shadow gives the whole trace: project something
    lying 250 m away onto a 981 m trace and it covers 106 to 981 m of it.

    What the stretch of a facet's `fit` should be is therefore not a geometric
    question. It is the stretch a geologist declared `exposure=exposed`, which
    is a person saying where this contact crops out -- so the claim is capped by
    its own licence, and `export_geology.py`'s `* *` over 3531 m of F0055 is
    refused by the curation and not by arithmetic. This function is the
    diagnostic that goes beside it: a facet whose cells are all 600 m away is
    worth looking at twice, and `off=` in the file is how the next reader sees
    that without recomputing anything.

    Point to polyline, in numpy, rather than through `gstruct.project` per cell:
    the quantity is the same and this module stays clear of the format.
    """

    points = np.asarray(
        [(float(p[0]), float(p[1])) for p in path], dtype=float
    )

    if len(points) < 2:
        return np.zeros(0)

    cells = cells_of(facet, cap=cap)

    if not len(cells):
        return np.zeros(0)

    starts = points[:-1]
    spans = np.diff(points, axis=0)
    lengths = np.einsum("ij,ij->i", spans, spans)
    lengths[lengths == 0.0] = 1.0

    # (cells, segments): where along each segment the foot of the perpendicular
    # falls, clamped into the segment so an obtuse corner answers with the
    # vertex rather than with a point off the end of the line.
    along = np.clip(
        np.einsum(
            "ijk,jk->ij", cells[:, None, :] - starts[None, :, :], spans
        ) / lengths,
        0.0,
        1.0,
    )

    feet = starts[None, :, :] + along[:, :, None] * spans[None, :, :]

    return np.min(np.linalg.norm(cells[:, None, :] - feet, axis=2), axis=1)
