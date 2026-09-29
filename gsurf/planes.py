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
