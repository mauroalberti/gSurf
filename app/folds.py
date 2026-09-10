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
from dataclasses import dataclass


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

        if result.n < self.min_points:
            return f"{result.n} attitudes, fewer than {self.min_points}"

        if math.isnan(result.k):
            return "no shape: the three axes are equal"

        if result.k > self.max_k:
            return f"K = {result.k:.2f}: a cluster, not a girdle"

        if result.c < self.min_c:
            return f"C = {result.c:.2f}: too weak to have a direction"

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
