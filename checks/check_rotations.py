"""
Does the rotation module measure a rotation that is known?

Three of the things checked here are not arithmetic but convention, and each
was a bug before it was a check: that bearings are taken on grid north, that
neighbours are gathered on a ring with their own distances, and that the sign
of a rotation about a horizontal axis is decided rather than left to the
rounding. A synthetic field has an imposed rotation in it, which is the only
way to check an answer instead of comparing it with itself.

    python check_rotations.py
"""

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("GSURF_REPO", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(REPO))

FAILURES = []


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}{'   ' + detail if detail else ''}")
    if not condition:
        FAILURES.append(label)


def main():
    from gsurf.rotations import (
        BaselineRotations,
        Lattice,
        localisation_exponent,
        principal_axis,
        ring,
        rotation_between_axes,
        rotation_field,
        triad_kagan,
        versors,
    )

    # -- one pair, where the answer can be written down ----------------------
    print("-- a rotation that is known --")

    # Two horizontal axes thirty degrees apart turn about the vertical.
    angle, trend, plunge = rotation_between_axes(
        versors([140.0], [0.0]), versors([170.0], [0.0])
    )
    check("thirty degrees in plan is thirty degrees about a vertical axis",
          abs(angle[0] - 30.0) < 1e-9 and abs(plunge[0] - 90.0) < 1e-9,
          f"{angle[0]:.3f} deg about {trend[0]:.1f}/{plunge[0]:.1f}")

    # The other way round is the same rotation with the other handedness.
    back, _, _ = rotation_between_axes(versors([170.0], [0.0]), versors([140.0], [0.0]))
    check("and the other way round is the same angle, other sign",
          abs(back[0] + angle[0]) < 1e-9,
          f"{angle[0]:+.3f} against {back[0]:+.3f}")

    # An axis onto itself: no angle, and no axis to turn about.
    same, same_t, _ = rotation_between_axes(versors([140.0], [20.0]), versors([140.0], [20.0]))
    check("an axis onto itself is zero, about nothing",
          same[0] == 0.0 and np.isnan(same_t[0]),
          f"angle {same[0]}, axis trend {same_t[0]}")

    # Sign-blindness: an axis and its opposite end are the same axis.
    flipped, _, _ = rotation_between_axes(versors([140.0], [20.0]), versors([320.0], [-20.0]))
    check("the far end of an axis is the same axis",
          abs(flipped[0]) < 1e-9, f"{flipped[0]:.2e} deg")

    # -- the sign on a horizontal axis, which the hemisphere cannot fix ------
    print("\n-- the trap: a rotation axis that is horizontal --")

    # Both of these are a plunging towards 140: the same rotation, applied
    # twice. Their rotation axis is horizontal by construction, so its third
    # component leaves the cross product as +-1e-16 and `z > 0` on that number
    # is a coin toss. Identical rotations must not come out with opposite signs.
    first, first_t, first_p = rotation_between_axes(versors([140.0], [0.0]), versors([140.0], [30.0]))
    second, second_t, second_p = rotation_between_axes(versors([140.0], [30.0]), versors([140.0], [60.0]))

    check("two identical plunge rotations agree in sign",
          np.sign(first[0]) == np.sign(second[0]),
          f"{first[0]:+.2f} and {second[0]:+.2f} deg")
    check("and in axis",
          abs(first_t[0] - second_t[0]) < 1e-6 and abs(first_p[0] - second_p[0]) < 1e-6,
          f"{first_t[0]:.2f}/{first_p[0]:.2f} and {second_t[0]:.2f}/{second_p[0]:.2f}")
    check("which is horizontal, and in [0, 180)",
          abs(first_p[0]) < 1e-6 and 0.0 <= first_t[0] < 180.0,
          f"{first_t[0]:.2f}/{first_p[0]:.2f}")

    # The same rotation asked for many times over, with the axis nudged through
    # the rounding: the sign must not depend on which side of zero it lands.
    plunges = np.linspace(0.0, 40.0, 41)
    many, _, _ = rotation_between_axes(versors(np.full(41, 50.0), plunges),
                                       versors(np.full(41, 50.0), plunges + 10.0))
    check("forty-one of them, all the same sign",
          len(np.unique(np.sign(many))) == 1,
          f"signs {sorted(set(np.sign(many).tolist()))}")

    # -- the ring, and the distances in it ----------------------------------
    print("\n-- the trap: neighbours on a ring, at their own distances --")

    offsets, distances = ring(500.0, 500.0)
    check("the ring at one step holds the eight around the cell",
          len(offsets) == 8, f"{len(offsets)} offsets")
    check("but not at one distance: four are the step, four are its diagonal",
          abs(sorted(distances)[0] - 500.0) < 1e-9
          and abs(sorted(distances)[-1] - 500.0 * np.sqrt(2.0)) < 1e-9,
          f"{sorted(distances)[0]:.1f} m and {sorted(distances)[-1]:.1f} m, "
          f"{(np.sqrt(2.0) - 1) * 100:.0f}% apart")

    offsets, distances = ring(500.0, 8000.0)
    check("a long baseline gathers many more than eight",
          len(offsets) > 50, f"{len(offsets)} offsets at 8 km")
    check("and every one of them is within half a step of the baseline",
          np.all(np.abs(distances - 8000.0) <= 250.0),
          f"worst {np.max(np.abs(distances - 8000.0)):.1f} m")

    # -- a ramp, whose rate is known ----------------------------------------
    print("\n-- a field whose rotation rate is imposed --")

    STEP = 500.0
    RATE = 2.0                  # degrees of trend per kilometre
    SPAN = 60                   # cells a side, so 29.5 km across

    gx, gy = np.meshgrid(np.arange(SPAN) * STEP, np.arange(SPAN) * STEP, indexing="ij")
    x, y = gx.ravel(), gy.ravel()
    trend = 100.0 + RATE * x / 1000.0

    lattice = Lattice.for_baselines(x, y, STEP, [4000.0, 8000.0])
    field = rotation_field(lattice, versors(trend, np.zeros_like(trend)), [4000.0])[0]

    # Away from the edges, where the ring is whole: a partial ring is a
    # one-sided sample of a symmetric quantity, which is the edge effect the
    # study measures separately.
    inside = (x > 6000.0) & (x < 23500.0) & (y > 6000.0) & (y < 23500.0)

    measured = np.nanmedian(field.gradient[inside])
    factor = measured / RATE

    # The ring averages over directions, and against a ramp that turns one way
    # only the along-ramp component goes as |cos|, whose mean over directions is
    # 2/pi. A property of the ring and not an error: the number is written down
    # so that a change in it is noticed rather than absorbed.
    check("the ring reads a unidirectional ramp low by 2/pi",
          abs(factor - 2.0 / np.pi) < 0.03,
          f"{measured:.3f} of {RATE:.1f} deg/km, factor {factor:.3f} against {2 / np.pi:.3f}")

    # And the other half of why there are two statistics. Around a linear ramp
    # the ring is symmetric: the neighbour ahead is reached by turning one way
    # and the one behind by turning the other, so the axis-angle vectors cancel
    # however large each of them is. Mean magnitude and magnitude of the mean
    # are not two spellings of one number.
    check("but the vector mean cancels on a ramp, while the magnitudes do not",
          np.nanmedian(field.net[inside]) < 1e-9 and np.nanmedian(field.rotation[inside]) > 1.0,
          f"net {np.nanmedian(field.net[inside]):.2e} deg, "
          f"rotation {np.nanmedian(field.rotation[inside]):.2f} deg")

    # -- grid north against true north --------------------------------------
    print("\n-- the trap: the bearings must be on grid north --")

    # A field whose axes are all the SAME on the grid, seen through a
    # convergence that varies west to east the way it does across a real map.
    # On grid north there is no rotation to find; on true north there is one,
    # and it is systematic, which is worse than noise.
    convergence = -0.5 + 1.0 * x / x.max()          # about a degree across
    grid_trend = np.full_like(x, 120.0)
    true_trend = (grid_trend + convergence) % 360.0

    on_grid = rotation_field(lattice, versors(grid_trend, np.zeros_like(x)), [8000.0])[0]
    on_true = rotation_field(lattice, versors(true_trend, np.zeros_like(x)), [8000.0])[0]

    inside = (x > 10000.0) & (x < 19500.0) & (y > 10000.0) & (y < 19500.0)

    check("on grid north a constant field rotates by nothing",
          np.nanmedian(on_grid.rotation[inside]) < 1e-9,
          f"{np.nanmedian(on_grid.rotation[inside]):.2e} deg")
    check("on true north the same field invents a rotation",
          np.nanmedian(on_true.rotation[inside]) > 0.1,
          f"{np.nanmedian(on_true.rotation[inside]):.3f} deg at 8 km, out of nothing")

    # -- no neighbour is not no rotation ------------------------------------
    print("\n-- a node with no neighbour --")

    lonely = Lattice.for_baselines(
        np.array([0.0, 500.0, 1000.0]), np.array([0.0, 0.0, 0.0]), STEP, [4000.0]
    )
    empty = rotation_field(lonely, versors([100.0, 110.0, 120.0], [0.0, 0.0, 0.0]), [4000.0])[0]

    check("comes out NaN and not zero",
          np.all(np.isnan(empty.rotation)) and np.all(empty.neighbours == 0),
          f"rotations {empty.rotation}")

    # -- a lattice that is not one ------------------------------------------
    print("\n-- nodes that are not on a grid --")

    try:
        Lattice(np.array([0.0, 500.0, 1030.0]), np.array([0.0, 0.0, 0.0]), 500.0, 2)
        refused = False
    except ValueError as problem:
        refused = "not a regular grid" in str(problem)

    check("are refused, rather than answered with empty neighbourhoods", refused)

    # -- the exponent, on curves whose slope is written down -----------------
    print("\n-- the localisation exponent --")

    baselines = [1000.0, 2000.0, 4000.0, 8000.0]

    def curve(values):
        return [
            BaselineRotations(baseline=h, rotation=np.array([v]), gradient=np.array([np.nan]),
                              net=np.array([np.nan]), coherence=np.array([np.nan]),
                              axis_trend=np.array([np.nan]), axis_plunge=np.array([np.nan]),
                              sense=np.array([np.nan]), neighbours=np.array([1]))
            for h, v in zip(baselines, values)
        ]

    ramp = curve([1.0, 2.0, 4.0, 8.0])          # doubles with the baseline
    slope, residual, used = localisation_exponent(ramp, radius=500.0)
    check("a rotation that doubles with the baseline has slope one",
          abs(slope[0] - 1.0) < 1e-9 and residual[0] < 1e-9,
          f"slope {slope[0]:.6f}, residual {residual[0]:.2e}, {len(used)} baselines")

    flat = curve([5.0, 5.0, 5.0, 5.0])          # all of it crossed at the first step
    slope, _, _ = localisation_exponent(flat, radius=500.0)
    check("one that saturates has slope zero",
          abs(slope[0]) < 1e-9, f"slope {slope[0]:.2e}")

    slope, _, used = localisation_exponent(ramp, radius=3000.0)
    check("and with fewer than three long enough baselines, no slope at all",
          slope is None, f"{len(used)} baselines at h >= 2R")

    # -- the triad, and the threshold under which it means nothing -----------
    print("\n-- the Kagan angle on the Bingham triad --")

    # A triad has to BE one. Two bearings picked to look perpendicular are not:
    # 30/10 and 120/20 are 3.4 degrees off, and `PTBAxes` refuses them, which is
    # the right answer to the wrong input. S2 is built in the plane normal to S1
    # so that the pair is orthonormal to machine precision, the way an exported
    # eigenframe is.
    def perpendicular_to(trend, plunge, twist):
        s1 = versors([trend], [plunge])[0]
        seed = np.array([0.0, 0.0, 1.0]) if abs(s1[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(s1, seed)
        u /= np.linalg.norm(u)
        v = np.cross(s1, u)
        s2 = np.cos(np.radians(twist)) * u + np.sin(np.radians(twist)) * v

        if s2[2] > 0:
            s2 = -s2

        return (
            float(np.degrees(np.arctan2(s2[0], s2[1])) % 360.0),
            float(np.degrees(np.arcsin(-s2[2]))),
        )

    s2_trend, s2_plunge = perpendicular_to(30.0, 10.0, 35.0)
    check("the synthetic triad is orthonormal, or it is not a triad",
          abs(float(versors([30.0], [10.0])[0] @ versors([s2_trend], [s2_plunge])[0])) < 1e-12,
          f"S1 30/10, S2 {s2_trend:.1f}/{s2_plunge:.1f}")

    n = len(x)
    s1 = np.column_stack([np.full(n, 30.0), np.full(n, 10.0)])
    s2 = np.column_stack([np.full(n, s2_trend), np.full(n, s2_plunge)])
    strong = np.tile(np.array([0.6, 0.3, 0.1]), (n, 1))

    angles, determined = triad_kagan(
        lattice, s1, s2, strong, np.zeros(n), [4000.0], min_ln_e12=0.35
    )
    check("identical triads are zero degrees apart",
          np.nanmax(angles[0][0]) < 1e-6 and determined.all(),
          f"worst {np.nanmax(angles[0][0]):.2e} deg, {determined.sum()} of {n} determined")

    weak = np.tile(np.array([0.45, 0.44, 0.11]), (n, 1))
    angles, determined = triad_kagan(
        lattice, s1, s2, weak, np.zeros(n), [4000.0], min_ln_e12=0.35
    )
    check("an undetermined S1/S2 gives no angle rather than a large one",
          not determined.any() and np.all(np.isnan(angles[0][0])),
          f"{determined.sum()} determined, ln(e1/e2) = {np.log(0.45 / 0.44):.3f}")

    # The convergence has to reach the triad too, or S3 is on the grid and its
    # own S1 and S2 are not.
    turned, _ = triad_kagan(lattice, s1, s2, strong, np.full(n, 1.0), [4000.0], min_ln_e12=0.35)
    check("a uniform convergence cancels between the two nodes of a pair",
          np.nanmax(turned[0][0]) < 1e-6,
          f"worst {np.nanmax(turned[0][0]):.2e} deg")

    # -- the mean axis ------------------------------------------------------
    print("\n-- the mean axis of a set --")

    # The far end of 120/5 is 300/-5 and not 300/5, which is a different line
    # altogether -- ten degrees of plunge away, and it drags the mean down with
    # it. The orientation tensor is sign-blind, so given the real far end it
    # should not notice which end it was handed.
    _, mean_trend, mean_plunge = principal_axis(
        np.array([118.0, 120.0, 122.0, 300.0]), np.array([4.0, 5.0, 6.0, -5.0])
    )
    check("is not thrown by an axis given from its far end",
          abs(mean_trend - 120.0) < 1.0 and abs(mean_plunge - 5.0) < 1.0,
          f"{mean_trend:.1f}/{mean_plunge:.1f}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed: {', '.join(FAILURES)}")
        return 1

    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
