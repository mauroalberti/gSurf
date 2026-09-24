"""
Which way does a normal point, on each side of the gstruct boundary?

Two numbers cross that boundary and the conversion invents nothing, so there
would be nothing here to check if the numbers were all a plane is. They are
not: a dip direction and a dip angle name a normal, and a normal has a sense.
FORMAT.md fixes it -- upward, horizontal component toward the dip -- and this
asks whether geogst says the same, through which method, and what happens if
the wrong one is used.

The last question is the reason for the file. An error of 180 degrees does not
show up where planes are compared, because that comparison takes the absolute
value and the sign cancels; it is invisible until something compares a plane
against a normal, and by then it looks like a result. So the flip is built here
on purpose, and both halves are measured: the one that cannot see it, and the
one that can.

The convention is written out a third time below, in `pole`, rather than
imported from either library. Comparing a library against itself would pass
whatever it does.

    python check_curation.py
"""

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("GSURF_REPO", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(REPO))

FAILURES = []

# Every quadrant, and the two places where the question degenerates: a
# horizontal plane, whose dip direction means nothing, and a vertical one,
# whose normal is horizontal and where up and down stop being different.
PLANES = [
    (90.0, 30.0),      # the case FORMAT.md works out by hand
    (0.0, 45.0),
    (180.0, 45.0),
    (270.0, 60.0),
    (45.0, 10.0),
    (135.0, 80.0),
    (200.0, 0.0),
    (200.0, 90.0),
]


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}{'   ' + detail if detail else ''}")
    if not condition:
        FAILURES.append(label)


def pole(dip_direction, dip):
    """(East, North, Up) of the upward normal, from the definition."""

    a, d = np.radians(dip_direction), np.radians(dip)

    return np.array([np.sin(a) * np.sin(d), np.cos(a) * np.sin(d), np.cos(d)])


def versor(direct):
    """(East, North, Up) of a geogst `Direct`. Its plunge is positive downward."""

    a, p = np.radians(direct.az.d), np.radians(direct.pl.d)

    return np.array([np.cos(p) * np.sin(a), np.cos(p) * np.cos(a), -np.sin(p)])


def separation(one, other):
    """Degrees between two poles, taken as axes -- which is what cannot see a flip."""

    return np.degrees(np.arccos(np.clip(abs(float(np.dot(one, other))), 0.0, 1.0)))


def main():
    try:
        import gstruct
    except ImportError:
        # Said plainly rather than left as a traceback, because this is the one
        # dependency gSurf cannot name in its pyproject: gstruct is on no index
        # -- the name there belongs to something else -- so it is installed from
        # its own repository or not at all.
        print("gstruct is not installed, and this check is about the boundary "
              "with it.\n  pip install -e <gstruct repo>")
        return 1

    from geogst.core.geology.orientations import Plane

    from gsurf.curation import geogst_plane, gstruct_plane

    print("-- the convention, on both sides --\n")

    canonical = pole(90.0, 30.0)

    check("a plane dipping 30 east has its normal leaning east and up",
          np.allclose(canonical, [0.5, 0.0, np.sqrt(3) / 2]),
          "(" + ", ".join(f"{c:+.4f}" for c in canonical) + ")")

    check("and gstruct says the same",
          np.allclose(gstruct.Plane(90.0, 30.0).normal(), canonical),
          "(" + ", ".join(f"{c:+.4f}" for c in gstruct.Plane(90.0, 30.0).normal()) + ")")

    check("and so does geogst, through norm_direct_up",
          np.allclose(versor(Plane(90.0, 30.0).norm_direct_up()), canonical),
          "(" + ", ".join(f"{c:+.4f}" for c in versor(Plane(90.0, 30.0).norm_direct_up())) + ")")

    agree = sum(np.allclose(gstruct.Plane(*p).normal(), pole(*p)) for p in PLANES)
    check("gstruct holds it in every quadrant, flat and vertical included",
          agree == len(PLANES), f"{agree} of {len(PLANES)}")

    agree = sum(np.allclose(versor(Plane(*p).norm_direct_up()), pole(*p)) for p in PLANES)
    check("and geogst holds it there too", agree == len(PLANES),
          f"{agree} of {len(PLANES)}")

    # The method gSurf reaches for by habit, and why it is not this one. Its use
    # in `attitudes.py` is right -- an orientation tensor is axial and a bed
    # overturned is the same pole -- but an axis handed to a format that states
    # a sense is a coin toss that lands wrong seven times in eight.
    axial = [
        np.allclose(versor(Plane(*p).normal_axis().as_direction()), pole(*p))
        for p in PLANES
    ]

    check("normal_axis points the other way, and is not the one to convert through",
          sum(axial) == 1 and axial[-1],
          f"agrees {sum(axial)} of {len(PLANES)}, and only where the plane is vertical")

    print("\n-- across and back --\n")

    same = []
    for dip_direction, dip in PLANES:
        there = gstruct_plane(Plane(dip_direction, dip))
        back = geogst_plane(there)
        same.append(
            np.allclose([there.dip_dir, there.dip], [dip_direction, dip])
            and np.allclose([back.dipazim, back.dipang], [dip_direction, dip])
        )

    check("a plane crosses and comes back as itself", all(same),
          f"{sum(same)} of {len(PLANES)}")

    # RHR strike is answered in the constructor, not kept: 0/45 built as a
    # strike is 90/45 built as a dip direction, and the same object afterwards.
    strike = gstruct_plane(Plane(0.0, 45.0, is_rhr_strike=True))

    check("a layer read as RHR strike crosses as a dip direction, once",
          np.allclose([strike.dip_dir, strike.dip], [90.0, 45.0]), str(strike))

    print("\n-- what a flipped normal would look like --\n")

    east = pole(90.0, 30.0)
    flipped = -east

    check("comparing two planes cannot see the flip",
          separation(east, flipped) < 1e-9,
          f"{separation(east, flipped):.1f} degrees apart, which is the whole trouble")

    check("comparing a plane against a normal can",
          abs(np.degrees(np.arccos(np.clip(float(np.dot(east, flipped)), -1.0, 1.0))) - 180.0) < 1e-9,
          "180 degrees, once the absolute value is not taken")

    # And the consequence, in the units the format actually uses: an apparent
    # dip in a section is signed, so a hemisphere error reverses the vergence.
    section = 90.0
    upright = gstruct.Plane(90.0, 30.0).apparent_dip(section)
    reversed_ = gstruct.Plane.from_normal(flipped)

    check("a flip would not survive from_normal, which re-points it upward",
          np.allclose([reversed_.dip_dir, reversed_.dip], [90.0, 30.0]), str(reversed_))

    check("so the apparent dip in a section keeps its sign",
          upright > 0 and abs(reversed_.apparent_dip(section) - upright) < 1e-9,
          f"{upright:+.1f} degrees toward B")

    print()

    if FAILURES:
        print(f"FAILED: {', '.join(FAILURES)}")
        return 1

    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
