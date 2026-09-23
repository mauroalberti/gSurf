"""
Does the trace module recover a plane that is known, and refuse one that is not?

The arithmetic is misah's, so what is checked here is the use of it: that the
attitude coming back off a synthetic outcrop trace is the plane the trace was
built on, that a trace too straight to carry one is refused rather than
answered, that the runs along a trace add up to the trace, and that the window
sweep finds a length of orientation domain that was put there on purpose.

A trace built on a plane has an attitude by construction, which is the only way
to check the answer instead of comparing it with itself. The V is built in plan
and lifted onto the plane, rather than intersected out of a DEM: the point is to
test this module, and a contouring step between the answer and the truth would
put its own errors in the middle.

    python check_traces.py
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("GSURF_REPO", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(REPO))

FAILURES = []


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}{'   ' + detail if detail else ''}")
    if not condition:
        FAILURES.append(label)


def pole(dip_direction, dip):
    """
    (East, North, Up) of the upward normal to a plane.

    Upward, so that a plane is one vector and not a vector and a sign. The
    downward normal of a plane dipping north points *south* and down, which is
    the trap: taking it as trend = dip direction builds a synthetic plane facing
    the opposite way from the one it is named after, and the check then measures
    a hundred and eighty degrees of nothing.
    """

    a, d = np.radians(dip_direction), np.radians(dip)

    return np.array([np.sin(a) * np.sin(d), np.cos(a) * np.sin(d), np.cos(d)])


def on_plane(xy, dip_direction, dip, z0=0.0):
    """Lift a plan-view polyline onto a plane of known attitude."""

    n = pole(dip_direction, dip)
    z = z0 - (n[0] * xy[:, 0] + n[1] * xy[:, 1]) / n[2]

    return np.column_stack([xy[:, 0], xy[:, 1], z])


def vee(half, amplitude, step=10.0):
    """A V in plan: the outcrop shape a contact makes crossing a valley."""

    t = np.arange(-half, half + step / 2, step)

    return np.column_stack([t, amplitude * np.abs(t) / half])


def sinuous(length, wavelength, amplitude, step=10.0):
    """
    A contact wandering across a dissected slope: a V is only its apex.

    Used wherever a stretch of trace has to turn *everywhere*. A single V has
    two straight limbs, so a window sliding along one reads a straight line and
    is refused -- correctly, and inconveniently for a test that wanted the whole
    half of a trace to be readable.
    """

    t = np.arange(0.0, length + step / 2, step)

    return np.column_stack([t, amplitude * np.sin(2.0 * np.pi * t / wavelength)])


def straight(half, step=10.0):
    """A trace that does not turn, whatever relief it crosses."""

    t = np.arange(-half, half + step / 2, step)

    return np.column_stack([t, np.zeros_like(t)])


def separation(a, b):
    """Degrees between two planes, sign-blind: a pole's sign is arbitrary."""

    dot = abs(float(np.dot(pole(*a), pole(*b))))

    return float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))


def progressives(coords):
    """Arc length along a polyline, as the module computes it."""

    steps = np.hypot(*np.diff(coords[:, :2], axis=0).T)

    return np.concatenate(([0.0], np.cumsum(steps)))


def part_of(coords):
    """One resampled fragment in the shape `trace_spans` expects."""

    return coords, progressives(coords)


class Line:
    """The least a mapped line has to be: something with `.coords`.

    `trace_points` and `digitising_jitter` both duck-type on that, so the check
    can exercise them without building geogst objects -- and, more to the point,
    can prove they do not secretly need anything else.
    """

    def __init__(self, coords):
        self.coords = coords


def main():
    from gsurf import traces

    print("\n-- a plane that is known ------------------------------------\n")

    for dip_direction, dip in ((45.0, 30.0), (135.0, 65.0), (300.0, 12.0)):
        points = on_plane(vee(400.0, 250.0), dip_direction, dip)
        window = traces.window_attitude(points)

        off = separation((dip_direction, dip), (window.dip_direction, window.dip))

        check(
            f"a V on {dip_direction:.0f}/{dip:.0f} gives back {dip_direction:.0f}/{dip:.0f}",
            off < 0.5,
            f"{window.dip_direction:.1f}/{window.dip:.1f}, {off:.2f} deg off",
        )

    print("\n-- a trace that is not a plane ------------------------------\n")

    gate = traces.TraceGate()
    turning = traces.window_attitude(on_plane(vee(400.0, 250.0), 135.0, 65.0))
    flat = traces.window_attitude(on_plane(straight(400.0), 135.0, 65.0))

    check(
        "a V is admitted, and held",
        gate.verdict_for(turning.n, *turning.singular_values) == "held",
        f"{turning.spread:.0f} m off straight, wobble {turning.wobble:.2f} deg",
    )
    check(
        "a straight trace is refused at stage one",
        gate.verdict_for(flat.n, *flat.singular_values) == "line",
        gate.refusal_for(flat.n, *flat.singular_values),
    )

    # The reason the refusal matters, rather than the fact of it. A straight
    # trace does not give an imprecise attitude, it gives an arbitrary one --
    # so the test is that the answer moves under a nudge too small to see,
    # while the V's answer does not.
    #
    # The nudge goes on all three coordinates and not on the elevation alone,
    # because a trace that is straight to the last decimal is a degenerate case
    # that does not occur: its scatter has an exact null direction, so the fit
    # returns the vertical plane through the line, stably and wrongly, and the
    # swing this measures would come out at zero for the wrong reason. Half a
    # metre of wander is what a digitised line has.
    rng = np.random.default_rng(20260923)

    def nudged(xy, true):
        got = []
        for _ in range(8):
            points = on_plane(xy, *true)
            points += rng.normal(0.0, 0.5, points.shape)
            window = traces.window_attitude(points)
            got.append((window.dip_direction, window.dip))
        return max(separation(got[0], other) for other in got[1:])

    swing_flat = nudged(straight(400.0), (135.0, 65.0))
    swing_vee = nudged(vee(400.0, 250.0), (135.0, 65.0))

    check(
        "half a metre of noise swings the straight trace and not the V",
        swing_flat > 10.0 * swing_vee,
        f"{swing_flat:.1f} deg against {swing_vee:.2f} deg",
    )

    print("\n-- the lever floor is a distance, and the scale of the sheet -\n")

    small = traces.window_attitude(on_plane(vee(400.0, 10.0), 135.0, 65.0))

    check(
        "a 10 m V is refused at 1:50.000",
        traces.TraceGate.for_scale(50_000).verdict_for(
            small.n, *small.singular_values
        ) == "line",
        f"{small.spread:.1f} m off straight against a "
        f"{traces.TraceGate.for_scale(50_000).min_lever:.1f} m floor",
    )
    check(
        "the same V is admitted at 1:10.000",
        traces.TraceGate.for_scale(10_000).verdict_for(
            small.n, *small.singular_values
        ) != "line",
        f"floor {traces.TraceGate.for_scale(10_000).min_lever:.1f} m",
    )

    print("\n-- the floor, measured instead of assumed -------------------\n")

    # A line rough by a known amount. The estimator has a constant in it that
    # was derived on paper and is checked here against noise that was put in on
    # purpose -- which is the only part of it that could quietly be wrong.
    #
    # Long, and not the two kilometres this was first written with. A median
    # over a hundred sagittas carries a standard error of twelve per cent, so a
    # tolerance tight enough to catch a wrong constant also fails on an unlucky
    # draw: the first run read 1.29 for a sigma of 1 and the constant was fine.
    # A thousand vertices brings that to four per cent.
    for sigma in (1.0, 5.0, 20.0):
        rough = straight(20_000.0, step=40.0)
        rough[:, 1] += rng.normal(0.0, sigma, len(rough))

        jitter = traces.digitising_jitter([Line(rough)])

        check(
            f"a line rough by {sigma:.0f} m is measured at {sigma:.0f} m",
            abs(jitter["sigma"] - sigma) / sigma < 0.10,
            f"sigma {jitter['sigma']:.2f} m, exponent {jitter['exponent']:+.2f}, "
            f"{jitter['vertices']} vertices every {jitter['spacing']:.0f} m",
        )

    # The case the single-number version got wrong, and the reason for the
    # stride sweep. A smooth arc is all curvature: its sagitta is large, it
    # falls away as the square of the baseline, and none of it is the pen.
    sweep = np.arange(0.0, 20_000.0, 40.0)
    arc = np.column_stack([sweep, sweep ** 2 / (2.0 * 40_000.0)])

    smooth_only = traces.digitising_jitter([Line(arc)])

    check(
        "a smooth arc is seen as curvature and not as a shaky hand",
        smooth_only["exponent"] > 1.5
        and (smooth_only["sigma"] is None or smooth_only["sigma"] < 0.5),
        f"exponent {smooth_only['exponent']:+.2f}, sigma "
        + ("none" if smooth_only["sigma"] is None
           else f"{smooth_only['sigma']:.2f} m"),
    )

    # And the two together: the jitter has to come back out of a line that is
    # curved as well as rough, which is every mapped trace there is.
    both = arc.copy()
    both[:, 1] += rng.normal(0.0, 3.0, len(both))

    mixed = traces.digitising_jitter([Line(both)])

    check(
        "3 m of jitter on a curved line is still measured at 3 m",
        mixed["sigma"] is not None and abs(mixed["sigma"] - 3.0) / 3.0 < 0.20,
        f"sigma {mixed['sigma']:.2f} m, exponent {mixed['exponent']:+.2f}, "
        f"against a raw median sagitta of {mixed['median']:.2f} m",
    )

    # The bug that started this: the same line decimated must give the same
    # answer. The one-stride version read a factor of 3.3 across two subsets of
    # one sheet, which is the pen changing with how much of the layer you hand it.
    halved = traces.digitising_jitter([Line(both[::2])])

    check(
        "decimating the layer does not change the pen",
        halved["sigma"] is not None
        and abs(halved["sigma"] - mixed["sigma"]) / mixed["sigma"] < 0.25,
        f"{mixed['sigma']:.2f} m at {mixed['spacing']:.0f} m spacing, "
        f"{halved['sigma']:.2f} m at {halved['spacing']:.0f} m",
    )

    # The trap the docstring warns about. Resampling puts the points on the
    # chords, so the short strides find no roughness at all and the long ones
    # find the original vertices again -- a roughness that climbs with the
    # baseline, which is the same signature as a self-affine line and gets the
    # same refusal. The guard catches this one for free, and the test is here to
    # keep it caught rather than to claim it was aimed at.
    original = traces.digitising_jitter([Line(rough)])
    walked = traces._resample(rough, 10.0)[0]
    resampled = traces.digitising_jitter([Line(walked)])

    check(
        "a resampled trace is refused rather than answered",
        resampled["sigma"] is None,
        f"exponent {resampled['exponent']:+.2f} against {original['exponent']:+.2f} "
        f"on the vertices it came from",
    )

    # A line rough at every scale: octaves of wobble with amplitude going as the
    # wavelength, which is what a mapped contact turns out to be. There is no
    # baseline at which its roughness stops, so there is no pen under it, and
    # the refusal is the correct answer rather than a limitation.
    x = np.arange(0.0, 20_000.0, 40.0)
    fractal = np.zeros_like(x)
    for octave in range(8):
        wavelength = 20_000.0 / 2 ** octave
        fractal += (wavelength ** 0.9) * 0.02 * np.sin(
            2.0 * np.pi * x / wavelength + rng.uniform(0, 2 * np.pi)
        )

    selfaffine = traces.digitising_jitter([Line(np.column_stack([x, fractal]))])

    check(
        "a line rough at every scale is refused, not extrapolated",
        selfaffine["sigma"] is None,
        f"exponent {selfaffine['exponent']:+.2f}, between a pen and a curve",
    )

    check(
        "a gate built off a rough layer stands clear of the roughness",
        abs(traces.TraceGate.from_traces([Line(rough)]).min_lever
            - 3.0 * original["sigma"]) < 1e-9,
        f"floor {traces.TraceGate.from_traces([Line(rough)]).min_lever:.1f} m",
    )
    check(
        "a layer with nothing measurable falls back rather than failing",
        traces.TraceGate.from_traces([Line(np.array([[0.0, 0.0], [1.0, 1.0]]))])
        == traces.TraceGate()
        and traces.TraceGate.from_traces(
            [Line(np.column_stack([x, fractal]))]
        ) == traces.TraceGate(),
    )

    # The reason the floor is read off `transverse_spread` and not off `s2`: a
    # singular value is a sum of squares over the points, so resampling the same
    # trace more densely inflates it. A gate laid on `s2` would let the step
    # decide which contacts carry an attitude.
    coarse = traces.window_attitude(on_plane(vee(400.0, 40.0, step=20.0), 135.0, 65.0))
    fine = traces.window_attitude(on_plane(vee(400.0, 40.0, step=5.0), 135.0, 65.0))

    # To within a couple of per cent and not exactly: the apex of a V falls
    # between samples differently at different steps, so the RMS moves a little.
    # What it does not do is track the count, which `s2` does exactly.
    check(
        "the same V resampled four times finer is the same distance off straight",
        abs(coarse.spread - fine.spread) / coarse.spread < 0.05,
        f"{coarse.spread:.2f} m against {fine.spread:.2f} m, a factor of "
        f"{fine.spread / coarse.spread:.2f}, while s2 goes "
        f"{coarse.singular_values[1]:.0f} to {fine.singular_values[1]:.0f}, "
        f"a factor of {fine.singular_values[1] / coarse.singular_values[1]:.2f}",
    )

    print("\n-- the trace, cut into runs ---------------------------------\n")

    # Straight for the first kilometre, then wandering for the second: the
    # verdicts should change where the shape does and not somewhere else.
    flat_half = straight(500.0)
    flat_half[:, 0] += 500.0
    turning_half = sinuous(1000.0, 300.0, 120.0)
    turning_half[:, 0] += 1000.0

    trace = np.vstack([flat_half, turning_half])
    spans = traces.trace_spans([part_of(on_plane(trace, 135.0, 65.0))], 250.0, step=25.0)

    totals = spans.metres()

    check(
        "every metre walked carries one verdict and no more",
        abs(sum(totals.values()) - spans.walked) < 1e-6,
        f"{sum(totals.values()):.0f} m of {spans.walked:.0f} walked",
    )
    check(
        "the straight half is unreadable and the turning half is not",
        totals["line"] > 400.0 and totals["held"] > 400.0,
        ", ".join(f"{k} {v:.0f} m" for k, v in totals.items()),
    )

    crossing = float(spans.progressive[-1])
    run = spans.run_at(crossing)

    check(
        "a crossing comes back with the stretch it was read on",
        run is not None and run[1] <= crossing <= run[2],
        f"{run[0]} over {run[2] - run[1]:.0f} m" if run else "no run",
    )
    check(
        "a progressive off the walked part has no window",
        spans.at(-500.0) is None,
    )

    # The head and tail of a trace are covered by windows that are not centred
    # on them, and the two lookups used to disagree about exactly that stretch:
    # an attitude came back with no extent to put it over, which is the one
    # pairing this module exists to keep together. Swept rather than spot-tested,
    # because the disagreement was only ever half a window wide.
    probes = np.arange(-300.0, spans.trace_length + 300.0, 17.0)

    check(
        "at and run_at agree everywhere about what is covered",
        all((spans.at(s) is None) == (spans.run_at(s) is None) for s in probes),
        f"{len(probes)} probes across the trace and past both ends",
    )

    print("\n-- runs do not cross a break in the trace -------------------\n")

    first = vee(300.0, 200.0)
    second = vee(300.0, 200.0)
    second[:, 0] += 5000.0            # a fragment far away: mapped, then cover

    parts = [part_of(on_plane(first, 135.0, 65.0))]
    coords, s = part_of(on_plane(second, 135.0, 65.0))
    parts.append((coords, s + parts[0][1][-1] + 2000.0))

    broken = traces.trace_spans(parts, 200.0, step=25.0)

    check(
        "the progressive carries on across the parts",
        broken.progressive[-1] > parts[0][1][-1],
        f"last centre at {broken.progressive[-1]:.0f} m",
    )
    check(
        "no run spans the gap between two fragments",
        all(
            not (s0 < parts[0][1][-1] < s1)
            for _, s0, s1 in broken.runs()
        ),
        f"{len(broken.runs())} runs",
    )

    print("\n-- the sweep finds a length that was put there --------------\n")

    # Four domains of 400 m, each on its own plane. A window inside a domain
    # sees one plane; a window spanning two sees curvature that is real, which
    # lands in s3. The held share should therefore turn over somewhere near the
    # domain, and not sit at an end of the sweep.
    pieces, offset = [], 0.0
    for index, dip in enumerate((40.0, 55.0, 25.0, 60.0)):
        shape = vee(200.0, 130.0)
        shape[:, 0] += offset + 200.0
        piece = on_plane(shape, 135.0 + 20.0 * index, dip)
        piece[:, 2] += 0.0 if not pieces else pieces[-1][-1, 2] - piece[0, 2]
        pieces.append(piece)
        offset += 400.0

    domained = [part_of(np.vstack(pieces))]
    sweep = traces.window_sweep(domained, (150.0, 250.0, 400.0, 700.0, 1000.0))

    shares = {
        length: span.metres()["held"] / span.walked if span.walked else 0.0
        for length, span in sweep.items()
    }

    print("      " + "  ".join(f"{int(k)}m {v:5.0%}" for k, v in sorted(shares.items())))

    found = traces.holding_length(sweep)

    check(
        "the sweep peaks inside it, at the scale of the domains",
        found is not None and 150.0 <= found <= 400.0,
        f"peak at {found:.0f} m" if found else "no peak",
    )
    check(
        "a trace of one plane throughout has no peak to report",
        traces.holding_length(
            traces.window_sweep(
                [part_of(on_plane(vee(1000.0, 600.0), 135.0, 65.0))],
                (150.0, 250.0, 400.0, 700.0),
            )
        ) is None,
    )

    print("\n-- a threshold is asked of a result, not computed into it ---\n")

    strict = traces.TraceGate(min_precision=40.0)
    regated = traces.trace_spans(
        [part_of(on_plane(vee(400.0, 250.0), 135.0, 65.0))], 250.0, step=25.0
    )
    before = regated.metres()["held"]
    fresh = traces.trace_spans(
        [part_of(on_plane(vee(400.0, 250.0), 135.0, 65.0))],
        250.0, step=25.0, gate=strict,
    )

    regated.regate(strict)

    check(
        "regating gives what computing with that gate would have given",
        np.array_equal(regated.held, fresh.held)
        and np.array_equal(regated.determined, fresh.determined),
        f"{before:.0f} m held before, {regated.metres()['held']:.0f} m after",
    )

    print("\n-- poles average, azimuths do not ---------------------------\n")

    spans.dip_directions[:2] = (350.0, 10.0)
    spans.dips[:2] = (60.0, 60.0)
    mean = spans.mean_attitude(spans.progressive[0], spans.progressive[1])

    check(
        "350 and 10 average to 0, not to 180",
        mean is not None and min(mean[0], 360.0 - mean[0]) < 0.5,
        f"{mean[0]:.1f}/{mean[1]:.1f}" if mean else "none",
    )

    print("\n-- the DEM, read once --------------------------------------\n")

    import rasterio
    from rasterio.transform import from_origin

    from gsurf.dem import Dem

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "tilt.tif"
        side, cell = 400, 5.0

        # A plane as a DEM, so that a trace hung on it has a known elevation at
        # every point and the sampling can be checked rather than eyeballed.
        rows, cols = np.mgrid[0:side, 0:side].astype(np.float64)
        band = 100.0 + 0.2 * cols * cell - 0.1 * (side - 1 - rows) * cell

        with rasterio.open(
            path, "w", driver="GTiff", height=side, width=side, count=1,
            dtype="float64", crs="EPSG:32633",
            transform=from_origin(500_000.0, 4_400_000.0, cell, cell),
            nodata=-9999.0,
        ) as sink:
            sink.write(band, 1)

        dem = Dem(str(path))

        a = np.column_stack([
            np.linspace(500_100.0, 500_500.0, 40),
            np.full(40, 4_399_500.0),
        ])
        b = np.column_stack([
            np.linspace(500_100.0, 500_500.0, 40),
            np.full(40, 4_399_200.0),
        ])

        hung = traces.trace_points([Line(a), Line(b)], dem, step=10.0)

        check("both fragments come back", len(hung) == 2, f"{len(hung)} parts")

        if len(hung) == 2:
            check(
                "the progressive of the second carries on from the first",
                abs(hung[1][1][0] - hung[0][1][-1]) < 15.0,
                f"{hung[0][1][-1]:.0f} m then {hung[1][1][0]:.0f} m",
            )

            # The DEM back from its own definition: col = (x - origin) / cell
            # and row = (origin - y) / cell, which is the transform the sampling
            # has to invert. Written out rather than simplified, so that a
            # change in the transform above cannot quietly agree with a change
            # here.
            x, y = hung[0][0][:, 0], hung[0][0][:, 1]
            col = (x - 500_000.0) / cell
            row = (4_400_000.0 - y) / cell
            expected = 100.0 + 0.2 * col * cell - 0.1 * (side - 1 - row) * cell

            worst = float(np.max(np.abs(hung[0][0][:, 2] - expected)))
            check(
                "the elevations are the DEM's, to within a cell",
                worst < 0.2 * cell,
                f"worst {worst:.2f} m",
            )

        dem.close()

    print("\n-- the whole way through, on a DEM --------------------------\n")

    import rasterio
    from rasterio.transform import from_origin

    from geogst.core.geology.orientations import Plane
    from geogst.core.geometries.shapes.lines import Ln

    from gsurf.attitudes import TraceRecord
    from gsurf.dem import Dem

    # An outcrop trace that is real rather than drawn: a corrugated topography
    # depending on x alone, and the plane that cuts it. Where the two meet is
    # solvable for one y per x, so the trace lies exactly on both -- which is
    # what makes the answer checkable. `fit_records` then does the whole thing,
    # DEM sampling and windowing and gating and records, and has to give the
    # plane back.
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "corrugated.tif"
        side, cell = 700, 5.0
        amplitude, wavelength = 120.0, 600.0
        west, north = 500_000.0, 4_400_000.0

        rows, cols = np.mgrid[0:side, 0:side].astype(np.float64)
        band = 800.0 + amplitude * np.sin(2.0 * np.pi * cols * cell / wavelength)

        with rasterio.open(
            path, "w", driver="GTiff", height=side, width=side, count=1,
            dtype="float64", crs="EPSG:32633",
            transform=from_origin(west, north, cell, cell), nodata=-9999.0,
        ) as sink:
            sink.write(band, 1)

        dem = Dem(str(path))

        truth = (170.0, 40.0)
        n = pole(*truth)
        slope_x, slope_y = -n[0] / n[2], -n[1] / n[2]

        xs = np.arange(700.0, 2800.0, 20.0)
        relief = 800.0 + amplitude * np.sin(2.0 * np.pi * xs / wavelength)

        # z0 chosen so the trace runs up the middle rather than off the sheet.
        mid = len(xs) // 2
        z0 = relief[mid] - slope_x * (xs[mid] - xs[0]) - slope_y * 0.0
        ys = 1750.0 + (relief - z0 - slope_x * (xs - xs[0])) / slope_y

        check(
            "the synthetic trace stays on the DEM",
            ys.min() > 100.0 and ys.max() < side * cell - 100.0,
            f"y from {ys.min():.0f} to {ys.max():.0f} m of {side * cell:.0f}",
        )

        line = Ln(np.column_stack([west + xs, north - side * cell + ys]))
        surveyed = TraceRecord(
            category="contatto", plane=Plane(0.0, 5.0), lines=[line],
            length=line.length_2d(), attrs={"src": "a wrong column"},
        )

        got, report = traces.fit_records([surveyed], dem)

        check(
            "one trace comes back as one or more fitted records",
            len(got) >= 1,
            traces.describe_fit(report).split("\n")[0],
        )

        if got:
            off = [
                separation(truth, (r.plane.dipazim, r.plane.dipang)) for r in got
            ]
            check(
                "and they carry the plane the topography implies, not the column's",
                max(off) < 5.0,
                f"worst {max(off):.1f} deg over {len(got)} stretches, against "
                f"{separation(truth, (0.0, 5.0)):.0f} deg for the column",
            )
            check(
                "each is marked as a fit, so curation can refuse to quote it",
                all(r.attrs.get("fitted") and "fit" in str(r.attrs.get("src"))
                    for r in got),
                str(got[0].attrs.get("src")),
            )
            check(
                "and keeps the whole trace, cutting it with the span",
                all(abs(r.length - line.length_2d()) < 1.0 and r.span is not None
                    for r in got),
                f"spans {', '.join(f'{b - a:.0f} m' for _, (a, b) in ((r, r.span) for r in got))}",
            )

        def fitted_from(name, coords):
            record = TraceRecord(
                category=name, plane=Plane(0.0, 5.0), lines=[Ln(coords)],
                length=Ln(coords).length_2d(),
            )
            return traces.fit_records([record], dem)

        # Straight in plan, but climbing the corrugations: this is *not* the
        # undetermined case, and writing the test as though it were is what
        # first failed here. A contact that runs dead straight across ridge and
        # valley alike is vertical -- the three-point rule says so, and the
        # points really do lie in one vertical plane. The fit has to say 90.
        across, _ = fitted_from("dritto sul rilievo", np.column_stack([
            west + xs, np.full(len(xs), north - side * cell + 1750.0)
        ]))

        check(
            "straight in plan across relief is a vertical plane, and is read as one",
            across and all(abs(r.plane.dipang - 90.0) < 1.0 for r in across),
            f"{across[0].plane.dipazim:.0f}/{across[0].plane.dipang:.0f}"
            if across else "nothing came back",
        )

        # The undetermined case is a trace along a contour. The DEM here depends
        # on x alone, so a line of constant x is level, its points are collinear
        # in three dimensions, and every plane through them fits equally well.
        ys_level = np.arange(800.0, 2600.0, 20.0)
        along, along_report = fitted_from("lungo la curva di livello", np.column_stack([
            np.full(len(ys_level), west + 1500.0),
            north - side * cell + ys_level,
        ]))

        check(
            "a trace along a contour gives no record at all",
            len(along) == 0 and along_report["silent"] == 1,
            "every plane through a level line fits it",
        )

        dem.close()

    print("\n-- a run is a record ----------------------------------------\n")

    from geogst.core.geometries.shapes.lines import Ln

    shape = np.vstack([flat_half, turning_half])
    line = Ln(shape)
    spans = traces.trace_spans([part_of(on_plane(shape, 135.0, 65.0))], 250.0, step=25.0)

    records = traces.records_from_spans(spans, [line], "contatto")

    check("a held run comes back as a record", len(records) >= 1, f"{len(records)}")

    if records:
        record = records[0]
        held = [run for run in spans.runs() if run[0] == "held"][0]

        check(
            "the record keeps the whole trace and cuts it with the span",
            abs(record.length - line.length_2d()) < 1.0
            and record.span == (held[1], held[2]),
            f"{record.length:.0f} m of trace, span {record.span[0]:.0f}-{record.span[1]:.0f}",
        )
        check(
            "the record's reach lands inside the run it came from",
            record.reach_endpoints(None) is not None,
        )
        check(
            "the plane on the record is the plane the trace was built on",
            separation((135.0, 65.0), (record.plane.dipazim, record.plane.dipang)) < 2.0,
            f"{record.plane.dipazim:.1f}/{record.plane.dipang:.1f}",
        )

    print()

    if FAILURES:
        print(f"FAILED: {', '.join(FAILURES)}")
        return 1

    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
