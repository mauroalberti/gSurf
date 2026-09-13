"""
Does the fold-axis module recover an axis that is known?

The arithmetic is geogst's, so what is tested here is the use of it: which
eigenvector is the axis, which way a pole points, whether a right-hand-rule
strike and a dip direction give the same answer, and whether the gate lets
through what it should. A synthetic cylindrical fold has an axis by
construction, which is the only way to check the answer rather than compare it
with itself.

    QT_QPA_PLATFORM=offscreen python check_folds.py
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


def versor(trend, plunge):
    """(East, North, Up) of a downward direction given as trend and plunge."""

    t, p = np.radians(trend), np.radians(plunge)

    return np.array([np.sin(t) * np.cos(p), np.cos(t) * np.cos(p), -np.sin(p)])


def angle_between_axes(a, b):
    """Degrees between two axes, sign-blind: 179 degrees apart is 1 degree."""

    return float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(a, b)))))))


def cylindrical_fold(axis_trend, axis_plunge, n=120, noise_deg=0.0, seed=0, half_opening=180.0):
    """
    Bedding attitudes around a fold of the given axis.

    Every pole of a cylindrical fold is perpendicular to its axis, so the poles
    lie on a great circle -- the girdle -- and generating them is a matter of
    walking round that circle. `half_opening` under 180 gives an open fold whose
    poles occupy an arc rather than the whole girdle, which is what real data
    looks like.
    """

    rng = np.random.default_rng(seed)
    axis = versor(axis_trend, axis_plunge)

    # Two directions spanning the plane normal to the axis.
    seed_vector = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(seed_vector, axis)) > 0.9:
        seed_vector = np.array([1.0, 0.0, 0.0])

    u = np.cross(axis, seed_vector)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    theta = np.radians(rng.uniform(-half_opening, half_opening, n))
    poles = np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v

    if noise_deg:
        poles = poles + rng.normal(0.0, np.radians(noise_deg), poles.shape)
        poles /= np.linalg.norm(poles, axis=1)[:, None]

    # As an axis, a pole is read downward.
    poles[poles[:, 2] > 0] *= -1.0

    plunge = np.degrees(np.arcsin(-poles[:, 2]))
    trend = np.degrees(np.arctan2(poles[:, 0], poles[:, 1])) % 360.0

    # A plane from its downward pole: it dips away from the pole's trend.
    return (trend + 180.0) % 360.0, 90.0 - plunge


def poles_about(axis_trends, axis_plunges, noise_deg=0.0, seed=0, half_opening=180.0):
    """
    One bedding attitude per station, each perpendicular to its own local axis.

    `cylindrical_fold` gives attitudes that share an axis. This gives attitudes
    that do not, which is the only way to write down a fold whose axis turns
    along strike: stacking domains built by the other function can produce a
    step and nothing else, and a step and a gradient are the two cases a field
    of axes most needs to be able to tell apart.
    """

    trends = np.asarray(axis_trends, dtype=float)
    plunges = np.asarray(axis_plunges, dtype=float)
    n = len(trends)
    rng = np.random.default_rng(seed)

    t, p = np.radians(trends), np.radians(plunges)
    axes = np.c_[np.sin(t) * np.cos(p), np.cos(t) * np.cos(p), -np.sin(p)]

    # Something to span the plane normal to each axis, swapped where the axis is
    # too near it for the cross product to be well conditioned.
    seeds = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    seeds[np.abs(axes[:, 2]) > 0.9] = np.array([1.0, 0.0, 0.0])

    u = np.cross(axes, seeds)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(axes, u)

    theta = np.radians(rng.uniform(-half_opening, half_opening, n))
    poles = np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v

    if noise_deg:
        poles = poles + rng.normal(0.0, np.radians(noise_deg), poles.shape)
        poles /= np.linalg.norm(poles, axis=1)[:, None]

    poles[poles[:, 2] > 0] *= -1.0

    plunge = np.degrees(np.arcsin(-poles[:, 2]))
    trend = np.degrees(np.arctan2(poles[:, 0], poles[:, 1])) % 360.0

    return (trend + 180.0) % 360.0, 90.0 - plunge


def as_layer(directory, name, xy, dip_dirs, dips, dip_dir_name="Immersione", dip_name="Inclinazione"):
    import geopandas as gpd
    from shapely.geometry import Point

    frame = gpd.GeoDataFrame(
        {dip_dir_name: dip_dirs, dip_name: dips},
        geometry=[Point(x, y) for x, y in xy],
        crs="EPSG:25833",
    )
    path = directory / f"{name}.gpkg"
    frame.to_file(path, layer=name, driver="GPKG")

    return path


def main():
    from PyQt6 import QtWidgets

    from gsurf.attitudes import AttitudeSource
    from gsurf.folds import Gate, fold_axis
    from gsurf.stereonet import StereonetView

    app = QtWidgets.QApplication(sys.argv)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # -- an axis that is known ------------------------------------------
        print("-- a cylindrical fold, axis 120/10 --")
        for noise in (0.0, 3.0, 10.0):
            dip_dirs, dips = cylindrical_fold(120.0, 10.0, n=120, noise_deg=noise, seed=1)
            xy = np.c_[np.zeros(len(dips)), np.zeros(len(dips))]
            path = as_layer(tmp, f"fold_{int(noise)}", xy, dip_dirs, dips)

            source = AttitudeSource(path, "EPSG:25833", layer=f"fold_{int(noise)}",
                                    dip_dir_field="Immersione", dip_field="Inclinazione")
            result = fold_axis(source.poles)

            recovered = versor(*result.axis)
            off = angle_between_axes(recovered, versor(120.0, 10.0))

            check(f"axis recovered, noise {noise:.0f} deg", off < max(1.0, noise / 3.0),
                  f"{result.axis[0]:.1f}/{result.axis[1]:.1f}, {off:.2f} deg off, K={result.k:.2f}")

        # -- and one that is not a fold --------------------------------------
        print("\n-- what the gate lets through --")
        gate = Gate()

        dip_dirs, dips = cylindrical_fold(120.0, 10.0, n=60, noise_deg=5.0, seed=2)
        girdle = fold_axis(AttitudeSource(
            as_layer(tmp, "girdle", np.zeros((60, 2)), dip_dirs, dips),
            "EPSG:25833", layer="girdle",
            dip_dir_field="Immersione", dip_field="Inclinazione").poles)
        check("a girdle is admitted", gate.admits(girdle),
              f"K={girdle.k:.2f} C={girdle.c:.2f}")

        # A homocline: every bed the same attitude, scattered a little.
        rng = np.random.default_rng(3)
        cluster_dd = 95.0 + rng.normal(0, 8, 60)
        cluster_d = 28.0 + rng.normal(0, 5, 60)
        cluster = fold_axis(AttitudeSource(
            as_layer(tmp, "cluster", np.zeros((60, 2)), cluster_dd, cluster_d),
            "EPSG:25833", layer="cluster",
            dip_dir_field="Immersione", dip_field="Inclinazione").poles)
        check("a homocline is refused", not gate.admits(cluster),
              gate.refusal(cluster))

        sparse = fold_axis(AttitudeSource(
            as_layer(tmp, "sparse", np.zeros((4, 2)), dip_dirs[:4], dips[:4]),
            "EPSG:25833", layer="sparse",
            dip_dir_field="Immersione", dip_field="Inclinazione").poles)
        check("too few attitudes is refused", not gate.admits(sparse), gate.refusal(sparse))
        check("an empty window has no result at all", fold_axis([]) is None)

        # -- the two ways of writing an azimuth --------------------------------
        print("\n-- conventions --")
        dip_dirs, dips = cylindrical_fold(120.0, 10.0, n=60, noise_deg=4.0, seed=4)
        strikes = (dip_dirs - 90.0) % 360.0

        by_dip_dir = fold_axis(AttitudeSource(
            as_layer(tmp, "dd", np.zeros((60, 2)), dip_dirs, dips),
            "EPSG:25833", layer="dd",
            dip_dir_field="Immersione", dip_field="Inclinazione").poles)

        rhr_source = AttitudeSource(
            as_layer(tmp, "rhr", np.zeros((60, 2)), strikes, dips),
            "EPSG:25833", layer="rhr",
            dip_dir_field="Immersione", dip_field="Inclinazione", is_rhr_strike=True)
        by_strike = fold_axis(rhr_source.poles)

        check("strike RHR and dip direction agree",
              angle_between_axes(versor(*by_dip_dir.axis), versor(*by_strike.axis)) < 1e-6,
              f"{by_dip_dir.axis[0]:.2f}/{by_dip_dir.axis[1]:.2f} vs "
              f"{by_strike.axis[0]:.2f}/{by_strike.axis[1]:.2f}")

        check("and dip_directions() undoes the strike",
              np.allclose(np.sort(rhr_source.dip_directions()), np.sort(dip_dirs % 360.0)))

        # -- reading what a survey actually writes -----------------------------
        print("\n-- the sentinels a real sheet carries --")
        dd = np.array([131.0, 999.0, 51.0, 168.0, 14.0, 999.0])
        da = np.array([33.0, 0.0, 99.0, 30.0, 28.0, 0.0])
        carg = AttitudeSource(
            as_layer(tmp, "carg", np.zeros((6, 2)), dd, da),
            "EPSG:25833", layer="carg",
            dip_dir_field="Immersione", dip_field="Inclinazione")

        # Six records, one unusable dip: the two horizontal beds survive their
        # 999 azimuth, which is the whole point of the exemption.
        check("horizontal beds are kept, sentinel azimuth and all", len(carg) == 5,
              f"{len(carg)} kept of 6")
        check("a dip of 99 is dropped, and said so",
              carg.dropped == {"dip outside 0-90": 1}, str(carg.dropped))

        horizontal = [i for i, d in enumerate(carg.dips) if d == 0.0]
        check("and their poles are vertical",
              all(abs(carg.poles[i].d[1] - 90.0) < 1e-9 for i in horizontal),
              f"plunge {carg.poles[horizontal[0]].d[1]:.6f}")

        # 360 is how north gets written: on the Marsico Nuovo sheet ten
        # attitudes carry it and none carries 0.
        north = AttitudeSource(
            as_layer(tmp, "north", np.zeros((3, 2)),
                     np.array([360.0, 0.0, 180.0]), np.array([30.0, 30.0, 30.0])),
            "EPSG:25833", layer="north",
            dip_dir_field="Immersione", dip_field="Inclinazione")

        check("a dip direction of 360 is north, not an error", len(north) == 3,
              f"{len(north)} kept of 3, dropped {north.dropped}")
        check("and it is the same attitude as 0",
              abs(north.poles[0].d[0] - north.poles[1].d[0]) < 1e-9,
              f"pole trends {north.poles[0].d[0]:.3f} and {north.poles[1].d[0]:.3f}")

        # -- windows ------------------------------------------------------------
        print("\n-- the moving window --")
        rng = np.random.default_rng(5)
        xy = rng.uniform(0, 10000, (200, 2))
        dip_dirs, dips = cylindrical_fold(120.0, 10.0, n=200, noise_deg=6.0, seed=6)
        spread = AttitudeSource(
            as_layer(tmp, "spread", xy, dip_dirs, dips),
            "EPSG:25833", layer="spread",
            dip_dir_field="Immersione", dip_field="Inclinazione")

        inside = spread.within(5000.0, 5000.0, 2000.0)
        offsets = spread.xy[inside] - np.array([5000.0, 5000.0])
        distances = np.hypot(offsets[:, 0], offsets[:, 1])
        outside = np.setdiff1d(np.arange(len(spread)), inside)
        far = np.hypot(*(spread.xy[outside] - np.array([5000.0, 5000.0])).T)

        check("the window holds what is inside it", distances.max() <= 2000.0,
              f"{len(inside)} stations, farthest {distances.max():.0f} m")
        check("and nothing that is not", far.min() > 2000.0, f"nearest excluded {far.min():.0f} m")

        # -- the net does not accumulate artists ---------------------------------
        print("\n-- the stereonet --")
        net = StereonetView()
        net.resize(360, 360)
        net.show()
        app.processEvents()

        before = len(net.axes.lines)
        for _ in range(50):
            idx = spread.within(rng.uniform(2000, 8000), rng.uniform(2000, 8000), 2500.0)
            result = fold_axis(spread.poles_at(idx))
            net.show_window(spread.dip_directions()[idx], spread.dips[idx], result,
                            gate.admits(result))
            app.processEvents()

        check("fifty windows leave the same artists behind",
              len(net.axes.lines) == before, f"{before} -> {len(net.axes.lines)}")
        check("and the poles are one artist, not one each",
              len(net.poles.get_xdata()) == len(idx), f"{len(net.poles.get_xdata())} points in 1 Line2D")

        # -- the grid ------------------------------------------------------------
        print("\n-- the grid --")

        from gsurf.folds import field_cost, fold_axis_field, grid_centres

        # Two structures side by side: a fold west of 5000, another east of it,
        # with axes 40 degrees apart. A field that cannot tell them apart is
        # averaging, which is the thing a moving window exists to avoid.
        west_dd, west_d = cylindrical_fold(120.0, 10.0, n=300, noise_deg=5.0, seed=7)
        east_dd, east_d = cylindrical_fold(160.0, 25.0, n=300, noise_deg=5.0, seed=8)

        rng = np.random.default_rng(9)
        west_xy = np.c_[rng.uniform(0, 4500, 300), rng.uniform(0, 10000, 300)]
        east_xy = np.c_[rng.uniform(5500, 10000, 300), rng.uniform(0, 10000, 300)]

        two = AttitudeSource(
            as_layer(tmp, "two", np.vstack([west_xy, east_xy]),
                     np.r_[west_dd, east_dd], np.r_[west_d, east_d]),
            "EPSG:25833", layer="two",
            dip_dir_field="Immersione", dip_field="Inclinazione")

        centres = grid_centres((0.0, 0.0, 10000.0, 10000.0), 1000.0)
        check("the grid covers the area", len(centres) == 11 * 11, f"{len(centres)} cells")
        check("and starts on its corner", tuple(centres[0]) == (0.0, 0.0))

        # How much a grid overlaps itself: the number the two controls never
        # showed, and the one that decides whether a thousand cells are a
        # thousand observations.
        from gsurf.folds import describe_sampling, sampling

        area = (0.0, 0.0, 10000.0, 10000.0)
        counts = sampling(area, 1500.0, 500.0)

        check("cells per attitude is the window over the cell",
              abs(counts["per_attitude"] - np.pi * 1500.0 ** 2 / 500.0 ** 2) < 1e-9,
              f"{counts['per_attitude']:.1f}")
        check("and the two numbers multiply back to the cell count",
              abs(counts["per_attitude"] * counts["tiling"] - 100e6 / 500.0 ** 2) < 1e-6,
              f"{counts['per_attitude']:.1f} x {counts['tiling']:.1f}")
        check("a step of twice the radius does not overlap",
              not sampling(area, 1500.0, 3000.0)["overlapping"]
              and "do not overlap" in describe_sampling(area, 1500.0, 3000.0))
        check("and just under it does",
              sampling(area, 1500.0, 2999.0)["overlapping"])

        estimate = field_cost(two, centres, 1500.0)
        field = fold_axis_field(two, centres, 1500.0)

        check("the estimate is the right order",
              0.2 < estimate["seconds"] / max(1e-6, 0.02) < 50,
              f"{estimate['seconds'] * 1000:.0f} ms predicted for {len(centres)} cells")
        check("the field answers every cell", len(field) == len(centres))
        check("and most of them have an axis", field.admitted.sum() > len(centres) * 0.5,
              field.summary())

        # Each side should recover its own axis, not the average of the two.
        for side, wanted, mask in (
            ("west", (120.0, 10.0), field.centres[:, 0] <= 3000),
            ("east", (160.0, 25.0), field.centres[:, 0] >= 7000),
        ):
            taken = mask & field.admitted
            offsets = [
                angle_between_axes(versor(t, p), versor(*wanted))
                for t, p in zip(field.trends[taken], field.plunges[taken])
            ]
            check(f"the {side} half recovers its own axis {wanted[0]:.0f}/{wanted[1]:.0f}",
                  taken.sum() > 3 and max(offsets) < 8.0,
                  f"{taken.sum()} cells, worst {max(offsets):.1f} deg off")

        # -- an axis that turns, rather than two that differ ---------------------
        print("\n-- a fold axis that rotates along strike --")

        # The other thing next to a step: a trend rising at a known rate, with
        # the plunge held still so that anything the plunge does is the window's
        # and not the model's. Two degrees per kilometre over twenty kilometres,
        # which is 100 to 140 -- the span the Basilicata axes actually cover.
        RATE = 2.0          # degrees of trend per kilometre
        BASE = 100.0
        SPAN = 20000.0

        rng = np.random.default_rng(11)
        ramp_xy = np.c_[rng.uniform(0.0, SPAN, 1200), rng.uniform(0.0, 10000.0, 1200)]
        imposed = BASE + RATE * ramp_xy[:, 0] / 1000.0

        ramp_dd, ramp_d = poles_about(imposed, np.full(1200, 10.0), noise_deg=5.0, seed=12)

        ramp = AttitudeSource(
            as_layer(tmp, "ramp", ramp_xy, ramp_dd, ramp_d),
            "EPSG:25833", layer="ramp",
            dip_dir_field="Immersione", dip_field="Inclinazione")

        ramp_centres = grid_centres((0.0, 0.0, SPAN, 10000.0), 1000.0)

        # A symmetric moving average leaves a straight line alone: every window
        # in the interior is centred on its own cell, and the rotation it
        # averages over is as much ahead of the centre as behind. So the rate
        # should come back whatever the radius -- which is exactly what a step
        # does not do, and the reason the two can be told apart at all.
        for radius, tolerance in ((1500.0, 0.15), (3000.0, 0.15)):
            rfield = fold_axis_field(ramp, ramp_centres, radius)

            inside = (
                rfield.admitted
                & (rfield.centres[:, 0] >= 3000.0)
                & (rfield.centres[:, 0] <= SPAN - 3000.0)
            )
            xs = rfield.centres[inside, 0] / 1000.0
            recovered = np.polyfit(xs, rfield.trends[inside], 1)[0]

            check(f"the rate comes back at r = {radius:.0f} m",
                  inside.sum() > 20 and abs(recovered - RATE) < tolerance * RATE,
                  f"{recovered:.2f} deg/km against {RATE:.2f} imposed, {inside.sum()} cells")

            residual = rfield.trends[inside] - (BASE + RATE * xs)
            check(f"and the trend sits on the ramp, not beside it, at r = {radius:.0f} m",
                  abs(float(np.mean(residual))) < 2.0,
                  f"mean residual {float(np.mean(residual)):+.2f} deg, "
                  f"scatter {float(np.std(residual)):.2f}")

            check(f"the plunge stays where it was put at r = {radius:.0f} m",
                  abs(float(np.median(rfield.plunges[inside])) - 10.0) < 2.0,
                  f"median {float(np.median(rfield.plunges[inside])):.1f} deg against 10")

        # -- regating must be the same answer, not a similar one -----------------
        for gate_now in (Gate(min_points=5, max_k=0.6),
                         Gate(min_points=20, max_k=2.0),
                         Gate(min_points=10, max_k=1.0)):
            fresh = fold_axis_field(two, centres, 1500.0, gate_now)
            again = fold_axis_field(two, centres, 1500.0).regate(gate_now)

            same = bool(np.array_equal(fresh.admitted, again.admitted))
            check(f"regate == recompute at n>={gate_now.min_points}, K<={gate_now.max_k}",
                  same and fresh.refusals == again.refusals,
                  f"{int(fresh.admitted.sum())} vs {int(again.admitted.sum())} axes")

        # -- and it can be stopped -----------------------------------------------
        seen = []

        def stop_early(done, total):
            seen.append(done)
            return done < 40

        partial = fold_axis_field(two, centres, 1500.0, progress=stop_early)
        check("a field can be given up on", partial.counts[-1] == 0 and len(seen) > 1,
              f"stopped after {seen[-1]} of {len(centres)}")

        # -- the net in a window of its own ---------------------------------------
        print("\n-- the net's window --")

        from gsurf.sources import open_session
        from gsurf.tools.fold_axes import FoldAxesWindow, read_attitudes

        spec = dict(
            path=str(as_layer(tmp, "windowed", np.vstack([west_xy, east_xy]),
                              np.r_[west_dd, east_dd], np.r_[west_d, east_d])),
            role="points", layer="windowed",
            dip_dir_field="Immersione", dip_field="Inclinazione",
        )
        session = open_session(dict(attitudes=spec))
        window = FoldAxesWindow(session, read_attitudes(session, spec), radius=2000.0)
        window.resize(1200, 860)
        window.show()
        app.processEvents()

        dock = window.stereonet_dock

        check("the net opens in a window of its own",
              dock.isFloating() and dock.isVisible())
        check("and what is in it is the stereonet", dock.widget() is window.stereonet)
        check("showing the window we are in", window._net_is_current,
              f"{len(window.stereonet.poles.get_xdata())} poles")

        # -- closed, it is not drawn ----------------------------------------------
        left_on_it = np.array(window.stereonet.poles.get_xdata()).copy()

        dock.toggleViewAction().trigger()
        app.processEvents()

        check("it closes from the panel's button", not dock.isVisible())
        check("which cannot disagree with it", not window.stereonet_button.isChecked())

        cx, cy = session.center()
        window._move_centre(cx + 2500.0, cy + 2500.0)
        window.update_window()
        app.processEvents()

        # Vacuous unless the window really changed under it, so that is checked
        # first: a net that was never going to move proves nothing about a net
        # that was not redrawn.
        check("the window moved to a different set of attitudes",
              len(window.indices) != len(left_on_it),
              f"{len(left_on_it)} -> {len(window.indices)}")
        check("but a net that is closed is not drawn",
              np.array_equal(left_on_it, window.stereonet.poles.get_xdata())
              and not window._net_is_current,
              f"{len(window.stereonet.poles.get_xdata())} poles left on it")

        # -- and reopening catches it up -------------------------------------------
        dock.toggleViewAction().trigger()
        app.processEvents()

        check("reopening shows the window we are in now, not the one it was closed on",
              window._net_is_current
              and len(window.stereonet.poles.get_xdata()) == len(window.indices),
              f"{len(window.stereonet.poles.get_xdata())} poles for "
              f"{len(window.indices)} attitudes in the window")

        # -- what the tool itself drew, taken off the map --------------------------
        print("\n-- the measurements, switched off --")

        view = window.map_view
        switches = view._legend_switches

        wired = {
            t.get_text() for t in view.legend.get_texts() if id(t) in switches
        }
        check("the tool's dense entries are clickable, its steering ones are not",
              wired == {"attitude", "in the window"},
              f"wired {sorted(wired)}")

        source, values = next(
            s for t, s in ((t, switches.get(id(t))) for t in view.legend.get_texts())
            if s and t.get_text() == "attitude"
        )
        source.toggle(values)
        check("switching 'attitude' takes the station dots off the map",
              not window.station_dots.get_visible())

        # The station dots are static -- they live in the blitting background,
        # not in the per-frame draw -- so this is the rebuild that has to put
        # them back, and the redraw that has to notice.
        view.refresh_legend()
        check("and they stay off across a legend rebuild",
              not window.station_dots.get_visible()
              and view.hidden_count() == 1,
              f"{view.hidden_count()} hidden")

        greyed = [t for t in view.legend.get_texts() if t.get_text() == "attitude"]
        check("with the entry greyed like a category's",
              greyed[0].get_color() == "#9a9a9a", greyed[0].get_color())

        # It counts as hidden, so the way back from a hidden legend covers it
        # too: without that, switching the attitudes off and the legend away
        # would leave them off with nothing to click.
        view.show_all_categories()
        check("and 'show all' puts what the tool drew back as well",
              window.station_dots.get_visible() and view.hidden_count() == 0)

        window.close()
        session.close()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed: {', '.join(FAILURES)}")
        return 1

    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
