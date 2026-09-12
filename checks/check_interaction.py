"""
The gestures the MapView extraction rewired, driven through synthesized
matplotlib events.

Click, drag, the pan guard, wheel zoom, the three legend placements, the saved
figure and the deferred hillshade. It runs against either side of the refactor
-- before it the window is its own map surface, after it the surface is
`window.map_view` -- so the answer is "unchanged", not merely "works".

    GSURF_REPO=/path/to/gSurf QT_QPA_PLATFORM=offscreen python check_interaction.py
"""

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(os.environ.get("GSURF_REPO", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(REPO))

from checks.synthetic import open_window, synthetic_dem  # noqa: E402

FAILURES = []


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}{'   ' + detail if detail else ''}")
    if not condition:
        FAILURES.append(label)


def surface(window):
    """Whatever owns the canvas: the MapView, or the window before it existed."""

    return getattr(window, "map_view", window)


def placements(ri):
    holder = getattr(ri, "MapView", ri.RealtimeWindow)

    return [mode for _, mode in holder.LEGEND_PLACEMENTS]


def display_xy(window, x, y):
    return surface(window).axes.transData.transform((x, y))


def mouse(window, name, x, y, button=1, step=0):
    """Fires a matplotlib mouse event at the map coordinate (x, y).

    The event carries integer display pixels, so the coordinate that comes out
    the other side is the click rounded to the screen -- which is what a real
    click is too. Positions are therefore compared in pixels, not in metres.
    """

    from matplotlib.backend_bases import MouseEvent

    canvas = surface(window).canvas
    px, py = display_xy(window, x, y)

    event = MouseEvent(name, canvas, int(px), int(py), button=button, step=step)
    canvas.callbacks.process(name, event)


def within_a_pixel(window, wanted, tolerance_px=2.0):
    """True if the source point sits where the click landed, to the pixel."""

    import math

    wx, wy = display_xy(window, *wanted)
    px, py = display_xy(window, *window.source_point[:2])

    return math.hypot(px - wx, py - wy) <= tolerance_px


def main():
    from matplotlib.backend_bases import MouseEvent
    from PyQt6 import QtWidgets

    from gsurf.tools import intersection as ri

    print(f"repository  {REPO}")
    print(f"surface     {'MapView' if hasattr(ri, 'MapView') else 'RealtimeWindow'}\n")

    app = QtWidgets.QApplication(sys.argv)

    with tempfile.TemporaryDirectory() as tmp:
        # Large enough that the background starts out decimated: on a DEM that
        # fits under display_max there is no reload to observe.
        source, window = open_window(
            ri,
            synthetic_dem(Path(tmp) / "synthetic.tif", side=3400),
            side=400,
            attitude=(90.0, 40.0),
        )
        dem = window.dem

        window.resize(1000, 800)
        window.show()
        app.processEvents()

        view = surface(window)
        left, right, bottom, top = dem.extent

        check("the background starts decimated", dem.decimation > 1, f"1:{dem.decimation}")

        # -- a click away from the point moves it there ---------------------
        target = (left + (right - left) * 0.3, bottom + (top - bottom) * 0.3)
        mouse(window, "button_press_event", *target)
        app.processEvents()

        check("click moves the source point", within_a_pixel(window, target),
              f"({window.source_point[0]:.0f}, {window.source_point[1]:.0f})")

        check("the boxes followed the click",
              abs(window.easting_spin.value() - window.source_point[0]) < 1.0)

        offset_after_click = window.window.offset

        # -- a press on the point starts a drag ------------------------------
        start = tuple(window.source_point[:2])
        mouse(window, "button_press_event", *start)
        check("a press on the point starts a drag", window.dragging is True)

        dragged_to = (start[0] + (right - left) * 0.05, start[1] - (top - bottom) * 0.05)
        mouse(window, "motion_notify_event", *dragged_to)
        app.processEvents()

        check("motion carries the point along", within_a_pixel(window, dragged_to))

        check("the window stays put during the drag",
              window.window.offset == offset_after_click,
              str(window.window.offset))

        mouse(window, "button_release_event", *dragged_to)
        app.processEvents()

        check("release ends the drag", window.dragging is False)
        check("release recentres the window",
              window.window.offset != offset_after_click,
              str(window.window.offset))

        # -- motion with no press does nothing --------------------------------
        resting = tuple(window.source_point[:2])
        mouse(window, "motion_notify_event", left + (right - left) * 0.8, top - 10.0)
        app.processEvents()
        check("motion without a press is ignored",
              tuple(window.source_point[:2]) == resting)

        # -- pan mode must swallow the gesture --------------------------------
        view.toolbar.pan()
        before = tuple(window.source_point[:2])
        mouse(window, "button_press_event", left + (right - left) * 0.7, bottom + 100.0)
        app.processEvents()
        check("pan mode swallows the click", tuple(window.source_point[:2]) == before)
        view.toolbar.pan()  # back off

        # -- the wheel zooms around the cursor --------------------------------
        span_before = view.axes.get_xlim()[1] - view.axes.get_xlim()[0]
        for _ in range(3):
            mouse(window, "scroll_event", *window.source_point[:2], button="up", step=1)
        app.processEvents()
        span_after = view.axes.get_xlim()[1] - view.axes.get_xlim()[0]
        check("the wheel zooms in", span_after < span_before,
              f"{span_before:.0f} m -> {span_after:.0f} m")

        check("zooming schedules the background reload", view.shade_timer.isActive())

        # The timer would fire 180 ms from now; here it is called outright.
        step_before = view.shade_step
        extent_before = list(view.shade_image.get_extent())
        view._refresh_shade()
        app.processEvents()

        check("the background is reread at the zoomed scale",
              view.shade_step < step_before,
              f"1:{step_before} -> 1:{view.shade_step}")
        check("and reread over the framed portion alone",
              list(view.shade_image.get_extent()) != extent_before)
        check("without moving the view",
              abs((view.axes.get_xlim()[1] - view.axes.get_xlim()[0]) - span_after) < 1.0)

        # -- the legend, in its three places ----------------------------------
        for mode in ("inside", "hidden", "beside"):
            window.legend_combo.setCurrentIndex(placements(ri).index(mode))
            app.processEvents()

            check(f"legend '{mode}'", (view.legend is not None) is (mode != "hidden"))

        check("the legend carries the tool's own entries",
              any(t.get_text() == "intersection" for t in view.legend.get_texts()))

        check("the way back from a switched-off category starts disabled",
              not window.legend_controls.show_all.isEnabled())

        # -- and a click on one of its entries switches it ----------------------
        # This session has no vector backdrop, so every entry is one this tool
        # drew: the intersection line and the compute window, which are what the
        # hand is steering. Those stay inert -- a window circle switched off
        # would leave a handle being dragged invisible. (The fold-axes tool does
        # have switchable entries of its own; check_folds.py covers them.)
        inert = all(
            id(t) not in view._legend_switches for t in view.legend.get_texts()
        )
        check("the entries a tool steers with are not clickable", inert,
              f"{len(view._legend_switches)} switches wired")

        # A click on a legend placed inside the map lands inside the axes too:
        # without the guard it would switch the entry and move the point at once.
        window.legend_combo.setCurrentIndex(placements(ri).index("inside"))
        app.processEvents()
        view.canvas.draw()
        app.processEvents()

        text = view.legend.get_texts()[0]
        box = text.get_window_extent(view.canvas.get_renderer())
        before = tuple(window.source_point[:2])

        event = MouseEvent(
            "button_press_event", view.canvas,
            int(box.x0 + box.width / 2), int(box.y0 + box.height / 2), button=1,
        )
        view.canvas.callbacks.process("button_press_event", event)
        app.processEvents()

        check("a click on a legend inside the map does not move the point",
              tuple(window.source_point[:2]) == before,
              f"({window.source_point[0]:.0f}, {window.source_point[1]:.0f})")

        window.legend_combo.setCurrentIndex(placements(ri).index("beside"))
        app.processEvents()

        # -- the saved figure keeps the trace ---------------------------------
        out = Path(tmp) / "shot.png"
        render = getattr(view, "rendered_figure", None) or window._rendered_figure
        render(out, dpi=60)
        check("the screenshot is written", out.exists() and out.stat().st_size > 5000,
              f"{out.stat().st_size} bytes")
        check("the animated flags come back on",
              all(a.get_animated() for a in (window.intersections,
                                             window.source_marker,
                                             window.window_patch)))

        # -- and the loop still runs ------------------------------------------
        window.set_dip_direction(200.0)
        app.processEvents()
        check("the intersection still recomputes",
              len(window.last_result[0]) > 0,
              f"{len(window.last_result[0])} points")

        source.close()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed: {', '.join(FAILURES)}")
        return 1

    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
