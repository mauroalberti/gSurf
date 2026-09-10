"""
Frame cost of the real-time intersection, off-screen.

Builds a synthetic DEM, opens the window as `main()` would, then sweeps the dip
direction dial for N frames and reports what the status bar reports. The point
is to have the same number before and after a refactor.

    QT_QPA_PLATFORM=offscreen python bench_realtime.py [--side 1000] [--frames 60]
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import rasterio
from rasterio.transform import from_origin

REPO = Path(os.environ.get("GSURF_REPO", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(REPO))


def synthetic_dem(path, side=2000, cell=5.0):
    """A sinusoidal surface: smooth, so the chord count stays predictable."""

    y, x = np.mgrid[0:side, 0:side].astype(np.float32)
    z = (
        800.0
        + 300.0 * np.sin(x / 180.0) * np.cos(y / 220.0)
        + 60.0 * np.sin(x / 37.0)
    ).astype(np.float32)

    profile = dict(
        driver="GTiff",
        height=side,
        width=side,
        count=1,
        dtype="float32",
        crs="EPSG:25833",
        transform=from_origin(600000.0, 4420000.0, cell, cell),
        nodata=-9999.0,
        tiled=True,
        compress="lzw",
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(z, 1)

    return path


def open_window(ri, dem_path, **kwargs):
    """The window, either side of the Session step. Returns (closeable, window)."""

    if hasattr(ri, "Session"):
        session = ri.Session.open(dem_path=dem_path)
        return session, ri.RealtimeWindow(session, **kwargs)

    dem = ri.Dem(dem_path)
    return dem, ri.RealtimeWindow(dem, ri.Overlay(), **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=1000, help="compute window side, in cells")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--dem-side", type=int, default=2000)
    args = parser.parse_args()

    from PyQt6 import QtWidgets

    import realtime_intersection as ri

    app = QtWidgets.QApplication(sys.argv)

    with tempfile.TemporaryDirectory() as tmp:
        dem_path = synthetic_dem(Path(tmp) / "synthetic.tif", side=args.dem_side)

        opened = perf_counter()
        source, window = open_window(
            ri, dem_path, side=args.side, attitude=(90.0, 40.0)
        )
        open_s = perf_counter() - opened

        window.resize(1180, 880)
        window.show()
        app.processEvents()

        # Warm up: the first frames pay for the blitting background.
        for _ in range(5):
            window.set_dip_direction(90.0)
            app.processEvents()

        window.frame_times.clear()

        started = perf_counter()
        for step in range(args.frames):
            window.set_dip_direction(45.0 + step * 0.7)
            app.processEvents()
        wall = perf_counter() - started

        times = list(window.frame_times)
        mean = sum(times) / len(times)
        points, segments = window.last_result

        print(f"open (DEM+window)  {open_s * 1000:7.1f} ms")
        print(f"window             {args.side}x{args.side}")
        print(f"intersection       {len(points)} points, {len(segments)} chords")
        print(f"frame (mean)       {mean * 1000:7.2f} ms   {1.0 / mean:6.1f} fps")
        print(f"frame (median)     {sorted(times)[len(times) // 2] * 1000:7.2f} ms")
        print(f"wall / {args.frames} frames  {wall * 1000:7.1f} ms")
        print(f"status bar         {window.statusBar().currentMessage()}")

        source.close()


if __name__ == "__main__":
    main()
