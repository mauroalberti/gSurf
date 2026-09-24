"""
What a tool has to know before it computes anything.

A projection, an area, and whatever has been opened in it. The DEM used to be
all three at once -- it carried the CRS, the extent and the centre, on top of
being the surface a plane is intersected with -- and a tool that has no use for
a topographic surface, fold axes from field attitudes being the first, would
have had to open one anyway to find out where it was.

So the session holds the frame and the DEM is one of the things in it. With no
DEM the frame comes from the layers themselves, read from their metadata: a
CRS, and the union of their extents projected onto it. Nothing is loaded to
find that out, which is what lets the answer arrive before the dialog closes.
"""

from __future__ import annotations

from pathlib import Path

from .convergence import MeridianConvergence
from .dem import Dem
from .vectors import Overlay, VectorSource


class Session:
    """
    The projection and the area, with the DEM and the vector layers in them.

    Built through `open`, which is where the order matters: the frame has to
    exist before the layers, because each of them is reprojected onto it and
    clipped to it as it is read.
    """

    def __init__(self, crs, bounds, dem=None, overlay=None, base_path=None):
        self.crs = crs
        self.bounds = tuple(float(v) for v in bounds)
        self.dem = dem
        self.overlay = overlay if overlay is not None else Overlay()

        # The path a suggested filename and a window title are built from. The
        # DEM's when there is one, because that is what the work is named after.
        self.base_path = Path(base_path) if base_path is not None else None

        # Every tool that reads a bearing off the field needs it, and it costs
        # one pyproj transformer: built here rather than once per tool.
        self.convergence = MeridianConvergence(crs)

    # -- opening -----------------------------------------------------------

    @classmethod
    def open(cls, dem_path=None, vectors=(), frame_layers=()):
        """
        Opens a session on a DEM, on some vector layers, or on both.

        At least one source: with none there is no projection and no area.

        `frame_layers` say where we are without being drawn. A tool's own data
        is usually one of these -- the fold-axis module reads its attitudes
        itself, and drawing them a second time as a backdrop would put two
        symbols on every station and an entry in the legend for the layer the
        whole window is about. Contributing to the frame and being drawn are
        two different jobs, and the same layer can do the first without the
        second.
        """

        specs = [dict(spec) for spec in vectors]

        # The frame layers go first, and that ordering is the point rather than
        # an accident: the projection is taken from the first layer that
        # declares one, and a tool's own data has a better claim to it than
        # whichever backdrop happens to have been listed first. Backdrops are
        # decoration and are often in whatever CRS they were downloaded in --
        # geographic, as often as not, which would put the whole session in
        # degrees.
        framing = [dict(spec) for spec in frame_layers] + specs

        dem = Dem(dem_path) if dem_path else None

        if dem is not None:
            crs, bounds = dem.crs, tuple(dem.bounds)
            base_path = dem.path
        elif framing:
            crs, bounds = cls._frame_from_vectors(framing)
            base_path = Path(framing[0]["path"])
        else:
            raise ValueError("a session needs a DEM or at least one vector layer")

        overlay = Overlay(
            VectorSource(
                spec["path"],
                spec["role"],
                crs,
                bounds,
                layer=spec.get("layer"),
                category_field=spec.get("category_field"),
                colors=spec.get("colors"),
                labels=spec.get("labels"),
                hidden=spec.get("hidden") or (),
            )
            for spec in specs
        )

        return cls(crs, bounds, dem=dem, overlay=overlay, base_path=base_path)

    @staticmethod
    def _frame_from_vectors(specs):
        """
        The CRS and the extent of a set of layers, from their metadata alone.

        The first layer that declares a CRS sets the projection, and the others
        are measured against it: not because it is the best choice, but because
        it is the one that can be made without opening anything. `read_info`
        answers off the header, so a half-gigabyte geopackage costs what an
        empty one costs.

        A geographic CRS is taken as it comes. It is a poor frame for a map --
        distances in degrees, and the aspect ratio wrong by the cosine of the
        latitude -- but refusing it would be refusing data over a choice that
        belongs to whoever opened it.
        """

        import pyogrio
        from pyproj import CRS, Transformer

        from .curation import frame_of, is_gstruct

        frames = []

        for spec in specs:
            # The one format here with no header to read. It is parsed instead,
            # which costs what parsing a text file costs and is still the
            # difference between a session that opens on it and one that cannot.
            if is_gstruct(spec["path"]):
                crs, bounds = frame_of(spec["path"])

                if crs is not None:
                    frames.append((crs, bounds))

                continue

            info = pyogrio.read_info(spec["path"], layer=spec.get("layer"))

            if info["crs"] is None or info["total_bounds"] is None:
                continue

            frames.append((CRS.from_user_input(info["crs"]), tuple(info["total_bounds"])))

        if not frames:
            raise ValueError("no vector layer declares both a CRS and an extent")

        crs = frames[0][0]
        left = bottom = float("inf")
        right = top = float("-inf")

        for source_crs, (x0, y0, x1, y1) in frames:
            if not source_crs.equals(crs):
                transformer = Transformer.from_crs(source_crs, crs, always_xy=True)

                # The four corners and not the two: a projected rectangle is
                # not a rectangle, and transforming the diagonal alone loses
                # whichever side bulges out.
                xs, ys = transformer.transform(
                    [x0, x1, x0, x1], [y0, y0, y1, y1]
                )
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)

            left, bottom = min(left, x0), min(bottom, y0)
            right, top = max(right, x1), max(top, y1)

        return crs, (left, bottom, right, top)

    # -- the frame ---------------------------------------------------------

    @property
    def extent(self):
        """The area as matplotlib wants it: left, right, bottom, top."""

        left, bottom, right, top = self.bounds

        return [left, right, bottom, top]

    @property
    def epsg(self):
        return self.crs.to_epsg() if self.crs else None

    def center(self):
        left, bottom, right, top = self.bounds

        return (left + right) / 2.0, (bottom + top) / 2.0

    @property
    def label(self):
        """What to put in a window title: the file the session is named after."""

        return self.base_path.name if self.base_path else "no source"

    def suggested_name(self, suffix, tag=None):
        """
        A filename next to the source, marked with what produced it.

        Exports land beside the data they came from rather than in whatever
        directory the application was started in, which is the one place nobody
        goes looking for them.
        """

        base = self.base_path or Path.cwd() / "gsurf"
        stem = base.stem + (f"_{tag}" if tag else "")

        return base.with_name(stem + suffix)

    def close(self):
        if self.dem is not None:
            self.dem.close()

    def summary(self):
        left, bottom, right, top = self.bounds
        epsg = f"EPSG:{self.epsg}" if self.epsg else "unknown CRS"
        size = f"{(right - left) / 1000.0:.1f} x {(top - bottom) / 1000.0:.1f} km"

        if self.dem is not None:
            pixels = self.dem.width * self.dem.height / 1e6
            source = (
                f"DEM {self.dem.width}x{self.dem.height} ({pixels:.1f} Mpx), "
                f"background decimated 1:{self.dem.decimation}"
            )
        else:
            source = "no DEM"

        return f"{epsg}, {size}, {source}"
