"""
The vector layers under whatever is being computed.

None of them is ever necessary: they answer where the work is being done, which
is a different question from the calculation and one no raster answers. They
are drawn once into the background that blitting recaptures, so per frame they
cost nothing.
"""

from __future__ import annotations

from pathlib import Path


class VectorSource:
    """
    One backdrop vector layer, in the role it was given.

    The roles are three -- polygons, lines, points -- and they are not a matter
    of style: they decide what it makes sense to ask of the layer. A field of
    polygons coloured per unit says which two formations a contact separates;
    the same twenty-three tints spread over four hundred faults cannot be read.
    So categorisation starts on for polygons and off for the rest, and it is
    the user who decides in the end.

    The layers are static: they are drawn once and end up in the background
    that blitting recaptures, so per frame they cost nothing. Each carries its
    own CRS -- in geology.gpkg the carbonates are in UTM 32N and the faults in
    geographic -- and is reprojected on its own onto the session's, never the
    file as a block.
    """

    ROLES = ("polygons", "lines", "points")

    # The OGR type suffix: 'Polygon' and 'MultiPolygon' both end in 'Polygon',
    # and so for the other two pairs. A layer with no geometry --
    # `fault_attitudes` in geology.gpkg is a pure table -- has no suffix and
    # stays out of all three roles, which is where it belongs.
    GEOMETRY_SUFFIX = {
        "polygons": "Polygon",
        "lines": "LineString",
        "points": "Point",
    }

    FLAT_STYLE = {
        "polygons": dict(facecolor="#4daf7c", edgecolor="#2f7a52", alpha=0.25, linewidth=0.5),
        "lines": dict(color="#1f4fd8", linewidth=1.0),
        "points": dict(color="#d95f02", markersize=26, marker="^", edgecolor="#4a2200"),
    }

    CATEGORY_STYLE = {
        "polygons": dict(edgecolor="#333333", linewidth=0.4, alpha=0.38),
        "lines": dict(linewidth=1.3),
        "points": dict(markersize=30, marker="^", edgecolor="#222222"),
    }

    # Points over lines, lines over polygons: the order in which a map is read.
    # With the single zorder of before, an outcrop drawn later covered the
    # faults you were using to find your way.
    ZORDER = {"polygons": 2, "lines": 3, "points": 4}

    # The field that plugs the holes in the chosen one: in geology.gpkg five
    # polygons have no `code`, and with no fallback they would end up in a
    # single "n/a" category mixing three different ones.
    CATEGORY_FALLBACK = "name"

    # Past a dozen entries the legend eats the map it is supposed to explain;
    # the categories in excess stay coloured, they are just not listed.
    MAX_LEGEND_ENTRIES = 12

    def __init__(self, path, role, crs, bounds, layer=None, category_field=None):
        import geopandas as gpd  # heavy to import: only when actually needed
        from shapely.geometry import box

        self.path = Path(path)
        self.role = role
        self.layer = layer
        self.category_field = category_field
        self.colors = {}
        self.labels = {}
        self.frame = None
        self.problem = None

        try:
            complete = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
        except Exception as err:
            self.problem = str(err).split("\n")[0]
            return

        if complete.crs is None:
            self.problem = "no CRS"
            return

        # A plain (left, bottom, right, top), which a rasterio BoundingBox also
        # is: the area is the session's, and a session need not have a raster.
        left, bottom, right, top = bounds
        window = box(left, bottom, right, top)
        visible = complete.to_crs(crs)
        visible = visible[visible.intersects(window)]

        if visible.empty:
            self.problem = "no feature in the area"
            return

        self.frame = self._categorize(complete, visible)

    # -- reading the container, without loading the data ------------------

    @staticmethod
    def candidate_layers(path, role):
        """
        The layers in the file whose geometry fits the role.

        Read from the metadata alone, so listing the layers of a half-gigabyte
        geopackage costs what listing an empty one costs -- and that is what
        lets the dialog filter while the user chooses.
        """

        import pyogrio

        suffix = VectorSource.GEOMETRY_SUFFIX[role]

        return [
            str(name)
            for name, geometry in pyogrio.list_layers(path)
            if geometry is not None and str(geometry).endswith(suffix)
        ]

    @staticmethod
    def text_fields(path, layer=None):
        """The layer's text fields: the only ones worth categorising on."""

        import pyogrio

        info = pyogrio.read_info(path, layer=layer) if layer else pyogrio.read_info(path)

        return [
            str(field)
            for field, dtype in zip(info["fields"], info["dtypes"])
            if str(dtype) == "object"
        ]

    # -- categories --------------------------------------------------------

    @property
    def is_loaded(self):
        return self.frame is not None

    def _values(self, frame):
        """The column to tell things apart by, with the holes plugged by the name."""

        if not self.category_field or self.category_field not in frame.columns:
            return None

        values = frame[self.category_field].astype("string")

        if self.CATEGORY_FALLBACK in frame.columns:
            values = values.fillna(frame[self.CATEGORY_FALLBACK].astype("string"))

        return values.fillna("n/a").astype(str)

    def _categorize(self, complete, visible):
        """
        Assigns one colour per category, decided on the layer's complete list.

        On the complete list and not on the visible one on purpose: if the
        colours came from whichever categories happen to fall in the window,
        the same formation would change colour as you pan or change DEM, and
        that is the one thing a legend cannot afford.
        """

        values = self._values(complete)

        if values is None:
            return visible

        from matplotlib import colormaps

        # Twenty plus twenty: the units mapped in geology.gpkg are twenty-three,
        # and with tab20 alone two of them would come out identical.
        wheel = list(colormaps["tab20"].colors) + list(colormaps["tab20b"].colors)
        order = sorted(values.unique())

        self.colors = {value: wheel[i % len(wheel)] for i, value in enumerate(order)}

        if self.CATEGORY_FALLBACK in complete.columns and self.category_field != self.CATEGORY_FALLBACK:
            named = complete[self.CATEGORY_FALLBACK].astype("string")
            self.labels = {
                value: (group.dropna().iloc[0] if len(group.dropna()) else "")
                for value, group in named.groupby(values)
            }

        return visible.assign(_gsurf_category=self._values(visible))

    # -- drawing -----------------------------------------------------------

    def draw(self, axes):
        table = self.CATEGORY_STYLE if self.colors else self.FLAT_STYLE
        style = dict(table[self.role])

        if self.colors:
            style["color"] = [self.colors[v] for v in self.frame["_gsurf_category"]]
        else:
            style.setdefault("label", self.layer or self.path.stem)

        self.frame.plot(ax=axes, zorder=self.ZORDER[self.role], **style)

    def _legend_label(self, value, width=28):
        name = self.labels.get(value, "")
        text = f"{value} - {name}" if name and name != value else str(value)

        return text if len(text) <= width else text[: width - 1] + "…"

    def _handle(self, label, color=None):
        """
        The dummy artist standing in for one legend entry.

        Needed because geopandas draws with collections matplotlib cannot
        represent on its own: without these the polygons would drop out of the
        legend silently. The shape follows the role, so across three
        categorised layers you can still tell whose entry is whose.
        """

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        style = dict((self.CATEGORY_STYLE if self.colors else self.FLAT_STYLE)[self.role])
        style.pop("markersize", None)
        style.pop("color", None)

        if self.role == "polygons":
            return Patch(facecolor=color or style.pop("facecolor", "#4daf7c"), label=label, **style)

        if self.role == "lines":
            return Line2D([], [], color=color or "#1f4fd8", label=label, **style)

        marker = style.pop("marker", "^")
        edge = style.pop("edgecolor", "#222222")

        return Line2D(
            [], [],
            linestyle="none",
            marker=marker,
            markerfacecolor=color or "#d95f02",
            markeredgecolor=edge,
            markersize=7,
            label=label,
        )

    def legend_handles(self):
        if not self.colors:
            return [self._handle(self.layer or self.path.stem)]

        # In the legend only the categories actually on show, and in order of
        # weight: alphabetically, the cut at twelve would throw out Qt, PL and
        # Op -- which are half the map -- to make room for AV, which is a
        # single polygon. Weight is area for polygons, length for lines, count
        # for points.
        frame = self.frame

        if self.role == "polygons":
            weight = frame.area
        elif self.role == "lines":
            weight = frame.length
        else:
            weight = 1.0

        present = list(
            frame.assign(_gsurf_weight=weight)
            .groupby("_gsurf_category")["_gsurf_weight"]
            .sum()
            .sort_values(ascending=False)
            .index
        )

        handles = [
            self._handle(self._legend_label(value), self.colors[value])
            for value in present[: self.MAX_LEGEND_ENTRIES]
        ]

        if len(present) > self.MAX_LEGEND_ENTRIES:
            from matplotlib.patches import Patch

            handles.append(
                Patch(
                    facecolor="none",
                    edgecolor="none",
                    label=f"+{len(present) - self.MAX_LEGEND_ENTRIES} more in {self.role}",
                )
            )

        return handles

    def summary(self):
        where = self.layer or self.path.name

        if not self.is_loaded:
            return f"{self.role}: {where} skipped ({self.problem})"

        if self.colors:
            distinct = len(set(self.frame["_gsurf_category"]))

            return (
                f"{self.role}: {where}, {len(self.frame)} in {distinct} "
                f"categories ({self.category_field})"
            )

        return f"{self.role}: {where}, {len(self.frame)}"


class Overlay:
    """
    The backdrop vector layers held together, in the order they are read in.

    None of them is necessary: the DEM alone is enough to intersect a plane.
    They answer where that plane is being laid down, which is a different
    question from the calculation and one the DEM does not answer.
    """

    def __init__(self, sources=()):
        sources = list(sources)

        self.sources = [s for s in sources if s.is_loaded]
        self.rejected = [s for s in sources if not s.is_loaded]

    def __bool__(self):
        return bool(self.sources)

    @property
    def is_categorized(self):
        return any(source.colors for source in self.sources)

    def draw(self, axes):
        """Draws on the axes, without letting geopandas rescale the view."""

        limits = axes.get_xlim(), axes.get_ylim()

        for source in sorted(self.sources, key=lambda s: VectorSource.ZORDER[s.role]):
            source.draw(axes)

        axes.set_xlim(limits[0])
        axes.set_ylim(limits[1])

    def legend_handles(self):
        handles = []

        for source in sorted(self.sources, key=lambda s: VectorSource.ZORDER[s.role]):
            handles.extend(source.legend_handles())

        return handles

    def summary(self):
        lines = [s.summary() for s in self.sources] + [s.summary() for s in self.rejected]

        return "; ".join(lines) if lines else "no vector layer"
