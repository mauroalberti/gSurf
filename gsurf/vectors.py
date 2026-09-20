"""
The vector layers under whatever is being computed.

None of them is ever necessary: they answer where the work is being done, which
is a different question from the calculation and one no raster answers. They
are drawn once into the background that blitting recaptures, so per frame they
cost nothing.
"""

from __future__ import annotations

from pathlib import Path


def single_parts(geometry, wanted):
    """
    The single-part pieces of one geometry of the wanted type, and how many
    pieces were of some other type.

    Flattened by recursion, because the containers nest and their names do not
    say so: a GeometryCollection holds MultiPolygons as readily as Polygons,
    and it is the one container whose type does not begin with 'Multi'. A test
    on the outer type therefore walks a collection straight into `.exterior`,
    which is how the sections tool came down on the `carbonates` layer of
    geology.gpkg -- one unit there, the Conglomerato di Santa Croce, is stored
    as its polygon plus two dangling edges 4 and 56 m long.

    What is of another type is counted rather than converted: a 4 m dangle put
    in with the lines would be drawn across a section as a mapped contact. The
    count is what lets the caller say so instead of swallowing it.
    """

    if geometry is None or geometry.is_empty:
        return [], 0

    if geometry.geom_type == wanted:
        return [geometry], 0

    if not hasattr(geometry, "geoms"):
        return [], 1

    kept, skipped = [], 0

    for part in geometry.geoms:
        part_kept, part_skipped = single_parts(part, wanted)
        kept.extend(part_kept)
        skipped += part_skipped

    return kept, skipped


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
        self.without_geometry = 0

        # One artist per category, and which of them are currently off the map.
        # A layer with no categories has a single artist under the key None, and
        # is turned off the same way: from the legend, by its one entry.
        self.artists = {}
        self.hidden = set()
        self.category_order = []

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

        # Counted apart from the window, because `intersects` is false for both
        # and they are not the same news: a unit outside the area is somewhere
        # else, one with no geometry is nowhere. Five of the 236 carbonates in
        # geology.gpkg carry none, and the Calabrian CASMEZ sheet 299 -- enough
        # that a count which does not add up should say why.
        self.without_geometry = int(
            (visible.geometry.isna() | visible.geometry.is_empty).sum()
        )

        visible = visible[visible.intersects(window)]

        if visible.empty:
            self.problem = "no feature in the area"
            return

        self.frame = self._categorize(complete, visible)
        self.category_order = self._order_by_weight()

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

    def _order_by_weight(self):
        """
        The categories actually on show, heaviest first.

        Weight is area for polygons, length for lines, count for points, and the
        order matters twice. In the legend, alphabetically the cut at twelve
        would throw out Qt, PL and Op -- which are half the map -- to make room
        for AV, which is a single polygon. On the map, drawn in this order the
        large units go down first and the small ones land on top of them
        instead of under; and the two orders being the same one, a reader going
        down the legend is going down the map as well.
        """

        if not self.colors:
            return []

        frame = self.frame

        if self.role == "polygons":
            weight = frame.area
        elif self.role == "lines":
            weight = frame.length
        else:
            weight = 1.0

        return list(
            frame.assign(_gsurf_weight=weight)
            .groupby("_gsurf_category")["_gsurf_weight"]
            .sum()
            .sort_values(ascending=False)
            .index
        )

    # -- drawing -----------------------------------------------------------

    def draw(self, axes):
        """
        Draws the layer, one artist per category.

        One per category rather than one for the whole layer, so that a category
        can be taken off the map without redrawing anything: `set_visible(False)`
        on its collection is the whole cost. That is what the legend's entries
        switch, and the reason a formation can be got out of the way of what is
        drawn over it -- a field of fold axes is unreadable through twenty
        tints, and reading it is what the map is for.
        """

        table = self.CATEGORY_STYLE if self.colors else self.FLAT_STYLE
        style = dict(table[self.role])

        self.artists = {}

        if not self.colors:
            style.setdefault("label", self.layer or self.path.stem)
            self.artists[None] = self._plot(axes, self.frame, style)
        else:
            for value in self.category_order:
                group = self.frame[self.frame["_gsurf_category"] == value]
                self.artists[value] = self._plot(
                    axes, group, dict(style, color=[self.colors[value]] * len(group))
                )

        self._apply_visibility()

    def _plot(self, axes, frame, style):
        """
        Draws one group, and hands back the artist geopandas left behind.

        There is no other way to it: `GeoDataFrame.plot` returns the axes, not
        what it drew. One call leaves exactly one collection -- a
        PatchCollection, a LineCollection or a PathCollection, by role -- so the
        one that was not there before is the one.
        """

        before = len(axes.collections)
        frame.plot(ax=axes, zorder=self.ZORDER[self.role], **style)
        added = axes.collections[before:]

        return added[-1] if added else None

    # -- what is on the map ------------------------------------------------

    def _apply_visibility(self):
        for value, artist in self.artists.items():
            if artist is not None:
                artist.set_visible(value not in self.hidden)

    def is_hidden(self, values):
        """True when every category in the block is off."""

        return all(value in self.hidden for value in values)

    def toggle(self, values):
        """
        Turns a category on or off -- or a block of them together -- and says
        which way it went.

        A block wholly off comes back on; a block even partly on goes off.
        Anything subtler would leave the '+N more' entry unreadable: clicking it
        twice has to be the same as not clicking it.
        """

        values = tuple(values)
        show = self.is_hidden(values)

        for value in values:
            if show:
                self.hidden.discard(value)
            else:
                self.hidden.add(value)

        self._apply_visibility()

        return show

    def show_all(self):
        self.hidden.clear()
        self._apply_visibility()

    def _legend_label(self, value, width=28):
        name = self.labels.get(value, "")
        text = f"{value} - {name}" if name and name != value else str(value)

        return text if len(text) <= width else text[: width - 1] + "…"

    def _handle(self, label, color=None, switches=()):
        """
        The dummy artist standing in for one legend entry.

        Needed because geopandas draws with collections matplotlib cannot
        represent on its own: without these the polygons would drop out of the
        legend silently. The shape follows the role, so across three
        categorised layers you can still tell whose entry is whose.

        It carries what it switches, so that a click on the entry knows which
        artists to take off the map. The map reads that attribute and nothing
        else about this class, and an entry a tool provided simply does not have
        it.
        """

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        style = dict((self.CATEGORY_STYLE if self.colors else self.FLAT_STYLE)[self.role])
        style.pop("markersize", None)
        style.pop("color", None)

        if self.role == "polygons":
            handle = Patch(
                facecolor=color or style.pop("facecolor", "#4daf7c"), label=label, **style
            )
        elif self.role == "lines":
            handle = Line2D([], [], color=color or "#1f4fd8", label=label, **style)
        else:
            marker = style.pop("marker", "^")
            edge = style.pop("edgecolor", "#222222")

            handle = Line2D(
                [], [],
                linestyle="none",
                marker=marker,
                markerfacecolor=color or "#d95f02",
                markeredgecolor=edge,
                markersize=7,
                label=label,
            )

        handle._gsurf_switch = (self, tuple(switches))

        return handle

    def _layer_handle(self):
        """
        The layer's own entry, standing over its categories.

        It is the one click that takes a whole layer off the map. Without it a
        backdrop of twenty units costs twenty-one clicks to clear, and clearing
        it is the thing one actually wants to do -- the backdrop is there to be
        read under what is drawn over it, not instead of it.

        It doubles as the only mark of where one layer's entries end and the
        next one's begin, which a run of category names alone does not say.
        """

        from matplotlib.patches import Patch

        handle = Patch(
            facecolor="none", edgecolor="none", label=self.layer or self.path.stem
        )
        handle._gsurf_switch = (self, tuple(self.category_order))
        handle._gsurf_heading = True

        return handle

    def legend_handles(self):
        if not self.colors:
            # One entry, named after the layer, and it already switches the
            # whole of it: a heading over a single line would say nothing twice.
            # Marked as one all the same, so that bold reads as "a layer" down
            # the whole legend and plain as "a category inside the one above".
            handle = self._handle(self.layer or self.path.stem, switches=(None,))
            handle._gsurf_heading = True

            return [handle]

        # Only the categories actually on show, heaviest first: see
        # _order_by_weight for why that order and not the alphabet.
        listed = self.category_order[: self.MAX_LEGEND_ENTRIES]
        rest = self.category_order[self.MAX_LEGEND_ENTRIES :]

        handles = [self._layer_handle()]
        handles += [
            self._handle(self._legend_label(value), self.colors[value], switches=(value,))
            for value in listed
        ]

        if rest:
            from matplotlib.patches import Patch

            # The categories past the cut stay coloured on the map, and this
            # entry is what they are switched by: without it they would be the
            # only ones that cannot be taken off it.
            handle = Patch(
                facecolor="none",
                edgecolor="none",
                label=f"+{len(rest)} more in {self.role}",
            )
            handle._gsurf_switch = (self, tuple(rest))
            handles.append(handle)

        return handles

    def summary(self):
        where = self.layer or self.path.name

        if not self.is_loaded:
            return f"{self.role}: {where} skipped ({self.problem})"

        if self.colors:
            distinct = len(set(self.frame["_gsurf_category"]))

            text = (
                f"{self.role}: {where}, {len(self.frame)} in {distinct} "
                f"categories ({self.category_field})"
            )
        else:
            text = f"{self.role}: {where}, {len(self.frame)}"

        if self.without_geometry:
            text += f"; {self.without_geometry} with no geometry"

        return text


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

    def show_all(self):
        for source in self.sources:
            source.show_all()

    def hidden_count(self):
        """How many legend entries are currently switched off, over all layers."""

        return sum(len(source.hidden) for source in self.sources)

    def summary(self):
        lines = [s.summary() for s in self.sources] + [s.summary() for s in self.rejected]

        return "; ".join(lines) if lines else "no vector layer"


def split_layer(spec, role):
    """
    `path` or `path:layer`, resolved without guessing.

    A path can hold a colon of its own, so the rule is to look at the disk
    rather than read the string: if the whole thing is an existing file, it is
    a path; otherwise the last piece is peeled off and tried.
    """

    if spec is None:
        return None

    if Path(spec).exists():
        return dict(path=spec, role=role, layer=None)

    head, _, tail = spec.rpartition(":")

    if head and Path(head).exists():
        return dict(path=head, role=role, layer=tail)

    return dict(path=spec, role=role, layer=None)
