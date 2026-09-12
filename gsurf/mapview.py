"""
The map, apart from what is drawn on it.

What made the real-time intersection work is not the intersection: it is the
loop around it -- a static background captured once, a handful of animated
artists blitted over it, an expensive redraw deferred to when the hand stops,
and the cost of every frame reported while you work. Every other tool that
belongs in gSurf has that same shape and only differs in what it computes, so
the loop is here and the computation is not.

The division is by what changes per frame. `MapView` owns the figure, the
canvas, the navigation bar, the hillshade underneath, the vector backdrop, the
legend and the blitting; a tool owns its own artists, registers them with
`add_animated`, and calls `blit` when it has something new to show. Mouse
gestures arrive as signals in map coordinates -- `pressed`, `dragged`,
`released` -- rather than as matplotlib events, because deciding what a click
means is the tool's business and filtering out the ones that belong to pan and
zoom is not.
"""

from __future__ import annotations

from time import perf_counter

# PyQt6 has to be imported before the backend: matplotlib picks the binding by
# looking at what is already in sys.modules.
import PyQt6.QtCore  # noqa: F401
from PyQt6 import QtCore, QtWidgets

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure


class Toolbar(NavigationToolbar2QT):
    """
    The navigation bar, with saving diverted.

    The toolbar's save button calls `savefig` on its own, and that path knows
    nothing of the animated artists: the file would come out with the map and
    without the trace on top. Here it ends up in the same place as the "Save
    screenshot" button, so the two cannot drift apart.
    """

    def __init__(self, canvas, parent, save_handler, view_changed):
        super().__init__(canvas, parent)
        self._save_handler = save_handler
        self._view_changed = view_changed

    def save_figure(self, *args):
        self._save_handler()

    # Every way the bar can change the framing has to report it, or the
    # background stays at the previous resolution.

    def release_pan(self, event):
        super().release_pan(event)
        self._view_changed()

    def release_zoom(self, event):
        super().release_zoom(event)
        self._view_changed()

    def home(self, *args):
        super().home(*args)
        self._view_changed()

    def back(self, *args):
        super().back(*args)
        self._view_changed()

    def forward(self, *args):
        super().forward(*args)
        self._view_changed()


class ToolLayers:
    """
    What a tool drew itself, switched from the legend by the label of its entry.

    The backdrop's categories keep their state in the VectorSource they were
    read from, which outlives every legend rebuild. A tool's artists have no
    such home, so the map holds one of these for them; same three methods, so
    the legend does not have to know which of the two kinds it is switching.
    """

    def __init__(self):
        self.artists = {}
        self.hidden = set()

    def register(self, label, artists):
        """By label and not by artist: a legend entry is rebuilt on every
        refresh, while what is off the map has to survive them."""

        self.artists[label] = list(artists)
        self._apply_visibility()

    def is_hidden(self, values):
        return all(value in self.hidden for value in values)

    def toggle(self, values):
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

    def _apply_visibility(self):
        for label, artists in self.artists.items():
            for artist in artists:
                artist.set_visible(label not in self.hidden)


class MapView(QtWidgets.QWidget):
    """
    Hillshaded DEM, vector backdrop and navigation, with a blitting surface on
    top for whatever a tool needs to draw.
    """

    ZOOM_STEP = 1.3

    # Where the legend goes. Categorised it runs to some thirty entries, and
    # inside the map it covers the corner you were most likely looking at:
    # beside it the map stays whole and the legend grows in its own column.
    # Hidden is for when the tints are already known and the map just has to be
    # read -- or for a figure whose caption lists them elsewhere.
    LEGEND_PLACEMENTS = (
        ("beside the map", "beside"),
        ("inside the map", "inside"),
        ("hidden", "hidden"),
    )

    # Map coordinates, not display pixels: a tool asked to interpret a click
    # should not have to know how the axes are transformed. Only gestures that
    # are not pan or rubber-band zoom get through.
    pressed = QtCore.pyqtSignal(float, float)
    dragged = QtCore.pyqtSignal(float, float)
    released = QtCore.pyqtSignal()

    save_requested = QtCore.pyqtSignal()
    status = QtCore.pyqtSignal(str)

    # How many legend entries are off, after one has just been switched. A
    # panel showing a way back to all of them needs to know when there is one.
    categories_changed = QtCore.pyqtSignal(int)

    def __init__(self, session, legend="beside", parent=None):
        super().__init__(parent)

        self.session = session
        self.dem = session.dem
        self.overlay = session.overlay
        self.background = None
        self.legend = None
        self.shade_image = None
        self.shade_step = None

        modes = [mode for _, mode in self.LEGEND_PLACEMENTS]
        self.legend_placement = legend if legend in modes else modes[0]

        # A tool fills this in with something returning its own legend entries;
        # the backdrop's are added here. Left unset the legend shows the
        # backdrop alone, which is what a tool that draws nothing would want.
        self.legend_handles_provider = None

        # Which backdrop categories each clickable legend artist switches, by
        # id: the legend draws copies of the handles it was given, so there is
        # nothing to hang this on but the artists it made.
        self._legend_switches = {}

        # And the same for what a tool draws, which has nowhere else to live.
        self.tool_layers = ToolLayers()

        self._animated = []
        self._pressing = False

        self.figure = Figure(figsize=(8, 8), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)

        self.toolbar = Toolbar(
            self.canvas, self, self.save_requested.emit, self.schedule_shade_refresh
        )

        # Reloading the hillshade is paid for in tens of milliseconds: too much
        # for every notch of the wheel, about right once the hand stops. Hence
        # the delay.
        self.shade_timer = QtCore.QTimer(self)
        self.shade_timer.setSingleShot(True)
        self.shade_timer.timeout.connect(self._refresh_shade)

        self.canvas.mpl_connect("draw_event", self._on_draw)
        self.canvas.mpl_connect("pick_event", self._on_legend_pick)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)

    # -- construction -----------------------------------------------------

    def add_animated(self, artist):
        """
        Registers an artist as one the blitting draws, and hands it back.

        animated=True keeps an artist out of the normal draw: only blitting
        redraws it, which is what keeps the loop inside the frame. Registering
        it here is also what puts it into the saved screenshot, where the
        animated flag has to come off for the duration.
        """

        artist.set_animated(True)
        self._animated.append(artist)

        return artist

    def draw_base_map(self):
        """The background: hillshade if there is one, axis labels, backdrop."""

        left, right, bottom, top = self.session.extent

        if self.dem is not None:
            self.shade_image = self.axes.imshow(
                self.dem.hillshade,
                cmap="gray",
                extent=self.session.extent,
                origin="upper",
                interpolation="bilinear",
            )
            self.shade_step = self.dem.decimation
        else:
            # Without a raster nothing sets the limits, and the vector layers
            # would decide them one at a time as they are drawn -- the last one
            # winning. The session's area is the frame the layers were clipped
            # to, so it is the honest one to show.
            self.axes.set_xlim(left, right)
            self.axes.set_ylim(bottom, top)

        epsg = self.session.epsg or "?"
        self.axes.set_xlabel(f"E (m, EPSG:{epsg})")
        self.axes.set_ylabel("N (m)")
        self.axes.set_aspect("equal")

        if self.overlay is not None:
            self.overlay.draw(self.axes)

    def anchor_home(self):
        """
        Puts the current framing at the bottom of the navigation bar's stack.

        Called once the base map is drawn, or "home" takes you back to the first
        framing the bar happened to see, which is some arbitrary point of the
        zoom and not the extent of the DEM.
        """

        self.toolbar.update()
        self.toolbar.push_current()

    # -- legend -----------------------------------------------------------

    def set_legend_placement(self, mode):
        self.legend_placement = mode
        self.refresh_legend()

    def legend_handles(self):
        """
        The legend has to be built by hand: animated artists do not show up in
        the normal draw, and geopandas polygons carry no handler.
        """

        handles = list(self.legend_handles_provider()) if self.legend_handles_provider else []

        if self.overlay is not None:
            handles.extend(self.overlay.legend_handles())

        return handles

    def switchable(self, handle, *artists):
        """
        Lets one of a tool's own legend entries switch what it stands for.

        For the data a tool puts on the map, not for what the hand is steering:
        twelve thousand station dots hide the field drawn over them and are
        worth taking off, while a window circle switched off leaves a handle
        being dragged invisible. Which entry is which only the tool knows, so
        it is the tool that asks.
        """

        self.tool_layers.register(handle.get_label(), artists)
        handle._gsurf_switch = (self.tool_layers, (handle.get_label(),))

        return handle

    def refresh_legend(self):
        """
        Rebuilds the legend where the placement says it goes, the map included.

        Rebuilt and not hidden: a legend made invisible beside the map would
        still hold its column, and the map would stay narrow to explain
        nothing. Outside the axes it is `figure.legend` and not `axes.legend`,
        because only that way does the layout reserve the column for it instead
        of letting it spill over; and it lives in the figure, not in a Qt widget
        alongside, or it would drop out of both outputs -- the saved screenshot
        and the copied one.
        """

        if self.legend is not None:
            self.legend.remove()
            self.legend = None

        self._legend_switches = {}

        if self.legend_placement != "hidden":
            # With distinct units the entries are a dozen instead of three, and
            # at normal body size they would not fit in height.
            categorized = self.overlay is not None and self.overlay.is_categorized
            handles = self.legend_handles()
            style = dict(
                handles=handles,
                fontsize="x-small" if categorized else "small",
                framealpha=0.85,
            )

            self.legend = (
                self.axes.legend(loc="upper right", **style)
                if self.legend_placement == "inside"
                else self.figure.legend(loc="outside right upper", **style)
            )

            self._wire_legend(handles)

        # The axes box has just moved: the blitting background cut on the
        # previous one would be worth nothing now. The draw_event fired from
        # here recaptures it.
        self.canvas.draw()

    def _wire_legend(self, handles):
        """
        Makes the backdrop's entries clickable, and greys the ones switched off.

        By position and not by identity: a legend does not draw the handles it
        was given, it draws copies its handlers make, so the switch has to be
        carried across on the index. An entry with no switch stays inert, which
        is what the tools' steering artists want.

        Greyed here rather than in the handle, so that the state lives in the
        layer and its appearance in one place: every rebuild reproduces it,
        and there is no second copy to fall out of step.
        """

        for handle, key, text in zip(
            handles, self.legend.legend_handles, self.legend.get_texts()
        ):
            # A layer's own entry reads as the heading of the block under it,
            # and not as one more thing drawn on the map.
            if getattr(handle, "_gsurf_heading", False):
                text.set_fontweight("bold")

            switch = getattr(handle, "_gsurf_switch", None)

            if switch is None:
                continue

            source, values = switch

            # Both, because the swatch is a few pixels across and the label is
            # what the hand goes for.
            for artist in (key, text):
                artist.set_picker(True)
                self._legend_switches[id(artist)] = switch

            if source.is_hidden(values):
                key.set_alpha(0.25)
                text.set_color("#9a9a9a")

    def _on_legend_pick(self, event):
        """A click on a legend entry takes its category off the map, or puts it back."""

        switch = self._legend_switches.get(id(event.artist))

        if switch is None:
            return

        source, values = switch
        source.toggle(values)

        # Rebuilt rather than touched up: the rebuild ends in a full draw, and
        # that draw is what recaptures the blitting background without what has
        # just been taken off it.
        self.refresh_legend()
        self._report_hidden()

    def show_all_categories(self):
        """
        Puts every switched-off category back on the map.

        The way out of the corner: with the legend hidden as a block there is
        nothing left to click, and a category switched off before that would
        otherwise have no way back.
        """

        if self.overlay is not None:
            self.overlay.show_all()

        self.tool_layers.show_all()
        self.refresh_legend()
        self._report_hidden()

    def hidden_count(self):
        backdrop = self.overlay.hidden_count() if self.overlay is not None else 0

        return backdrop + len(self.tool_layers.hidden)

    def _report_hidden(self):
        hidden = self.hidden_count()

        self.categories_changed.emit(hidden)
        self.status.emit(
            f"{hidden} legend entries hidden" if hidden else "every category is on the map"
        )

    # -- the frame --------------------------------------------------------

    def _on_draw(self, event):
        """The background only changes on resize or zoom: here it is recaptured."""

        self.background = self.canvas.copy_from_bbox(self.axes.bbox)
        self._draw_animated()

    def _draw_animated(self):
        for artist in self._animated:
            self.axes.draw_artist(artist)

    def blit(self):
        """
        Shows what the animated artists now hold, without redrawing the map.

        Before the first draw_event there is no background to restore, and the
        only way through is the slow one.
        """

        if self.background is None:
            self.canvas.draw()
        else:
            self.canvas.restore_region(self.background)
            self._draw_animated()
            self.canvas.blit(self.axes.bbox)

        self.canvas.flush_events()

    # -- interaction ------------------------------------------------------

    def is_navigating(self):
        """True while pan or rubber-band zoom are active in the bar.

        Without this check a pan would drag a tool's handle along too, because
        the two gestures are the same one: left button held down and moved."""

        return bool(self.toolbar.mode)

    def display_xy(self, x, y):
        """Map coordinates as screen pixels, for thresholds that must not
        change with the zoom scale."""

        return self.axes.transData.transform((x, y))

    def _on_press(self, event):
        if self.is_navigating() or event.inaxes is not self.axes or event.xdata is None:
            return

        # A legend inside the map is inside the axes as well, and its entries
        # are now clickable: without this a click on one would switch the
        # category and move the tool's point at the same time.
        if self.legend is not None and self.legend.contains(event)[0]:
            return

        self._pressing = True
        self.pressed.emit(event.xdata, event.ydata)

    def _on_motion(self, event):
        if not self._pressing or event.inaxes is not self.axes or event.xdata is None:
            return

        self.dragged.emit(event.xdata, event.ydata)

    def _on_release(self, event):
        if not self._pressing:
            return

        self._pressing = False
        self.released.emit()

    def _on_scroll(self, event):
        """Zoom around the cursor, which stays put on the point it was on."""

        if event.inaxes is not self.axes or event.xdata is None:
            return

        factor = 1.0 / self.ZOOM_STEP if event.button == "up" else self.ZOOM_STEP

        for axis, limits, anchor in (
            (self.axes.set_xlim, self.axes.get_xlim(), event.xdata),
            (self.axes.set_ylim, self.axes.get_ylim(), event.ydata),
        ):
            low, high = limits
            axis((anchor + (low - anchor) * factor, anchor + (high - anchor) * factor))

        # Every notch goes on the stack, so the bar's back/forward arrows
        # retrace the zooms made with the wheel as well.
        self.toolbar.push_current()

        # The background has changed: a full draw is needed, and the draw_event
        # recaptures it for the blitting of the frames that follow.
        self.canvas.draw()
        self.schedule_shade_refresh()

    # -- background at the right scale ------------------------------------

    def schedule_shade_refresh(self, delay_ms=180):
        """
        Asks for the background at the scale the view now needs, in a moment.

        With no DEM there is no background to reread: the vector layers are
        drawn from geometry and are as sharp at any zoom, so the timer is not
        even started.
        """

        if self.dem is None:
            return

        self.shade_timer.start(delay_ms)

    def _refresh_shade(self):
        """Re-reads the hillshade for the current view, if anything changes."""

        if self.dem is None:
            return

        xmin, xmax = self.axes.get_xlim()
        ymin, ymax = self.axes.get_ylim()

        result = self.dem.shade_for(xmin, xmax, ymin, ymax)
        if result is None:
            return

        shade, extent, step = result
        if step == self.shade_step and extent == list(self.shade_image.get_extent()):
            return

        started = perf_counter()

        # set_extent rescales the axes if you let it, and the view would jump on
        # every reload: the limits have to be put back the way they were.
        limits = self.axes.get_xlim(), self.axes.get_ylim()
        self.shade_image.set_data(shade)
        self.shade_image.set_extent(extent)
        self.axes.set_xlim(limits[0])
        self.axes.set_ylim(limits[1])
        self.shade_step = step

        self.canvas.draw()

        metres = self.dem.res_x * step
        self.status.emit(
            f"background redrawn at {metres:.0f} m/cell in "
            f"{(perf_counter() - started) * 1000:.0f} ms"
        )

    # -- outputs ----------------------------------------------------------

    def rendered_figure(self, path, dpi=150):
        """
        Saves the figure with the animated artists in it too.

        An animated artist takes no part in the normal draw, so savefig on its
        own would give back the map without the trace. Here they are turned
        off, it is saved, they are turned back on, and the final draw remakes
        the blitting background.
        """

        for artist in self._animated:
            artist.set_animated(False)

        try:
            self.figure.savefig(path, dpi=dpi)
        finally:
            for artist in self._animated:
                artist.set_animated(True)
            self.canvas.draw()

    def copy_screenshot(self):
        QtWidgets.QApplication.clipboard().setPixmap(self.canvas.grab())
        self.status.emit("screenshot copied to the clipboard")


class LegendControls(QtWidgets.QWidget):
    """
    Where the legend goes, and the way back from a category switched off.

    Both tools want the same two controls, so they are here and not twice over
    in the panels. The placement includes "hidden", which takes the whole legend
    off in one go: on a backdrop of twenty tints that is the difference between
    reading the map and reading the legend.
    """

    def __init__(self, map_view, placement="beside", parent=None):
        super().__init__(parent)

        self.map_view = map_view

        self.combo = QtWidgets.QComboBox()
        for text, mode in MapView.LEGEND_PLACEMENTS:
            self.combo.addItem(text, mode)

        modes = [mode for _, mode in MapView.LEGEND_PLACEMENTS]
        self.combo.setCurrentIndex(modes.index(placement) if placement in modes else 0)
        self.combo.currentIndexChanged.connect(
            lambda _: map_view.set_legend_placement(self.combo.currentData())
        )

        # Enabled only when there is something to bring back -- and it is the
        # only way back once the legend itself is hidden, there being nothing
        # left to click on.
        self.show_all = QtWidgets.QPushButton("Show all categories")
        self.show_all.setEnabled(False)
        self.show_all.clicked.connect(map_view.show_all_categories)
        map_view.categories_changed.connect(
            lambda hidden: self.show_all.setEnabled(hidden > 0)
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(QtWidgets.QLabel("Legend"))
        layout.addWidget(self.combo)
        layout.addWidget(self.show_all)


def fit_to_screen(window, width, height):
    """
    Shows the window at the wanted size, or maximised if it does not fit.

    1180x880 is the right measure for the map plus the panel, but on a 1366x768
    screen the bottom of the window -- the buttons and the status bar -- ends up
    under the edge, and unlike the buttons the status bar cannot be reached by
    scrolling the panel.

    Maximised rather than resized to the available area, because the window
    frame is not ours to measure: right after show() the title bar does not
    exist yet, and it only appears once the window manager has reparented the
    window, some hundred milliseconds later -- a delay there is no honest way to
    wait for. Maximising hands the arithmetic to the window manager, which knows
    how thick its own decorations are. The one case this leaves rough is a
    screen tall enough for the client area but not for the title bar too, where
    the window sticks out by the height of that bar.
    """

    available = window.screen().availableGeometry()

    if width > available.width() or height > available.height():
        window.showMaximized()
        return

    window.resize(width, height)
    window.show()
