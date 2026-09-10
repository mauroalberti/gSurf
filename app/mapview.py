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

        if self.legend_placement != "hidden":
            # With distinct units the entries are a dozen instead of three, and
            # at normal body size they would not fit in height.
            categorized = self.overlay is not None and self.overlay.is_categorized
            style = dict(
                handles=self.legend_handles(),
                fontsize="x-small" if categorized else "small",
                framealpha=0.85,
            )

            self.legend = (
                self.axes.legend(loc="upper right", **style)
                if self.legend_placement == "inside"
                else self.figure.legend(loc="outside right upper", **style)
            )

        # The axes box has just moved: the blitting background cut on the
        # previous one would be worth nothing now. The draw_event fired from
        # here recaptures it.
        self.canvas.draw()

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
