"""
Intersezione piano geologico / DEM in tempo reale.

Il kernel misah interseca un piano illimitato con la griglia e restituisce le
corde marching-squares; qui intorno c'e' il minimo che serve a guidarlo con la
mano e a vedere il risultato mentre si muove. La barra di stato riporta kernel,
disegno e fps a ogni frame, cosi' il costo resta visibile durante l'uso.

Uso:
    python realtime_intersection.py <dem.tif> [--geologia <geology.gpkg>]
                                    [--finestra N] [--assetto <file.json>]

Il DEM puo' essere grande quanto si vuole: non viene caricato in memoria. Lo
sfondo e' una overview decimata, mentre il kernel gira su una finestra a piena
risoluzione centrata sul punto di appoggio, il cui lato --finestra decide la
fluidita' (1000 px stanno sui 38 fps, 500 px sui 100).

Nella finestra:
    - quadrante e cursore per immersione e inclinazione;
    - rotella per zoomare attorno al cursore, barra di navigazione per pan,
      zoom a rettangolo e ritorno alla vista piena;
    - clic sulla mappa per spostare il punto di appoggio, oppure trascinamento
      del punto stesso (da fare con pan e zoom disattivati, altrimenti i due
      gesti sono lo stesso);
    - schermata negli appunti o su file, assetto corrente in JSON, traccia
      calcolata in shapefile.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np
import rasterio
from rasterio.windows import Window

# PyQt6 va importato prima del backend: matplotlib sceglie il binding
# guardando quello gia' presente in sys.modules.
import PyQt6.QtCore  # noqa: F401
from PyQt6 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from misah.kernels import intersect_plane_grid


def hillshade(z, dx, dy, azimuth=315.0, altitude=45.0):
    """Ombreggiatura secondo la convenzione ESRI, con le righe che vanno a sud."""

    d_row, d_col = np.gradient(z, dy, dx)
    dz_dx, dz_dy = d_col, -d_row

    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(dz_dy, -dz_dx)

    zenith = np.radians(90.0 - altitude)
    az = np.radians(360.0 - azimuth + 90.0)

    shaded = np.cos(zenith) * np.cos(slope) + np.sin(zenith) * np.sin(slope) * np.cos(az - aspect)

    return np.clip(shaded, 0.0, 1.0)


class ComputeWindow:
    """
    Il ritaglio a piena risoluzione su cui gira il kernel.

    Esiste separato dal DEM perche' il costo di un frame va con il numero di
    celle scandite, non con la dimensione del file: su un mosaico da 314 Mpx il
    kernel impiegherebbe secondi, su una finestra da 1000x1000 sta in 22 ms.
    """

    def __init__(self, data, geotransform, bounds, offset):
        self.data = data
        self.geotransform = geotransform
        self.bounds = bounds
        self.offset = offset

    @property
    def shape(self):
        return self.data.shape

    def covers(self, x, y):
        left, bottom, right, top = self.bounds

        return left <= x <= right and bottom <= y <= top

    def rectangle_xy(self):
        left, bottom, right, top = self.bounds

        return (left, bottom), right - left, top - bottom


class Dem:
    """
    Il DEM aperto senza caricarlo: overview per lo sfondo, finestre a piena
    risoluzione a richiesta.

    Un mosaico da 314 Mpx sarebbe 2,5 GB in float64, quindi tenerlo tutto in
    memoria non e' una possibilita' e nemmeno serve: il kernel legge una
    finestra per volta, e la lettura costa 4,9 ms su 1000x1000.
    """

    def __init__(self, path, display_max=1600):
        self.path = Path(path)
        self._src = rasterio.open(path)

        self.crs = self._src.crs
        self.nodata = self._src.nodata
        self.bounds = self._src.bounds
        self.width = self._src.width
        self.height = self._src.height
        self.res_x = abs(self._src.transform.a)
        self.res_y = abs(self._src.transform.e)

        self.extent = [
            self.bounds.left,
            self.bounds.right,
            self.bounds.bottom,
            self.bounds.top,
        ]

        # Lo sfondo non ha bisogno della piena risoluzione: oltre il paio di
        # migliaia di pixel non si vedrebbe comunque, e l'ombreggiatura di un
        # mosaico intero costerebbe minuti.
        self.decimation = max(1, math.ceil(max(self.width, self.height) / display_max))
        shape = (self.height // self.decimation, self.width // self.decimation)
        overview = self._src.read(1, out_shape=shape).astype(float)

        if self.nodata is not None:
            overview[overview == self.nodata] = np.nan

        self.hillshade = hillshade(
            overview,
            self.res_x * self.decimation,
            self.res_y * self.decimation,
        )
        self.z_median = float(np.nanmedian(overview))

    def close(self):
        self._src.close()

    def center(self):
        return (
            (self.bounds.left + self.bounds.right) / 2.0,
            (self.bounds.bottom + self.bounds.top) / 2.0,
        )

    def elevation_at(self, x, y):
        """Quota alla coordinata mappa, o None fuori griglia / su nodata."""

        row, col = self._src.index(x, y)
        row, col = int(row), int(col)

        if not (0 <= row < self.height and 0 <= col < self.width):
            return None

        z = float(self._src.read(1, window=Window(col, row, 1, 1))[0, 0])

        return None if self.nodata is not None and z == self.nodata else z

    def shade_for(self, xmin, xmax, ymin, ymax, max_px=1200):
        """
        Ombreggiatura della sola vista corrente, alla risoluzione che serve.

        Lo sfondo iniziale e' decimato sull'intero DEM: su un mosaico grande
        vuol dire celle da decine di metri, e zoomando resta poltiglia proprio
        mentre la traccia diventa dettagliata. Qui si rilegge la porzione
        inquadrata con la decimazione giusta per quella scala.

        Torna None se la vista e' del tutto fuori dal DEM.
        """

        left = max(xmin, self.bounds.left)
        right = min(xmax, self.bounds.right)
        bottom = max(ymin, self.bounds.bottom)
        top = min(ymax, self.bounds.top)

        if right <= left or top <= bottom:
            return None

        window = rasterio.windows.from_bounds(
            left, bottom, right, top, self._src.transform
        ).round_offsets().round_lengths()

        window = window.intersection(Window(0, 0, self.width, self.height))

        if window.width < 2 or window.height < 2:
            return None

        step = max(1, math.ceil(max(window.width, window.height) / max_px))
        shape = (max(2, int(window.height) // step), max(2, int(window.width) // step))

        band = self._src.read(1, window=window, out_shape=shape).astype(float)

        if self.nodata is not None:
            band[band == self.nodata] = np.nan

        shade = hillshade(band, self.res_x * step, self.res_y * step)
        left, bottom, right, top = rasterio.windows.bounds(window, self._src.transform)

        return shade, [left, right, bottom, top], step

    def window_at(self, x, y, side):
        """Finestra di `side` celle centrata su (x, y), tagliata sul DEM."""

        row, col = self._src.index(x, y)
        col_off = int(col) - side // 2
        row_off = int(row) - side // 2

        # Sui bordi la finestra si sposta invece di rimpicciolirsi, cosi' il
        # costo per frame resta quello annunciato ovunque la si porti.
        col_off = max(0, min(col_off, self.width - side))
        row_off = max(0, min(row_off, self.height - side))

        width = min(side, self.width)
        height = min(side, self.height)

        window = Window(col_off, row_off, width, height)
        band = self._src.read(1, window=window)
        transform = rasterio.windows.transform(window, self._src.transform)
        bounds = rasterio.windows.bounds(window, self._src.transform)

        # misah vuole f64 contiguo. I DEM reali sono spesso f32: la conversione
        # si paga qui, non dentro il ciclo.
        return ComputeWindow(
            np.ascontiguousarray(band.astype(np.float64)),
            list(transform.to_gdal()),
            bounds,
            (col_off, row_off),
        )


class GeologyOverlay:
    """
    Faglie e carbonati come facilitatori di posizionamento.

    Sono statici, quindi vanno disegnati una volta e finiscono nel fondale che
    il blitting ricattura: per frame costano zero. Il geopackage tiene i layer
    in CRS diversi fra loro (i carbonati in UTM 32N, le faglie in geografiche),
    quindi ognuno va riproiettato per conto suo su quello del DEM.
    """

    STYLES = {
        "carbonates": dict(facecolor="#4daf7c", edgecolor="#2f7a52", alpha=0.22, linewidth=0.5),
        "faults": dict(color="#1f4fd8", linewidth=1.0),
    }

    def __init__(self, path, crs, bounds, layers=("carbonates", "faults")):
        import geopandas as gpd  # pesante da importare: solo se serve davvero
        from shapely.geometry import box

        window = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

        self.layers = {}
        self.skipped = {}

        for name in layers:
            try:
                data = gpd.read_file(path, layer=name)
            except Exception as err:
                self.skipped[name] = str(err).split("\n")[0]
                continue

            if data.crs is None:
                self.skipped[name] = "CRS assente"
                continue

            visible = data.to_crs(crs)
            visible = visible[visible.intersects(window)]

            if visible.empty:
                self.skipped[name] = "nessun elemento nella finestra"
                continue

            self.layers[name] = visible

    def draw(self, axes):
        """Disegna sugli assi, senza lasciare che geopandas riscali la vista."""

        limits = axes.get_xlim(), axes.get_ylim()

        for name, data in self.layers.items():
            data.plot(ax=axes, zorder=2, label=name, **self.STYLES[name])

        axes.set_xlim(limits[0])
        axes.set_ylim(limits[1])

    def legend_handles(self):
        """
        Artisti fittizi per la legenda: geopandas disegna i poligoni con una
        collection che matplotlib non sa rappresentare da sola, e senza questi
        i carbonati sparirebbero dalla legenda in silenzio.
        """

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        handles = []

        for name in self.layers:
            style = self.STYLES[name]

            if "facecolor" in style:
                handles.append(Patch(label=name, **style))
            else:
                handles.append(Line2D([], [], label=name, **style))

        return handles

    def summary(self):
        found = ", ".join(f"{n} {len(g)}" for n, g in self.layers.items())
        missing = ", ".join(f"{n} ({why})" for n, why in self.skipped.items())

        if found and missing:
            return f"{found} -- saltati: {missing}"

        return found or f"nessun layer usabile: {missing}"


def merged_traces(points, segments):
    """
    Le corde marching-squares saldate in polilinee, con la quota.

    Il kernel restituisce migliaia di segmenti da due vertici: scritti cosi'
    sono inutilizzabili in un GIS. `linemerge` li ricuce nelle tracce continue
    che sono davvero, e conserva la Z.
    """

    from shapely.geometry import LineString
    from shapely.ops import linemerge

    if not len(segments):
        return []

    chords = [LineString(points[pair]) for pair in segments]
    merged = linemerge(chords)

    return list(merged.geoms) if hasattr(merged, "geoms") else [merged]


class Toolbar(NavigationToolbar2QT):
    """
    La barra di navigazione, con il salvataggio dirottato.

    Il pulsante di salvataggio del toolbar chiama `savefig` per conto suo, e
    quel percorso non sa nulla degli artisti animati: il file uscirebbe con la
    mappa e senza la traccia sopra. Qui va a finire nello stesso posto del
    bottone "Salva schermata", cosi' i due non possono divergere.
    """

    def __init__(self, canvas, parent, save_handler, view_changed):
        super().__init__(canvas, parent)
        self._save_handler = save_handler
        self._view_changed = view_changed

    def save_figure(self, *args):
        self._save_handler()

    # Ogni via con cui la barra cambia inquadratura deve avvisare, o lo sfondo
    # resta alla risoluzione di prima.

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


class RealtimeWindow(QtWidgets.QMainWindow):

    PICK_RADIUS_PX = 12
    ZOOM_STEP = 1.3

    # QDial mette il minimo alle ore 6, non alle 12: misurato afferrando il
    # widget e cercando la lancetta, il valore 0 punta a 181 gradi dalle ore 12
    # e il valore 180 a 360. Il verso e' orario, come l'azimut, quindi fra la
    # scala del widget e l'immersione geologica c'e' solo mezzo giro di scarto.
    DIAL_NORTH_OFFSET = 180

    def __init__(self, dem, overlay=None, side=1000, attitude=(90.0, 30.0), source=None):
        super().__init__()

        self.dem = dem
        self.overlay = overlay
        self.background = None
        self.dragging = False
        self.frame_times = deque(maxlen=20)
        self.last_result = ([], [])

        if source is None:
            cx, cy = dem.center()
            source = [cx, cy, dem.elevation_at(cx, cy) or dem.z_median]
        self.source_point = list(source)

        self.side = min(side, dem.width, dem.height)
        self.window = dem.window_at(self.source_point[0], self.source_point[1], self.side)

        self.setWindowTitle(f"gSurf - intersezione in tempo reale - {dem.path.name}")
        self._build_ui(attitude)
        self._draw_base_map()

        self.update_intersection()

    # -- costruzione ------------------------------------------------------

    def _build_ui(self, attitude):
        self.figure = Figure(figsize=(8, 8), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)

        self.toolbar = Toolbar(self.canvas, self, self.save_screenshot, self.schedule_shade_refresh)

        # Il ricarico dell'ombreggiatura si paga in decine di millisecondi:
        # troppo per farlo a ogni scatto di rotella, giusto una volta quando la
        # mano si ferma. Da qui il ritardo.
        self.shade_timer = QtCore.QTimer(self)
        self.shade_timer.setSingleShot(True)
        self.shade_timer.timeout.connect(self._refresh_shade)

        self.canvas.mpl_connect("draw_event", self._on_draw)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        self.dip_dir_dial = QtWidgets.QDial()
        self.dip_dir_dial.setRange(0, 359)
        self.dip_dir_dial.setValue(
            int(round(attitude[0] - self.DIAL_NORTH_OFFSET)) % 360
        )
        self.dip_dir_dial.setWrapping(True)
        self.dip_dir_dial.setNotchesVisible(True)
        self.dip_dir_dial.setMinimumSize(140, 140)

        self.dip_angle_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
        self.dip_angle_slider.setRange(0, 90)
        self.dip_angle_slider.setValue(int(attitude[1]))
        self.dip_angle_slider.setTickInterval(10)
        self.dip_angle_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksRight)

        self.attitude_label = QtWidgets.QLabel()
        self.attitude_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # Il punto: si ricalcola mentre si trascina, non su un bottone.
        self.dip_dir_dial.valueChanged.connect(self.update_intersection)
        self.dip_angle_slider.valueChanged.connect(self.update_intersection)

        self.side_spin = QtWidgets.QSpinBox()
        self.side_spin.setRange(100, min(4000, max(self.dem.width, self.dem.height)))
        self.side_spin.setSingleStep(100)
        self.side_spin.setValue(self.side)
        self.side_spin.setSuffix(" px")
        self.side_spin.valueChanged.connect(self._on_side_changed)

        self.side_label = QtWidgets.QLabel()
        self.side_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        controls = QtWidgets.QWidget()
        controls.setMaximumWidth(200)
        layout = QtWidgets.QVBoxLayout(controls)
        layout.addWidget(QtWidgets.QLabel("Immersione"))
        layout.addWidget(self.dip_dir_dial)
        layout.addWidget(QtWidgets.QLabel("Inclinazione"))
        layout.addWidget(self.dip_angle_slider, stretch=1)
        layout.addWidget(self.attitude_label)

        layout.addSpacing(8)
        layout.addWidget(QtWidgets.QLabel("Finestra di calcolo"))
        layout.addWidget(self.side_spin)
        layout.addWidget(self.side_label)

        layout.addSpacing(8)
        for text, slot in (
            ("Copia schermata", self.copy_screenshot),
            ("Salva schermata...", self.save_screenshot),
            ("Salva assetto...", self.save_settings),
            ("Esporta traccia...", self.export_traces),
        ):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            layout.addWidget(button)

        map_side = QtWidgets.QWidget()
        map_layout = QtWidgets.QVBoxLayout(map_side)
        map_layout.setContentsMargins(0, 0, 0, 0)
        map_layout.addWidget(self.toolbar)
        map_layout.addWidget(self.canvas, stretch=1)

        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.addWidget(map_side, stretch=1)
        main_layout.addWidget(controls)
        self.setCentralWidget(central)

        self.statusBar().showMessage(
            "rotella per zoomare; trascina il punto giallo, o clicca altrove per spostarlo"
        )

    def _draw_base_map(self):
        self.shade_image = self.axes.imshow(
            self.dem.hillshade,
            cmap="gray",
            extent=self.dem.extent,
            origin="upper",
            interpolation="bilinear",
        )
        self.shade_step = self.dem.decimation
        epsg = self.dem.crs.to_epsg() if self.dem.crs else "?"
        self.axes.set_xlabel(f"E (m, EPSG:{epsg})")
        self.axes.set_ylabel("N (m)")
        self.axes.set_aspect("equal")

        if self.overlay is not None:
            self.overlay.draw(self.axes)

        # animated=True tiene questi artisti fuori dal draw normale: li ridisegna
        # solo il blitting, che e' cio' che tiene il ciclo dentro il frame.
        #
        # Una sola Line2D con separatori NaN, non una LineCollection: le corde
        # marching-squares sono migliaia di segmenti sciolti, e per matplotlib
        # un unico percorso spezzato costa 4-5 volte meno di altrettanti path
        # separati (misurato: 2,0 ms contro 9,1 su 1000x1000).
        (self.intersections,) = self.axes.plot(
            [], [], "-", color="red", linewidth=1.2, animated=True
        )

        (self.source_marker,) = self.axes.plot(
            [self.source_point[0]],
            [self.source_point[1]],
            marker="o",
            color="yellow",
            markeredgecolor="black",
            markersize=8,
            animated=True,
        )

        corner, width, height = self.window.rectangle_xy()
        self.window_patch = Rectangle(
            corner,
            width,
            height,
            fill=False,
            edgecolor="orange",
            linestyle="--",
            linewidth=1.0,
            animated=True,
        )
        self.axes.add_patch(self.window_patch)

        # La legenda va costruita a mano: gli artisti animati non compaiono nel
        # draw normale, e i poligoni di geopandas non portano un handler.
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        handles = [
            Line2D([], [], color="red", linewidth=1.2, label="intersezione"),
            Patch(facecolor="none", edgecolor="orange", linestyle="--", label="finestra di calcolo"),
        ]
        if self.overlay is not None:
            handles.extend(self.overlay.legend_handles())
        self.axes.legend(handles=handles, loc="upper right", fontsize="small", framealpha=0.85)

        self.canvas.draw()

        # La vista piena va messa in fondo alla pila della barra, altrimenti
        # "casa" riporta al primo inquadramento che la barra ha visto passare,
        # che e' un punto qualsiasi dello zoom e non l'estensione del DEM.
        self.toolbar.update()
        self.toolbar.push_current()

    # -- interazione ------------------------------------------------------

    def _on_draw(self, event):
        """Il fondale cambia solo su resize o zoom: qui lo si ricattura."""

        self.background = self.canvas.copy_from_bbox(self.axes.bbox)
        self._draw_animated()

    def _draw_animated(self):
        self.axes.draw_artist(self.window_patch)
        self.axes.draw_artist(self.intersections)
        self.axes.draw_artist(self.source_marker)

    def _near_source(self, event):
        """Vicinanza misurata in pixel di schermo, non in metri: la soglia deve
        restare la stessa a ogni scala di zoom."""

        px, py = self.axes.transData.transform(self.source_point[:2])

        return math.hypot(event.x - px, event.y - py) <= self.PICK_RADIUS_PX

    def _move_source(self, x, y):
        z = self.dem.elevation_at(x, y)

        # Su nodata si tiene la quota precedente invece di rifiutare lo
        # spostamento: interrompere un trascinamento a meta' e' peggio che
        # appoggiare il piano a una quota vecchia di qualche pixel.
        self.source_point = [x, y, z if z is not None else self.source_point[2]]
        self.source_marker.set_data([x], [y])

        return z is not None

    def _recenter_window(self):
        """Rilegge la finestra se il punto ne e' uscito. Torna True se cambiata."""

        fresh = self.dem.window_at(self.source_point[0], self.source_point[1], self.side)

        if fresh.offset == self.window.offset:
            return False

        self.window = fresh
        corner, width, height = fresh.rectangle_xy()
        self.window_patch.set_xy(corner)
        self.window_patch.set_width(width)
        self.window_patch.set_height(height)

        return True

    def dip_direction(self):
        """L'immersione in azimut, non il numero grezzo del quadrante."""

        return float((self.dip_dir_dial.value() + self.DIAL_NORTH_OFFSET) % 360)

    def set_dip_direction(self, azimuth):
        self.dip_dir_dial.setValue(int(round(azimuth - self.DIAL_NORTH_OFFSET)) % 360)

    def dip_angle(self):
        return float(self.dip_angle_slider.value())

    def _navigating(self):
        """Vero mentre pan o zoom-rettangolo sono attivi nella barra.

        Senza questo controllo un pan trascinerebbe anche il punto di appoggio,
        perche' i due gesti sono lo stesso: tasto sinistro premuto e mosso."""

        return bool(self.toolbar.mode)

    def _on_press(self, event):
        if self._navigating() or event.inaxes is not self.axes or event.xdata is None:
            return

        if self._near_source(event):
            self.dragging = True
            return

        self._move_source(event.xdata, event.ydata)
        self._recenter_window()
        self.update_intersection()

    def _on_scroll(self, event):
        """Zoom attorno al cursore, che resta fermo sul punto che indicava."""

        if event.inaxes is not self.axes or event.xdata is None:
            return

        factor = 1.0 / self.ZOOM_STEP if event.button == "up" else self.ZOOM_STEP

        for axis, limits, anchor in (
            (self.axes.set_xlim, self.axes.get_xlim(), event.xdata),
            (self.axes.set_ylim, self.axes.get_ylim(), event.ydata),
        ):
            low, high = limits
            axis((anchor + (low - anchor) * factor, anchor + (high - anchor) * factor))

        # Ogni scatto entra nella pila, cosi' le frecce avanti/indietro della
        # barra ripercorrono anche gli zoom fatti con la rotella.
        self.toolbar.push_current()

        # Il fondale e' cambiato: serve un draw pieno, e il draw_event lo
        # ricattura per il blitting dei frame successivi.
        self.canvas.draw()
        self.schedule_shade_refresh()

    def schedule_shade_refresh(self, delay_ms=180):
        self.shade_timer.start(delay_ms)

    def _refresh_shade(self):
        """Rilegge l'ombreggiatura per la vista corrente, se cambia qualcosa."""

        xmin, xmax = self.axes.get_xlim()
        ymin, ymax = self.axes.get_ylim()

        result = self.dem.shade_for(xmin, xmax, ymin, ymax)
        if result is None:
            return

        shade, extent, step = result
        if step == self.shade_step and extent == list(self.shade_image.get_extent()):
            return

        started = perf_counter()

        # set_extent riscala gli assi se lo si lascia fare, e la vista
        # salterebbe a ogni ricarico: i limiti vanno rimessi come stavano.
        limits = self.axes.get_xlim(), self.axes.get_ylim()
        self.shade_image.set_data(shade)
        self.shade_image.set_extent(extent)
        self.axes.set_xlim(limits[0])
        self.axes.set_ylim(limits[1])
        self.shade_step = step

        self.canvas.draw()

        metres = self.dem.res_x * step
        self.statusBar().showMessage(
            f"sfondo ridisegnato a {metres:.0f} m/cella in {(perf_counter() - started) * 1000:.0f} ms"
        )

    def _on_motion(self, event):
        if not self.dragging or event.inaxes is not self.axes or event.xdata is None:
            return

        # Durante il trascinamento la finestra resta ferma: rileggerla a ogni
        # passo costerebbe 4,9 ms su 1000x1000, e la traccia dentro la finestra
        # e' corretta comunque, perche' il piano e' illimitato e il punto di
        # appoggio non deve starci dentro. Si ricentra al rilascio.
        self._move_source(event.xdata, event.ydata)
        self.update_intersection()

    def _on_release(self, event):
        if not self.dragging:
            return

        self.dragging = False

        if self._recenter_window():
            self.update_intersection()

    def _on_side_changed(self, value):
        self.side = value
        self.window = self.dem.window_at(self.source_point[0], self.source_point[1], self.side)

        corner, width, height = self.window.rectangle_xy()
        self.window_patch.set_xy(corner)
        self.window_patch.set_width(width)
        self.window_patch.set_height(height)

        self.update_intersection()

    # -- ciclo ------------------------------------------------------------

    def update_intersection(self):
        dip_dir = self.dip_direction()
        dip_angle = self.dip_angle()

        start = perf_counter()
        points, segments = intersect_plane_grid(
            self.window.data,
            self.window.geotransform,
            self.source_point,
            dip_dir,
            dip_angle,
            self.dem.nodata,
        )
        kernel_done = perf_counter()

        self.last_result = (points, segments)

        # points e' (N, 3) in coordinate mappa, segments (M, 2) di indici. Le
        # corde diventano un percorso unico: estremo, estremo, NaN, e la NaN
        # spezza la linea fra una corda e la successiva.
        if len(segments):
            chords = points[segments][:, :, :2]
            path = np.full((len(chords) * 3, 2), np.nan)
            path[0::3] = chords[:, 0]
            path[1::3] = chords[:, 1]
            self.intersections.set_data(path[:, 0], path[:, 1])
        else:
            self.intersections.set_data([], [])

        if self.background is None:
            self.canvas.draw()
        else:
            self.canvas.restore_region(self.background)
            self._draw_animated()
            self.canvas.blit(self.axes.bbox)

        self.canvas.flush_events()
        drawn = perf_counter()

        self.frame_times.append(drawn - start)
        self._report(dip_dir, dip_angle, len(points), kernel_done - start, drawn - kernel_done)

    def _report(self, dip_dir, dip_angle, n_points, kernel_s, draw_s):
        self.attitude_label.setText(f"{dip_dir:03.0f} / {dip_angle:02.0f}")

        rows, cols = self.window.shape
        km = cols * self.dem.res_x / 1000.0
        self.side_label.setText(f"{cols}x{rows} = {km:.1f} km")

        mean_frame = sum(self.frame_times) / len(self.frame_times)
        fps = 1.0 / mean_frame if mean_frame else 0.0

        self.statusBar().showMessage(
            f"{n_points} punti   "
            f"kernel {kernel_s * 1000:5.1f} ms   "
            f"disegno {draw_s * 1000:5.1f} ms   "
            f"totale {(kernel_s + draw_s) * 1000:5.1f} ms   "
            f"{fps:4.1f} fps"
        )

    # -- uscite -----------------------------------------------------------

    def _rendered_figure(self, path, dpi=150):
        """
        Salva la figura con dentro anche gli artisti animati.

        Un artista animato non partecipa al draw normale, quindi savefig da solo
        restituirebbe la mappa senza la traccia. Qui si spengono, si salva, si
        riaccendono, e il draw finale rifa' il fondale del blitting.
        """

        animated = [self.intersections, self.source_marker, self.window_patch]

        for artist in animated:
            artist.set_animated(False)

        try:
            self.figure.savefig(path, dpi=dpi)
        finally:
            for artist in animated:
                artist.set_animated(True)
            self.canvas.draw()

    def copy_screenshot(self):
        QtWidgets.QApplication.clipboard().setPixmap(self.canvas.grab())
        self.statusBar().showMessage("schermata copiata negli appunti")

    def save_screenshot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Salva schermata", str(self._suggested_name(".png")), "PNG (*.png)"
        )
        if not path:
            return

        self._rendered_figure(path)
        self.statusBar().showMessage(f"schermata salvata in {path}")

    def save_settings(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Salva assetto", str(self._suggested_name(".json")), "JSON (*.json)"
        )
        if not path:
            return

        settings = {
            "dem": str(self.dem.path),
            "dip_dir": self.dip_direction(),
            "dip_angle": self.dip_angle(),
            "source_point": [float(v) for v in self.source_point],
            "finestra_px": int(self.side),
            "epsg": self.dem.crs.to_epsg() if self.dem.crs else None,
        }

        Path(path).write_text(json.dumps(settings, indent=2), encoding="utf-8")
        self.statusBar().showMessage(f"assetto salvato in {path}")

    def export_traces(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Esporta traccia", str(self._suggested_name(".shp")), "Shapefile (*.shp)"
        )
        if not path:
            return

        import geopandas as gpd

        points, segments = self.last_result
        traces = merged_traces(points, segments)

        if not traces:
            self.statusBar().showMessage("nessuna intersezione da esportare")
            return

        dip_dir = self.dip_direction()
        dip_angle = self.dip_angle()

        frame = gpd.GeoDataFrame(
            {
                "dip_dir": [dip_dir] * len(traces),
                "dip": [dip_angle] * len(traces),
                "src_x": [self.source_point[0]] * len(traces),
                "src_y": [self.source_point[1]] * len(traces),
                "src_z": [self.source_point[2]] * len(traces),
            },
            geometry=traces,
            crs=self.dem.crs,
        )
        frame.to_file(path, driver="ESRI Shapefile")

        self.statusBar().showMessage(f"{len(traces)} tracce esportate in {path}")

    def _suggested_name(self, suffix):
        stem = self.dem.path.stem
        attitude = f"{int(self.dip_direction()):03d}-{int(self.dip_angle()):02d}"

        return self.dem.path.with_name(f"{stem}_{attitude}{suffix}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dem", help="DEM in un formato leggibile da rasterio")
    parser.add_argument(
        "--geologia",
        metavar="GPKG",
        help="geopackage da cui prendere faglie e carbonati come riferimento",
    )
    parser.add_argument(
        "--finestra",
        type=int,
        default=1000,
        metavar="N",
        help="lato in celle della finestra di calcolo (default 1000)",
    )
    parser.add_argument(
        "--assetto",
        metavar="JSON",
        help="assetto salvato da cui ripartire",
    )
    args = parser.parse_args()

    attitude = (90.0, 30.0)
    source = None
    side = args.finestra

    if args.assetto:
        saved = json.loads(Path(args.assetto).read_text(encoding="utf-8"))
        attitude = (saved["dip_dir"], saved["dip_angle"])
        source = saved["source_point"]
        side = saved.get("finestra_px", side)
        print(f"assetto: {attitude[0]:.0f}/{attitude[1]:.0f}, finestra {side} px")

    dem = Dem(args.dem)
    print(
        f"DEM {dem.width}x{dem.height} ({dem.width * dem.height / 1e6:.1f} Mpx), "
        f"EPSG:{dem.crs.to_epsg()}, nodata={dem.nodata}, "
        f"sfondo decimato 1:{dem.decimation}"
    )

    overlay = None
    if args.geologia:
        overlay = GeologyOverlay(args.geologia, dem.crs, dem.bounds)
        print(f"geologia: {overlay.summary()}")

    app = QtWidgets.QApplication(sys.argv)
    window = RealtimeWindow(dem, overlay, side=side, attitude=attitude, source=source)
    window.resize(1180, 880)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
