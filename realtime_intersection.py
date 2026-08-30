"""
Intersezione piano geologico / DEM in tempo reale.

Il kernel misah interseca un piano illimitato con la griglia e restituisce le
corde marching-squares; qui intorno c'e' il minimo che serve a guidarlo con la
mano e a vedere il risultato mentre si muove. La barra di stato riporta kernel,
disegno e fps a ogni frame, cosi' il costo resta visibile durante l'uso.

Uso:
    python realtime_intersection.py <dem.tif> [--geologia <geology.gpkg>]
                                    [--categorie CAMPO] [--finestra N]
                                    [--assetto <file.json>]

Il DEM puo' essere grande quanto si vuole: non viene caricato in memoria. Lo
sfondo e' una overview decimata, mentre il kernel gira su una finestra a piena
risoluzione centrata sul punto di appoggio, il cui lato --finestra decide la
fluidita' (1000 px stanno sui 38 fps, 500 px sui 100).

L'immersione si legge e si scrive in **azimut vero**, come la si misura sul
terreno. Il DEM pero' e' sulla griglia della proiezione, e i due nord non
coincidono: la convergenza dei meridiani viene tolta prima di chiamare il
kernel, e mostrata sotto l'assetto. Sull'Appennino meridionale vale da +0,41 a
+1,04 gradi, che su cinque chilometri di traccia sono fino a 91 metri.

Nella finestra:
    - quadrante e cursore per immersione e inclinazione;
    - rotella per zoomare attorno al cursore, barra di navigazione per pan,
      zoom a rettangolo e ritorno alla vista piena;
    - clic sulla mappa per spostare il punto di appoggio, oppure trascinamento
      del punto stesso (da fare con pan e zoom disattivati, altrimenti i due
      gesti sono lo stesso);
    - schermata negli appunti o su file, assetto corrente in JSON, traccia
      calcolata in shapefile. Il punto di appoggio esce in entrambi nelle due
      forme, proiettata e geografica.

Gli affioramenti poligonali sono colorati per unita' -- il campo lo decide
--categorie, di default `code`. I colori escono dall'elenco completo del layer
e non da quali unita' inquadri, cosi' la stessa formazione tiene il suo colore
mentre ti sposti.
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


class MeridianConvergence:
    """
    L'angolo fra il nord della carta e il nord geografico, punto per punto.

    In una proiezione le linee verticali della griglia non sono meridiani: solo
    sul meridiano centrale i due nord coincidono. Sull'Appennino meridionale in
    EPSG:25833 lo scarto va da +0,41 a +1,04 gradi, che su cinque chilometri di
    traccia sono fino a 91 metri -- venti volte la cella del DEM.

    Il valore si ricava misurandolo invece di leggerlo da una formula: si fa un
    passo di un centinaio di metri lungo il nord vero e si guarda che azimut ha
    assunto sulla griglia. Costa 8 microsecondi e vale per qualunque proiezione,
    anche quelle che non si lasciano scrivere in PROJ.

    Il segno, verificato su tre punti a quattro decimali:

        azimut_di_griglia = azimut_vero - convergenza
    """

    STEP_M = 100.0

    def __init__(self, crs):
        self.available = False
        self._to_geographic = None

        if crs is None:
            return

        import pyproj

        try:
            self._to_geographic = pyproj.Transformer.from_crs(crs, 4326, always_xy=True)
            self._to_projected = pyproj.Transformer.from_crs(4326, crs, always_xy=True)
            self._geod = pyproj.CRS.from_user_input(crs).get_geod()
        except Exception:
            return

        self.available = self._geod is not None

    def at(self, x, y):
        """Convergenza in gradi nel punto, positiva a est del meridiano centrale."""

        if not self.available:
            return 0.0

        lon, lat = self._to_geographic.transform(x, y)
        lon_n, lat_n, _ = self._geod.fwd(lon, lat, 0.0, self.STEP_M)
        x_n, y_n = self._to_projected.transform(lon_n, lat_n)

        return -math.degrees(math.atan2(x_n - x, y_n - y))

    def to_grid(self, true_azimuth, x, y):
        return (true_azimuth - self.at(x, y)) % 360.0

    def geographic(self, x, y):
        """
        Longitudine e latitudine del punto, o None senza un CRS utilizzabile.

        La trasformazione esiste gia' perche' serve alla convergenza a ogni
        frame: qui viene solo esposta, perche' un punto scritto nelle sole
        coordinate proiettate e' inutilizzabile fuori dal suo EPSG -- in un
        taccuino, in un GPS, in un articolo.

        Fuori dal dominio della proiezione pyproj restituisce infinito invece
        di sollevare: il controllo sui finiti e' quello che distingue il fuori
        campo da una coordinata buona.
        """

        if self._to_geographic is None:
            return None

        lon, lat = self._to_geographic.transform(x, y)

        if not (math.isfinite(lon) and math.isfinite(lat)):
            return None

        return lon, lat


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
    Faglie e affioramenti come facilitatori di posizionamento.

    Sono statici, quindi vanno disegnati una volta e finiscono nel fondale che
    il blitting ricattura: per frame costano zero. Il geopackage tiene i layer
    in CRS diversi fra loro (i carbonati in UTM 32N, le faglie in geografiche),
    quindi ognuno va riproiettato per conto suo su quello del DEM.

    I poligoni si colorano per unita' invece che in blocco: appoggiare un piano
    a un contatto vuol dire sapere quali due unita' lo fanno, e un campo verde
    uniforme quel contatto non lo mostra. Le linee restano di un colore solo --
    quattrocento faglie divise in ventitre colori non si leggono.
    """

    STYLES = {
        "carbonates": dict(facecolor="#4daf7c", edgecolor="#2f7a52", alpha=0.22, linewidth=0.5),
        "faults": dict(color="#1f4fd8", linewidth=1.0),
    }

    CATEGORY_STYLE = dict(edgecolor="#333333", linewidth=0.4, alpha=0.38)

    # Per un layer che non e' in STYLES e che la categorizzazione non prende:
    # un grigio qualsiasi disegnato e' meglio di un KeyError a meta' mappa.
    DEFAULT_STYLE = dict(facecolor="#9e9e9e", edgecolor="#5e5e5e", alpha=0.25, linewidth=0.5)

    # Il campo che tappa i buchi di quello scelto: in geology.gpkg cinque
    # poligoni non hanno `code`, e senza fallback finirebbero tutti in un'unica
    # categoria "n.d." che ne mescola tre di diverse.
    CATEGORY_FALLBACK = "name"

    # Oltre una dozzina di voci la legenda mangia la mappa che dovrebbe
    # spiegare; le unita' in eccesso restano colorate, solo non elencate.
    MAX_LEGEND_ENTRIES = 12

    def __init__(
        self,
        path,
        crs,
        bounds,
        layers=("carbonates", "faults"),
        category_field="code",
    ):
        import geopandas as gpd  # pesante da importare: solo se serve davvero
        from shapely.geometry import box

        window = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

        self.category_field = category_field
        self.layers = {}
        self.skipped = {}
        self.category_colors = {}
        self.category_labels = {}

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

            self.layers[name] = self._categorize(name, data, visible)

    @property
    def is_categorized(self):
        return bool(self.category_colors)

    def _values(self, frame):
        """La colonna su cui distinguere, con i buchi tappati dal nome."""

        if not self.category_field or self.category_field not in frame.columns:
            return None

        values = frame[self.category_field].astype("string")

        if self.CATEGORY_FALLBACK in frame.columns:
            values = values.fillna(frame[self.CATEGORY_FALLBACK].astype("string"))

        return values.fillna("n.d.").astype(str)

    def _categorize(self, layer, complete, visible):
        """
        Assegna un colore per unita', deciso sull'elenco completo del layer.

        Sull'elenco completo e non su quello visibile di proposito: se i colori
        uscissero da quali unita' capitano nella finestra, la stessa formazione
        cambierebbe colore spostandosi o cambiando DEM, ed e' l'unica cosa che
        una legenda non puo' permettersi.
        """

        if not visible.geom_type.astype(str).str.endswith("Polygon").any():
            return visible

        values = self._values(complete)

        if values is None:
            return visible

        from matplotlib import colormaps

        # Venti piu' venti: le unita' cartografate qui sono ventitre, e con la
        # sola tab20 due di esse uscirebbero identiche.
        wheel = list(colormaps["tab20"].colors) + list(colormaps["tab20b"].colors)
        order = sorted(values.unique())

        self.category_colors[layer] = {
            value: wheel[i % len(wheel)] for i, value in enumerate(order)
        }

        if self.CATEGORY_FALLBACK in complete.columns and self.category_field != self.CATEGORY_FALLBACK:
            named = complete[self.CATEGORY_FALLBACK].astype("string")
            self.category_labels[layer] = {
                value: (group.dropna().iloc[0] if len(group.dropna()) else "")
                for value, group in named.groupby(values)
            }

        return visible.assign(_gsurf_category=self._values(visible))

    def draw(self, axes):
        """Disegna sugli assi, senza lasciare che geopandas riscali la vista."""

        limits = axes.get_xlim(), axes.get_ylim()

        for name, data in self.layers.items():
            colors = self.category_colors.get(name)

            if colors:
                data.plot(
                    ax=axes,
                    zorder=2,
                    color=[colors[v] for v in data["_gsurf_category"]],
                    **self.CATEGORY_STYLE,
                )
            else:
                data.plot(
                    ax=axes,
                    zorder=2,
                    label=name,
                    **self.STYLES.get(name, self.DEFAULT_STYLE),
                )

        axes.set_xlim(limits[0])
        axes.set_ylim(limits[1])

    def _legend_label(self, layer, value, width=28):
        name = self.category_labels.get(layer, {}).get(value, "")
        text = f"{value} - {name}" if name and name != value else str(value)

        return text if len(text) <= width else text[: width - 1] + "…"

    def legend_handles(self):
        """
        Artisti fittizi per la legenda: geopandas disegna i poligoni con una
        collection che matplotlib non sa rappresentare da sola, e senza questi
        i carbonati sparirebbero dalla legenda in silenzio.
        """

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        handles = []

        for name, data in self.layers.items():
            colors = self.category_colors.get(name)

            if not colors:
                style = self.STYLES.get(name, self.DEFAULT_STYLE)

                if "facecolor" in style:
                    handles.append(Patch(label=name, **style))
                else:
                    handles.append(Line2D([], [], label=name, **style))

                continue

            # In legenda solo le unita' che si vedono davvero, e in ordine di
            # superficie affiorante: in ordine alfabetico il taglio a dodici
            # butterebbe fuori Qt, PL e Op -- che sono meta' della mappa --
            # per far posto ad AV, che e' un poligono solo.
            present = list(
                data.assign(_gsurf_area=data.area)
                .groupby("_gsurf_category")["_gsurf_area"]
                .sum()
                .sort_values(ascending=False)
                .index
            )

            for value in present[: self.MAX_LEGEND_ENTRIES]:
                handles.append(
                    Patch(
                        facecolor=colors[value],
                        label=self._legend_label(name, value),
                        **self.CATEGORY_STYLE,
                    )
                )

            if len(present) > self.MAX_LEGEND_ENTRIES:
                handles.append(
                    Patch(
                        facecolor="none",
                        edgecolor="none",
                        label=f"+{len(present) - self.MAX_LEGEND_ENTRIES} altre unita'",
                    )
                )

        return handles

    def summary(self):
        parts = []

        for name, data in self.layers.items():
            colors = self.category_colors.get(name)

            if colors:
                distinct = len(set(data["_gsurf_category"]))
                parts.append(f"{name} {len(data)} in {distinct} unita' ({self.category_field})")
            else:
                parts.append(f"{name} {len(data)}")

        found = ", ".join(parts)
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
        self.convergence = MeridianConvergence(dem.crs)
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

        # Quadrante per la mano, casella per il numero. Il quadrante da solo va
        # a passo di un grado, e la convergenza dei meridiani qui e' 0,8: senza
        # il decimo di grado la correzione sarebbe piu' piccola del controllo
        # che dovrebbe correggere, e quindi inutile. La casella e' la fonte
        # autorevole, il quadrante la insegue.
        self.dip_dir_dial = QtWidgets.QDial()
        self.dip_dir_dial.setRange(0, 359)
        self.dip_dir_dial.setWrapping(True)
        self.dip_dir_dial.setNotchesVisible(True)
        self.dip_dir_dial.setMinimumSize(140, 140)

        self.dip_dir_spin = QtWidgets.QDoubleSpinBox()
        self.dip_dir_spin.setRange(0.0, 359.9)
        self.dip_dir_spin.setDecimals(1)
        self.dip_dir_spin.setSingleStep(0.1)
        self.dip_dir_spin.setWrapping(True)
        self.dip_dir_spin.setSuffix("°  immersione")

        self.dip_angle_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
        self.dip_angle_slider.setRange(0, 90)
        self.dip_angle_slider.setTickInterval(10)
        self.dip_angle_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksRight)

        self.dip_angle_spin = QtWidgets.QDoubleSpinBox()
        self.dip_angle_spin.setRange(0.0, 90.0)
        self.dip_angle_spin.setDecimals(1)
        self.dip_angle_spin.setSingleStep(0.1)
        self.dip_angle_spin.setSuffix("°  inclinazione")

        self.attitude_label = QtWidgets.QLabel()
        self.attitude_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.convergence_label = QtWidgets.QLabel()
        self.convergence_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.convergence_label.setStyleSheet("color: gray; font-size: 10px;")

        self.dip_dir_spin.setValue(float(attitude[0]) % 360.0)
        self.dip_angle_spin.setValue(float(attitude[1]))
        self._sync_dial_from_spin()
        self._sync_slider_from_spin()

        # Il punto: si ricalcola mentre si trascina, non su un bottone.
        self.dip_dir_dial.valueChanged.connect(self._on_dial_moved)
        self.dip_dir_spin.valueChanged.connect(self._on_dip_dir_typed)
        self.dip_angle_slider.valueChanged.connect(self._on_slider_moved)
        self.dip_angle_spin.valueChanged.connect(self._on_dip_angle_typed)

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
        layout.addWidget(self.dip_dir_spin)
        layout.addWidget(QtWidgets.QLabel("Inclinazione"))
        layout.addWidget(self.dip_angle_slider, stretch=1)
        layout.addWidget(self.dip_angle_spin)
        layout.addWidget(self.attitude_label)
        layout.addWidget(self.convergence_label)

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
        categorized = self.overlay is not None and self.overlay.is_categorized

        if self.overlay is not None:
            handles.extend(self.overlay.legend_handles())

        # Con le unita' distinte le voci sono una dozzina invece di tre: al
        # corpo normale la legenda coprirebbe un quarto della mappa.
        self.axes.legend(
            handles=handles,
            loc="upper right",
            fontsize="x-small" if categorized else "small",
            framealpha=0.85,
        )

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
        """
        L'immersione come la si misura sul terreno: azimut dal nord geografico.

        E' questo il numero che il quadrante mostra e che finisce negli export,
        perche' e' quello che una bussola legge una volta corretta la
        declinazione. Il kernel invece lavora sulla griglia, e vuole
        `grid_dip_direction`.
        """

        return float(self.dip_dir_spin.value())

    def set_dip_direction(self, azimuth):
        self.dip_dir_spin.setValue(float(azimuth) % 360.0)

    def _sync_dial_from_spin(self):
        with QtCore.QSignalBlocker(self.dip_dir_dial):
            self.dip_dir_dial.setValue(
                int(round(self.dip_dir_spin.value() - self.DIAL_NORTH_OFFSET)) % 360
            )

    def _sync_slider_from_spin(self):
        with QtCore.QSignalBlocker(self.dip_angle_slider):
            self.dip_angle_slider.setValue(int(round(self.dip_angle_spin.value())))

    def _on_dial_moved(self, value):
        """Il quadrante muove la casella, che e' quella che comanda."""

        with QtCore.QSignalBlocker(self.dip_dir_spin):
            self.dip_dir_spin.setValue(float((value + self.DIAL_NORTH_OFFSET) % 360))

        self.update_intersection()

    def _on_dip_dir_typed(self, value):
        self._sync_dial_from_spin()
        self.update_intersection()

    def _on_slider_moved(self, value):
        with QtCore.QSignalBlocker(self.dip_angle_spin):
            self.dip_angle_spin.setValue(float(value))

        self.update_intersection()

    def _on_dip_angle_typed(self, value):
        self._sync_slider_from_spin()
        self.update_intersection()

    def grid_dip_direction(self):
        """L'immersione ruotata sul nord della carta, che e' cio' che il DEM ha."""

        return self.convergence.to_grid(
            self.dip_direction(), self.source_point[0], self.source_point[1]
        )

    def convergence_here(self):
        return self.convergence.at(self.source_point[0], self.source_point[1])

    def source_geographic(self):
        """Il punto di appoggio in longitudine e latitudine, o None."""

        return self.convergence.geographic(self.source_point[0], self.source_point[1])

    def dip_angle(self):
        return float(self.dip_angle_spin.value())

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
        dip_angle = self.dip_angle()

        # Il quadrante e' in azimut vero, il DEM e' sulla griglia: la
        # convergenza sta in mezzo e va tolta prima di chiamare il kernel.
        convergence = self.convergence_here()
        grid_dip_dir = (self.dip_direction() - convergence) % 360.0

        start = perf_counter()
        points, segments = intersect_plane_grid(
            self.window.data,
            self.window.geotransform,
            self.source_point,
            grid_dip_dir,
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
        self._report(
            grid_dip_dir,
            convergence,
            dip_angle,
            len(points),
            kernel_done - start,
            drawn - kernel_done,
        )

    def _report(self, grid_dip_dir, convergence, dip_angle, n_points, kernel_s, draw_s):
        # Sull'etichetta il numero vero, che e' quello che si misura e si
        # scrive nel taccuino; sotto, per esteso, che cosa ne fa la griglia.
        self.attitude_label.setText(f"{self.dip_direction():03.0f} / {dip_angle:02.0f}")
        self.convergence_label.setText(
            f"nord vero\nconvergenza {convergence:+.2f}°\ngriglia {grid_dip_dir:05.1f}°"
            if self.convergence.available
            else "nord della griglia\n(convergenza non\ncalcolabile)"
        )

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

        # Il riferimento va scritto per esteso: un'immersione senza il nord a
        # cui si appoggia e' ambigua di quasi un grado, da queste parti.
        #
        # E il punto va in entrambe le forme. Le proiettate sono quelle su cui
        # gira il kernel, ma senza le geografiche il file non si legge fuori
        # dal suo EPSG -- e l'EPSG e' proprio la riga che si perde per prima.
        lon_lat = self.source_geographic()

        settings = {
            "dem": str(self.dem.path),
            "dip_dir": self.dip_direction(),
            "dip_dir_riferimento": "nord geografico",
            "dip_dir_griglia": self.grid_dip_direction(),
            "convergenza_meridiani": self.convergence_here(),
            "dip_angle": self.dip_angle(),
            "source_point": [float(v) for v in self.source_point],
            "source_point_riferimento": (
                f"EPSG:{self.dem.crs.to_epsg()}" if self.dem.crs else "sconosciuto"
            ),
            "source_lon": lon_lat[0] if lon_lat else None,
            "source_lat": lon_lat[1] if lon_lat else None,
            "source_lon_lat_riferimento": "EPSG:4326",
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

        # Nomi entro i dieci caratteri che lo shapefile concede, e i due azimut
        # entrambi presenti: chi riapre il file non deve indovinare quale nord.
        # Stessa ragione per le due coppie di coordinate: il .prj dice in che
        # EPSG stanno src_x e src_y, ma il .prj e' il file che si perde.
        lon_lat = self.source_geographic()

        frame = gpd.GeoDataFrame(
            {
                "dip_dir": [dip_dir] * len(traces),
                "dipdir_grd": [self.grid_dip_direction()] * len(traces),
                "converg": [self.convergence_here()] * len(traces),
                "dip": [dip_angle] * len(traces),
                "src_x": [self.source_point[0]] * len(traces),
                "src_y": [self.source_point[1]] * len(traces),
                "src_z": [self.source_point[2]] * len(traces),
                "src_lon": [lon_lat[0] if lon_lat else None] * len(traces),
                "src_lat": [lon_lat[1] if lon_lat else None] * len(traces),
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
        help="geopackage da cui prendere faglie e affioramenti come riferimento",
    )
    parser.add_argument(
        "--categorie",
        metavar="CAMPO",
        default="code",
        help=(
            "campo su cui colorare gli affioramenti poligonali (default 'code'); "
            "'nessuna' per il colore unico di prima"
        ),
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
        overlay = GeologyOverlay(
            args.geologia,
            dem.crs,
            dem.bounds,
            category_field=None if args.categorie == "nessuna" else args.categorie,
        )
        print(f"geologia: {overlay.summary()}")

    app = QtWidgets.QApplication(sys.argv)
    window = RealtimeWindow(dem, overlay, side=side, attitude=attitude, source=source)
    window.resize(1180, 880)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
