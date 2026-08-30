"""
Intersezione piano geologico / DEM in tempo reale.

Il kernel misah interseca un piano illimitato con la griglia e restituisce le
corde marching-squares; qui intorno c'e' il minimo che serve a guidarlo con la
mano e a vedere il risultato mentre si muove. La barra di stato riporta kernel,
disegno e fps a ogni frame, cosi' il costo resta visibile durante l'uso.

Uso:
    python realtime_intersection.py
    python realtime_intersection.py <dem.tif> [--poligoni PATH[:LAYER]]
                                    [--linee PATH[:LAYER]] [--punti PATH[:LAYER]]
                                    [--categorie CAMPO] [--x E] [--y N] [--z Q]
                                    [--finestra N] [--assetto <file.json>]

Senza argomenti si apre un dialogo che chiede le stesse cose. L'unica
obbligatoria e' il DEM: i tre slot vettoriali -- poligoni, linee, punti --
servono a sapere dove si sta appoggiando il piano, che e' una domanda diversa
dal calcolo. I layer offerti in ciascuno slot sono filtrati sulla geometria,
letta dai metadati, quindi fra i poligoni le faglie non compaiono.

Il DEM puo' essere grande quanto si vuole: non viene caricato in memoria. Lo
sfondo e' una overview decimata, mentre il kernel gira su una finestra a piena
risoluzione centrata sul punto di appoggio, il cui lato --finestra decide la
fluidita' (1000 px stanno sui 38 fps, 500 px sui 100).

Il punto di appoggio si fissa componente per componente, e cio' che si lascia
vuoto lo decide il DEM: senza --x e --y si va al centro, senza --z si prende la
quota del suolo. Una quota data invece resta quella anche spostando il punto --
e' cosi' che si appoggia un piano a un orizzonte che passa sopra o sotto la
topografia di oggi. La spunta "quota dal DEM" nel pannello fa e disfa il legame
in qualsiasi momento.

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

# Gli slot vettoriali sono tre e generici, ma questo strumento e' nato con un
# geopackage solo: `--geologia` resta come scorciatoia per quello, e mette
# `carbonates` fra i poligoni e `faults` fra le linee.

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


class VectorSource:
    """
    Un layer vettoriale di sfondo, nel ruolo che gli si e' dato.

    I ruoli sono tre -- poligoni, linee, punti -- e non sono una scelta di
    stile: decidono che cosa ha senso chiedere al layer. Un campo di poligoni
    colorato per unita' dice a quali due formazioni appartiene un contatto; le
    stesse ventitre tinte spalmate su quattrocento faglie non si leggono.
    Quindi la categorizzazione parte accesa sui poligoni e spenta sul resto,
    ed e' comunque l'utente a decidere.

    I layer sono statici: si disegnano una volta e finiscono nel fondale che il
    blitting ricattura, quindi per frame costano zero. Ognuno porta il proprio
    CRS -- in geology.gpkg i carbonati sono in UTM 32N e le faglie in
    geografiche -- e va riproiettato per conto suo su quello del DEM, mai il
    file in blocco.
    """

    ROLES = ("poligoni", "linee", "punti")

    # Il suffisso del tipo OGR: 'Polygon' e 'MultiPolygon' finiscono entrambi
    # in 'Polygon', e cosi' le altre due coppie. Un layer senza geometria --
    # `fault_attitudes` in geology.gpkg e' una tabella pura -- non ha suffisso
    # e resta fuori da tutti e tre i ruoli, che e' dove deve stare.
    GEOMETRY_SUFFIX = {
        "poligoni": "Polygon",
        "linee": "LineString",
        "punti": "Point",
    }

    FLAT_STYLE = {
        "poligoni": dict(facecolor="#4daf7c", edgecolor="#2f7a52", alpha=0.25, linewidth=0.5),
        "linee": dict(color="#1f4fd8", linewidth=1.0),
        "punti": dict(color="#d95f02", markersize=26, marker="^", edgecolor="#4a2200"),
    }

    CATEGORY_STYLE = {
        "poligoni": dict(edgecolor="#333333", linewidth=0.4, alpha=0.38),
        "linee": dict(linewidth=1.3),
        "punti": dict(markersize=30, marker="^", edgecolor="#222222"),
    }

    # I punti sopra le linee, le linee sopra i poligoni: l'ordine in cui una
    # carta si legge. Con lo zorder unico di prima un affioramento disegnato
    # dopo copriva le faglie che servivano a orientarsi.
    ZORDER = {"poligoni": 2, "linee": 3, "punti": 4}

    # Il campo che tappa i buchi di quello scelto: in geology.gpkg cinque
    # poligoni non hanno `code`, e senza fallback finirebbero in un'unica
    # categoria "n.d." che ne mescola tre di diverse.
    CATEGORY_FALLBACK = "name"

    # Oltre una dozzina di voci la legenda mangia la mappa che dovrebbe
    # spiegare; le categorie in eccesso restano colorate, solo non elencate.
    MAX_LEGEND_ENTRIES = 12

    def __init__(self, path, role, crs, bounds, layer=None, category_field=None):
        import geopandas as gpd  # pesante da importare: solo se serve davvero
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
            self.problem = "CRS assente"
            return

        window = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        visible = complete.to_crs(crs)
        visible = visible[visible.intersects(window)]

        if visible.empty:
            self.problem = "nessun elemento sul DEM"
            return

        self.frame = self._categorize(complete, visible)

    # -- lettura del contenitore, senza caricare i dati -------------------

    @staticmethod
    def candidate_layers(path, role):
        """
        I layer del file che hanno la geometria giusta per il ruolo.

        Si legge dai soli metadati, quindi elencare i layer di un geopackage
        da mezzo giga costa quanto elencarne uno vuoto -- ed e' cio' che
        permette al dialogo di filtrare mentre l'utente sceglie.
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
        """I campi testuali del layer: gli unici su cui categorizzare abbia senso."""

        import pyogrio

        info = pyogrio.read_info(path, layer=layer) if layer else pyogrio.read_info(path)

        return [
            str(field)
            for field, dtype in zip(info["fields"], info["dtypes"])
            if str(dtype) == "object"
        ]

    # -- categorie ---------------------------------------------------------

    @property
    def is_loaded(self):
        return self.frame is not None

    def _values(self, frame):
        """La colonna su cui distinguere, con i buchi tappati dal nome."""

        if not self.category_field or self.category_field not in frame.columns:
            return None

        values = frame[self.category_field].astype("string")

        if self.CATEGORY_FALLBACK in frame.columns:
            values = values.fillna(frame[self.CATEGORY_FALLBACK].astype("string"))

        return values.fillna("n.d.").astype(str)

    def _categorize(self, complete, visible):
        """
        Assegna un colore per categoria, deciso sull'elenco completo del layer.

        Sull'elenco completo e non su quello visibile di proposito: se i colori
        uscissero da quali categorie capitano nella finestra, la stessa
        formazione cambierebbe colore spostandosi o cambiando DEM, ed e'
        l'unica cosa che una legenda non puo' permettersi.
        """

        values = self._values(complete)

        if values is None:
            return visible

        from matplotlib import colormaps

        # Venti piu' venti: le unita' cartografate in geology.gpkg sono
        # ventitre, e con la sola tab20 due di esse uscirebbero identiche.
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

    # -- disegno -----------------------------------------------------------

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
        L'artista fittizio per una voce di legenda.

        Serve perche' geopandas disegna con collection che matplotlib non sa
        rappresentare da sola: senza questi, i poligoni sparirebbero dalla
        legenda in silenzio. La forma segue il ruolo, cosi' fra tre layer
        categorizzati si capisce comunque quale voce e' di chi.
        """

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        style = dict((self.CATEGORY_STYLE if self.colors else self.FLAT_STYLE)[self.role])
        style.pop("markersize", None)
        style.pop("color", None)

        if self.role == "poligoni":
            return Patch(facecolor=color or style.pop("facecolor", "#4daf7c"), label=label, **style)

        if self.role == "linee":
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

        # In legenda solo le categorie che si vedono davvero, e in ordine di
        # peso: alfabeticamente il taglio a dodici butterebbe fuori Qt, PL e
        # Op -- che sono meta' della mappa -- per far posto ad AV, che e' un
        # poligono solo. Il peso e' l'area per i poligoni, la lunghezza per le
        # linee, il conteggio per i punti.
        frame = self.frame

        if self.role == "poligoni":
            weight = frame.area
        elif self.role == "linee":
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
                    label=f"+{len(present) - self.MAX_LEGEND_ENTRIES} altre in {self.role}",
                )
            )

        return handles

    def summary(self):
        where = self.layer or self.path.name

        if not self.is_loaded:
            return f"{self.role}: {where} saltato ({self.problem})"

        if self.colors:
            distinct = len(set(self.frame["_gsurf_category"]))

            return (
                f"{self.role}: {where}, {len(self.frame)} in {distinct} "
                f"categorie ({self.category_field})"
            )

        return f"{self.role}: {where}, {len(self.frame)}"


class Overlay:
    """
    I layer vettoriali di sfondo tenuti insieme, nell'ordine in cui si leggono.

    Nessuno di essi e' necessario: il DEM da solo basta a intersecare un piano.
    Servono a sapere dove si sta appoggiando quel piano, che e' una domanda
    diversa dal calcolo e a cui il DEM non risponde.
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
        """Disegna sugli assi, senza lasciare che geopandas riscali la vista."""

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

        return "; ".join(lines) if lines else "nessun layer vettoriale"


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


VECTOR_FILTER = (
    "Vettoriali (*.gpkg *.shp *.geojson *.json *.gml *.kml *.sqlite *.fgb);;"
    "Tutti i file (*)"
)

RASTER_FILTER = "Raster (*.tif *.tiff *.vrt *.asc *.img *.dt2 *.hgt);;Tutti i file (*)"


def as_number(text):
    """Il testo come numero, o None se e' vuoto o non lo e'.

    La virgola vale il punto: la tastiera italiana mette la virgola sul
    tastierino, e rifiutare '1187,4' sarebbe una piccola crudelta'."""

    text = (text or "").strip().replace(",", ".")

    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        return None


class VectorPicker(QtWidgets.QGroupBox):
    """
    La scelta di un layer per un ruolo: file, layer dentro il file, categorie.

    I layer offerti sono filtrati sulla geometria del ruolo, letta dai soli
    metadati: nello slot dei poligoni le faglie non compaiono proprio, e una
    tabella senza geometria non compare da nessuna parte. E' un errore in meno
    da diagnosticare a valle, al costo di una lettura che non tocca i dati.
    """

    # I nomi con cui una colonna di categoria si presenta di solito. Sui
    # poligoni la categorizzazione parte accesa perche' e' quasi sempre cio'
    # che si vuole; su linee e punti parte spenta, che venti tinte su
    # quattrocento faglie non si leggono.
    PREFERRED_FIELDS = ("code", "sigla", "unit", "unita", "type", "tipo", "name", "nome")

    def __init__(self, role, parent=None):
        super().__init__(role.capitalize(), parent)

        self.role = role
        self._path = None

        self.path_label = QtWidgets.QLineEdit()
        self.path_label.setReadOnly(True)
        self.path_label.setPlaceholderText("nessuno (opzionale)")

        browse = QtWidgets.QPushButton("Sfoglia...")
        browse.clicked.connect(self._browse)

        self.clear_button = QtWidgets.QPushButton("Togli")
        self.clear_button.clicked.connect(self.clear)
        self.clear_button.setEnabled(False)

        self.layer_combo = QtWidgets.QComboBox()
        self.layer_combo.setEnabled(False)
        self.layer_combo.currentTextChanged.connect(self._on_layer_changed)

        self.category_combo = QtWidgets.QComboBox()
        self.category_combo.setEnabled(False)

        grid = QtWidgets.QGridLayout(self)
        grid.addWidget(self.path_label, 0, 0, 1, 2)
        grid.addWidget(browse, 0, 2)
        grid.addWidget(self.clear_button, 0, 3)
        grid.addWidget(QtWidgets.QLabel("layer"), 1, 0)
        grid.addWidget(self.layer_combo, 1, 1, 1, 3)
        grid.addWidget(QtWidgets.QLabel("categorie"), 2, 0)
        grid.addWidget(self.category_combo, 2, 1, 1, 3)
        grid.setColumnStretch(1, 1)

    def _browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Scegli il file: {self.role}", "", VECTOR_FILTER
        )

        if path:
            self.set_path(path)

    def set_path(self, path, layer=None, category_field=None):
        """Carica l'elenco dei layer adatti al ruolo. Torna False se non ce ne sono."""

        try:
            candidates = VectorSource.candidate_layers(path, self.role)
        except Exception as err:
            QtWidgets.QMessageBox.warning(
                self, "File illeggibile", f"{Path(path).name}\n\n{str(err).splitlines()[0]}"
            )
            return False

        if not candidates:
            QtWidgets.QMessageBox.information(
                self,
                "Nessun layer adatto",
                f"{Path(path).name} non contiene layer di tipo {self.role}.",
            )
            return False

        self._path = Path(path)
        self.path_label.setText(str(path))
        self.path_label.setToolTip(str(path))
        self.clear_button.setEnabled(True)

        with QtCore.QSignalBlocker(self.layer_combo):
            self.layer_combo.clear()
            self.layer_combo.addItems(candidates)

            if layer and layer in candidates:
                self.layer_combo.setCurrentText(layer)

        self.layer_combo.setEnabled(True)
        self._on_layer_changed(self.layer_combo.currentText(), preferred=category_field)

        return True

    def _on_layer_changed(self, layer, preferred=None):
        if not self._path or not layer:
            return

        try:
            fields = VectorSource.text_fields(self._path, layer)
        except Exception:
            fields = []

        with QtCore.QSignalBlocker(self.category_combo):
            self.category_combo.clear()
            self.category_combo.addItem("(nessuna)")
            self.category_combo.addItems(fields)

            chosen = None

            if preferred and preferred in fields:
                chosen = preferred
            elif self.role == "poligoni":
                chosen = next((f for f in self.PREFERRED_FIELDS if f in fields), None)

            self.category_combo.setCurrentText(chosen or "(nessuna)")

        self.category_combo.setEnabled(bool(fields))

    def clear(self):
        self._path = None
        self.path_label.clear()
        self.path_label.setToolTip("")
        self.clear_button.setEnabled(False)
        self.layer_combo.clear()
        self.layer_combo.setEnabled(False)
        self.category_combo.clear()
        self.category_combo.setEnabled(False)

    def value(self):
        """Il ruolo scelto come dizionario, o None se lo slot e' vuoto."""

        if self._path is None:
            return None

        field = self.category_combo.currentText()

        return dict(
            path=str(self._path),
            role=self.role,
            layer=self.layer_combo.currentText() or None,
            category_field=None if field in ("", "(nessuna)") else field,
        )


class SourcesDialog(QtWidgets.QDialog):
    """
    Che cosa aprire, chiesto prima di aprire la finestra di lavoro.

    Il DEM e' l'unico obbligatorio, perche' e' l'unico di cui il kernel ha
    bisogno: gli altri tre servono a sapere dove si sta appoggiando il piano,
    che e' una domanda diversa dal calcolo.

    Anche il punto di appoggio si puo' fissare qui, componente per componente:
    lasciando vuoto si va al centro del DEM con la quota del suolo, e una
    quota scritta a mano vale piu' del suolo -- e' cosi' che si appoggia un
    piano a un orizzonte che passa sopra la topografia di oggi.
    """

    def __init__(self, parent=None, dem=None, vectors=(), point=(None, None, None)):
        super().__init__(parent)

        self.setWindowTitle("gSurf - sorgenti")
        self.setMinimumWidth(560)

        self.dem_label = QtWidgets.QLineEdit()
        self.dem_label.setReadOnly(True)
        self.dem_label.setPlaceholderText("obbligatorio")

        dem_browse = QtWidgets.QPushButton("Sfoglia...")
        dem_browse.clicked.connect(self._browse_dem)

        self.dem_info = QtWidgets.QLabel()
        self.dem_info.setStyleSheet("color: gray; font-size: 10px;")

        dem_box = QtWidgets.QGroupBox("DEM")
        dem_grid = QtWidgets.QGridLayout(dem_box)
        dem_grid.addWidget(self.dem_label, 0, 0)
        dem_grid.addWidget(dem_browse, 0, 1)
        dem_grid.addWidget(self.dem_info, 1, 0, 1, 2)
        dem_grid.setColumnStretch(0, 1)

        self.pickers = {role: VectorPicker(role) for role in VectorSource.ROLES}

        # Le caselle restano di testo e non spinbox: una spinbox non sa stare
        # vuota, e "vuoto" e' proprio il valore che qui vuol dire "decidi tu".
        numeric = QtGui.QDoubleValidator()
        numeric.setLocale(QtCore.QLocale.c())

        self.easting_edit = QtWidgets.QLineEdit()
        self.northing_edit = QtWidgets.QLineEdit()
        self.elevation_edit = QtWidgets.QLineEdit()

        for edit in (self.easting_edit, self.northing_edit, self.elevation_edit):
            edit.setValidator(numeric)

        self.easting_edit.setPlaceholderText("centro del DEM")
        self.northing_edit.setPlaceholderText("centro del DEM")
        self.elevation_edit.setPlaceholderText("quota del DEM")

        point_box = QtWidgets.QGroupBox("Punto di appoggio (opzionale)")
        point_grid = QtWidgets.QGridLayout(point_box)
        for column, (caption, edit) in enumerate(
            (
                ("E", self.easting_edit),
                ("N", self.northing_edit),
                ("Z", self.elevation_edit),
            )
        ):
            point_grid.addWidget(QtWidgets.QLabel(caption), 0, column * 2)
            point_grid.addWidget(edit, 0, column * 2 + 1)
            point_grid.setColumnStretch(column * 2 + 1, 1)

        note = QtWidgets.QLabel(
            "Una quota scritta qui non segue il DEM: il piano si appoggia a quella."
        )
        note.setStyleSheet("color: gray; font-size: 10px;")
        point_grid.addWidget(note, 1, 0, 1, 6)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Open
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        # I pulsanti standard di Qt prendono la lingua da un file di traduzione
        # che qui non c'e', e uscirebbero in inglese in mezzo a un'interfaccia
        # italiana. Scriverli a mano costa meno che installare un traduttore.
        for button, text in (
            (QtWidgets.QDialogButtonBox.StandardButton.Open, "Apri"),
            (QtWidgets.QDialogButtonBox.StandardButton.Cancel, "Annulla"),
        ):
            self.buttons.button(button).setText(text)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(dem_box)
        for role in VectorSource.ROLES:
            layout.addWidget(self.pickers[role])
        layout.addWidget(point_box)
        layout.addWidget(self.buttons)

        self._dem_path = None
        self._set_dem(dem)

        for spec in vectors or ():
            picker = self.pickers.get(spec.get("role"))

            if picker is not None:
                picker.set_path(spec["path"], spec.get("layer"), spec.get("category_field"))

        for edit, value in zip(
            (self.easting_edit, self.northing_edit, self.elevation_edit), point
        ):
            if value is not None:
                edit.setText(f"{float(value):g}")

    def _browse_dem(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Scegli il DEM", "", RASTER_FILTER
        )

        if path:
            self._set_dem(path)

    def _set_dem(self, path):
        """
        Apre il DEM per i soli metadati, e con quelli spiega che cosa e'.

        Serve anche a scoprire subito che il file non e' un raster, invece che
        dopo aver chiuso il dialogo -- e a scrivere nei segnaposto le
        coordinate vere del centro, che sono il valore che si otterrebbe
        lasciando vuoto.
        """

        self._refresh_ok()

        if not path:
            return

        try:
            with rasterio.open(path) as src:
                epsg = src.crs.to_epsg() if src.crs else None
                centre_x = (src.bounds.left + src.bounds.right) / 2.0
                centre_y = (src.bounds.bottom + src.bounds.top) / 2.0
                info = (
                    f"{src.width}x{src.height} "
                    f"({src.width * src.height / 1e6:.1f} Mpx), "
                    f"EPSG:{epsg or '?'}, cella {abs(src.transform.a):g} m"
                )
        except Exception as err:
            QtWidgets.QMessageBox.warning(
                self, "DEM illeggibile", f"{Path(path).name}\n\n{str(err).splitlines()[0]}"
            )
            return

        self._dem_path = str(path)
        self.dem_label.setText(str(path))
        self.dem_label.setToolTip(str(path))
        self.dem_info.setText(info)

        self.easting_edit.setPlaceholderText(f"centro: {centre_x:.0f}")
        self.northing_edit.setPlaceholderText(f"centro: {centre_y:.0f}")

        self._refresh_ok()

    def _refresh_ok(self):
        self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Open).setEnabled(
            self._dem_path is not None
        )

    def choices(self):
        """DEM, layer vettoriali e punto, nella forma che `main` sa usare."""

        vectors = [p.value() for p in self.pickers.values()]

        return dict(
            dem=self._dem_path,
            vectors=[v for v in vectors if v],
            point=(
                as_number(self.easting_edit.text()),
                as_number(self.northing_edit.text()),
                as_number(self.elevation_edit.text()),
            ),
        )


class RealtimeWindow(QtWidgets.QMainWindow):

    PICK_RADIUS_PX = 12
    ZOOM_STEP = 1.3

    # QDial mette il minimo alle ore 6, non alle 12: misurato afferrando il
    # widget e cercando la lancetta, il valore 0 punta a 181 gradi dalle ore 12
    # e il valore 180 a 360. Il verso e' orario, come l'azimut, quindi fra la
    # scala del widget e l'immersione geologica c'e' solo mezzo giro di scarto.
    DIAL_NORTH_OFFSET = 180

    def __init__(
        self,
        dem,
        overlay=None,
        side=1000,
        attitude=(90.0, 30.0),
        source=None,
        z_follows_dem=None,
    ):
        super().__init__()

        self.dem = dem
        self.overlay = overlay
        self.background = None
        self.dragging = False
        self.frame_times = deque(maxlen=20)
        self.convergence = MeridianConvergence(dem.crs)
        self.last_result = ([], [])

        # Le tre componenti sono indipendenti: si puo' fissare la sola quota e
        # lasciare che il punto stia al centro, o il contrario.
        x, y, z = (tuple(source) + (None, None, None))[:3] if source else (None, None, None)
        centre_x, centre_y = dem.center()

        x = centre_x if x is None else float(x)
        y = centre_y if y is None else float(y)

        # Una quota data esplicitamente vuole restare quella: e' il caso di un
        # orizzonte proiettato, o di una misura presa sopra o sotto il suolo.
        # Darla e' quindi anche il modo di dire che non deve seguire il DEM,
        # salvo che qualcuno lo chieda esplicitamente.
        self.z_follows_dem = (z is None) if z_follows_dem is None else bool(z_follows_dem)

        if z is None:
            surface = dem.elevation_at(x, y)
            z = dem.z_median if surface is None else surface

        self.source_point = [x, y, float(z)]

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

        # Il punto di appoggio, scrivibile oltre che trascinabile: sul terreno
        # una stazione ha delle coordinate, e ridigitarle cercandole con il
        # mouse e' un modo di perderle.
        # L'intervallo esce dal DEM di una sua larghezza per lato, invece di
        # fermarsi al bordo: il piano e' illimitato e il punto che lo regge non
        # deve starci sopra. Fermarsi al bordo vorrebbe dire che un --x fuori
        # DEM verrebbe silenziosamente riportato dentro, e la casella direbbe
        # una cosa diversa dal punto che sta calcolando.
        span_x = self.dem.bounds.right - self.dem.bounds.left
        span_y = self.dem.bounds.top - self.dem.bounds.bottom

        self.easting_spin = QtWidgets.QDoubleSpinBox()
        self.easting_spin.setDecimals(1)
        self.easting_spin.setSingleStep(50.0)
        self.easting_spin.setRange(self.dem.bounds.left - span_x, self.dem.bounds.right + span_x)
        self.easting_spin.setPrefix("E ")

        self.northing_spin = QtWidgets.QDoubleSpinBox()
        self.northing_spin.setDecimals(1)
        self.northing_spin.setSingleStep(50.0)
        self.northing_spin.setRange(self.dem.bounds.bottom - span_y, self.dem.bounds.top + span_y)
        self.northing_spin.setPrefix("N ")

        # La quota va sotto il livello del mare -- qui l'avanfossa ci arriva --
        # e sopra la cima piu' alta, perche' un piano puo' appoggiarsi a un
        # orizzonte che sta in aria sopra la topografia attuale.
        self.elevation_spin = QtWidgets.QDoubleSpinBox()
        self.elevation_spin.setDecimals(1)
        self.elevation_spin.setSingleStep(10.0)
        self.elevation_spin.setRange(-6000.0, 9000.0)
        self.elevation_spin.setPrefix("Z ")
        self.elevation_spin.setSuffix(" m")

        # Acceso, il punto striscia sulla topografia. Spento, la quota resta
        # quella scritta e il piano si stacca dal suolo: e' cio' che serve per
        # un orizzonte proiettato, o per una misura presa in parete.
        self.follow_dem_check = QtWidgets.QCheckBox("quota dal DEM")
        self.follow_dem_check.setChecked(self.z_follows_dem)

        self.elevation_label = QtWidgets.QLabel()
        self.elevation_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.elevation_label.setStyleSheet("color: gray; font-size: 10px;")

        self._sync_point_boxes()

        self.easting_spin.valueChanged.connect(self._on_point_typed)
        self.northing_spin.valueChanged.connect(self._on_point_typed)
        self.elevation_spin.valueChanged.connect(self._on_elevation_typed)
        self.follow_dem_check.toggled.connect(self._on_follow_dem_toggled)

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
        layout.addWidget(QtWidgets.QLabel("Punto di appoggio"))
        layout.addWidget(self.easting_spin)
        layout.addWidget(self.northing_spin)
        layout.addWidget(self.elevation_spin)
        layout.addWidget(self.follow_dem_check)
        layout.addWidget(self.elevation_label)

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

        # Il pannello ha cinque gruppi e non ci sta piu' su uno schermo basso:
        # dentro un'area scorrevole si accorcia invece di tagliare i bottoni.
        panel = QtWidgets.QScrollArea()
        panel.setWidget(controls)
        panel.setWidgetResizable(True)
        panel.setMaximumWidth(224)
        panel.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.addWidget(map_side, stretch=1)
        main_layout.addWidget(panel)
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
        surface = self.dem.elevation_at(x, y)

        # Con la spunta tolta la quota non si tocca: chi l'ha scritta la vuole
        # dov'e', e un trascinamento in mappa e' un gesto sul piano orizzontale.
        # Su nodata si tiene comunque la precedente invece di rifiutare lo
        # spostamento: interrompere un trascinamento a meta' e' peggio che
        # appoggiare il piano a una quota vecchia di qualche pixel.
        if self.z_follows_dem and surface is not None:
            z = surface
        else:
            z = self.source_point[2]

        self.source_point = [x, y, z]
        self.source_marker.set_data([x], [y])
        self._sync_point_boxes()

        return surface is not None

    def _sync_point_boxes(self):
        """Riporta le tre caselle sul punto, senza farle rispondere."""

        for box, value in (
            (self.easting_spin, self.source_point[0]),
            (self.northing_spin, self.source_point[1]),
            (self.elevation_spin, self.source_point[2]),
        ):
            with QtCore.QSignalBlocker(box):
                box.setValue(value)

        self.elevation_spin.setEnabled(not self.z_follows_dem)
        self._report_elevation()

    def _report_elevation(self):
        """Lo scarto dal suolo, che e' l'unica cosa che la quota da sola non dice."""

        surface = self.dem.elevation_at(self.source_point[0], self.source_point[1])

        if surface is None:
            self.elevation_label.setText("fuori DEM")
            return

        if self.z_follows_dem:
            self.elevation_label.setText(f"suolo {surface:.0f} m")
            return

        gap = self.source_point[2] - surface
        self.elevation_label.setText(f"suolo {surface:.0f} m\n{gap:+.0f} m dal suolo")

    def _on_point_typed(self, value):
        """Coordinate scritte a mano: il punto va dove dicono, e la finestra lo segue."""

        x = float(self.easting_spin.value())
        y = float(self.northing_spin.value())

        self._move_source(x, y)
        self._recenter_window()
        self.update_intersection()

    def _on_elevation_typed(self, value):
        self.source_point[2] = float(value)
        self._report_elevation()
        self.update_intersection()

    def _on_follow_dem_toggled(self, checked):
        """
        Riagganciare la quota al DEM la riporta subito sul suolo.

        Il contrario no: togliendo la spunta la quota resta quella che era, che
        e' il punto di partenza naturale per spostarla di poco.
        """

        self.z_follows_dem = bool(checked)

        if checked:
            surface = self.dem.elevation_at(self.source_point[0], self.source_point[1])

            if surface is not None:
                self.source_point[2] = surface

        self._sync_point_boxes()
        self.update_intersection()

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
            # Senza queste due righe una quota staccata dal suolo si rilegge
            # come un errore di lettura del DEM, invece che come la scelta che
            # era: va detto che il distacco e' voluto e di quanto.
            "source_z_dal_dem": bool(self.z_follows_dem),
            "quota_dem": self.dem.elevation_at(self.source_point[0], self.source_point[1]),
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
        # `src_z_dem` e' la quota del suolo sotto il punto: se differisce da
        # `src_z` il piano e' stato staccato apposta, e senza il confronto
        # sembrerebbe uno sbaglio.
        lon_lat = self.source_geographic()
        surface = self.dem.elevation_at(self.source_point[0], self.source_point[1])

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
                "src_z_dem": [surface] * len(traces),
                "z_da_dem": [bool(self.z_follows_dem)] * len(traces),
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


def split_layer(spec, role):
    """
    `percorso` oppure `percorso:layer`, sciolto senza indovinare.

    Un percorso puo' contenere due punti per conto suo, quindi la regola e'
    guardare il disco invece di leggere la stringa: se il tutto e' un file
    esistente, e' un percorso; altrimenti si prova a staccare l'ultimo pezzo.
    """

    if spec is None:
        return None

    if Path(spec).exists():
        return dict(path=spec, role=role, layer=None)

    head, _, tail = spec.rpartition(":")

    if head and Path(head).exists():
        return dict(path=head, role=role, layer=tail)

    return dict(path=spec, role=role, layer=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dem",
        nargs="?",
        help="DEM in un formato leggibile da rasterio; se manca lo chiede un dialogo",
    )
    for role, explanation in (
        ("poligoni", "affioramenti, unita', qualunque campitura"),
        ("linee", "faglie, contatti, tracce"),
        ("punti", "stazioni, misure, campioni"),
    ):
        parser.add_argument(
            f"--{role}",
            metavar="PATH[:LAYER]",
            help=f"layer {role} da mettere sotto il piano ({explanation})",
        )
    parser.add_argument(
        "--geologia",
        metavar="GPKG",
        help="scorciatoia: carbonates come poligoni e faults come linee dallo stesso geopackage",
    )
    parser.add_argument(
        "--categorie",
        metavar="CAMPO",
        default="code",
        help=(
            "campo su cui colorare i poligoni (default 'code'); "
            "'nessuna' per il colore unico"
        ),
    )
    parser.add_argument("--x", type=float, help="est del punto di appoggio (default: centro del DEM)")
    parser.add_argument("--y", type=float, help="nord del punto di appoggio (default: centro del DEM)")
    parser.add_argument(
        "--z",
        type=float,
        help=(
            "quota del punto di appoggio (default: quella del DEM); "
            "se data, il piano resta a questa quota anche spostandolo"
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
    point = (args.x, args.y, args.z)
    z_follows_dem = None
    side = args.finestra

    if args.assetto:
        saved = json.loads(Path(args.assetto).read_text(encoding="utf-8"))
        attitude = (saved["dip_dir"], saved["dip_angle"])
        side = saved.get("finestra_px", side)
        z_follows_dem = saved.get("source_z_dal_dem")

        # Gli argomenti espliciti battono il file: se si passa --z insieme a un
        # assetto, e' la riga di comando che si e' scritta adesso.
        stored = saved.get("source_point") or (None, None, None)
        point = tuple(
            given if given is not None else was for given, was in zip(point, stored)
        )

        if args.z is not None:
            z_follows_dem = False

        print(f"assetto: {attitude[0]:.0f}/{attitude[1]:.0f}, finestra {side} px")

    vectors = [
        spec
        for spec in (
            split_layer(args.poligoni, "poligoni"),
            split_layer(args.linee, "linee"),
            split_layer(args.punti, "punti"),
        )
        if spec
    ]

    # La scorciatoia storica, tenuta perche' e' come questo strumento si e'
    # sempre lanciato: i due layer di geology.gpkg nei loro ruoli naturali.
    if args.geologia:
        vectors.append(dict(path=args.geologia, role="poligoni", layer="carbonates"))
        vectors.append(dict(path=args.geologia, role="linee", layer="faults"))

    for spec in vectors:
        spec.setdefault(
            "category_field",
            None if args.categorie == "nessuna" or spec["role"] != "poligoni" else args.categorie,
        )

    # La QApplication prima di ogni dialogo, altrimenti Qt esce senza dire
    # perche'.
    app = QtWidgets.QApplication(sys.argv)

    if not args.dem:
        dialog = SourcesDialog(dem=args.dem, vectors=vectors, point=point)

        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        chosen = dialog.choices()
        args.dem = chosen["dem"]
        vectors = chosen["vectors"]
        point = chosen["point"]

        # Una quota scritta nel dialogo e' una scelta come quella da riga di
        # comando, e vale allo stesso modo.
        if point[2] is not None:
            z_follows_dem = False

    dem = Dem(args.dem)
    print(
        f"DEM {dem.width}x{dem.height} ({dem.width * dem.height / 1e6:.1f} Mpx), "
        f"EPSG:{dem.crs.to_epsg()}, nodata={dem.nodata}, "
        f"sfondo decimato 1:{dem.decimation}"
    )

    overlay = Overlay(
        VectorSource(
            spec["path"],
            spec["role"],
            dem.crs,
            dem.bounds,
            layer=spec.get("layer"),
            category_field=spec.get("category_field"),
        )
        for spec in vectors
    )

    if vectors:
        print(f"vettoriali: {overlay.summary()}")

    window = RealtimeWindow(
        dem,
        overlay,
        side=side,
        attitude=attitude,
        source=point,
        z_follows_dem=z_follows_dem,
    )
    window.resize(1180, 880)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
