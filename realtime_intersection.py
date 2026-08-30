"""
Intersezione piano geologico / DEM in tempo reale.

Fetta verticale: serve a misurare se il ciclo completo -- kernel misah, poi
disegno -- sta dentro il budget di un frame mentre si trascina il quadrante.
Il kernel e' gia' noto (~24 ms su 1000x1000): la domanda aperta e' quanto resta
per disegnare, ed e' quello che la barra di stato riporta a ogni frame.

Uso:
    python realtime_intersection.py <dem.tif>

Il DEM va tenuto piccolo: 1000x1000 e' il bersaglio, 500x500 il margine comodo.
Clic sulla mappa per spostare il punto di appoggio del piano.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from time import perf_counter

import numpy as np
import rasterio

# PyQt6 va importato prima del backend: matplotlib sceglie il binding
# guardando quello gia' presente in sys.modules.
import PyQt6.QtCore  # noqa: F401
from PyQt6 import QtCore, QtWidgets

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

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


class Dem:
    """Il DEM nella forma che il kernel vuole, letta una volta sola."""

    def __init__(self, path):
        with rasterio.open(path) as src:
            band = src.read(1)
            self.geotransform = list(src.transform.to_gdal())
            self.nodata = src.nodata
            self.bounds = src.bounds

        # misah vuole f64 contiguo. I DEM reali sono spesso f32: la conversione
        # si paga qui, al caricamento, non a ogni frame.
        self.data = np.ascontiguousarray(band.astype(np.float64))

        self.extent = [
            self.bounds.left,
            self.bounds.right,
            self.bounds.bottom,
            self.bounds.top,
        ]

        valid = self.data if self.nodata is None else self.data[self.data != self.nodata]
        self.z_median = float(np.median(valid))

        shaded = self.data.copy()
        if self.nodata is not None:
            shaded[shaded == self.nodata] = np.nan
        self.hillshade = hillshade(shaded, abs(self.geotransform[1]), abs(self.geotransform[5]))

    @property
    def shape(self):
        return self.data.shape

    def center(self):
        return (
            (self.bounds.left + self.bounds.right) / 2.0,
            (self.bounds.bottom + self.bounds.top) / 2.0,
        )

    def elevation_at(self, x, y):
        """Quota alla coordinata mappa, o None fuori griglia / su nodata."""

        origin_x, px_w, _, origin_y, _, px_h = self.geotransform
        col = int((x - origin_x) / px_w)
        row = int((y - origin_y) / px_h)

        n_rows, n_cols = self.data.shape
        if not (0 <= row < n_rows and 0 <= col < n_cols):
            return None

        z = float(self.data[row, col])

        return None if self.nodata is not None and z == self.nodata else z


class RealtimeWindow(QtWidgets.QMainWindow):

    def __init__(self, dem):
        super().__init__()

        self.dem = dem
        self.background = None
        self.frame_times = deque(maxlen=20)

        cx, cy = dem.center()
        self.source_point = [cx, cy, dem.elevation_at(cx, cy) or dem.z_median]

        self.setWindowTitle(f"gSurf - intersezione in tempo reale ({dem.shape[1]}x{dem.shape[0]})")
        self._build_ui()
        self._draw_base_map()

        self.update_intersection()

    # -- costruzione ------------------------------------------------------

    def _build_ui(self):
        self.figure = Figure(figsize=(8, 8), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)

        self.canvas.mpl_connect("draw_event", self._on_draw)
        self.canvas.mpl_connect("button_press_event", self._on_click)

        self.dip_dir_dial = QtWidgets.QDial()
        self.dip_dir_dial.setRange(0, 359)
        self.dip_dir_dial.setValue(90)
        self.dip_dir_dial.setWrapping(True)
        self.dip_dir_dial.setNotchesVisible(True)
        self.dip_dir_dial.setMinimumSize(140, 140)

        self.dip_angle_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
        self.dip_angle_slider.setRange(0, 90)
        self.dip_angle_slider.setValue(30)
        self.dip_angle_slider.setTickInterval(10)
        self.dip_angle_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksRight)

        self.attitude_label = QtWidgets.QLabel()
        self.attitude_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # Il punto: si ricalcola mentre si trascina, non su un bottone.
        self.dip_dir_dial.valueChanged.connect(self.update_intersection)
        self.dip_angle_slider.valueChanged.connect(self.update_intersection)

        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.addWidget(QtWidgets.QLabel("Immersione"))
        controls_layout.addWidget(self.dip_dir_dial)
        controls_layout.addWidget(QtWidgets.QLabel("Inclinazione"))
        controls_layout.addWidget(self.dip_angle_slider, stretch=1)
        controls_layout.addWidget(self.attitude_label)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(controls)
        self.setCentralWidget(central)

        self.statusBar().showMessage("clic sulla mappa per spostare il punto di appoggio")

    def _draw_base_map(self):
        self.axes.imshow(
            self.dem.hillshade,
            cmap="gray",
            extent=self.dem.extent,
            origin="upper",
            interpolation="bilinear",
        )
        self.axes.set_xlabel("E (m, EPSG:25833)")
        self.axes.set_ylabel("N (m)")

        # animated=True tiene i due artisti fuori dal draw normale: li ridisegna
        # solo il blitting, che e' tutto il punto dell'esercizio.
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
            markersize=7,
            animated=True,
        )

        self.canvas.draw()

    # -- ciclo di aggiornamento -------------------------------------------

    def _on_draw(self, event):
        """Il fondale cambia solo su resize o zoom: qui lo si ricattura."""

        self.background = self.canvas.copy_from_bbox(self.axes.bbox)
        self.axes.draw_artist(self.intersections)
        self.axes.draw_artist(self.source_marker)

    def _on_click(self, event):
        if event.inaxes is not self.axes or event.xdata is None:
            return

        z = self.dem.elevation_at(event.xdata, event.ydata)
        if z is None:
            self.statusBar().showMessage("punto fuori dal DEM o su nodata")
            return

        self.source_point = [event.xdata, event.ydata, z]
        self.source_marker.set_data([event.xdata], [event.ydata])
        self.update_intersection()

    def update_intersection(self):
        dip_dir = float(self.dip_dir_dial.value())
        dip_angle = float(self.dip_angle_slider.value())

        start = perf_counter()
        points, segments = intersect_plane_grid(
            self.dem.data,
            self.dem.geotransform,
            self.source_point,
            dip_dir,
            dip_angle,
            self.dem.nodata,
        )
        kernel_done = perf_counter()

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
            self.axes.draw_artist(self.intersections)
            self.axes.draw_artist(self.source_marker)
            self.canvas.blit(self.axes.bbox)

        self.canvas.flush_events()
        drawn = perf_counter()

        self.frame_times.append(drawn - start)
        self._report(dip_dir, dip_angle, len(points), kernel_done - start, drawn - kernel_done)

    def _report(self, dip_dir, dip_angle, n_points, kernel_s, draw_s):
        self.attitude_label.setText(f"{dip_dir:03.0f} / {dip_angle:02.0f}")

        mean_frame = sum(self.frame_times) / len(self.frame_times)
        fps = 1.0 / mean_frame if mean_frame else 0.0

        self.statusBar().showMessage(
            f"{n_points} punti   "
            f"kernel {kernel_s * 1000:5.1f} ms   "
            f"disegno {draw_s * 1000:5.1f} ms   "
            f"totale {(kernel_s + draw_s) * 1000:5.1f} ms   "
            f"{fps:4.1f} fps"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dem", help="DEM in un formato leggibile da rasterio")
    args = parser.parse_args()

    dem = Dem(args.dem)
    print(f"DEM {dem.shape[1]}x{dem.shape[0]}, nodata={dem.nodata}")

    app = QtWidgets.QApplication(sys.argv)
    window = RealtimeWindow(dem)
    window.resize(1100, 850)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
