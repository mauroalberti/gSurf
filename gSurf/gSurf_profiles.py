import sys

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox


from pygsf.spatial.rasters.io import read_raster, read_band
from pygsf.spatial.vectorial.io import read_linestring_geometries

from pygsf.spatial.rasters.geoarray import GeoArray


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):

        super().__init__()
        uic.loadUi('gSurf_0.3.0.ui', self)

        # File menu

        self.actLoadDem.triggered.connect(self.load_dem)
        self.actLoadLineShapefile.triggered.connect(self.load_line_shapefile)

        # Profiles menu

        self.actChooseDems.triggered.connect(self.choose_dems)
        self.actDefineLines.triggered.connect(self.define_lines)

        # data storage

        self.dems = []
        self.lines = []

        # window visibility

        self.show()

    def load_dem(self):

        fileName, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Open DEM file (using GDAL)"),
            "",
            "*.*"
        )

        if not fileName:
            return

        dataset, geotransform, num_bands, projection = read_raster(fileName)

        if num_bands != 1:
            QMessageBox.warning(
                None,
                "Raster warning",
                "Number of bands is {}, not 1 as required".format(num_bands)
             )
            return

        band_params, data = read_band(dataset)
        ga = GeoArray(
            inGeotransform=geotransform,
            epsg_cd=-1,
            inLevels=[data]
        )

        self.dems.append(ga)

        QMessageBox.information(
            None,
            "Raster loading",
            "Raster read".format(fileName)
        )

    def load_line_shapefile(self):

        fileName, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Open line shapefile (using GDAL)"),
            "",
            "*.shp"
        )

        if not fileName:
            return

        try:
            multiline = read_linestring_geometries(line_shp_path=fileName)
        except Exception as e:
            QMessageBox.critical(
                None,
                "Line shapefile error",
                "Exception: {}".format(e)
             )
            return

        if not multiline:
            QMessageBox.warning(
                None,
                "Line shapefile warning",
                "Unable to read line shapefile"
             )
            return

        self.lines.append(multiline)

        QMessageBox.information(
            None,
            "Line shapefile",
            "Shapefile read ({} lines)".format(len(multiline))
        )

    def choose_dems(self):
        """
        Chooses DEM(s) to use for profile creation.

        :return:
        """

        pass

    def define_lines(self):
        """
        Defines lines to use for profile creation.

        :return:
        """

        pass


if __name__ == "__main__":
    
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    sys.exit(app.exec_())


