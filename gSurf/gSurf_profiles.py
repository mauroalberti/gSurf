import sys

from typing import List, Optional

from collections import namedtuple

import numbers

from PyQt5 import QtWidgets, uic


from pygsf.spatial.rasters.io import read_raster, read_band, read_raster_band
from pygsf.spatial.vectorial.io import read_linestring_geometries
from pygsf.spatial.rasters.geoarray import GeoArray
from pygsf.utils.qt.tools import *

from pygsf.spatial.geology.profiles.geoprofiles import GeoProfile
from pygsf.spatial.geology.profiles.profilers import LinearProfiler
from pygsf.spatial.geology.profiles.plot import plot

from gSurf_ui_classes import ChooseSourceDataDialog


DataParameters = namedtuple("DataParameters", 'filePath, data')


def get_selected_layers_indices(
    treeWidgetDataList: QtWidgets.QTreeWidget
) -> List[numbers.Integral]:

    selected_data_indices = []

    for data_ndx in range(treeWidgetDataList.topLevelItemCount()):
        curr_data_item = treeWidgetDataList.topLevelItem(data_ndx)
        if curr_data_item.checkState(0) == 2:
            selected_data_indices.append(data_ndx)

    return selected_data_indices


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):

        super().__init__()
        uic.loadUi('gSurf_0.3.0.ui', self)

        self.plugin_name = "gSurf"

        # File menu

        self.actLoadDem.triggered.connect(self.load_dem)
        self.actLoadLineShapefile.triggered.connect(self.load_line_shapefile)

        # Profiles menu

        self.actChooseDEMs.triggered.connect(self.choose_dems)
        self.actChooseLines.triggered.connect(self.define_lines)

        self.actCreateTopoProfile.triggered.connect(self.create_profile)
        self.actCalcProfileStats.triggered.connect(self.calculate_statistics)

        # data storage

        self.dems = []
        self.lines = []

        # data choices

        self.selected_dems_indices = []
        self.selected_lines_indices = []

        # window visibility

        self.show()

    def load_dem(self):

        filePath, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Open DEM file (using GDAL)"),
            "",
            "*.*"
        )

        if not filePath:
            return

        dataset, geotransform, num_bands, projection = read_raster(filePath)

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

        self.dems.append(DataParameters(filePath, ga))

        QMessageBox.information(
            None,
            "DEM loading",
            "DEM read".format(filePath)
        )

    def load_line_shapefile(self):

        filePath, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Open line shapefile (using GDAL)"),
            "",
            "*.shp"
        )

        if not filePath:
            return

        try:
            multiline = read_linestring_geometries(line_shp_path=filePath)
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

        self.lines.append(DataParameters(filePath, multiline))

        QMessageBox.information(
            None,
            "Line shapefile",
            "Shapefile read ({} lines)".format(len(multiline))
        )

    def choose_data(
        self,
        data: List[str]
    ) -> Optional[List[numbers.Integral]]:
        """
        Choose data to use for profile creation.

        :param data: the list of data sources
        :type data: List[str]
        :return: the selected data, as indices of the input list
        :rtype: List[numbers.Integral]
        """

        selected_data_indices = []

        dialog = ChooseSourceDataDialog(
            self.plugin_name,
            data_sources_paths=data)

        if dialog.exec_():
            selected_data_indices = get_selected_layers_indices(dialog.listData_treeWidget)

        return selected_data_indices

    def choose_dems(self):
        """
        Chooses DEM(s) to use for profile creation.

        :return:
        """

        self.selected_dems_indices = self.choose_data(
            data=[dem.filePath for dem in self.dems]
        )

        if not self.selected_dems_indices:
            warn(self,
                 self.plugin_name,
                 "No chosen data")
            return []

    def define_lines(self):
        """
        Defines lines to use for profile creation.

        :return:
        """

        self.selected_lines_indices = self.choose_data(
            data=[line.filePath for line in self.lines]
        )

        if not self.selected_lines_indices:
            warn(self,
                 self.plugin_name,
                 "No chosen data")
            return []

    def create_profile(self):

        # DEM
        geoarray = self.dems[self.selected_dems_indices[0]].data

        # profile
        profiles = self.lines[self.selected_lines_indices[0]].data
        line = profiles.line()

        geoprofile = GeoProfile()
        profiler = LinearProfiler(start_pt=line.start_pt(), end_pt=line.end_pt(), densify_distance=5)

        topo_profile = profiler.profile_grid(geoarray)
        geoprofile.topo_profile = topo_profile
        self.fig = plot(geoprofile)
        #self.fig.show()

    def calculate_statistics(self):

        pass


if __name__ == "__main__":
    
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    sys.exit(app.exec_())


