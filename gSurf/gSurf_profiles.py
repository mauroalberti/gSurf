import sys
import os

from typing import List, Optional

from collections import namedtuple

import numbers

import matplotlib.pyplot as plt

from PyQt5 import QtWidgets, uic


from pygsf.spatial.rasters.io import read_raster, read_band, read_raster_band
from pygsf.spatial.vectorial.io import try_read_as_geodataframe
from pygsf.spatial.vectorial.geodataframes import geodataframe_geom_types, containsLines
from pygsf.spatial.rasters.geoarray import GeoArray
from pygsf.utils.qt.tools import *

from pygsf.spatial.geology.profiles.geoprofiles import GeoProfile, GeoProfileSet
from pygsf.spatial.geology.profiles.profilers import LinearProfiler, ParallelProfilers
from pygsf.spatial.geology.profiles.plot import plot

from gSurf_ui_classes import ChooseSourceDataDialog

DataPametersFldNms = [
    "filePath",
    "data",
    "type"
]
DataParameters = namedtuple(
    "DataParameters",
    DataPametersFldNms
)

multiple_profiles_choices = [
    "central",
    "left",
    "right"
]


def get_selected_layer_index(
    treeWidgetDataList: QtWidgets.QTreeWidget
) -> numbers.Integral:

    return treeWidgetDataList.selectedIndexes()[0].row()


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):

        super().__init__()
        uic.loadUi('./widgets/gSurf_0.3.0.ui', self)

        self.plugin_name = "gSurf"
        self.chosen_dem = None
        self.chosen_profile = None

        # File menu

        self.actLoadDem.triggered.connect(self.load_dem)
        #self.actLoadLineShapefile.triggered.connect(self.load_line_shapefile)
        self.actLoadVectorLayer.triggered.connect(self.load_vector_layer)

        # Profiles menu

        self.actChooseDEMs.triggered.connect(self.define_used_dem)
        self.actChooseLines.triggered.connect(self.define_used_profile_dataset)

        self.actionCreateSingleProfile.triggered.connect(self.create_single_profile)
        self.actionCreateMultipleParallelProfiles.triggered.connect(self.create_multi_parallel_profiles)
        self.actProjectGeolAttitudes.triggered.connect(self.project_attitudes)

        # data storage

        self.dems = []
        self.vector_datasets = []

        # data choices

        self.selected_dem_index = []
        self.selected_profile_index = []

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

        self.dems.append(DataParameters(filePath, ga, "DEM"))

        QMessageBox.information(
            None,
            "DEM loading",
            "DEM read".format(filePath)
        )

    '''
    def load_line_shapefile(self):
        """
        Load a line shapefile data.

        :return:
        """

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

        self.vector_datasets.append(DataParameters(filePath, multiline))

        QMessageBox.information(
            None,
            "Line shapefile",
            "Shapefile read ({} lines)".format(len(multiline))
        )
    '''

    def load_vector_layer(self):

        filePath, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Open vector layer (using geopandas)"),
            "",
            "*.*"
        )

        filePath = filePath.strip()

        if not filePath:
            return

        success, result = try_read_as_geodataframe(
            path=filePath
        )

        if not success:
            msg = result
            QMessageBox.critical(
                None,
                "Input",
                "Exception: {}".format(msg)
             )
            return
        else:
            geodataframe = result
            self.vector_datasets.append(DataParameters(filePath, geodataframe, "vector"))
            # self.vector_datasets.append(DataParameters(filePath, multiline))
            QMessageBox.information(
                None,
                "Input",
                "Dataset read"
            )

    def choose_dataset_index(
        self,
        datasets_paths: List[str]
    ) -> Optional[List[numbers.Integral]]:
        """
        Choose data to use for profile creation.

        :param datasets_paths: the list of data sources
        :type datasets_paths: List[str]
        :return: the selected data, as index of the input list
        :rtype: numbers.Integral
        """

        #selected_data_index = []

        dialog = ChooseSourceDataDialog(
            self.plugin_name,
            data_sources=map(lambda pth: os.path.basename(pth), datasets_paths)
        )

        if dialog.exec_():
            return get_selected_layer_index(dialog.listData_treeWidget)
        else:
            return None

        #return selected_data_index

    def define_used_dem(self):
        """
        Define DEM to use for profile creation.

        :return:
        """

        self.selected_dem_index = self.choose_dataset_index(
            datasets_paths=[dem.filePath for dem in self.dems]
        )

        if self.selected_dem_index is None:
            warn(self,
                 self.plugin_name,
                 "No chosen data")
        else:
            self.chosen_dem = self.dems[self.selected_dem_index].data

    def define_used_profile_dataset(self):
        """
        Defines vector dataset for profile creation.

        :return:
        """

        lines_datasets = list(filter(lambda dataset: containsLines(dataset.data), self.vector_datasets))
        self.selected_profile_index = self.choose_dataset_index(
            datasets_paths=[line_dataset.filePath for line_dataset in lines_datasets]
        )

        if not self.selected_profile_index:
            warn(self,
                 self.plugin_name,
                 "No chosen data")
        else:
            self.chosen_profile = lines_datasets[self.selected_profile_index].data

    def create_single_profile(self):

        baseline = self.chosen_profile.line()

        geoprofile = GeoProfile()
        profiler = LinearProfiler(
            start_pt=baseline.start_pt(),
            end_pt=baseline.end_pt(),
            densify_distance=5
        )

        topo_profile = profiler.profile_grid(self.chosen_dem)
        geoprofile.topo_profile = topo_profile

        self.fig = plot(geoprofile)

        self.fig.show()

    def create_multi_parallel_profiles(self):

        dialog = MultiProfilesDefWindow()

        if dialog.exec_():
            densify_distance = dialog.densifyDistanceDoubleSpinBox.value()
            total_profiles_number = dialog.numberOfProfilesSpinBox.value()
            profiles_offset = dialog.profilesOffsetDoubleSpinBox.value()
            profiles_arrangement = dialog.profilesLocationComboBox.currentText()
            superposed_profiles = dialog.superposedProfilesCheckBox.isChecked()
        else:
            return

        # profile
        baseline = self.chosen_profile.line()

        geoprofiles = GeoProfileSet()
        base_profiler = LinearProfiler(
            start_pt=baseline.start_pt(),
            end_pt=baseline.end_pt(),
            densify_distance=densify_distance
        )

        multiple_profilers = ParallelProfilers.fromProfiler(
            base_profiler=base_profiler,
            profs_num=total_profiles_number,
            profs_offset=profiles_offset,
            profs_arr=profiles_arrangement
        )

        topo_profiles = multiple_profilers.profile_grid(self.chosen_dem)

        geoprofiles.topo_profiles_set = topo_profiles

        self.fig = plot(
            geoprofiles,
            superposed=superposed_profiles
        )

        self.fig.show()

    def project_attitudes(self):

        dialog = ProjectAttitudesDefWindow()

        if dialog.exec_():
            print("Hoora")
            """
            densify_distance = dialog.densifyDistanceDoubleSpinBox.value()
            total_profiles_number = dialog.numberOfProfilesSpinBox.value()
            profiles_offset = dialog.profilesOffsetDoubleSpinBox.value()
            profiles_arrangement = dialog.profilesLocationComboBox.currentText()
            superposed_profiles = dialog.superposedProfilesCheckBox.isChecked()
            """
        else:
            return


class MultiProfilesDefWindow(QtWidgets.QDialog):

    def __init__(self):

        super().__init__()
        uic.loadUi('./widgets/multiple_profiles.ui', self)

        self.densifyDistanceDoubleSpinBox.setValue(5.0)
        self.numberOfProfilesSpinBox.setValue(10)
        self.profilesOffsetDoubleSpinBox.setValue(500)
        self.profilesLocationComboBox.addItems(multiple_profiles_choices)

        self.setWindowTitle("Multiple parallel profiles")


class ProjectAttitudesDefWindow(QtWidgets.QDialog):

    def __init__(self):

        super().__init__()
        uic.loadUi('./widgets/project_attitudes.ui', self)

        """
        self.densifyDistanceDoubleSpinBox.setValue(5.0)
        self.numberOfProfilesSpinBox.setValue(10)
        self.profilesOffsetDoubleSpinBox.setValue(500)
        self.profilesLocationComboBox.addItems(multiple_profiles_choices)
        """

        self.setWindowTitle("Project geological attitudes")


if __name__ == "__main__":
    
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    sys.exit(app.exec_())


