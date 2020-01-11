import sys

from typing import List

from collections import namedtuple

import numbers

from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets, uic

from pygsf.spatial.rasters.io import *
from pygsf.spatial.vectorial.io import try_read_as_geodataframe
from pygsf.spatial.vectorial.geodataframes import *
from pygsf.spatial.rasters.geoarray import GeoArray
from pygsf.utils.qt.tools import *

from pygsf.spatial.geology.profiles.geoprofiles import GeoProfile, GeoProfileSet
from pygsf.spatial.geology.profiles.profilers import LinearProfiler, ParallelProfilers
from pygsf.spatial.geology.profiles.plot import plot
from pygsf.spatial.geology.convert import extract_georeferenced_attitudes


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

attitude_colors = [
    "red",
    "blue",
    "orange"
]


def get_selected_layer_index(
    treeWidgetDataList: QtWidgets.QTreeWidget
) -> Optional[numbers.Integral]:

    if not treeWidgetDataList:
        return None
    elif not treeWidgetDataList.selectedIndexes():
        return None
    else:
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
        self.actLoadVectorLayer.triggered.connect(self.load_vector_layer)

        # Profiles menu

        self.actChooseDEMs.triggered.connect(self.define_used_dem)
        self.actChooseLines.triggered.connect(self.define_used_profile_dataset)

        self.actionCreateSingleProfile.triggered.connect(self.create_single_profile)
        self.actionCreateMultipleParallelProfiles.triggered.connect(self.create_multi_parallel_profiles)
        self.actProjectGeolAttitudes.triggered.connect(self.project_attitudes)
        self.actIntersectLineLayer.triggered.connect(self.intersect_lines)

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
            self.tr("Open DEM file (using rasterio)"),
            "",
            "*.*"
        )

        if not filePath:
            return

        success, result = try_read_rasterio_band(
            filePath
        )

        if not success:
            msg = result
            QMessageBox.warning(
                None,
                "Raster input",
                "Error: {}".format(msg)
             )
            return

        array, affine_transform, epsg = result

        ga = GeoArray.fromRasterio(
            array=array,
            affine_transform=affine_transform,
            epsg=epsg
        )

        self.dems.append(DataParameters(filePath, ga, "DEM"))

        QMessageBox.information(
            None,
            "DEM loading",
            "DEM read".format(filePath)
        )

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

        dialog = ChooseSourceDataDialog(
            self.plugin_name,
            data_sources=map(lambda pth: os.path.basename(pth), datasets_paths)
        )

        if dialog.exec_():
            return get_selected_layer_index(dialog.listData_treeWidget)
        else:
            return None

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
                 "No dataset selected")
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

        if self.selected_profile_index is None:
            warn(self,
                 self.plugin_name,
                 "No dataset selected")
        else:
            self.chosen_profile = lines_datasets[self.selected_profile_index].data

    def create_single_profile(self):

        self.superposed_profiles = False

        pts = extract_line_points(
            geodataframe=self.chosen_profile,
            ndx=0
        )

        if len(pts) != 2:
            warn(self,
                 self.plugin_name,
                 "Input must be a line with two points")
            return

        self.geoprofiles = GeoProfile()
        self.profiler = LinearProfiler(
            start_pt=pts[0],
            end_pt=pts[1],
            densify_distance=5
        )

        topo_profile = self.profiler.profile_grid(self.chosen_dem)
        self.geoprofiles.topo_profile = topo_profile

        self.fig = plot(self.geoprofiles)

        self.fig.show()

    def create_multi_parallel_profiles(self):

        dialog = MultiProfilesDefWindow()

        if dialog.exec_():
            densify_distance = dialog.densifyDistanceDoubleSpinBox.value()
            total_profiles_number = dialog.numberOfProfilesSpinBox.value()
            profiles_offset = dialog.profilesOffsetDoubleSpinBox.value()
            profiles_arrangement = dialog.profilesLocationComboBox.currentText()
            self.superposed_profiles = dialog.superposedProfilesCheckBox.isChecked()
        else:
            return

        pts = extract_line_points(
            geodataframe=self.chosen_profile,
            ndx=0
        )

        if len(pts) != 2:
            warn(self,
                 self.plugin_name,
                 "Input must be a line with two points")
            return

        self.geoprofiles = GeoProfileSet()
        base_profiler = LinearProfiler(
            start_pt=pts[0],
            end_pt=pts[1],
            densify_distance=densify_distance
        )

        self.profiler = ParallelProfilers.fromProfiler(
            base_profiler=base_profiler,
            profs_num=total_profiles_number,
            profs_offset=profiles_offset,
            profs_arr=profiles_arrangement
        )

        topo_profiles = self.profiler.profile_grid(self.chosen_dem)

        self.geoprofiles.topo_profiles_set = topo_profiles

        self.fig = plot(
            self.geoprofiles,
            superposed=self.superposed_profiles
        )

        self.fig.show()

    def project_attitudes(self):

        point_layers = list(filter(lambda dataset: containsPoints(dataset.data), self.vector_datasets))

        if not point_layers:
            warn(self,
                 self.plugin_name,
                 "No point layer available")
            return

        dialog = ProjectAttitudesDefWindow(
            self.plugin_name,
            point_layers
        )

        if dialog.exec_():

            input_layer_index = dialog.inputPtLayerComboBox.currentIndex()

            azimuth_is_dipdir = dialog.azimuthDipDirRadioButton.isChecked()
            azimuth_is_strikerhr = dialog.azimuthRHRStrikeRadioButton.isChecked()

            attitude_id_fldnm = dialog.idFldNmComboBox.currentText()
            attitude_azimuth_angle_fldnm = dialog.attitudeAzimAngleFldNmComboBox.currentText()
            attitude_dip_angle_fldnm = dialog.attitudeDipAngleFldNmcomboBox.currentText()

            projection_nearest_intersection = dialog.projectNearestIntersectionRadioButton.isChecked()
            projection_constant_axis = dialog.projectAxisWithTrendRadioButton.isChecked()
            projection_axes_from_fields = dialog.projectAxesFromFieldsRadioButton.isChecked()

            projection_axis_trend_angle = dialog.projectAxisTrendAngDblSpinBox.value()
            projection_axis_plunge_angle = dialog.projectAxisPlungeAngDblSpinBox.value()

            projection_axes_trend_fldnm = dialog.projectAxesTrendFldNmComboBox.currentText()
            projection_axes_plunge_fldnm = dialog.projectAxesPlungeFldNmComboBox.currentText()

            labels_add_orientdip = dialog.labelsOrDipCheckBox.isChecked()
            labels_add_id = dialog.labelsIdCheckBox.isChecked()

            attitudes_color = dialog.attitudesColorComboBox.currentText()

        else:
            return

        attitudes = point_layers[input_layer_index].data

        georef_attitudes = extract_georeferenced_attitudes(
            geodataframe=attitudes,
            azim_fldnm=attitude_azimuth_angle_fldnm,
            dip_ang_fldnm=attitude_dip_angle_fldnm,
            id_fldnm=attitude_id_fldnm,
            is_rhrstrike=azimuth_is_strikerhr
        )

        mapping_method = {}
        if projection_nearest_intersection:
            mapping_method['method'] = 'nearest'
        elif projection_constant_axis:
            mapping_method['method'] = 'common axis'
            mapping_method['trend'] = projection_axis_trend_angle
            mapping_method['plunge'] = projection_axis_plunge_angle
        elif projection_axes_from_fields:
            mapping_method['method'] = 'individual axes'
            axes_values = []
            for projection_axes_trend, projection_axes_plunge in zip(attitudes[projection_axes_trend_fldnm], attitudes[projection_axes_plunge_fldnm]):
                axes_values.append((projection_axes_trend, projection_axes_plunge))
            mapping_method['individual_axes_values'] = axes_values
        else:
            raise Exception("Debug_ mapping method not correctly defined")

        att_projs = self.profiler.map_georef_attitudes_to_section(
            structural_data=georef_attitudes,
            mapping_method=mapping_method,
            height_source=self.chosen_dem
        )

        self.geoprofiles.profile_attitudes = att_projs

        print("Plotting")

        self.fig = plot(
            self.geoprofiles,
            superposed=self.superposed_profiles,
            attitude_color=attitudes_color,
            labels_add_orientdip=labels_add_orientdip,
            labels_add_id=labels_add_id
        )

        self.fig.show()

    def intersect_lines(self):

        line_layers = list(filter(lambda dataset: containsLines(dataset.data), self.vector_datasets))

        if not line_layers:
            warn(self,
                 self.plugin_name,
                 "No line layer available")
            return

        dialog = LinesIntersectionDefWindow(
            self.plugin_name,
            line_layers
        )

        if dialog.exec_():

            input_layer_index = dialog.inputLayercomboBox.currentIndex()

            """
            azimuth_is_dipdir = dialog.azimuthDipDirRadioButton.isChecked()
            azimuth_is_strikerhr = dialog.azimuthRHRStrikeRadioButton.isChecked()

            attitude_id_fldnm = dialog.idFldNmComboBox.currentText()
            attitude_azimuth_angle_fldnm = dialog.attitudeAzimAngleFldNmComboBox.currentText()
            attitude_dip_angle_fldnm = dialog.attitudeDipAngleFldNmcomboBox.currentText()

            projection_nearest_intersection = dialog.projectNearestIntersectionRadioButton.isChecked()
            projection_constant_axis = dialog.projectAxisWithTrendRadioButton.isChecked()
            projection_axes_from_fields = dialog.projectAxesFromFieldsRadioButton.isChecked()

            projection_axis_trend_angle = dialog.projectAxisTrendAngDblSpinBox.value()
            projection_axis_plunge_angle = dialog.projectAxisPlungeAngDblSpinBox.value()

            projection_axes_trend_fldnm = dialog.projectAxesTrendFldNmComboBox.currentText()
            projection_axes_plunge_fldnm = dialog.projectAxesPlungeFldNmComboBox.currentText()

            labels_add_orientdip = dialog.labelsOrDipCheckBox.isChecked()
            labels_add_id = dialog.labelsIdCheckBox.isChecked()

            attitudes_color = dialog.attitudesColorComboBox.currentText()
            """

        else:
            return

        #attitudes = line_layers[input_layer_index].data


class ChooseSourceDataDialog(QDialog):

    def __init__(
        self,
        plugin_name: str,
        data_sources: List[str],
        parent=None
    ):

        super(ChooseSourceDataDialog, self).__init__(parent)

        self.plugin_name = plugin_name

        self.data_layers = data_sources

        self.listData_treeWidget = QTreeWidget()
        self.listData_treeWidget.setColumnCount(1)
        self.listData_treeWidget.headerItem().setText(0, "Name")
        self.listData_treeWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.listData_treeWidget.setDragEnabled(False)
        self.listData_treeWidget.setDragDropMode(QAbstractItemView.NoDragDrop)
        self.listData_treeWidget.setAlternatingRowColors(True)
        self.listData_treeWidget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.listData_treeWidget.setTextElideMode(Qt.ElideLeft)

        self.populate_layers_treewidget()

        self.listData_treeWidget.resizeColumnToContents(0)
        self.listData_treeWidget.resizeColumnToContents(1)

        okButton = QPushButton("&OK")
        cancelButton = QPushButton("Cancel")

        buttonLayout = QHBoxLayout()
        buttonLayout.addStretch()
        buttonLayout.addWidget(okButton)
        buttonLayout.addWidget(cancelButton)

        layout = QGridLayout()

        layout.addWidget(self.listData_treeWidget, 0, 0, 1, 3)
        layout.addLayout(buttonLayout, 1, 0, 1, 3)

        self.setLayout(layout)

        okButton.clicked.connect(self.accept)
        cancelButton.clicked.connect(self.reject)

        self.setWindowTitle("Define data source")

    def populate_layers_treewidget(self):

        self.listData_treeWidget.clear()

        for raster_layer in self.data_layers:
            tree_item = QTreeWidgetItem(self.listData_treeWidget)
            tree_item.setText(0, raster_layer)


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

    def __init__(self,
        plugin_name: str,
        point_layers: List
    ):

        super().__init__()

        self.plugin_name = plugin_name

        uic.loadUi('./widgets/project_attitudes.ui', self)

        self.point_layers = point_layers

        data_sources = map(lambda data_par: os.path.basename(data_par.filePath), self.point_layers)

        self.inputPtLayerComboBox.insertItems(0, data_sources)
        self.inputPtLayerComboBox.currentIndexChanged.connect(self.layer_index_changed)

        start_layer = self.point_layers[0]
        fields = start_layer.data.columns

        self.idFldNmComboBox.insertItems(0, fields)
        self.attitudeAzimAngleFldNmComboBox.insertItems(0, fields)
        self.attitudeDipAngleFldNmcomboBox.insertItems(0, fields)

        self.azimuthDipDirRadioButton.setChecked(True)
        self.projectNearestIntersectionRadioButton.setChecked(True)

        self.projectAxesTrendFldNmComboBox.insertItems(0, fields)
        self.projectAxesPlungeFldNmComboBox.insertItems(0, fields)

        self.attitudesColorComboBox.insertItems(0, attitude_colors)

        self.setWindowTitle("Project geological attitudes")

    def layer_index_changed(self, ndx: numbers.Integral):
        """

        :param ndx:
        :return:
        """

        current_lyr = self.point_layers[ndx]
        fields = current_lyr.data.columns

        self.idFldNmComboBox.clear()
        self.idFldNmComboBox.insertItems(0, fields)

        self.attitudeAzimAngleFldNmComboBox.clear()
        self.attitudeAzimAngleFldNmComboBox.insertItems(0, fields)

        self.attitudeDipAngleFldNmcomboBox.clear()
        self.attitudeDipAngleFldNmcomboBox.insertItems(0, fields)

        self.projectAxesTrendFldNmComboBox.clear()
        self.projectAxesTrendFldNmComboBox.insertItems(0, fields)

        self.projectAxesPlungeFldNmComboBox.clear()
        self.projectAxesPlungeFldNmComboBox.insertItems(0, fields)


class LinesIntersectionDefWindow(QtWidgets.QDialog):

    def __init__(self,
        plugin_name: str,
        line_layers: List
    ):

        super().__init__()

        self.plugin_name = plugin_name

        uic.loadUi('./widgets/line_intersections.ui', self)

        self.line_layers = line_layers

        data_sources = map(lambda data_par: os.path.basename(data_par.filePath), self.line_layers)

        self.inputLayercomboBox.insertItems(0, data_sources)
        self.inputLayercomboBox.currentIndexChanged.connect(self.layer_index_changed)

        start_layer = self.point_layers[0]
        fields = start_layer.data.columns

        self.idFldNmComboBox.insertItems(0, fields)
        self.attitudeAzimAngleFldNmComboBox.insertItems(0, fields)
        self.attitudeDipAngleFldNmcomboBox.insertItems(0, fields)

        self.azimuthDipDirRadioButton.setChecked(True)
        self.projectNearestIntersectionRadioButton.setChecked(True)

        self.projectAxesTrendFldNmComboBox.insertItems(0, fields)
        self.projectAxesPlungeFldNmComboBox.insertItems(0, fields)

        self.attitudesColorComboBox.insertItems(0, attitude_colors)



        self.setWindowTitle("Line intersections")


if __name__ == "__main__":
    
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    sys.exit(app.exec_())


