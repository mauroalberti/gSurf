import sys

from typing import List

from collections import namedtuple

import pyproj
import fiona


from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets #, uic

from pygsf.spatial.rasters.io import *
from pygsf.spatial.vectorial.io import try_read_as_geodataframe
from pygsf.spatial.vectorial.geodataframes import *
from pygsf.utils.qt.tools import *

from pygsf.spatial.geology.profiles.geoprofiles import GeoProfile, GeoProfileSet
from pygsf.spatial.geology.profiles.profilers import *
from pygsf.spatial.geology.profiles.plot import plot
from pygsf.spatial.geology.convert import try_extract_georeferenced_attitudes


DataPametersFldNms = [
    "filePath",
    "data",
    "type",
    "epsg_code"
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

color_palettes = [
    "Pastel1",
    "Pastel2",
    "Paired",
    "Accent",
    "Dark2",
    "Set1",
    "Set2",
    "Set3",
    "tab10",
    "tab20",
    "tab20b",
    "tab20c"
]


def get_selected_layer_index(
    treewidget_data_list: QtWidgets.QTreeWidget
) -> Optional[numbers.Integral]:

    if not treewidget_data_list:
        return None
    elif not treewidget_data_list.selectedIndexes():
        return None
    else:
        return treewidget_data_list.selectedIndexes()[0].row()


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):

        self.menubar = QtWidgets.QMenuBar(MainWindow)

        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setTitle("File")

        self.actLoadDem = QtWidgets.QAction(MainWindow)
        self.actLoadDem.setText("Load DEM")
        self.menuFile.addAction(self.actLoadDem)

        self.actLoadVectorLayer = QtWidgets.QAction(MainWindow)
        self.actLoadVectorLayer.setText("Load vector layer")
        self.menuFile.addAction(self.actLoadVectorLayer)

        self.menuFile.addSeparator()

        self.actionQuit = QtWidgets.QAction(MainWindow)
        self.actionQuit.setText("Quit")
        self.menuFile.addAction(self.actionQuit)

        self.menubar.addAction(self.menuFile.menuAction())

        self.menuProcessing = QtWidgets.QMenu(self.menubar)
        self.menuProcessing.setTitle("Processing")

        self.actOpenDemIntersection = QtWidgets.QAction(MainWindow)
        self.actOpenDemIntersection.setText("Plane-DEM intersections")
        self.menuProcessing.addAction(self.actOpenDemIntersection)

        self.menuProfiles = QtWidgets.QMenu(self.menuProcessing)
        self.menuProfiles.setTitle("Profiles")

        self.actChooseDEMs = QtWidgets.QAction(MainWindow)
        self.actChooseDEMs.setText("Choose DEM")
        self.menuProfiles.addAction(self.actChooseDEMs)

        self.menuProcessing.addAction(self.menuProfiles.menuAction())

        self.actOpenStereoplot = QtWidgets.QAction(MainWindow)
        self.actOpenStereoplot.setText("Stereoplot")
        self.menuProcessing.addAction(self.actOpenStereoplot)

        self.menubar.addAction(self.menuProcessing.menuAction())

        self.menuInfo = QtWidgets.QMenu(self.menubar)
        self.menuInfo.setTitle("Info")

        self.actionHelp = QtWidgets.QAction(MainWindow)
        self.actionHelp.setText("Help")
        self.menuInfo.addAction(self.actionHelp)

        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setText("About")
        self.menuInfo.addAction(self.actionAbout)

        self.menubar.addAction(self.menuInfo.menuAction())

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)

        MainWindow.setMenuBar(self.menubar)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):

        super().__init__(parent)

        '''
        uic.loadUi(
            './widgets/gSurf_0.3.0.ui',
            self
        )
        '''

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.plugin_name = "gSurf"
        self.chosen_dem = None
        self.working_epsg_code = None
        self.chosen_profile_data = None
        self.fig = None

        # File menu

        self.ui.actLoadDem.triggered.connect(self.load_dem)
        self.ui.actLoadVectorLayer.triggered.connect(self.load_vector_layer)

        # Plane-DEM intersections menu

        self.ui.actOpenDemIntersection.triggered.connect(self.open_dem_intersection_win)

        # Profiles menu

        self.ui.actChooseDEMs.triggered.connect(self.define_used_dem)
        #self.actChooseLines.triggered.connect(self.define_used_profile_dataset)
        """
        self.actionCreateSingleProfile.triggered.connect(self.create_single_profile)
        self.actionCreateMultipleParallelProfiles.triggered.connect(self.create_multi_parallel_profiles)
        self.actProjectGeolAttitudes.triggered.connect(self.project_attitudes)
        self.actIntersectLineLayer.triggered.connect(self.intersect_lines)
        self.actIntersectPolygonLayer.triggered.connect(self.intersect_polygons)
        """

        # data storage

        self.dems = []
        self.vector_datasets = []

        # data choices

        self.selected_dem_index = []
        self.selected_profile_index = []


        # window visibility

        self.show()

    def open_dem_intersection_win(self):

        line_layers = lines_datasets = list(filter(lambda dataset: containsLines(dataset.data), self.vector_datasets))

        dialog = PlaneDemIntersWindow(
            self.plugin_name,
            self.dems,
            line_layers
        )

        dialog.exec_()

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

        array, affine_transform, epsg_code = result

        if epsg_code == -1:

            dialog = EPSGCodeDefineWindow(
                self.plugin_name
            )

            if dialog.exec_():

                epsg_code = dialog.EPSGCodeSpinBox.value()

            else:

                QMessageBox.warning(
                    None,
                    "Raster input",
                    "No EPSG code defined"
                )
                return

        ga = GeoArray.fromRasterio(
            array=array,
            affine_transform=affine_transform,
            epsg_code=epsg_code
        )

        self.dems.append(
            DataParameters(
                filePath,
                ga,
                "DEM",
                epsg_code
            )
        )

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

        geodataframe = result

        crs = geodataframe.crs

        if isinstance(crs, str):
            crs = fiona.crs.from_string(crs)

        if crs and "init" in crs and crs["init"].lower().startswith("epsg:"):

            epsg_code = int(crs["init"].split(":")[1])

        else:

            epsg_code = -1

        if epsg_code == -1:

            dialog = EPSGCodeDefineWindow(
                self.plugin_name
            )

            if dialog.exec_():

                epsg_code = dialog.EPSGCodeSpinBox.value()

            else:

                QMessageBox.warning(
                    None,
                    "Raster input",
                    "No EPSG code defined"
                )
                return

        self.vector_datasets.append(
            DataParameters(
                filePath,
                geodataframe,
                "vector",
                epsg_code
            )
        )

        QMessageBox.information(
            None,
            "Input",
            "Dataset read"
        )

    def choose_dataset_index(
        self,
        datasets_paths: List[str]
    ) -> Optional[numbers.Integral]:
        """
        Choose data to use for profile creation.

        :param datasets_paths: the list of data sources
        :type datasets_paths: List[str]
        :return: the selected data, as index of the input list
        :rtype: Optional[numbers.Integral]
        """

        dialog = ChooseSourceDataDialog(
            self.plugin_name,
            data_sources=list(map(lambda pth: os.path.basename(pth), datasets_paths))
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
            print(f"DEM EPSG code: {self.chosen_dem.epsg_code()}")
            self.working_epsg_code = self.chosen_dem.epsg_code()

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
            self.chosen_profile = lines_datasets[self.selected_profile_index]
            self.chosen_profile_data = lines_datasets[self.selected_profile_index].data

    def create_single_profile(self):

        self.superposed_profiles = False

        pts = extract_line_points(
            geodataframe=self.chosen_profile.data,
            ndx=0,
            epsg_code=self.chosen_profile.epsg_code
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

        if self.fig:
            self.fig.show()
        else:
            warn(self,
                 self.plugin_name,
                 "Figure cannot be generated.\nPossible DEM-profile extent mismatch?"
                 )
            return

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
            geodataframe=self.chosen_profile.data,
            ndx=0,
            epsg_code=self.chosen_profile.epsg_code
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

        self.profiler = ParallelProfiler.fromBaseProfiler(
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

        if self.fig:
            self.fig.show()
        else:
            warn(
                self,
                self.plugin_name,
                "Figure cannot be generated.\nPossible DEM-profile extent mismatch?"
            )

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

            projection_max_distance_from_profile = dialog.maxDistFromProfDoubleSpinBox.value()

            labels_add_orientdip = dialog.labelsOrDipCheckBox.isChecked()
            labels_add_id = dialog.labelsIdCheckBox.isChecked()

            attitudes_color = dialog.attitudesColorComboBox.currentText()

        else:

            return

        attitudes = point_layers[input_layer_index].data

        success, result = try_extract_georeferenced_attitudes(
            geodataframe=attitudes,
            azim_fldnm=attitude_azimuth_angle_fldnm,
            dip_ang_fldnm=attitude_dip_angle_fldnm,
            id_fldnm=attitude_id_fldnm,
            is_rhrstrike=azimuth_is_strikerhr
        )

        if not success:
            msg = result
            warn(
                self,
                self.plugin_name,
                "Error with georeferenced attitudes extraction: {}".format(msg)
            )
            return

        georef_attitudes = result

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

        attitudes_3d = georef_attitudes_3d_from_grid(
            structural_data=georef_attitudes,
            height_source=self.chosen_dem,
        )

        att_projs = self.profiler.map_georef_attitudes_to_section(
            attitudes_3d=attitudes_3d,
            mapping_method=mapping_method,
            max_profile_distance=projection_max_distance_from_profile
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

        if self.fig:

            self.fig.show()

        else:

            warn(
                self,
                self.plugin_name,
                "Unable to create figure"
            )

    def intersect_lines(self):

        mline_layers = list(filter(lambda dataset: containsLines(dataset.data), self.vector_datasets))

        if not mline_layers:
            warn(self,
                 self.plugin_name,
                 "No line layer available")
            return

        dialog = LinesIntersectionDefWindow(
            self.plugin_name,
            mline_layers
        )

        if dialog.exec_():

            input_layer_index = dialog.inputLayercomboBox.currentIndex()
            category_fldnm = dialog.labelFieldcomboBox.currentText()
            add_labels = dialog.addLabelcheckBox.isChecked()

        else:

            return

        mlines_geoms = mline_layers[input_layer_index].data
        mlines_geoms = mlines_geoms[~mlines_geoms.is_empty]

        profiler_pyproj_epsg = f"EPSG:{self.profiler.epsg_code()}"
        if not mlines_geoms.crs == pyproj.Proj(profiler_pyproj_epsg):
            mlines_geoms = mlines_geoms.to_crs(
                epsg=self.profiler.epsg_code()
            )

        toprocess_geometries = []

        print("Input geometries")

        for index, row in mlines_geoms.iterrows():

            print(index, row)

            geom_cat = row[category_fldnm]
            mline_geometry = row["geometry"]

            if mline_geometry:

                geometry = line_from_shapely(
                    shapely_geom=mline_geometry,
                    epsg_code=self.profiler.epsg_code()
                )

                print(geometry)

                toprocess_geometries.append(
                    (geom_cat, geometry)
                )

        if isinstance(self.profiler, LinearProfiler):

            print("Intersections")

            intersections_cat_geom = []

            for geom_cat, geometry in toprocess_geometries:

                ptsegm_intersections = self.profiler.intersect_line(
                    mline=geometry
                )

                print(geom_cat, ptsegm_intersections)

                if ptsegm_intersections:

                    intersections_cat_geom.append((geom_cat, ptsegm_intersections))

            profile_intersections = PointSegmentCollections(intersections_cat_geom)

            self.geoprofiles.lines_intersections = self.profiler.parse_intersections_for_profile(profile_intersections)

        elif isinstance(self.profiler, ParallelProfiler):

            profiles_intersections = []

            for profile in self.profiler:

                intersections_cat_geom = []

                for geom_cat, geometry in toprocess_geometries:

                    pt_segm_collection = PointSegmentCollection(
                        element_id=geom_cat,
                        geoms=profile.intersect_line(geometry)
                    )

                    intersections_cat_geom.append(pt_segm_collection)

                profile_intersections = PointSegmentCollections(intersections_cat_geom)
                profiles_intersections.append(profile_intersections)

            lines_intersections_set = PointSegmentCollectionsSet(profiles_intersections)
            self.geoprofiles.lines_intersections_set = lines_intersections_set

        else:

            raise Exception("Expected LinearProfiler or ParallelProfiles, got {}".format(type(self.profiler)))

        print("Plotting")

        self.fig = plot(
            self.geoprofiles,
            superposed=self.superposed_profiles,
            inters_label=add_labels
        )

        if self.fig:

            self.fig.show()

        else:

            warn(
                self,
                self.plugin_name,
                "Unable to create figure"
            )

    def intersect_polygons(self):

        mpolygon_layers = list(filter(lambda dataset: containsPolygons(dataset.data), self.vector_datasets))

        if not mpolygon_layers:
            warn(self,
                 self.plugin_name,
                 "No polygon layer available")
            return

        dialog = PolygonsIntersectionDefWindow(
            self.plugin_name,
            mpolygon_layers
        )

        if dialog.exec_():

            input_layer_index = dialog.polygonLayercomboBox.currentIndex()
            category_fldnm = dialog.classificationFieldcomboBox.currentText()
            add_labels = dialog.addLabelcheckBox.isChecked()

        else:

            return

        mpolygons_geoms = mpolygon_layers[input_layer_index].data
        mpolygons_geoms = mpolygons_geoms[~mpolygons_geoms.is_empty]

        profiler_pyproj_epsg = f"EPSG:{self.profiler.epsg_code()}"
        if not mpolygons_geoms.crs == pyproj.Proj(profiler_pyproj_epsg):
            mpolygons_geoms = mpolygons_geoms.to_crs(
                epsg=self.profiler.epsg_code()
            )

        toprocess_geometries = []

        for index, row in mpolygons_geoms.iterrows():

            geom_cat = row[category_fldnm]
            mpolygon_geometry = row["geometry"]

            if mpolygon_geometry:

                geometry = MPolygon(
                        shapely_geom=mpolygon_geometry,
                        epsg_code=self.profiler.epsg_code()
                )

                toprocess_geometries.append(
                    (geom_cat, geometry)
                )

        pt_segment_intersections = []

        if isinstance(self.profiler, LinearProfiler):

            for geom_cat, geometry in toprocess_geometries:

                #print(type(geometry))
                intersections = self.profiler.intersect_polygon(
                    mpolygon=geometry
                )

                if intersections:
                    pt_segment_intersections.append((geom_cat, intersections))

        for cat, inters in pt_segment_intersections:
            print(cat, inters)

        """
        for cat, intersections in imported_polygons:
            print(cat)
            for intersection in intersections:
                print(intersection)
        """

        """
        polygons_intersections_set = PointSegmentCollectionsSet(intersection_sections)

        self.geoprofiles.polygons_intersections_set = polygons_intersections_set
        print("Plotting")

        self.fig = plot(
            self.geoprofiles,
            superposed=self.superposed_profiles,
            inters_label=add_labels
        )

        if self.fig:

            self.fig.show()

        else:

            warn(
                self,
                self.plugin_name,
                "Unable to create figure"
            )
        """


class PlaneDemIntersWindow(QDialog):

    def __init__(self,
        plugin_name: str,
        dems,
        line_layers
    ):

        super().__init__()

        self.plugin_name = plugin_name

        uic.loadUi('./widgets/intersections.ui', self)

        dem_sources = [os.path.basename(dem.filePath) for dem in dems]
        self.InputDemComboBox.insertItems(0, dem_sources)

        lines_sources = map(lambda data_par: os.path.basename(data_par.filePath), line_layers)
        self.InputTracesComboBox.insertItems(0, lines_sources)


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
        self.numberOfProfilesSpinBox.setValue(5)
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

        start_layer = self.line_layers[0]
        fields = start_layer.data.columns

        self.labelFieldcomboBox.insertItems(0, fields)

        self.setWindowTitle("Line intersections")

    def layer_index_changed(self, ndx: numbers.Integral):
        """

        :param ndx:
        :return:
        """

        current_lyr = self.line_layers[ndx]
        fields = current_lyr.data.columns

        self.labelFieldcomboBox.clear()
        self.labelFieldcomboBox.insertItems(0, fields)


class PolygonsIntersectionDefWindow(QtWidgets.QDialog):

    def __init__(self,
                 plugin_name: str,
                 polygon_layers: List
                 ):

        super().__init__()

        self.plugin_name = plugin_name

        uic.loadUi('./widgets/polygons_intersections.ui', self)

        self.polygon_layers = polygon_layers

        data_sources = map(lambda data_par: os.path.basename(data_par.filePath), self.polygon_layers)

        self.polygonLayercomboBox.insertItems(0, data_sources)
        self.polygonLayercomboBox.currentIndexChanged.connect(self.layer_index_changed)

        start_layer = self.polygon_layers[0]
        fields = start_layer.data.columns

        self.classificationFieldcomboBox.insertItems(0, fields)

        self.setWindowTitle("Polygon intersections")

    def layer_index_changed(self, ndx: numbers.Integral):
        """

        :param ndx:
        :return:
        """

        current_lyr = self.polygon_layers[ndx]
        fields = current_lyr.data.columns

        self.classificationFieldcomboBox.clear()
        self.classificationFieldcomboBox.insertItems(0, fields)


class EPSGCodeDefineWindow(QtWidgets.QDialog):

    def __init__(self,
        plugin_name: str
    ):

        super().__init__()

        self.plugin_name = plugin_name

        uic.loadUi('./widgets/define_epsg_code.ui', self)


if __name__ == "__main__":
    
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    sys.exit(app.exec_())


