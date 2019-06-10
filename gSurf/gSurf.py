# -*- coding: utf-8 -*-

"""
/***************************************************************************
 geoSurf

 DEM - planes intersections
                              -------------------
        begin                : 2011-12-21
        version              : 0.1.2 - 2012-03-10
        copyright            : (C) 2011-2012 by Mauro Alberti - www.malg.eu
        email                : alberti.m65@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

import os
import sys

from math import isnan, floor

import webbrowser

from PyQt5 import QtWidgets

from matplotlib.offsetbox import AnchoredOffsetbox, TextArea

from gSurf_ui import Ui_MainWindow

from gis_utils.rasters import *
from gsf.geometry import *
from gsf.grids import *


__version__ = "0.2.0"


class IntersectionParameters(object):
    """
    IntersectionParameters class.
    Manages the metadata for spdata results (DEM source filename, source point, plane attitude.

    """

    def __init__(self, sourcename, src_pt, src_plane_attitude):
        """
        Class constructor.

        @param sourcename: name of the DEM used to create the grid.
        @type sourcename: String.
        @param src_plane_attitude: orientation of the plane used to calculate the spdata.
        @type src_plane_attitude: class StructPlane.

        @return: self
        """
        self._sourcename = sourcename
        self._srcPt = src_pt
        self._srcPlaneAttitude = src_plane_attitude


class Traces(object):

    def __init__(self):
        self.lines_x, self.lines_y = [], []
        self.extent_x = [0, 100]
        self.extent_y = [0, 100]


class Intersections(object):

    def __init__(self):
        self.parameters = None

        self.xcoords_x = []
        self.xcoords_y = []
        self.ycoords_x = []
        self.ycoords_y = []

        self.links = None
        self.networks = {}


class GeoData(object):

    def set_dem_default(self):

        self.dem = None

    def set_vector_default(self):

        # Fault traces data
        self.traces = Traces()

    def set_intersections_default(self):
        """
        Set result values to null.
        """
        self.inters = Intersections()

    def __init__(self):

        self.set_dem_default()
        self.set_vector_default()
        self.set_intersections_default()

    def read_traces(self, in_traces_shp):
        """
        Read line shapefile.

        @param  in_traces_shp:  parameter to check.
        @type  in_traces_shp:  QString or string

        """
        # reset layer parameters

        self.set_vector_default()

        if in_traces_shp is None or in_traces_shp == '':
            return

            # open input shapefile
        shape_driver = ogr.GetDriverByName("ESRI Shapefile")

        in_shape = shape_driver.Open(str(in_traces_shp), 0)

        if in_shape is None:
            return

        # get internal layer
        lnLayer = in_shape.GetLayer(0)

        # set vector layer extent
        self.traces.extent_x[0], self.traces.extent_x[1], \
        self.traces.extent_y[0], self.traces.extent_y[1] \
            = lnLayer.GetExtent()

        # start reading layer features
        curr_line = lnLayer.GetNextFeature()

        # loop in layer features
        while curr_line:

            line_vert_x, line_vert_y = [], []

            line_geom = curr_line.GetGeometryRef()

            if line_geom is None:
                in_shape.Destroy()
                return

            if line_geom.GetGeometryType() != ogr.wkbLineString and \
                    line_geom.GetGeometryType() != ogr.wkbMultiLineString:
                in_shape.Destroy()
                return

            for i in range(line_geom.GetPointCount()):
                x, y = line_geom.GetX(i), line_geom.GetY(i)

                line_vert_x.append(x)
                line_vert_y.append(y)

            self.traces.lines_x.append(line_vert_x)
            self.traces.lines_y.append(line_vert_y)

            curr_line = lnLayer.GetNextFeature()

        in_shape.Destroy()

    def get_intersections(self):
        """
        Initialize a structured array of the possible and found links for each intersection.
        It will store a list of the possible connections for each intersection,
        together with the found connections.
        """

        # data type for structured array storing intersection parameters
        dt = np.dtype([('id', np.uint16),
                       ('i', np.uint16),
                       ('j', np.uint16),
                       ('pi_dir', np.str_, 1),
                       ('conn_from', np.uint16),
                       ('conn_to', np.uint16),
                       ('start', np.bool_)])

        # number of valid intersections
        num_intersections = len(list(self.inters.xcoords_x[np.logical_not(np.isnan(self.inters.xcoords_x))])) + \
                            len(list(self.inters.ycoords_y[np.logical_not(np.isnan(self.inters.ycoords_y))]))

        # creation and initialization of structured array of valid intersections in the x-direction
        links = np.zeros(num_intersections, dtype=dt)

        # filling array with values

        curr_ndx = 0
        for i in range(self.inters.xcoords_x.shape[0]):
            for j in range(self.inters.xcoords_x.shape[1]):
                if not isnan(self.inters.xcoords_x[i, j]):
                    links[curr_ndx] = (curr_ndx + 1, i, j, 'x', 0, 0, False)
                    curr_ndx += 1

        for i in range(self.inters.ycoords_y.shape[0]):
            for j in range(self.inters.ycoords_y.shape[1]):
                if not isnan(self.inters.ycoords_y[i, j]):
                    links[curr_ndx] = (curr_ndx + 1, i, j, 'y', 0, 0, False)
                    curr_ndx += 1

        return links

    def set_neighbours(self):

        # shape of input arrays (equal shapes)
        num_rows, num_cols = self.inters.xcoords_x.shape

        # dictionary storing intersection links
        neighbours = {}

        # search and connect intersection points
        for curr_ndx in range(self.inters.links.shape[0]):

            # get current point location (i, j) and direction type (pi_dir)
            curr_id = self.inters.links[curr_ndx]['id']
            curr_i = self.inters.links[curr_ndx]['i']
            curr_j = self.inters.links[curr_ndx]['j']
            curr_dir = self.inters.links[curr_ndx]['pi_dir']

            # check possible connected spdata
            near_intersections = []

            if curr_dir == 'x':

                if curr_i < num_rows - 1 and curr_j < num_cols - 1:

                    try:  # -- A
                        id_link = self.inters.links[(self.inters.links['i'] == curr_i + 1) & \
                                                    (self.inters.links['j'] == curr_j + 1) & \
                                                    (self.inters.links['pi_dir'] == 'y')]['id']
                        if len(list(id_link)) == 1:
                            near_intersections.append(id_link[0])
                    except:
                        pass
                    try:  # -- B
                        id_link = self.inters.links[(self.inters.links['i'] == curr_i + 1) & \
                                                    (self.inters.links['j'] == curr_j) & \
                                                    (self.inters.links['pi_dir'] == 'x')]['id']
                        if len(list(id_link)) == 1:
                            near_intersections.append(id_link[0])
                    except:
                        pass
                    try:  # -- C
                        id_link = self.inters.links[(self.inters.links['i'] == curr_i + 1) & \
                                                    (self.inters.links['j'] == curr_j) & \
                                                    (self.inters.links['pi_dir'] == 'y')]['id']
                        if len(list(id_link)) == 1:
                            near_intersections.append(id_link[0])
                    except:
                        pass

                if curr_i > 0 and curr_j < num_cols - 1:

                    try:  # -- E
                        id_link = self.inters.links[(self.inters.links['i'] == curr_i) & \
                                                    (self.inters.links['j'] == curr_j) & \
                                                    (self.inters.links['pi_dir'] == 'y')]['id']
                        if len(list(id_link)) == 1:
                            near_intersections.append(id_link[0])
                    except:
                        pass
                    try:  # -- F
                        id_link = self.inters.links[(self.inters.links['i'] == curr_i - 1) & \
                                                    (self.inters.links['j'] == curr_j) & \
                                                    (self.inters.links['pi_dir'] == 'x')]['id']
                        if len(list(id_link)) == 1:
                            near_intersections.append(id_link[0])
                    except:
                        pass
                    try:  # -- G
                        id_link = self.inters.links[(self.inters.links['i'] == curr_i) & \
                                                    (self.inters.links['j'] == curr_j + 1) & \
                                                    (self.inters.links['pi_dir'] == 'y')]['id']
                        if len(list(id_link)) == 1:
                            near_intersections.append(id_link[0])
                    except:
                        pass

            if curr_dir == 'y':

                if curr_i > 0 and curr_j < num_cols - 1:

                    try:  # -- D
                        id_link = self.inters.links[(self.inters.links['i'] == curr_i) & \
                                                    (self.inters.links['j'] == curr_j) & \
                                                    (self.inters.links['pi_dir'] == 'x')]['id']
                        if len(list(id_link)) == 1:
                            near_intersections.append(id_link[0])
                    except:
                        pass
                    try:  # -- F
                        id_link = self.inters.links[(self.inters.links['i'] == curr_i - 1) & \
                                                    (self.inters.links['j'] == curr_j) & \
                                                    (self.inters.links['pi_dir'] == 'x')]['id']
                        if len(list(id_link)) == 1:
                            near_intersections.append(id_link[0])
                    except:
                        pass
                    try:  # -- G
                        id_link = self.inters.links[(self.inters.links['i'] == curr_i) & \
                                                    (self.inters.links['j'] == curr_j + 1) & \
                                                    (self.inters.links['pi_dir'] == 'y')]['id']
                        if len(list(id_link)) == 1:
                            near_intersections.append(id_link[0])
                    except:
                        pass

                if curr_i > 0 and curr_j > 0:

                    try:  # -- H
                        id_link = self.inters.links[(self.inters.links['i'] == curr_i) & \
                                                    (self.inters.links['j'] == curr_j - 1) & \
                                                    (self.inters.links['pi_dir'] == 'x')]['id']
                        if len(list(id_link)) == 1:
                            near_intersections.append(id_link[0])
                    except:
                        pass
                    try:  # -- I
                        id_link = self.inters.links[(self.inters.links['i'] == curr_i) & \
                                                    (self.inters.links['j'] == curr_j - 1) & \
                                                    (self.inters.links['pi_dir'] == 'y')]['id']
                        if len(list(id_link)) == 1:
                            near_intersections.append(id_link[0])
                    except:
                        pass
                    try:  # -- L
                        id_link = self.inters.links[(self.inters.links['i'] == curr_i - 1) & \
                                                    (self.inters.links['j'] == curr_j - 1) & \
                                                    (self.inters.links['pi_dir'] == 'x')]['id']
                        if len(list(id_link)) == 1:
                            near_intersections.append(id_link[0])
                    except:
                        pass

            neighbours[curr_id] = near_intersections

        return neighbours

    def follow_path(self, start_id):
        """
        Creates a path of connected intersections from a given start intersection.

        """
        from_id = start_id

        while self.inters.links[from_id - 1]['conn_to'] == 0:

            conns = self.inters.neighbours[from_id]
            num_conn = len(conns)
            if num_conn == 0:
                raise ConnectionError('no connected intersection')
            elif num_conn == 1:
                if self.inters.links[conns[0] - 1]['conn_from'] == 0 and self.inters.links[conns[0] - 1][
                    'conn_to'] != from_id:
                    to_id = conns[0]
                else:
                    raise ConnectionError('no free connection')
            elif num_conn == 2:
                if self.inters.links[conns[0] - 1]['conn_from'] == 0 and self.inters.links[conns[0] - 1][
                    'conn_to'] != from_id:
                    to_id = conns[0]
                elif self.inters.links[conns[1] - 1]['conn_from'] == 0 and self.inters.links[conns[1] - 1][
                    'conn_to'] != from_id:
                    to_id = conns[1]
                else:
                    raise ConnectionError('no free connection')
            else:
                raise ConnectionError('multiple connection')

            # set connection
            self.inters.links[to_id - 1]['conn_from'] = from_id
            self.inters.links[from_id - 1]['conn_to'] = to_id

            # prepare for next step
            from_id = to_id

    def path_closed(self, start_id):

        from_id = start_id

        while self.inters.links[from_id - 1]['conn_to'] != 0:

            to_id = self.inters.links[from_id - 1]['conn_to']

            if to_id == start_id: return True

            from_id = to_id

        return False

    def invert_path(self, start_id):

        self.inters.links[start_id - 1]['start'] = False

        curr_id = start_id

        while curr_id != 0:

            prev_from_id = self.inters.links[curr_id - 1]['conn_from']
            prev_to_id = self.inters.links[curr_id - 1]['conn_to']

            self.inters.links[curr_id - 1]['conn_from'] = prev_to_id
            self.inters.links[curr_id - 1]['conn_to'] = prev_from_id

            if self.inters.links[curr_id - 1]['conn_from'] == 0:
                self.inters.links[curr_id - 1]['start'] = True

            curr_id = prev_to_id

        return

    def patch_path(self, start_id):

        if self.path_closed(start_id):
            return

        from_id = start_id

        conns = self.inters.neighbours[from_id]
        try:
            conns.remove(self.inters.links[from_id - 1]['conn_to'])
        except:
            pass

        num_conn = len(conns)

        if num_conn != 1: return

        new_toid = self.inters.links[conns[0] - 1]

        if self.inters.links[new_toid]['conn_to'] > 0 \
                and self.inters.links[new_toid]['conn_to'] != from_id \
                and self.inters.links[new_toid]['conn_from'] == 0:

            if self.path_closed(new_toid): return
            self.invert_path(from_id)
            self.self.inters.links[from_id - 1]['conn_to'] = new_toid
            self.self.inters.links[new_toid - 1]['conn_from'] = from_id
            self.self.inters.links[new_toid - 1]['start'] = False

    def define_paths(self):

        # simple networks starting from border
        for ndx in range(self.inters.links.shape[0]):

            if len(self.inters.neighbours[ndx + 1]) != 1 or \
                    self.inters.links[ndx]['conn_from'] > 0 or \
                    self.inters.links[ndx]['conn_to'] > 0:
                continue

            try:
                self.follow_path(ndx + 1)
            except:
                continue

        # inner, simple networks

        for ndx in range(self.inters.links.shape[0]):

            if len(self.inters.neighbours[ndx + 1]) != 2 or \
                    self.inters.links[ndx]['conn_to'] > 0 or \
                    self.inters.links[ndx]['start'] == True:
                continue

            try:
                self.inters.links[ndx]['start'] = True
                self.follow_path(ndx + 1)
            except:
                continue

        # inner, simple networks, connection of FROM

        for ndx in range(self.inters.links.shape[0]):

            if len(self.inters.neighbours[ndx + 1]) == 2 and \
                    self.inters.links[ndx]['conn_from'] == 0:
                try:
                    self.patch_path(ndx + 1)
                except:
                    continue

    def define_networks(self):
        """
        Creates list of connected intersections,
        to output as line shapefile.
        """

        pid = 0
        networks = {}

        # open, simple networks
        for ndx in range(self.inters.links.shape[0]):

            if len(self.inters.neighbours[ndx + 1]) != 1:
                continue

            network_list = []

            to_ndx = ndx + 1

            while to_ndx != 0:
                network_list.append(to_ndx)

                to_ndx = self.inters.links[to_ndx - 1]['conn_to']

            if len(network_list) > 1:
                pid += 1

                networks[pid] = network_list

                # closed, simple networks
        for ndx in range(self.inters.links.shape[0]):

            if len(self.inters.neighbours[ndx + 1]) != 2 or not self.inters.links[ndx]['start']:
                continue

            start_id = ndx + 1

            network_list = []

            to_ndx = ndx + 1

            while to_ndx != 0:

                network_list.append(to_ndx)

                to_ndx = self.inters.links[to_ndx - 1]['conn_to']

                if to_ndx == start_id:
                    network_list.append(to_ndx)
                    break

            if len(network_list) > 1:
                pid += 1

                networks[pid] = network_list

        return networks


class MainWindow(QtWidgets.QMainWindow):
    """
    Principal GUI class
    
    """
    
    def __init__(self, parent=None):
        """
        Constructor
        
        """
        super(MainWindow, self).__init__(parent)
        
        # Set up the user interface from Designer.
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # initialize intersection drawing
        self.valid_intersections = False
        
        # initialize spdata

        self.spdata = GeoData()        

        # DEM connections

        self.ui.actionInput_DEM.triggered.connect(self.select_dem_file)
        self.ui.DEM_lineEdit.textChanged['QString'].connect(self.selected_dem)

        self.ui.show_DEM_checkBox.stateChanged['int'].connect(self.redraw_map)
        self.ui.DEM_cmap_comboBox.currentIndexChanged['QString'].connect(self.redraw_map)

        # Fault traces connections

        self.ui.actionInput_line_shapefile.triggered.connect(self.select_traces_file)
        self.ui.Trace_lineEdit.textChanged['QString'].connect(self.reading_traces)
        self.ui.show_Fault_checkBox.stateChanged['int'].connect(self.redraw_map)

        # Full zoom

        self.ui.mplwidget.zoom_to_full_view.connect(self.zoom_full_view)

        # Source point

        self.ui.mplwidget.map_press.connect(self.update_src_pt)

        self.ui.Pt_spinBox_x.valueChanged['int'].connect(self.set_z)
        self.ui.Pt_spinBox_y.valueChanged['int'].connect(self.set_z)
        self.ui.Z_fix2DEM_checkBox_z.stateChanged['int'].connect(self.set_z)
        self.ui.Pt_spinBox_z.valueChanged['int'].connect(self.set_z)

        self.ui.Pt_spinBox_x.valueChanged['int'].connect(self.redraw_map)
        self.ui.Pt_spinBox_y.valueChanged['int'].connect(self.redraw_map)
        self.ui.Pt_spinBox_z.valueChanged['int'].connect(self.redraw_map)

        self.ui.show_SrcPt_checkBox.stateChanged['int'].connect(self.redraw_map)

        # Plane orientation

        self.ui.DDirection_dial.valueChanged['int'].connect(self.update_dipdir_spinbox)
        self.ui.DDirection_spinBox.valueChanged['int'].connect(self.update_dipdir_slider)

        self.ui.DAngle_verticalSlider.valueChanged['int'].connect(self.update_dipang_spinbox)
        self.ui.DAngle_spinBox.valueChanged['int'].connect(self.update_dipang_slider)

        # Intersections
        self.ui.Intersection_calculate_pushButton.clicked['bool'].connect(self.calc_intersections)
        self.ui.Intersection_show_checkBox.stateChanged['int'].connect(self.redraw_map)
        self.ui.Intersection_color_comboBox.currentIndexChanged['QString'].connect(self.redraw_map)

        # Write result
        self.ui.actionPoints.triggered.connect(self.write_intersections_as_points)
        self.ui.actionLines.triggered.connect(self.write_intersections_as_lines)

        # Other actions
        self.ui.actionHelp.triggered.connect(self.openHelp)
        self.ui.actionAbout.triggered.connect(self.helpAbout)
        self.ui.actionQuit.triggered.connect(sys.exit)

    def draw_map(self, map_extent_x, map_extent_y):            
        """
        Draw the map content.
    
        @param  map_extent_x:  map extent along the x axis.
        @type  map_extent_x:  list of two float values (min x and max x).
        @param  map_extent_y:  map extent along the y axis.
        @type  map_extent_y:  list of two float values (min y and max y).
        """

        self.ui.mplwidget.canvas.ax.cla()
        
        # DEM processing

        if self.spdata.dem is not None:
                           
            geo_extent = [
                self.spdata.dem.domain.g_llcorner().x, self.spdata.dem.domain.g_trcorner().x,
                self.spdata.dem.domain.g_llcorner().y, self.spdata.dem.domain.g_trcorner().y]
            
            if self.ui.show_DEM_checkBox.isChecked():  # DEM check is on
                
                curr_colormap = str(self.ui.DEM_cmap_comboBox.currentText())
                     
                self.ui.mplwidget.canvas.ax.imshow(self.spdata.dem.data, extent=geo_extent,  cmap=curr_colormap)

        # Fault traces proc.

        if self.spdata.traces.lines_x is not None and self.spdata.traces.lines_y is not None \
           and self.ui.show_Fault_checkBox.isChecked():  # Fault check is on
 
            for currLine_x, currLine_y in zip(self.spdata.traces.lines_x, self.spdata.traces.lines_y):
                    self.ui.mplwidget.canvas.ax.plot(currLine_x, currLine_y,'-')

        # Intersections proc.

        if self.ui.Intersection_show_checkBox.isChecked() and self.valid_intersections:
            
            curr_color = str(self.ui.Intersection_color_comboBox.currentText())
                        
            intersections_x = list(self.spdata.inters.xcoords_x[np.logical_not(np.isnan(self.spdata.inters.xcoords_x))]) + \
                              list(self.spdata.inters.ycoords_x[np.logical_not(np.isnan(self.spdata.inters.ycoords_y))])
        
            intersections_y = list(self.spdata.inters.xcoords_y[np.logical_not(np.isnan(self.spdata.inters.xcoords_x))]) + \
                              list(self.spdata.inters.ycoords_y[np.logical_not(np.isnan(self.spdata.inters.ycoords_y))])

            self.ui.mplwidget.canvas.ax.plot(intersections_x, intersections_y,  "w+",  ms=2,  mec=curr_color,  mew=2)
                                
            legend_text = "Plane dip dir., angle: (%d, %d)\nSource point x, y, z: (%d, %d, %d)" % \
                (self.spdata.inters.parameters._srcPlaneAttitude._dipdir, self.spdata.inters.parameters._srcPlaneAttitude._dipangle,
                 self.spdata.inters.parameters._srcPt.x, self.spdata.inters.parameters._srcPt.y, self.spdata.inters.parameters._srcPt.z) 
                                             
            at = AnchoredText(
                legend_text,
                loc=2,
                frameon=True)
            
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            at.patch.set_alpha(0.5)
            self.ui.mplwidget.canvas.ax.add_artist(at)    

        # Source point proc.         
        curr_x = self.ui.Pt_spinBox_x.value()
        curr_y = self.ui.Pt_spinBox_y.value() 
               
        if self.ui.show_SrcPt_checkBox.isChecked():                  

            self.ui.mplwidget.canvas.ax.plot(curr_x, curr_y, "ro")
          
        self.ui.mplwidget.canvas.draw()           
          
        self.ui.mplwidget.canvas.ax.set_xlim(map_extent_x) 
        self.ui.mplwidget.canvas.ax.set_ylim(map_extent_y) 

    def refresh_map(self, map_extent_x=None, map_extent_y=None):
        """
        Update map.
    
        @param  map_extent_x:  map extent along the x axis.
        @type  map_extent_x:  list of two float values (min x and max x).
        @param  map_extent_y:  map extent along the y axis.
        @type  map_extent_y:  list of two float values (min y and max y).
        """

        if map_extent_x is None:
            map_extent_x = self.ui.mplwidget.canvas.ax.get_xlim()

        if map_extent_y is None:
            map_extent_y = self.ui.mplwidget.canvas.ax.get_ylim()
                
        self.draw_map(map_extent_x, map_extent_y)

    def redraw_map(self):
        """
        Convenience function for drawing the map.
        """
                      
        self.refresh_map()
                
    def zoom_full_view(self):
        """
        Update map view to the DEM extent or otherwise, if available, to the shapefile extent.
        """
       
        if self.spdata.dem is not None:
            map_extent_x = [self.spdata.dem.domain.g_llcorner().x, self.spdata.dem.domain.g_trcorner().x]
            map_extent_y = [self.spdata.dem.domain.g_llcorner().y, self.spdata.dem.domain.g_trcorner().y]
            
        elif self.spdata.traces.extent_x != [] and self.spdata.traces.extent_y != []:
            map_extent_x = self.spdata.traces.extent_x
            map_extent_y = self.spdata.traces.extent_y

        else:
            map_extent_x = [0, 100]
            map_extent_y = [0, 100]
                                
        self.refresh_map(map_extent_x, map_extent_y)
        
    def select_dem_file(self):
        """
        Select input DEM file
        
        """

        fileName = QtWidgets.QFileDialog.getOpenFileName(self, self.tr("Open DEM file (using GDAL)"), '', "*.*")
        file_path = fileName[0]
        if not file_path:
            return          

        self.ui.DEM_lineEdit.setText(file_path)

    def selected_dem(self):

        in_dem_fn = self.ui.DEM_lineEdit.text()

        if not in_dem_fn:
            return

        try:
            self.spdata.dem = read_dem(in_dem_fn)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "DEM", "Unable to read file: {}".format(e.message))
            self.ui.DEM_lineEdit.clear()
            return
        
        # set map limits

        self.ui.mplwidget.canvas.ax.set_xlim(self.spdata.dem.domain.g_llcorner().x, self.spdata.dem.domain.g_trcorner().x)
        self.ui.mplwidget.canvas.ax.set_ylim(self.spdata.dem.domain.g_llcorner().y, self.spdata.dem.domain.g_trcorner().y) 

        # fix z to DEM if required

        self.set_z()
        #self.ui.Z_fix2DEM_checkBox_z.emit(QtCore.SIGNAL(" stateChanged (int) "), 1)
             
        # set DEM visibility on

        self.ui.show_DEM_checkBox.setCheckState(2)
        
        # set intersection validity to False

        self.valid_intersections = False
        
        # zoom to full view

        self.zoom_full_view()

    def select_traces_file(self):        
        """
        Selection of the linear shapefile to be opened.
        
        """            
        fileName = QtWidgets.QFileDialog.getOpenFileName(self, self.tr("Open shapefile"), '', "shp (*.shp *.SHP)")
        file_path = fileName[0]
        if not file_path:
            return          

        self.ui.Trace_lineEdit.setText(file_path)
                
    def reading_traces(self, in_traces_shp):
        """
        Read line shapefile.
    
        @param  in_traces_shp:  parameter to check.
        @type  in_traces_shp:  QString or string
        
        """ 
        
        try:
            self.spdata.read_traces(in_traces_shp)
        except:
            QtWidgets.QMessageBox.critical(self, "Traces", "Unable to read shapefile")
            return
        else:
            if self.spdata.traces.lines_x is None or self.spdata.traces.lines_y is None:
                self.ui.Trace_lineEdit.setText('')
                QtWidgets.QMessageBox.critical(self, "Traces", "Unable to read shapefile")
                return           
                      
        # set layer visibility on
        self.ui.show_Fault_checkBox.setCheckState(2) 
        
        # zoom to full view
        self.zoom_full_view()

    def update_src_pt (self, x, y):
        """
        Update the source point position from user input (click event in map).
          
        @param x: x coordinate of clicked point.
        @param y: y coordinate of clicked point.
        @type: list of two float values.      
        """         
        
        self.ui.Pt_spinBox_x.setValue(int(x))
        self.ui.Pt_spinBox_y.setValue(int(y))
        
        # set intersection validity to False
        self.valid_intersections = False

    def set_z(self):
        """
        Update z value.
        
        """ 
        
        # set intersection validity to False
        self.valid_intersections = False        
        
        if self.spdata.dem is None:
            return
        
        if self.ui.Z_fix2DEM_checkBox_z.isChecked():    
   
            curr_x = self.ui.Pt_spinBox_x.value()
            curr_y = self.ui.Pt_spinBox_y.value()
            
            if curr_x <= self.spdata.dem.domain.g_llcorner().x or curr_x >= self.spdata.dem.domain.g_trcorner().x or \
               curr_y <= self.spdata.dem.domain.g_llcorner().y or curr_y >= self.spdata.dem.domain.g_trcorner().y:
                return
           
            curr_point = Point(curr_x, curr_y)
            currArrCoord = self.spdata.dem.geog2array_coord(curr_point)
    
            z = floor(self.spdata.dem.interpolate_bilinear(currArrCoord))
    
            self.ui.Pt_spinBox_z.setValue(int(z))

    def update_dipdir_slider(self):
        """
        Update the value of the dip direction in the slider.
        """

        real_dipdirection = self.ui.DDirection_spinBox.value()
        
        transformed_dipdirection = real_dipdirection + 180.0
        if transformed_dipdirection > 360.0:
            transformed_dipdirection = transformed_dipdirection - 360 
                       
        self.ui.DDirection_dial.setValue(transformed_dipdirection) 
           
        # set intersection validity to False
        self.valid_intersections = False
         
    def update_dipdir_spinbox(self):            
        """
        Update the value of the dip direction in the spinbox.
        """        
        transformed_dipdirection = self.ui.DDirection_dial.value()
        
        real_dipdirection = transformed_dipdirection - 180.0
        if real_dipdirection < 0.0:
            real_dipdirection = real_dipdirection + 360.0
            
        self.ui.DDirection_spinBox.setValue(real_dipdirection) 

        # set intersection validity to False
        self.valid_intersections = False
         
    def update_dipang_slider(self):
        """
        Update the value of the dip angle in the slider.
        """
        self.ui.DAngle_verticalSlider.setValue(self.ui.DAngle_spinBox.value())    
                  
        # set intersection validity to False
        self.valid_intersections = False
                
    def update_dipang_spinbox(self):            
        """
        Update the value of the dip angle in the spinbox.
        """        
        self.ui.DAngle_spinBox.setValue(self.ui.DAngle_verticalSlider.value()) 

        # set intersection validity to False
        self.valid_intersections = False
        
    def calc_intersections(self):
        """
        Calculate intersection points.
        """                
                      
        curr_x = self.ui.Pt_spinBox_x.value()
        curr_y = self.ui.Pt_spinBox_y.value()
        curr_z = self.ui.Pt_spinBox_z.value()
                
        srcPt = Point(curr_x, curr_y, curr_z)

        srcDipDir = self.ui.DDirection_spinBox.value()
        srcDipAngle = self.ui.DAngle_verticalSlider.value()

        srcPlaneAttitude = GPlane(srcDipDir, srcDipAngle)

        # intersection arrays
        self.spdata.set_intersections_default()
        
        intersection_results = self.spdata.dem.intersection_with_surface('plane', srcPt, srcPlaneAttitude)
        
        self.spdata.inters.xcoords_x = intersection_results[0]
        self.spdata.inters.xcoords_y = intersection_results[1]
        self.spdata.inters.ycoords_x = intersection_results[2]
        self.spdata.inters.ycoords_y = intersection_results[3]
            
        self.spdata.inters.parameters = IntersectionParameters(self.spdata.dem._sourcename, srcPt, srcPlaneAttitude)
        
        self.valid_intersections = True

        self.refresh_map()

    def write_intersections_as_points(self):
        """
        Write intersection results in the output shapefile.
        """
        
        if self.spdata.inters.xcoords_x == []:
            QtWidgets.QMessageBox.critical(self, "Save results", "No results available")
            return

        srcPt = self.spdata.inters.parameters._srcPt
        srcPlaneAttitude = self.spdata.inters.parameters._srcPlaneAttitude     
                
        plane_z = plane_from_geo(srcPt, srcPlaneAttitude)   
                                
        x_filtered_coord_x = self.spdata.inters.xcoords_x[np.logical_not(np.isnan(self.spdata.inters.xcoords_x))] 
        x_filtered_coord_y = self.spdata.inters.xcoords_y[np.logical_not(np.isnan(self.spdata.inters.xcoords_x))]            
        x_filtered_coord_z = plane_z(x_filtered_coord_x, x_filtered_coord_y)

        y_filtered_coord_x = self.spdata.inters.ycoords_x[np.logical_not(np.isnan(self.spdata.inters.ycoords_y))] 
        y_filtered_coord_y = self.spdata.inters.ycoords_y[np.logical_not(np.isnan(self.spdata.inters.ycoords_y))]             
        y_filtered_coord_z = plane_z(y_filtered_coord_x, y_filtered_coord_y)        
        
        intersections_x = list(x_filtered_coord_x) + list(y_filtered_coord_x)    
        intersections_y = list(x_filtered_coord_y) + list(y_filtered_coord_y)                                           
        intersections_z = list(x_filtered_coord_z) + list(y_filtered_coord_z)       

        # creation of output shapefile

        fileName = QtWidgets.QFileDialog.getSaveFileName(self, self.tr("Save as shapefile"), 'points.shp', "shp (*.shp *.SHP)")
        fileName = fileName[0]
        if not fileName:
            return  

        fileName = str(fileName)
        
        shape_driver = ogr.GetDriverByName("ESRI Shapefile")
              
        out_shape = shape_driver.CreateDataSource(fileName)
        if out_shape is None:
            QtWidgets.QMessageBox.critical(self, "Results", "Unable to create output shapefile: %s" % fileName)
            return
        out_layer = out_shape.CreateLayer('output_points', geom_type=ogr.wkbPoint)
        if out_layer is None:
            QtWidgets.QMessageBox.critical(self, "Results", "Unable to create output shapefile: %s" % fileName)
            return        
        
        # add fields to the output shapefile

        id_fieldDef = ogr.FieldDefn('id', ogr.OFTInteger)
        out_layer.CreateField(id_fieldDef)
                
        x_fieldDef = ogr.FieldDefn('x', ogr.OFTReal)
        out_layer.CreateField(x_fieldDef)

        y_fieldDef = ogr.FieldDefn('y', ogr.OFTReal)
        out_layer.CreateField(y_fieldDef)

        z_fieldDef = ogr.FieldDefn('z', ogr.OFTReal)
        out_layer.CreateField(z_fieldDef)

        srcPt_x_fieldDef = ogr.FieldDefn('srcPt_x', ogr.OFTReal)
        out_layer.CreateField(srcPt_x_fieldDef)

        srcPt_y_fieldDef = ogr.FieldDefn('srcPt_y', ogr.OFTReal)
        out_layer.CreateField(srcPt_y_fieldDef)

        srcPt_z_fieldDef = ogr.FieldDefn('srcPt_z', ogr.OFTReal)
        out_layer.CreateField(srcPt_z_fieldDef)        
        
        DipDir_fieldDef = ogr.FieldDefn('dip_dir', ogr.OFTReal)
        out_layer.CreateField(DipDir_fieldDef)

        DipAng_fieldDef = ogr.FieldDefn('dip_ang', ogr.OFTReal)
        out_layer.CreateField(DipAng_fieldDef)        
                        
        # get the layer definition of the output shapefile
        outshape_featdef = out_layer.GetLayerDefn()  
        
        curr_Pt_id = 0                    

        for curr_Pt in zip(intersections_x, intersections_y, intersections_z):
            
            curr_Pt_id += 1
            
            # pre-processing for new feature in output layer
            curr_Pt_geom = ogr.Geometry(ogr.wkbPoint)
            curr_Pt_geom.AddPoint(float(curr_Pt[0]), float(curr_Pt[1]), float(curr_Pt[2]))
                
            # create a new feature
            curr_Pt_shape = ogr.Feature(outshape_featdef)
            curr_Pt_shape.SetGeometry(curr_Pt_geom)
            curr_Pt_shape.SetField('id', curr_Pt_id)                                       
            curr_Pt_shape.SetField('x', curr_Pt[0])
            curr_Pt_shape.SetField('y', curr_Pt[1]) 
            curr_Pt_shape.SetField('z', curr_Pt[2]) 

            curr_Pt_shape.SetField('srcPt_x', srcPt.x)
            curr_Pt_shape.SetField('srcPt_y', srcPt.y) 
            curr_Pt_shape.SetField('srcPt_z', srcPt.z)

            curr_Pt_shape.SetField('dip_dir', srcPlaneAttitude._dipdir)
            curr_Pt_shape.SetField('dip_ang', srcPlaneAttitude._dipangle)             

            # add the feature to the output layer
            out_layer.CreateFeature(curr_Pt_shape)            
            
            # destroy no longer used objects
            curr_Pt_geom.Destroy()
            curr_Pt_shape.Destroy()
                            
        # destroy output geometry
        out_shape.Destroy()

        QtWidgets.QMessageBox.information(self, "Result", "Saved to shapefile: %s" % fileName)

    def write_intersections_as_lines(self):
        """
        Write intersection results in a line shapefile.
        """
        
        if self.spdata.inters.xcoords_x == []:
            QtWidgets.QMessageBox.critical(self, "Save results", "No results available")
            return        

        srcPt = self.spdata.inters.parameters._srcPt
        srcPlaneAttitude = self.spdata.inters.parameters._srcPlaneAttitude     

        plane_z = plane_from_geo(srcPt, srcPlaneAttitude) 
                
        # create dictionary of cell with intersection points
        
        self.spdata.inters.links = self.spdata.get_intersections() 
            
        self.spdata.inters.neighbours = self.spdata.set_neighbours()        
                
        self.spdata.define_paths()  
        
        # networks of connected intersections
        self.spdata.inters.networks = self.spdata.define_networks()        

        # creation of output shapefile

        fileName = QtWidgets.QFileDialog.getSaveFileName(self, self.tr("Save as shapefile"), 'lines.shp', "shp (*.shp *.SHP)")
        fileName = fileName[0]
        if not fileName:
            return  

        fileName = str(fileName)
               
        shape_driver = ogr.GetDriverByName("ESRI Shapefile")
              
        out_shape = shape_driver.CreateDataSource(fileName)
        if out_shape is None:
            QtWidgets.QMessageBox.critical(self, "Results", "Unable to create output shapefile: %s" % fileName)
            return
        out_layer = out_shape.CreateLayer('output_lines', geom_type=ogr.wkbLineString)
        if out_layer is None:
            QtWidgets.QMessageBox.critical(self, "Results", "Unable to create output shapefile: %s" % fileName)
            return        
        
        # add fields to the output shapefile      

        pathId_fieldDef = ogr.FieldDefn('id', ogr.OFTInteger)
        out_layer.CreateField(pathId_fieldDef)
                
        srcPt_x_fieldDef = ogr.FieldDefn('srcPt_x', ogr.OFTReal)
        out_layer.CreateField(srcPt_x_fieldDef)

        srcPt_y_fieldDef = ogr.FieldDefn('srcPt_y', ogr.OFTReal)
        out_layer.CreateField(srcPt_y_fieldDef)

        srcPt_z_fieldDef = ogr.FieldDefn('srcPt_z', ogr.OFTReal)
        out_layer.CreateField(srcPt_z_fieldDef)        
        
        DipDir_fieldDef = ogr.FieldDefn('dip_dir', ogr.OFTReal)
        out_layer.CreateField(DipDir_fieldDef)

        DipAng_fieldDef = ogr.FieldDefn('dip_ang', ogr.OFTReal)
        out_layer.CreateField(DipAng_fieldDef)        
                        
        # get the layer definition of the output shapefile
        outshape_featdef = out_layer.GetLayerDefn()  

        for curr_path_id in sorted(self.spdata.inters.networks.keys()):

            curr_path_points = self.spdata.inters.networks[curr_path_id]
                                    
            line = ogr.Geometry(ogr.wkbLineString)
            
            for curr_point_id in curr_path_points:  
                          
                curr_intersection = self.spdata.inters.links[curr_point_id-1]
                           
                i, j, direct = curr_intersection['i'], curr_intersection['j'], curr_intersection['pi_dir']
                
                if direct == 'x':
                    x, y = self.spdata.inters.xcoords_x[i, j], self.spdata.inters.xcoords_y[i, j]
                if direct == 'y':
                    x, y = self.spdata.inters.ycoords_x[i, j], self.spdata.inters.ycoords_y[i, j]
                                       
                z = plane_z(x, y)
 
                line.AddPoint(x, y, z)

            # create a new feature
            line_shape = ogr.Feature(outshape_featdef)
            line_shape.SetGeometry(line)                           

            line_shape.SetField('id', curr_path_id)
            line_shape.SetField('srcPt_x', srcPt.x)
            line_shape.SetField('srcPt_y', srcPt.y) 
            line_shape.SetField('srcPt_z', srcPt.z)
    
            line_shape.SetField('dip_dir', srcPlaneAttitude._dipdir)
            line_shape.SetField('dip_ang', srcPlaneAttitude._dipangle)             
    
            # add the feature to the output layer
            out_layer.CreateFeature(line_shape)            
            
            # destroy no longer used objects
            line.Destroy()
            line_shape.Destroy()
                            
        # destroy output geometry
        out_shape.Destroy()

        QtWidgets.QMessageBox.information(self, "Result", "Saved to shapefile: %s" % fileName)

    def helpAbout(self):
        """
        Visualize an About window.
        """
        QtWidgets.QMessageBox.about(self, "About gSurf",
        """
            <p>gSurf version 0.2.0 for Python 3 / Qt5</p>
            <p>M. Alberti, <a href="http://www.malg.eu">www.malg.eu</a></p> 
            <p>This program calculates the intersection between a plane and a DEM in an interactive way.
            The result can be saved as a point/linear shapefile.</p>            
             <p>Report any bug to <a href="mailto:alberti.m65@gmail.com">alberti.m65@gmail.com</a></p>
        """)        

    def openHelp(self):
        """
        Open an Help HTML file
        from CADTOOLS module in QGIS
        """
        help_path = os.path.join(os.path.dirname(__file__), 'help', 'help.html')         
        webbrowser.open(help_path)
 
 
class AnchoredText(AnchoredOffsetbox):
    """
    Creation of an info box in the plot
    
    """
    def __init__(self, s, loc, pad=0.4, borderpad=0.5, prop=None, frameon=True):

        self.txt = TextArea(s, minimumdescent=False)

        super(AnchoredText, self).__init__(loc, pad=pad, borderpad=borderpad,
                                           child=self.txt,
                                           prop=prop,
                                           frameon=frameon)


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)

    form = MainWindow()
    form.show()
    sys.exit(app.exec_())




