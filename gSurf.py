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

import webbrowser

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication


from matplotlib.offsetbox import AnchoredOffsetbox, TextArea

from gSurf_ui import Ui_MainWindow
from gSurf_data import *


class MainWindow(QMainWindow):
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
        
        # DEM
        QtCore.QObject.connect(self.ui.actionInput_DEM, QtCore.SIGNAL(" triggered() "), self.select_dem_file)
        QtCore.QObject.connect(self.ui.DEM_lineEdit, QtCore.SIGNAL(" textChanged (QString) "), self.selected_dem) 
                       
        QtCore.QObject.connect(self.ui.show_DEM_checkBox, QtCore.SIGNAL(" stateChanged (int) "), self.redraw_map)
        QtCore.QObject.connect(self.ui.DEM_cmap_comboBox, QtCore.SIGNAL(" currentIndexChanged (QString) "), self.redraw_map)
                
        # Fault traces
        QtCore.QObject.connect(self.ui.actionInput_line_shapefile, QtCore.SIGNAL(" triggered() "), self.select_traces_file)
        QtCore.QObject.connect(self.ui.Trace_lineEdit, QtCore.SIGNAL(" textChanged (QString) "), self.reading_traces) 
                       
        QtCore.QObject.connect(self.ui.show_Fault_checkBox, QtCore.SIGNAL(" stateChanged (int) "), self.redraw_map)
        
        # Full zoom
        QtCore.QObject.connect(self.ui.mplwidget.canvas, QtCore.SIGNAL(" zoom_to_full_view "), self.zoom_to_full_view)
 
        # Source point
        QtCore.QObject.connect(self.ui.mplwidget.canvas, QtCore.SIGNAL(" map_press "), self.update_srcpt) # event from matplotlib widget
                
        QtCore.QObject.connect(self.ui.Pt_spinBox_x, QtCore.SIGNAL(" valueChanged (int) "), self.set_z)
        QtCore.QObject.connect(self.ui.Pt_spinBox_y, QtCore.SIGNAL(" valueChanged (int) "), self.set_z)
        QtCore.QObject.connect(self.ui.Z_fix2DEM_checkBox_z, QtCore.SIGNAL(" stateChanged (int) "), self.set_z)         
        QtCore.QObject.connect(self.ui.Pt_spinBox_z, QtCore.SIGNAL(" valueChanged (int) "), self.set_z)
                                           
        QtCore.QObject.connect(self.ui.Pt_spinBox_x, QtCore.SIGNAL(" valueChanged (int) "), self.redraw_map)
        QtCore.QObject.connect(self.ui.Pt_spinBox_y, QtCore.SIGNAL(" valueChanged (int) "), self.redraw_map) 
        QtCore.QObject.connect(self.ui.Pt_spinBox_z, QtCore.SIGNAL(" valueChanged (int) "), self.redraw_map) 
        QtCore.QObject.connect(self.ui.show_SrcPt_checkBox, QtCore.SIGNAL(" stateChanged (int) "), self.redraw_map)
 
        # Plane orientation      
        QtCore.QObject.connect(self.ui.DDirection_dial, QtCore.SIGNAL(" valueChanged (int) "), self.update_dipdir_spinbox)
        QtCore.QObject.connect(self.ui.DDirection_spinBox, QtCore.SIGNAL(" valueChanged (int) "), self.update_dipdir_slider)
               
        QtCore.QObject.connect(self.ui.DAngle_verticalSlider, QtCore.SIGNAL(" valueChanged (int) "), self.update_dipang_spinbox)
        QtCore.QObject.connect(self.ui.DAngle_spinBox, QtCore.SIGNAL(" valueChanged (int) "), self.update_dipang_slider)

        # Intersections        
        QtCore.QObject.connect(self.ui.Intersection_calculate_pushButton, QtCore.SIGNAL(" clicked(bool) "), self.calc_intersections) 
        QtCore.QObject.connect(self.ui.Intersection_show_checkBox, QtCore.SIGNAL(" stateChanged (int) "), self.redraw_map)
        QtCore.QObject.connect(self.ui.Intersection_color_comboBox, QtCore.SIGNAL(" currentIndexChanged (QString) "), self.redraw_map)     

        # Write result
        QtCore.QObject.connect(self.ui.actionPoints, QtCore.SIGNAL(" triggered() "), self.write_intersections_as_points)
        QtCore.QObject.connect(self.ui.actionLines, QtCore.SIGNAL(" triggered() "), self.write_intersections_as_lines)

        # Other actions
        QtCore.QObject.connect(self.ui.actionHelp, QtCore.SIGNAL(" triggered() "), self.openHelp)           
        QtCore.QObject.connect(self.ui.actionAbout, QtCore.SIGNAL(" triggered() "), self.helpAbout)          
        QtCore.QObject.connect(self.ui.actionQuit, QtCore.SIGNAL(" triggered() "), sys.exit)        

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
                           
            geo_extent = [self.spdata.dem.domain.g_llcorner().x, self.spdata.dem.domain.g_trcorner().x, \
                           self.spdata.dem.domain.g_llcorner().y, self.spdata.dem.domain.g_trcorner().y]
            
            if self.ui.show_DEM_checkBox.isChecked(): # DEM check is on
                
                curr_colormap = str(self.ui.DEM_cmap_comboBox.currentText())
                     
                self.ui.mplwidget.canvas.ax.imshow(self.spdata.dem.data, extent = geo_extent,  cmap= curr_colormap)

        # Fault traces proc.
        if self.spdata.traces.lines_x is not None and self.spdata.traces.lines_y is not None \
           and self.ui.show_Fault_checkBox.isChecked(): # Fault check is on 
 
            for currLine_x, currLine_y  in zip(self.spdata.traces.lines_x, self.spdata.traces.lines_y):                
                    self.ui.mplwidget.canvas.ax.plot(currLine_x, currLine_y,'-')

        # Intersections proc.
        if self.ui.Intersection_show_checkBox.isChecked() and self.valid_intersections == True:
            
            curr_color = str(self.ui.Intersection_color_comboBox.currentText())
                        
            intersections_x = list(self.spdata.inters.xcoords_x[np.logical_not(np.isnan(self.spdata.inters.xcoords_x))]) + \
                              list(self.spdata.inters.ycoords_x[np.logical_not(np.isnan(self.spdata.inters.ycoords_y))])
        
            intersections_y = list(self.spdata.inters.xcoords_y[np.logical_not(np.isnan(self.spdata.inters.xcoords_x))]) + \
                              list(self.spdata.inters.ycoords_y[np.logical_not(np.isnan(self.spdata.inters.ycoords_y))])

            self.ui.mplwidget.canvas.ax.plot(intersections_x, intersections_y,  "w+",  ms=2,  mec=curr_color,  mew=2)
                                
            legend_text = "Plane dip dir., angle: (%d, %d)\nSource point x, y, z: (%d, %d, %d)" % \
                (self.spdata.inters.parameters._srcPlaneAttitude._dipdir, self.spdata.inters.parameters._srcPlaneAttitude._dipangle, \
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
        if map_extent_x == None:
            map_extent_x = self.ui.mplwidget.canvas.ax.get_xlim()

        if map_extent_y == None:            
            map_extent_y = self.ui.mplwidget.canvas.ax.get_ylim()
                
        self.draw_map(map_extent_x, map_extent_y)

    def redraw_map(self):
        """
        Convenience function for drawing the map.
                
        """
                      
        self.refresh_map()
                
    def zoom_to_full_view(self, map_extent_x=[0, 100], map_extent_y=[0, 100]):
        """
        Update map view to the DEM extent or otherwise, if available, to the shapefile extent.
    
        @param  map_extent_x:  map extent along the x axis.
        @type  map_extent_x:  list of two float values (min x and max x).
        @param  map_extent_y:  map extent along the y axis.
        @type  map_extent_y:  list of two float values (min y and max y).
                
        """      
       
        if self.spdata.dem is not None:
            map_extent_x = [self.spdata.dem.domain.g_llcorner().x, self.spdata.dem.domain.g_trcorner().x]
            map_extent_y = [self.spdata.dem.domain.g_llcorner().y, self.spdata.dem.domain.g_trcorner().y]
            
        elif self.spdata.traces.extent_x != [] and self.spdata.traces.extent_y != []:
            map_extent_x = self.spdata.traces.extent_x
            map_extent_y = self.spdata.traces.extent_y                 
                                
        self.refresh_map(map_extent_x, map_extent_y)
        
    def select_dem_file(self):
        """
        Select input DEM file
        
        """            
        fileName = QtGui.QFileDialog.getOpenFileName(self, self.tr("Open DEM file (using GDAL)"), '', "*.*")
        if fileName.isEmpty():
            return          

        self.ui.DEM_lineEdit.setText(fileName)

    def selected_dem(self, in_dem_fn):        

        try:
            self.spdata.dem = self.spdata.read_dem(in_dem_fn)
        except:
            self.ui.DEM_lineEdit.clear()
            QtGui.QMessageBox.critical(self, "DEM", "Unable to read file")
            return
        
        # set map limits
        self.ui.mplwidget.canvas.ax.set_xlim(self.spdata.dem.domain.g_llcorner().x, self.spdata.dem.domain.g_trcorner().x)
        self.ui.mplwidget.canvas.ax.set_ylim(self.spdata.dem.domain.g_llcorner().y, self.spdata.dem.domain.g_trcorner().y) 

        # fix z to DEM if required
        self.ui.Z_fix2DEM_checkBox_z.emit(QtCore.SIGNAL(" stateChanged (int) "), 1)
             
        # set DEM visibility on
        self.ui.show_DEM_checkBox.setCheckState(2)
        
        # set intersection validity to False
        self.valid_intersections = False
        
        # zoom to full view        
        self.zoom_to_full_view()

    def select_traces_file(self):        
        """
        Selection of the linear shapefile to be opened.
        
        """            
        fileName = QtGui.QFileDialog.getOpenFileName(self, self.tr("Open shapefile"), '', "shp (*.shp *.SHP)")
        if fileName.isEmpty():
            return          

        self.ui.Trace_lineEdit.setText(fileName)
                
    def reading_traces(self, in_traces_shp):
        """
        Read line shapefile.
    
        @param  in_traces_shp:  parameter to check.
        @type  in_traces_shp:  QString or string
        
        """ 
        
        try:
            self.spdata.read_traces(in_traces_shp)
        except:
            QtGui.QMessageBox.critical(self, "Traces", "Unable to read shapefile")
            return
        else:
            if self.spdata.traces.lines_x is None or self.spdata.traces.lines_y is None:
                self.ui.Trace_lineEdit.setText('')
                QtGui.QMessageBox.critical(self, "Traces", "Unable to read shapefile")
                return           
                      
        # set layer visibility on
        self.ui.show_Fault_checkBox.setCheckState(2) 
        
        # zoom to full view
        self.zoom_to_full_view()

    def update_srcpt (self, pos_values):
        """
        Update the source point position from user input (click event in map).
          
        @param pos_values: location of clicked point.
        @type: list of two float values.      
        """         
        x = pos_values[0]
        y = pos_values[1]
        
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
        
        if self.spdata.dem is None: return
        
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

        srcPlaneAttitude = StructPlane(srcDipDir, srcDipAngle)

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
            QtGui.QMessageBox.critical(self, "Save results", "No results available") 
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

        fileName = QtGui.QFileDialog.getSaveFileName(self, self.tr("Save as shapefile"), 'points.shp', "shp (*.shp *.SHP)")
        if fileName.isEmpty():
            return  

        fileName = str(fileName)
        
        shape_driver = ogr.GetDriverByName("ESRI Shapefile")
              
        out_shape = shape_driver.CreateDataSource(fileName)
        if out_shape is None:
            QtGui.QMessageBox.critical(self, "Results", "Unable to create output shapefile: %s" % fileName)
            return
        out_layer = out_shape.CreateLayer('output_points', geom_type=ogr.wkbPoint)
        if out_layer is None:
            QtGui.QMessageBox.critical(self, "Results", "Unable to create output shapefile: %s" % fileName) 
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
        
        QtGui.QMessageBox.information(self, "Result", "Saved to shapefile: %s" % fileName)

    def write_intersections_as_lines(self):
        """
        Write intersection results in a line shapefile.
        """
        
        if self.spdata.inters.xcoords_x == []:
            QtGui.QMessageBox.critical(self, "Save results", "No results available") 
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

        fileName = QtGui.QFileDialog.getSaveFileName(self, self.tr("Save as shapefile"), 'lines.shp', "shp (*.shp *.SHP)")
        if fileName.isEmpty():
            return  

        fileName = str(fileName)
               
        shape_driver = ogr.GetDriverByName("ESRI Shapefile")
              
        out_shape = shape_driver.CreateDataSource(fileName)
        if out_shape is None:
            QtGui.QMessageBox.critical(self, "Results", "Unable to create output shapefile: %s" % fileName)
            return
        out_layer = out_shape.CreateLayer('output_lines', geom_type=ogr.wkbLineString)
        if out_layer is None:
            QtGui.QMessageBox.critical(self, "Results", "Unable to create output shapefile: %s" % fileName) 
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

        for curr_path_id, curr_path_points in self.spdata.inters.networks.iteritems():
                                    
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

        QtGui.QMessageBox.information(self, "Result", "Saved to shapefile: %s" % fileName)

    def helpAbout(self):
        """
        Visualize an About window.
        """
        QtGui.QMessageBox.about(self, "About gSurf", 
        """
            <p>gSurf version 0.2.0</p>
            <p>M. Alberti, <a href="http://www.malg.eu">www.malg.eu</a></p> 
            <p>This program calculates the intersection between a plane and a DEM in an interactive way.
            The result can be saved as a point/linear shapefile.</p>            
            
             <p>Created and tested with Python 2.7 and Eclipse/PyDev.</p>
             <p>Tested in Windows Vista (Python 2.7.2) and Ubuntu Lucid Lynx (Python 2.6.5)</p>
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

    app = QApplication(sys.argv)

    form = MainWindow()
    # form.show()
    sys.exit(app.exec_())




