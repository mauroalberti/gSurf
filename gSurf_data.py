# -*- coding: utf-8 -*-

"""
/***************************************************************************
 geoSurf

 DEM - planes intersections
                              -------------------
        begin                : 2011-12-21
        version              : 0.1.2 - 2012-04-07
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

from __future__  import division

from math import *

from osgeo import gdal
from osgeo.gdalconst import *

try:
    from osgeo import ogr
except: 
    import ogr

from gSurf_errors import Raster_Parameters_Errors
from gSurf_spatial import *
   


def read_raster_band( raster_name ):
    """
    Read an input raster band, based on GDAL module.
    
    @param raster_name: name of the raster to be read.
    @type raster_name: QString.
    
    @return: tuple of a GDALParameters instance and a 2D numpy.array instance. 
    
    @raise IOError: unable to open or read data from raster.
    @raise TypeError: more than one band in raster.    
    """
            
    # GDAL register
    gdal.AllRegister
    
    # open raster file and check operation success 
    raster_data = gdal.Open( str( raster_name ), GA_ReadOnly )    
    if raster_data is None:
        raise IOError, 'Unable to open raster band' 

    # initialize DEM parameters
    raster_params = GDALParameters()
    
    # get driver type for current raster 
    raster_params.driverShortName = raster_data.GetDriver().ShortName

    # get current raster projection
    raster_params.projection = raster_data.GetProjection()   

    # get row and column numbers    
    raster_params.rows = raster_data.RasterYSize
    raster_params.cols = raster_data.RasterXSize
    
    # get and check number of raster bands - it must be one
    raster_bands = raster_data.RasterCount
    if raster_bands > 1:
        raise TypeError, 'More than one raster band in raster' 
    
    # set critical grid values from geotransform array
    raster_params.topLeftX = raster_data.GetGeoTransform()[0]
    raster_params.pixSizeEW = raster_data.GetGeoTransform()[1]
    raster_params.rotGT2 = raster_data.GetGeoTransform()[2]
    raster_params.topLeftY = raster_data.GetGeoTransform()[3]
    raster_params.rotGT4 = raster_data.GetGeoTransform()[4]
    raster_params.pixSizeNS = raster_data.GetGeoTransform()[5]
 
    # get single band 
    band = raster_data.GetRasterBand(1)
    
    # get no data value for current band
    try: 
        raster_params.noDataValue = band.GetNoDataValue()
    except:
        pass
    # read data from band 
    grid_values = band.ReadAsArray( 0,0, raster_params.cols, raster_params.rows )
    if grid_values is None:
        raise IOError, 'Unable to read data from raster'
     
    # transform data into numpy array
    data = np.asarray( grid_values ) 

    # if nodatavalue exists, set null values to NaN in numpy array
    if raster_params.noDataValue is not None:
        data = np.where( abs( data - raster_params.noDataValue ) > 1e-05, data, np.NaN ) 

    return raster_params, data
    


class Intersection_Parameters(object):
    """
    Intersection_Parameters class.
    Manages the metadata for spdata results (DEM source filename, source point, plane attitude.
    
    """
    
    def __init__( self, sourcename, srcPt, srcPlaneAttitude ): 
        """
        Class constructor.
        
        @param sourcename: name of the DEM used to create the grid.
        @type sourcename: String.
        @param srcPlaneAttitude: orientation of the plane used to calculate the spdata.
        @type srcPlaneAttitude: class StructPlane.
        
        @return: self
        """
        self._sourcename = sourcename
        self._srcPt = srcPt
        self._srcPlaneAttitude = srcPlaneAttitude
                            

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
       

    def read_dem( self, in_dem_fn ):
        """
        Read input DEM file.
    
        @param  in_dem_fn: name of file to be read.
        @type  in_dem_fn:  string
        
        """
                
        # try reading DEM data
        try:
            dem_params, dem_array = read_raster_band( in_dem_fn )
            dem_params.check_params()
        except ( IOError, TypeError, Raster_Parameters_Errors ), e:                    
            raise IOError, 'Unable to read data from raster'
               
        # create current grid
        return Grid(in_dem_fn, dem_params, dem_array)
        
                
    def read_traces( self, in_traces_shp ):
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
        self.traces.extent_y[0], self.traces.extent_y[1]  \
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

            for i in range( line_geom.GetPointCount() ):
                                
                x, y = line_geom.GetX(i), line_geom.GetY(i)
                            
                line_vert_x.append(x)
                line_vert_y.append(y)

            self.traces.lines_x.append(line_vert_x)    
            self.traces.lines_y.append(line_vert_y)
                        
            curr_line = lnLayer.GetNextFeature()

        in_shape.Destroy()
    
    
    def get_intersections( self ):
        """
        Initialize a structured array of the possible and found links for each intersection.
        It will store a list of the possible connections for each intersection,
        together with the found connections.
        
        
        """
                
        # data type for structured array storing intersection parameters
        dt = np.dtype([ ('id', np.uint16),
                        ('i', np.uint16),
                        ('j', np.uint16),
                        ('pi_dir', np.str_,  1 ),                     
                        ('conn_from', np.uint16),
                        ('conn_to', np.uint16),
                        ('start', np.bool_)                            
                      ])
                 
        # number of valid intersections
        num_intersections = len( list( self.inters.xcoords_x[ np.logical_not( np.isnan(self.inters.xcoords_x)) ] )) + \
                            len( list( self.inters.ycoords_y[ np.logical_not( np.isnan(self.inters.ycoords_y)) ] ))
        
        
        # creation and initialization of structured array of valid intersections in the x-direction 
        links = np.zeros( ( num_intersections ), dtype=dt )
        
        # filling array with values
        
        print 'links'
        curr_ndx = 0
        for i in xrange(self.inters.xcoords_x.shape[0]):
            for j in xrange(self.inters.xcoords_x.shape[1]): 
                if not isnan(self.inters.xcoords_x[i, j]):
                    print curr_ndx+1, i, j, 'x'               
                    links[curr_ndx] = (curr_ndx+1, i, j, 'x', 0, 0, False)                
                    curr_ndx += 1
    
        for i in xrange(self.inters.ycoords_y.shape[0]):
            for j in xrange(self.inters.ycoords_y.shape[1]): 
                if not isnan(self.inters.ycoords_y[i, j]):
                    print curr_ndx+1, i, j, 'y'                                     
                    links[curr_ndx] = (curr_ndx+1, i, j, 'y', 0, 0, False)                
                    curr_ndx += 1
        

        
        return links                                     


    def set_neighbours( self ):
        
        # shape of input arrays (equal shapes)
        num_rows, num_cols = self.inters.xcoords_x.shape
        
        # dictionary storing intersection links
        neighbours = {}
        
        # search and connect intersection points   
        for curr_ndx in xrange( self.inters.links.shape[0] ):            
            
            # get current point location (i, j) and direction type (pi_dir)
            curr_id = self.inters.links[curr_ndx]['id']
            curr_i = self.inters.links[curr_ndx]['i']
            curr_j = self.inters.links[curr_ndx]['j'] 
            curr_dir = self.inters.links[curr_ndx]['pi_dir']         
            
            # check possible connected spdata
            near_intersections = []
            
            if curr_dir == 'x':
                
                if curr_i < num_rows - 1 and curr_j < num_cols - 1 :
                     
                    try: # -- A
                        id_link = self.inters.links[(self.inters.links['i'] == curr_i+1) & \
                                                         (self.inters.links['j'] == curr_j+1) & \
                                                         (self.inters.links['pi_dir'] == 'y') ] ['id']
                        if len(list(id_link)) == 1:
                            near_intersections.append(id_link[0])
                    except:
                        pass       
                    try: # -- B
                        id_link = self.inters.links[(self.inters.links['i'] == curr_i+1) & \
                                                         (self.inters.links['j'] == curr_j) & \
                                                         (self.inters.links['pi_dir'] == 'x') ] ['id']
                        if len(list(id_link)) == 1:
                            near_intersections.append(id_link[0])
                    except:
                        pass
                    try: # -- C
                        id_link = self.inters.links[(self.inters.links['i'] == curr_i+1) & \
                                                         (self.inters.links['j'] == curr_j) & \
                                                         (self.inters.links['pi_dir'] == 'y') ] ['id']
                        if len(list(id_link)) == 1:
                            near_intersections.append(id_link[0])
                    except:
                        pass            
                
                if curr_i > 0 and curr_j < num_cols - 1:
                    
                    try: # -- E
                        id_link = self.inters.links[(self.inters.links['i'] == curr_i) & \
                                                         (self.inters.links['j'] == curr_j) & \
                                                         (self.inters.links['pi_dir'] == 'y') ] ['id']
                        if len(list(id_link)) == 1:
                            near_intersections.append(id_link[0])
                    except:
                        pass                
                    try: # -- F
                        id_link = self.inters.links[(self.inters.links['i'] == curr_i-1) & \
                                                         (self.inters.links['j'] == curr_j) & \
                                                         (self.inters.links['pi_dir'] == 'x') ] ['id']
                        if len(list(id_link)) == 1:
                            near_intersections.append(id_link[0])
                    except:
                        pass                
                    try: # -- G
                        id_link = self.inters.links[(self.inters.links['i'] == curr_i) & \
                                                         (self.inters.links['j'] == curr_j+1) & \
                                                         (self.inters.links['pi_dir'] == 'y') ] ['id']
                        if len(list(id_link)) == 1:
                            near_intersections.append(id_link[0])
                    except:
                        pass 
                            
                
            if curr_dir == 'y':
                
                if curr_i > 0 and curr_j < num_cols - 1:
    
                    try: # -- D
                        id_link = self.inters.links[(self.inters.links['i'] == curr_i) & \
                                                         (self.inters.links['j'] == curr_j) & \
                                                         (self.inters.links['pi_dir'] == 'x') ] ['id']
                        if len(list(id_link)) == 1:
                            near_intersections.append(id_link[0])
                    except:
                        pass 
                    try: # -- F
                        id_link = self.inters.links[(self.inters.links['i'] == curr_i-1) & \
                                                         (self.inters.links['j'] == curr_j) & \
                                                         (self.inters.links['pi_dir'] == 'x') ] ['id']
                        if len(list(id_link)) == 1:
                            near_intersections.append(id_link[0])
                    except:
                        pass                
                    try: # -- G
                        id_link = self.inters.links[(self.inters.links['i'] == curr_i) & \
                                                         (self.inters.links['j'] == curr_j+1) & \
                                                         (self.inters.links['pi_dir'] == 'y') ] ['id']
                        if len(list(id_link)) == 1:
                            near_intersections.append(id_link[0])
                    except:
                        pass 
    
                if curr_i > 0 and curr_j > 0:
                    
                    try: # -- H
                        id_link = self.inters.links[(self.inters.links['i'] == curr_i) & \
                                                         (self.inters.links['j'] == curr_j-1) & \
                                                         (self.inters.links['pi_dir'] == 'x') ] ['id']
                        if len(list(id_link)) == 1:
                            near_intersections.append(id_link[0])
                    except:
                        pass  
                    try: # -- I
                        id_link = self.inters.links[(self.inters.links['i'] == curr_i) & \
                                                         (self.inters.links['j'] == curr_j-1) & \
                                                         (self.inters.links['pi_dir'] == 'y') ] ['id']
                        if len(list(id_link)) == 1:
                            near_intersections.append(id_link[0])
                    except:
                        pass             
                    try: # -- L
                        id_link = self.inters.links[(self.inters.links['i'] == curr_i-1) & \
                                                         (self.inters.links['j'] == curr_j-1) & \
                                                         (self.inters.links['pi_dir'] == 'x') ] ['id']
                        if len(list(id_link)) == 1:
                            near_intersections.append(id_link[0])
                    except:
                        pass 
                  
            
            neighbours[curr_id] = near_intersections
        
        return neighbours
    
        
    def follow_path( self, start_id ):
        """
        Creates a path of connected intersections from a given start intersection.
        
        """
        from_id = start_id
        
        while self.inters.links[from_id-1]['conn_to'] == 0:

            conns = self.inters.neighbours[from_id]
            num_conn = len(conns)
            if num_conn == 0:
                raise ConnectionError, 'no connected intersection'  
            elif num_conn == 1:
                if self.inters.links[conns[0]-1]['conn_from'] == 0 and self.inters.links[conns[0]-1]['conn_to'] != from_id:
                    to_id = conns[0]
                else:
                    raise ConnectionError, 'no free connection'
            elif num_conn == 2:
                if self.inters.links[conns[0]-1]['conn_from'] == 0 and self.inters.links[conns[0]-1]['conn_to'] != from_id:
                    to_id = conns[0]
                elif self.inters.links[conns[1]-1]['conn_from'] == 0 and self.inters.links[conns[1]-1]['conn_to'] != from_id:
                    to_id = conns[1]
                else:
                    raise ConnectionError, 'no free connection' 
            else:
                raise ConnectionError, 'multiple connection'
            
            # set connection
            self.inters.links[to_id-1]['conn_from'] = from_id
            self.inters.links[from_id-1]['conn_to'] = to_id
                        
            # prepare for next step
            from_id = to_id
            

    def path_closed( self, start_id ):
        
        from_id = start_id
        
        while self.inters.links[from_id-1]['conn_to'] != 0:
            
            to_id = self.inters.links[from_id-1]['conn_to']
            
            if to_id == start_id: return True
            
            from_id = to_id
            
        return False
        

    def invert_path( self, start_id ):
        
        self.inters.links[start_id-1]['start'] = False
        
        curr_id = start_id
        
        while curr_id != 0:
            
            prev_from_id = self.inters.links[curr_id-1]['conn_from'] 
            prev_to_id = self.inters.links[curr_id-1]['conn_to']        
        
            self.inters.links[curr_id-1]['conn_from'] = prev_to_id
            self.inters.links[curr_id-1]['conn_to'] = prev_from_id
            
            if  self.inters.links[curr_id-1]['conn_from'] == 0:
                self.inters.links[curr_id-1]['start'] = True
                
            curr_id = prev_to_id
         
        return
        
        
    def patch_path( self, start_id ):
        """
        
        """
        
        if self.path_closed( start_id ) : return
        
        from_id = start_id
        
        conns = self.inters.neighbours[from_id]
        try:
            conns.remove(self.inters.links[from_id-1]['conn_to'])
        except:
            pass
        
        num_conn = len(conns)
        
        if num_conn != 1: return
        
        new_toid = self.inters.links[conns[0]-1]        
        
        if self.inters.links[new_toid]['conn_to'] > 0 \
           and self.inters.links[new_toid]['conn_to'] != from_id \
           and self.inters.links[new_toid]['conn_from'] == 0:            
            
            if self.path_closed( new_toid ): return
            self.invert_path( from_id )
            self.self.inters.links[from_id-1]['conn_to'] = new_toid
            self.self.inters.links[new_toid-1]['conn_from'] = from_id            
            self.self.inters.links[new_toid-1]['start'] = False            
              
        
    def define_paths( self ):
        
        # simple networks starting from border
        for ndx in xrange(self.inters.links.shape[0]):
            
            if len(self.inters.neighbours[ndx+1]) != 1 or \
               self.inters.links[ndx]['conn_from'] > 0 or \
               self.inters.links[ndx]['conn_to'] > 0: 
                continue
            
            try:
                self.follow_path( ndx+1 )
            except:
                continue
            
        
        # inner, simple networks
        
        for ndx in xrange(self.inters.links.shape[0]):

            if len(self.inters.neighbours[ndx+1]) != 2 or \
               self.inters.links[ndx]['conn_to'] > 0 or \
               self.inters.links[ndx]['start'] == True:
                continue
                        
            try:
                self.inters.links[ndx]['start'] = True                
                self.follow_path( ndx+1 )
            except:
                continue
    

        # inner, simple networks, connection of FROM
        
        for ndx in xrange(self.inters.links.shape[0]):

            if len(self.inters.neighbours[ndx+1]) == 2 and \
               self.inters.links[ndx]['conn_from'] == 0:                        
                try:                
                    self.patch_path( ndx+1 )
                except:
                    continue
                
                     
   
    def define_networks( self ):
        """
        Creates list of connected intersections,
        to output as line shapefile
        
        
        """
    
        pid = 0
        networks = {}
        
        # open, simple networks
        for ndx in xrange( self.inters.links.shape[0] ):
            
            if len(self.inters.neighbours[ndx+1]) != 1: continue       
            
            network_list = []
            
            to_ndx = ndx + 1
            
            while to_ndx != 0:
                                        
                network_list.append(to_ndx)
                
                to_ndx = self.inters.links[ to_ndx-1 ][ 'conn_to' ]
                
            if len(network_list) > 1:
                
                pid += 1
                 
                networks[pid] = network_list
    
            
        # closed, simple networks
        for ndx in xrange( self.inters.links.shape[0] ):

            if len(self.inters.neighbours[ndx+1]) != 2 or \
               self.inters.links[ndx]['start'] == False: 
                continue
            
            start_id = ndx+1
            
            network_list = []
            
            to_ndx = ndx + 1
            
            while to_ndx != 0:
                                        
                network_list.append(to_ndx)
                
                to_ndx = self.inters.links[ to_ndx-1 ][ 'conn_to' ]
                
                if  to_ndx == start_id:
                    network_list.append(to_ndx)
                    break                
                
            if len(network_list) > 1:
                
                pid += 1
                 
                networks[pid] = network_list
                
        
        return networks        
    
                           
    
