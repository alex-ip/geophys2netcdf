'''
Created on 2Sep.,2016

@author: Alex Ip
'''
import os
import sys
import errno
import logging
import re
import subprocess
import tempfile
from shutil import rmtree
import glob
from osgeo import gdal, gdalconst
import netCDF4
from geophys2netcdf.array_pieces import array_pieces
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Initial logging level for this module

class ERS2NetCDFChecker(object):
    '''
    Class definition for ERS2NetCDFChecker to check conversion of ERS to NetCDF datasets
    '''
    FILE_EXTENSION = 'zip'
    def __init__(self, dataset_dir=None, debug=False):
        '''
        Constructor for class ERS2NetCDFChecker
        '''
        self._zipdir = None
        self._debug = False
        self.debug = debug # Set property
        
        if dataset_dir:
            self.check_ERS2NetCDF(dataset_dir)
            
    
    def __del__(self):
        '''
        Destructor for class ERS2NetCDFChecker
        '''
        if self._zipdir and not self._debug:
            logger.info('Removing temporary directory %s', self._zipdir)
            rmtree(self._zipdir)

    def check_ERS2NetCDF(self, dataset_dir):
        '''
        Function to check NetCDF file against zipped ERS. Assumes only one .zip file and one .nc file exist in dataset_dir
        '''
        assert os.path.isdir(dataset_dir), '%s is not a directory' % dataset_dir
        zip_list = glob.glob(os.path.join(dataset_dir, '*.zip'))
        assert len(zip_list) == 1, 'Unable to find single zip file in %s. (%s)' % (dataset_dir, zip_list.join(', '))
        zip_path = zip_list[0]
        nc_list = glob.glob(os.path.join(dataset_dir, '*.nc'))
        assert len(nc_list) == 1, 'Unable to find single NetCDF file in %s. (%s)' % (dataset_dir, nc_list.join(', '))
        nc_path = nc_list[0]
        
        # Remove any existing zip directory
        if self._zipdir:
            logger.info('Removing previous temporary directory %s', self._zipdir)
            os.removedirs(self._zipdir)

        base_path = os.path.join(tempfile.gettempdir(), os.path.splitext(os.path.basename(zip_path))[0])
        self._zipdir = base_path
        
        # Unzip file into fresh directory
        zipdir_revision = 0
        while os.path.exists(self._zipdir):
            zipdir_revision += 1
            self._zipdir = '%s_%s' % (base_path, zipdir_revision)
        logger.debug('self._zipdir = %s', self._zipdir)
        
        try:
            os.makedirs(self._zipdir)
        except OSError, exception:
            if exception.errno != errno.EEXIST or not os.path.isdir(self._zipdir):
                raise exception

        unzip_command = ['unzip',
                         zip_path,
                         '-d',
                         self._zipdir
                         ]
        
        subprocess.check_call(unzip_command)
        logger.info('%s unzipped into %s', zip_path, self._zipdir)

        file_list = os.listdir(self._zipdir)
        extension_set = set([os.path.splitext(file_path)[1].lower() for file_path in file_list])
        logger.debug('file_list = %s', file_list)
        logger.debug('extension_set = %s', extension_set)

        if set(['.ers', '']) <= extension_set: # N.B: .isi files are optional
            logger.info('%s contains an ERS dataset', self._zipdir)
            ers_list = [file_path for file_path in file_list if file_path.lower().endswith('.ers')]
            assert len(ers_list) == 1, 'Multiple .ers files found in %s' % self._zipdir
            
            ers_path = os.path.join(self._zipdir, ers_list[0])
            if os.path.exists(ers_path):
                self.compare_ERS2NetCDF(ers_path, nc_path)
            
        else:
            raise Exception('Unhandled file types in zip file %s' % zip_path)      
        
    def compare_ERS2NetCDF(self, ers_path, nc_path):
        '''
        Function to compare ERS file to NetCDF file
        Currently retrieves data from ERS file using GDAL, and from NetCDF file using netCDF4.
        Note that NetCDF array is YX ordered, with a LL origin, while the ERS is XY ordered with a UL origin.
        '''
        
        def check_ers_extent(ers_path, geotransform):
            '''
            Function to check extent shown in ERS header file against GDAL geotransform.
            Used to check for bug in GDAL 1.9 and earlier
            
            Sample ERS header file is as follows:
            
            DatasetHeader Begin
                LastUpdated    = Tue Feb 28 05:16:57 GMT 2012
                Version    = "5.0"
                DataSetType    = ERStorage
                DataType    = Raster
                HeaderOffset    = 512
                CoordinateSpace Begin
                    Projection    = "GEODETIC"
                    CoordinateType    = LATLONG
                    Datum    = "GDA94"
                    Rotation    = 0:0:0 
                CoordinateSpace End
                ByteOrder    = LSBFirst
                RasterInfo Begin
                    CellType    = IEEE4ByteReal
                    NrOfLines    = 4182
                    NrOfCellsPerLine    = 5717
                    NrOfBands    = 1
                    NullCellValue    = -99999.00000000
                    CellInfo Begin
                        Xdimension    =      0.00833333
                        Ydimension    =      0.00833333
                    CellInfo End
                    RegistrationCellX    = 0
                    RegistrationCellY    = 0
                    RegistrationCoord Begin
                        Longitude    = 109:6:0.843442298
                        Latitude    = -9:21:17.81304202
                    RegistrationCoord End
                    BandId Begin
                        Value    = "IR_gravity_anomaly V1"
                    BandId End
                RasterInfo End
            DatasetHeader End

            Sample geotransform = (109.1002342895272 0.00833333 0 -9.354948067227777 0 -0.00833333)
            '''
            
            def dms2degrees(dms_string):
                '''
                Function to translate degrees:minutes:seconds string into float degrees
                '''
                dms_match = re.match('(\-|\+)?(\d+):(\d+):(\d+(\.\d*)?)', dms_string)
                assert dms_match, 'Unable to parse degrees:minutes:seconds string %s' % dms_string
                sign = dms_match.group(1)
                try:
                    degrees = float(dms_match.group(2))
                except:
                    degrees = 0.0
                    
                try:
                    minutes = float(dms_match.group(3))
                except:
                    minutes = 0.0
                    
                try:
                    seconds = float(dms_match.group(4))
                except:
                    seconds = 0.0
                    
                value = degrees + minutes / 60 + seconds / 360
                
                if sign == '-':
                    value = -value
                    
                return value
                
            #TODO: Make this work with UTM datasets
            ers_file = open(ers_path)
            lines =  ers_file.readlines()
            ers_file.close()
            
            ers_dict = {}
            parent_dicts = [ers_dict]
            section_dict = ers_dict
            for line in lines:
                begin_match = re.match('\s*([^\s]+) Begin', line, re.IGNORECASE)
                if begin_match:
                    current_section = begin_match.group(1)
                    parent_dicts.append(section_dict)
                    section_dict = {}
                    parent_dicts[-1][current_section] = section_dict
                    continue
                
                end_match = re.match('\s*([^\s]+) End', line, re.IGNORECASE)
                if end_match:
                    end_section = end_match.group(1)
                    assert end_section == current_section, 'Malformed sections in ERS file: End found for %s when current section is %s' % (end_section, current_section)
                    parent_dicts.pop()
                    section_dict = parent_dicts[-1]
                    continue
                
                keyvalue_match = re.match('\s*([^\s]+)\s*=\s*([^\s]+)', line, re.IGNORECASE)
                if keyvalue_match:
                    key = keyvalue_match.group(1)
                    value = keyvalue_match.group(2)
                    section_dict[key] = re.sub('^"|"$', '', value) # Strip double quotes from string
                
            assert parent_dicts[-1] == ers_dict, 'Section not closed in ERS file'
            
            assert (float(ers_dict['DatasetHeader']['RasterInfo']['CellInfo']['Xdimension']) - geotransform[1] < 0.000001), 'ERS & GDAL pixel X size are not equal'
            assert (float(ers_dict['DatasetHeader']['RasterInfo']['CellInfo']['Ydimension']) - geotransform[5] < 0.000001), 'ERS & GDAL pixel Y size are not equal'
            assert (dms2degrees(ers_dict['DatasetHeader']['RasterInfo']['CellInfo']['RegistrationCoord']['Longitude']) - geotransform[0] < 0.000001), 'ERS & GDAL X origin are not equal'
            assert (dms2degrees(ers_dict['DatasetHeader']['RasterInfo']['CellInfo']['RegistrationCoord']['Latitude']) - geotransform[3] < 0.000001), 'ERS & GDAL Y origin are not equal'
            
            return True
        
        assert os.path.isfile(ers_path), 'ERS file %s does not exist' % ers_path
        assert os.path.isfile(nc_path), 'NetCDF file %s does not exist' % nc_path
        
        try:
            ers_gdal_dataset = gdal.Open(ers_path, gdalconst.GF_Read)
            assert ers_gdal_dataset, 'Unable to open ERS file %s using GDAL' % ers_path
            
            if check_ers_extent(ers_path, ers_gdal_dataset.GetGeoTransform()):
                print 'ERS extents translated correctly by GDAL'
                        
            nc_gdal_dataset = gdal.Open(nc_path, gdalconst.GF_Read)
            assert nc_gdal_dataset, 'Unable to open NetCDF file %s using GDAL' % nc_path
            nc_dataset = netCDF4.Dataset(nc_path, 'r')
            assert nc_dataset, 'Unable to open NetCDF file %s using netCDF4' % nc_path
            
            if (ers_gdal_dataset.RasterCount == 1) and (nc_gdal_dataset.RasterCount == 1):
                print 'PASS: Both datasets have a single data variable'
            else:
                raise Exception('FAIL: Both datasets DO NOT have a single data variable')
            
            if (ers_gdal_dataset.RasterXSize == nc_gdal_dataset.RasterXSize) and (ers_gdal_dataset.RasterYSize == nc_gdal_dataset.RasterYSize):
                print 'PASS: Both datasets are the same shape'
            else:
                raise Exception('FAIL: Both datasets are not of the same shape')
            
            if ers_gdal_dataset.GetProjection() == nc_gdal_dataset.GetProjection():
                print 'PASS: Both datasets have the same projection'
            else:
                raise Exception('FAIL: Both datasets do not have the same projection')
            
            if not [nc_gdal_dataset.GetGeoTransform()[index] for index in range(5) if ers_gdal_dataset.GetGeoTransform()[index] - nc_gdal_dataset.GetGeoTransform()[index] > 0.000001]:
                print 'PASS: Both datasets have the same spatial extent and resolution'
            else:
                raise Exception('FAIL: Both datasets do not have the same spatial extent and resolution')
            
            ers_band = ers_gdal_dataset.GetRasterBand(1)
            nc_band = ers_gdal_dataset.GetRasterBand(1)
            
            if ers_band.GetNoDataValue() == nc_band.GetNoDataValue():
                print 'PASS: Both datasets have the same no-data value'
            else:
                raise Exception('FAIL: Both datasets do not have the same no-data value')
            
            
            
            # Find variable with "grid_mapping" attribute in NetCDF dataset - assumed to be 2D data variable
            try:
                data_variable = [variable for variable in nc_dataset.variables.values() if hasattr(variable, 'grid_mapping')][0]
            except:
                raise Exception('Unable to determine data variable (must have "grid_mapping" attribute')
        
            pixel_count = 0
            weighted_mean_nc_value = 0
            weighted_mean_ers_value = 0
            weighted_mean_percentage_difference = 0
            for nc_piece_array, start_indices in array_pieces(data_variable, 1000000000): # 1GB Pieces  
                piece_size = reduce(lambda x, y: x*y, nc_piece_array.shape)
                #print start_indices, nc_piece_array.shape, piece_size
                              
                if type(nc_piece_array) == np.ma.core.MaskedArray:
                    nc_piece_array = nc_piece_array.data

                # Invert NetCDF array to convert LL origin to UL
                nc_piece_array = np.flipud(nc_piece_array)

                # Note reversed indices to match YX ordering in NetCDF with XY ordering in ERS
                ers_piece_array = ers_band.ReadAsArray(start_indices[1], ers_gdal_dataset.RasterYSize - start_indices[0] - nc_piece_array.shape[0], nc_piece_array.shape[1], nc_piece_array.shape[0])

                percentage_difference_piece_array = np.absolute(1 - nc_piece_array / ers_piece_array) * 100

                try:
                    min_nc_value = min(min_nc_value, np.nanmin(nc_piece_array))
                except:
                    min_nc_value = np.nanmin(nc_piece_array)
        
                try:
                    max_nc_value = max(max_nc_value, np.nanmax(nc_piece_array))
                except:
                    max_nc_value = np.nanmax(nc_piece_array)

                try:
                    min_ers_value = min(min_ers_value, np.nanmin(ers_piece_array))
                except:
                    min_ers_value = np.nanmin(ers_piece_array)
        
                try:
                    max_ers_value = max(max_ers_value, np.nanmax(ers_piece_array))
                except:
                    max_ers_value = np.nanmax(ers_piece_array)

                try:
                    min_percentage_difference = min(min_percentage_difference, np.nanmin(percentage_difference_piece_array))
                except:
                    min_percentage_difference = np.nanmin(percentage_difference_piece_array)
        
                try:
                    max_percentage_difference = max(max_percentage_difference, np.nanmax(percentage_difference_piece_array))
                except:
                    max_percentage_difference = np.nanmax(percentage_difference_piece_array)

                weighted_mean_nc_value = weighted_mean_nc_value + np.nanmean(nc_piece_array) * piece_size
                weighted_mean_ers_value = weighted_mean_ers_value + np.nanmean(ers_piece_array) * piece_size
                weighted_mean_percentage_difference = weighted_mean_percentage_difference + np.nanmean(percentage_difference_piece_array) * piece_size
                
                pixel_count += piece_size

            mean_nc_value = weighted_mean_nc_value / pixel_count 
            mean_ers_value = weighted_mean_ers_value / pixel_count 
            mean_percentage_difference = weighted_mean_percentage_difference / pixel_count 
           
            if max_percentage_difference < 0.000001:
                print 'PASS: There is less than 0.000001% percentage_difference in all data values'
            else:
                #raise Exception('FAIL: There is more than 0.000001 percentage_difference in data values')
                pass
  
            print 'min nc_value = %f, mean nc_value = %f, max nc_value = %f' % (min_nc_value, mean_nc_value, max_nc_value)
            print 'min ers_value = %f, mean ers_value = %f, max ers_value = %f' % (min_ers_value, mean_ers_value, max_ers_value)
            print 'min percentage_difference = %f%%, mean percentage_difference = %f%%, max percentage_difference = %f%%' % (min_percentage_difference, mean_percentage_difference, max_percentage_difference)
             
        except Exception, e:
            print 'File comparison failed: %s' % e.message
            
            
def main():
    if len(sys.argv) == 2: # Only directory provided
        e2nchecker = ERS2NetCDFChecker(sys.argv[1])
    if len(sys.argv) == 3: # Only directory provided
        e2nchecker = ERS2NetCDFChecker()
        e2nchecker.compare_ERS2NetCDF(sys.argv[1], sys.argv[2])
        
if __name__ == '__main__':
    main()
