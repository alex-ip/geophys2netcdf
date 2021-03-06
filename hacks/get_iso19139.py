'''
Created on 23Feb.,2017

@author: u76345
'''
import re
import os
from datetime import datetime
from owslib import fes
import argparse
from owslib.csw import CatalogueServiceWeb

class CSWUtils(object):
    '''
    CSW query utilities
    '''
    DEFAULT_CSW_URL = 'http://ecat.ga.gov.au/geonetwork/srv/eng/csw' # GA's externally-facing eCat
    DEFAULT_TIMEOUT = 30 # Timeout in seconds
    DEFAULT_CRS = 'EPSG:4326' # Unprojected WGS84
    DEFAULT_MAXQUERYRECORDS = 100 # Retrieve only this many datasets per CSW query
    DEFAULT_MAXTOTALRECORDS = 2000 # Maximum total number of records to retrieve

    def __init__(self, 
                 csw_url=None, 
                 timeout=None,
                 debug=False
                 ):
        '''
        Constructor for CSWUtils class
        @param csw_url: URL for CSW service. Defaults to value of CSWUtils.DEFAULT_CSW_URL
        @param timeout: Timeout in seconds. Defaults to value of CSWUtils.DEFAULT_TIMEOUT
        '''
        csw_url = csw_url or CSWUtils.DEFAULT_CSW_URL
        timeout = timeout or CSWUtils.DEFAULT_TIMEOUT
        self.debug = debug

        self.csw = CatalogueServiceWeb(csw_url, timeout=timeout)

    def list_from_comma_separated_string(self, comma_separated_string):
        '''
        Helper function to return list of strings from a comma-separated string
        Substitute single-character wildcard for whitespace characters
        @param comma_separated_string: comma-separated string
        @return: list of strings
        '''
        return [re.sub('(\s)', '_', keyword.strip()) for keyword in comma_separated_string.split(',')]

    # A helper function for date range filtering
    def get_date_filter(self, start_datetime=None, stop_datetime=None, constraint='overlaps'):
        '''
        Helper function to return a list containing a pair of FES filters for a date range
        @param  start_datetime: datetime object for start of time period to search
        @param stop_datetime: datetime object for end of time period to search
        @param constraint: string value of either 'overlaps' or 'within' to indicate type of temporal search

        @return: list containing a pair of FES filters for a date range
        '''
        if start_datetime:
            start_date_string = start_datetime.date().isoformat()
        else:
            start_date_string = '1900-01-01' # Distant past

        if start_datetime:
            stop_date_string = stop_datetime.date().isoformat()
        else:
            stop_date_string = '2100-01-01' # Distant future
            
        if constraint == 'overlaps':
            start_filter = fes.PropertyIsLessThanOrEqualTo(propertyname='ows100:TempExtent_begin', literal=stop_date_string)
            stop_filter = fes.PropertyIsGreaterThanOrEqualTo(propertyname='ows100:TempExtent_end', literal=start_date_string)
        elif constraint == 'within':
            start_filter = fes.PropertyIsGreaterThanOrEqualTo(propertyname='ows100:TempExtent_begin', literal=start_date_string)
            stop_filter = fes.PropertyIsLessThanOrEqualTo(propertyname='ows100:TempExtent_end', literal=stop_date_string)

        return [start_filter, stop_filter]
    
    def any_case_match_filters(self, propertyname, word_list):
        '''
        Helper function to return and/or filter from a word list for partial case insensitivity
        Resolves issue where "matchCase=False" is ignored
        @param propertyname: String denoting CSW property to search
        @param word_list: List of strings for (partially) case-insensitive "and" search
        
        @return: List of FES filters expressing "and" query
        '''
        filter_list = []
        for word in word_list:
            word_variant_set = set([word, word.upper(), word.lower(), word.title()])
            
            if len(word_variant_set) == 1: # Use single filter if no "or" required
                filter_list.append(fes.PropertyIsLike(propertyname=propertyname, literal=word_variant_set.pop(), matchCase=False))
            else:
                filter_list.append(fes.Or([fes.PropertyIsLike(propertyname=propertyname, literal=word_variant, matchCase=False)
                                           for word_variant in set([word, word.upper(), word.lower(), word.title()])
                                           ]
                                          )
                                   )
                
        return filter_list

    def get_csw_records(self, 
                        fes_filters, 
                        max_query_records=None, 
                        max_total_records=None
                        ):
        '''
        Generator yeilding nested dicts containing information about each CSW query result record including distributions
        @param fes_filters: List of fes filters to apply to CSW query
        @param max_query_records: Maximum number of records to return per CSW query. Defaults to value of CSWUtils.DEFAULT_MAXRECORDS
        @param max_total_records: Maximum total number of records to return. Defaults to value of CSWUtils.DEFAULT_MAXTOTALRECORDS
 
        '''
        max_query_records = max_query_records or CSWUtils.DEFAULT_MAXQUERYRECORDS
        max_total_records = max_total_records or CSWUtils.DEFAULT_MAXTOTALRECORDS

        startposition = 1 # N.B: This is 1-based, not 0-based
        record_count = 0

        # Keep querying until all results have been retrieved
        while record_count < max_total_records:
            # apply all the filters using the "and" syntax: [[filter1, filter2]]
            self.csw.getrecords2(constraints=[fes_filters],
                                 esn='full',
                                 outputschema='http://www.isotc211.org/2005/gmd',
                                 maxrecords=max_query_records,
                                 startposition=startposition)

            if self.debug:
                print 'CSW request:\n%s' % self.csw.request
                print 'CSW response:\n %s' % self.csw.response

            query_record_count = len(self.csw.records)

            for uuid in self.csw.records.keys():
                record = self.csw.records[uuid]

                record_count += 1
                yield uuid, record.xml
                #print '%d distribution(s) found for "%s"' % (len(info_list), title)

                if record_count >= max_total_records:  # Don't go around again for another query - maximum retrieved
                    raise Exception('Maximum number of records retrieved ($d)' % max_total_records)
    
            if query_record_count < max_query_records:  # Don't go around again for another query - should be the end
                break

            # Increment start position and repeat query
            startposition += max_query_records

        #print '%d records found.' % record_count


    def query_csw(self,
                  keyword_list=None,
                  bounding_box=None,
                  bounding_box_crs=None,
                  anytext_list=None,
                  titleword_list=None,
                  start_datetime=None,
                  stop_datetime=None,
                  max_total_records=None
                  ):
        '''
        Function to query CSW using AND combination of provided search parameters and return generator object
            yielding nested dicts containing information about each record including distributions
        @param keyword_list: List of strings or comma-separated string containing keyword search terms
        @param bounding_box: Bounding box to search as a list of ordinates [bbox.minx, bbox.minx, bbox.maxx, bbox.maxy]
        @param bounding_box_crs: Coordinate reference system for bounding box. Defaults to value of CSWUtils.DEFAULT_CRS
        @param anytext_list: List of strings or comma-separated string containing any text search terms
        @param titleword: List of strings or comma-separated string containing title search terms
        @param start_datetime: Datetime object defining start of temporal search period
        @param stop_datetime: Datetime object defining end of temporal search period
        @param max_total_records: Maximum total number of records to return. Defaults to value of CSWUtils.DEFAULT_MAXTOTALRECORDS 
        
        @return: generator object yielding nested dicts containing information about each record including distributions
        '''
        bounding_box_crs = bounding_box_crs or CSWUtils.DEFAULT_CRS

        # Convert strings to lists if required
        if type(keyword_list) == str:
            keyword_list = self.list_from_comma_separated_string(keyword_list)

        if type(anytext_list) == str:
            anytext_list = self.list_from_comma_separated_string(anytext_list)

        if type(titleword_list) == str:
            titleword_list = self.list_from_comma_separated_string(titleword_list)

        # Build filter list
        fes_filter_list = []
        
        # Check for unchanged, upper-case, lower-case and capitalised keywords
        if keyword_list:
            fes_filter_list += self.any_case_match_filters('Subject', keyword_list)
          
        if anytext_list:
            fes_filter_list += [fes.PropertyIsLike(propertyname='anyText', literal=phrase, matchCase=False) for phrase in anytext_list]
        
        if start_datetime or stop_datetime:
            fes_filter_list += self.get_date_filter(start_datetime, stop_datetime)
        
        if titleword_list:
            fes_filter_list += [fes.PropertyIsLike(propertyname='title', literal=titleword, matchCase=False) for titleword in titleword_list]
        
        if bounding_box:
            fes_filter_list += [fes.BBox(bounding_box, crs=bounding_box_crs)]

        assert fes_filter_list, 'No search criteria defined'

        # Use single filter if no "and" required
        if len(fes_filter_list) == 1:
            fes_filter_list = fes_filter_list[0]

        # Return generator object
        return self.get_csw_records(fes_filter_list, 
                                    max_total_records=max_total_records)


def date_string2datetime(date_string):
    '''
    Helper function to convert date string in one of several possible formats to a datetime object
    @param date_string: date string in one of several possible formats
    
    @return datetime object
    '''
    #?
    DATE_FORMAT_LIST = ['%Y%m%d', '%Y-%m-%d', '%d/%m/%y', '%d/%m/%Y']
    #If there is a date_string (start date or end date from argparse) then for each format type
    # in the DATE FORMAT LIST try to use the datetime.strptime method.
    #datetime.strptime() returns a datetime variable from the input parsed into the correct format.
    #  OR does it check if it is the correct format?
    datetime_result = None
    
    if date_string:
        for format_string in DATE_FORMAT_LIST:
            try:
                datetime_result = datetime.strptime(date_string, format_string)
                break
            except ValueError:
                pass
            
    #if successful return the input as a datetime class object or None.
    return datetime_result


def main():
    '''
    Main function to take command line parameters, perform CSW query and print required output
    '''
    # Define command line arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-k", "--keywords", help="comma-separated list of keywords for search", type=str)
    parser.add_argument("-t", "--titlewords", help="comma-separated list of titlewords for search", type=str)
    parser.add_argument("-a", "--anytext", help="comma-separated list of text snippets for search", type=str)
    parser.add_argument("-b", "--bounds", help="comma-separated <minx>,<miny>,<maxx>,<maxy> ordinates of bounding box for search",
                        type=str)
    parser.add_argument("-c", "--crs", help='coordinate reference system for bounding box coordinates for search. Defaults to "EPSG:4326".',
                        type=str)
    parser.add_argument("-s", "--start_date", help="start date for search", type=str)
    parser.add_argument("-e", "--end_date", help="end date for search", type=str)
    parser.add_argument("-u", "--url", help="CSW URL to query - Defaults to GA's external eCat (%s)" %
                        CSWUtils.DEFAULT_CSW_URL, type=str)
    parser.add_argument("-m", "--max_results", help="Maximum number of records to return. Defaults to %d" % CSWUtils.DEFAULT_MAXTOTALRECORDS, type=int)
    parser.add_argument("-o", "--output_dir", help="output directory for XML files", type=str, default='.')    
    parser.add_argument('--debug', action='store_const', const=True, default=False,
                        help='output debug information. Default is no debug info')
    
    args = parser.parse_args()


#CONVERTING INPUT TO CORRECT DATA TYPES AND FORMATS

    # Convert string to list of floats
    # if the user inputs an argument for bounds, then convert this string to a list of floats
    #  whereas each list object is split at the commas.
    # If the user does not call the bounds argument, then don't use it.
    if args.bounds:
        bounds = [float(ordinate) for ordinate in args.bounds.split(',')]
    else:
        bounds = None

    # Convert string to datetime
    start_date = date_string2datetime(args.start_date)

    # Convert string to datetime
    end_date = date_string2datetime(args.end_date)
    
    output_dir = os.path.abspath(args.output_dir)

    #creatse a CSW object and populates the parameters with argparse inputs

    cswu = CSWUtils(args.url,
                    debug=args.debug)
    
    record_generator = cswu.query_csw(keyword_list=args.keywords,
                                 anytext_list=args.anytext,
                                 titleword_list=args.titlewords,
                                 bounding_box=bounds,
                                 start_datetime=start_date,
                                 stop_datetime=end_date,
                                 max_total_records=args.max_results
                                 )

    for uuid, record_xml in record_generator:
        output_path = os.path.join(output_dir, uuid + '.xml')
        print 'Writing XML to %s' % output_path
        output_file = open(output_path, 'w')
        
        output_file.write(record_xml)
        output_file.close()


if __name__ == '__main__':
    main()
