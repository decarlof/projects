# -*- coding: utf-8 -*-
"""
.. module:: add_dataExchange_entries.py
   :platform: Unix
   :synopsis: add Data Exchange entries to an HDF5 file
   :INPUT
      Data Exchange file name 

.. moduleauthor:: David Vine <djvine@gmail.com>, Francesco De Carlo <decarlof@gmail.com>

This module is largely John Hammonds work to which I'll be adding
some scripts as needed.
""" 

import os
import pytz
import datetime
from data_exchange import DataExchangeFile, DataExchangeEntry
import process_variables as pv


from suds.wsse import Security, UsernameToken
from suds.client import Client
from suds.transport.https import HttpAuthenticated
import logging
import sys
import datetime
import traceback
import urllib2
import httplib
from xml.sax import SAXParseException
import ipdb
from collections import defaultdict
import ConfigParser

debug = False

""" You must use the APS web password. You can check it by logging into
the proposal system. Be careful because this system also accepts LDAP
account info.

The credentials are stored in a '.ini' file and read by python.
 - Create a file called 'credentials.ini',
 - Put the following text in it:
 [credentials]
 username = YOUR BADGE NUMBER
 password = YOUR APS WEB PASSWORD

 that's it.

"""

cf = ConfigParser.ConfigParser()
cf.read('credentials.ini')
username = cf.get('credentials', 'username')
password = cf.get('credentials', 'password')

# Uncomment one if using ANL INTERNAL or EXTERNAL network
base = cf.get('hosts', 'internal')
#base = cf.get('hosts', 'external')

class HTTPSConnectionV3(httplib.HTTPSConnection):
    def __init__(self, *args, **kwargs):
        httplib.HTTPSConnection.__init__(self, *args, **kwargs)

    def connect(self):
        sock = socket.create_connection((self.host, self.port), self.timeout)
        if self._tunnel_host:
            self.sock = sock
            self._tunnel()
        try:
            print ("using connection")
            self.sock = ssl.wrap_socket(sock, self.key_file, self.cert_file, \
                                        ssl_version=ssl.PROTOCOL_SSLv3)
        except ssl.SSLError, e:
            print("Trying SSLv3.")
            self.sock = ssl.wrap_socket(sock, self.key_file, self.cert_file, \
                                        ssl_version=ssl.PROTOCOL_SSLv23)

class HTTPSHandlerV3(urllib2.HTTPSHandler):
    def https_open(self, req):
        print "using this opener"
        return self.do_open(HTTPSConnectionV3, req)

def setSoapHeader(client, username, password):
    security = Security()
    token = UsernameToken(username, password)
    token.setcreated()
    security.tokens.append(token)
    if debug:
        print security
    client.set_options(wsse=security)

def findRunName(startDate, endDate):
    """Find the official run name for the run that spans the
    given startDate and endDate

    Returns string."""
    runScheduleServiceClient, beamlineScheduleServiceClient = setup_connection()
    try:
        result = runScheduleServiceClient.service.findAllRuns()
    except Exception:
        print "ERROR in findRunName"
        sys.exit(2)
    except soapFault:
        print "ERROR in findRunName"
        print soapFault
        sys.exit(2)
    runArray = result.run
    runName = None
    for run in runArray:

        try:
            if startDate >= run.startTime and endDate <= run.endTime:
                runName = run.runName
                break
        except Exception as ex:
            print "ERROR caught in findRunName:" + str(ex)
            print startDate
            print run.startTime
            print endDate
            print run.endTime
            raise ex
    return runName


def findBeamlineSchedule(beamlineName, runName):
    """Find beamline schedule for given beamlineName and runName"""

    runScheduleServiceClient, beamlineScheduleServiceClient = setup_connection()
    try:
        result  = beamlineScheduleServiceClient.service.findBeamlineSchedule(beamlineName, runName)
    except SAXParseException as ex:
        print "ERROR in findBeamlineSchedule\n"
        traceback.print_exc()
        sys.exit(2)

    return result

def findBeamtimeRequestsByBeamline(beamlineName, runName):
    """Find beamline schedule for given beamlineName and runName

    Returns schedule object."""
    try:
        result  = beamlineScheduleServiceClient.service.findBeamtimeRequestsByBeamline(beamlineName, runName)
    except SAXParseException:
        print "ERROR in findBeamtimeRequestsByBeamline"
    except Exception:
        print "ERROR in findBeamtimeRequestByBeamline\n"
        traceback.print_exc()
        sys.exit(2)
    return result

def setup_connection():
    result = urllib2.install_opener(urllib2.build_opener(HTTPSHandlerV3()))
    logging.raiseExceptions = 0

    beamlineScheduleServiceURL = base + \
         'beamlineScheduleService/beamlineScheduleWebService.wsdl'

    runScheduleServiceURL = base + \
         'runScheduleService/runScheduleWebService.wsdl'

    try:
        credentials = dict(username=username, password=password)
        t = HttpAuthenticated(**credentials)
        if debug:
            print t.u2handlers()
            print t.credentials()
        runScheduleServiceClient = Client(runScheduleServiceURL)
        runScheduleServiceClient.options.cache.setduration(seconds=10)
        result = setSoapHeader(runScheduleServiceClient, username, password)
        beamlineScheduleServiceClient = Client(beamlineScheduleServiceURL)
        beamlineScheduleServiceClient.options.cache.setduration(seconds=10)
        result = setSoapHeader(beamlineScheduleServiceClient, username, password)
    except Exception, ex:
        print "CANNOT OPEN SERVICES:" + str(ex)
        raise
        exit(-1)

    return runScheduleServiceClient, beamlineScheduleServiceClient

def get_users(beamline='2-BM-A,B', date=None):
    """Find all users listed in the proposal for a given beamline and date

    Returns users."""
    runScheduleServiceClient, beamlineScheduleServiceClient = setup_connection()
    if not date:
        date = datetime.datetime.now()
    run_name = findRunName(date, date)
    schedule = findBeamlineSchedule(beamline, run_name)

    events = schedule.activities.activity
    users = defaultdict(dict)
    for event in events:
        try:
            if event.activityType.activityTypeName in ['GUP', 'PUP', 'rapid-access', 'sector staff']:
                if date >= event.startTime and date <= event.endTime:
                        for experimenter in event.beamtimeRequest.proposal.experimenters.experimenter:
                            for key in experimenter.__keylist__:
                                users[experimenter.lastName][key] = getattr(experimenter, key)
        except:
            ipdb.set_trace()
            raise

    return users

def get_proposal_id(beamline='2-BM-A,B', date=None):
    """Find the proposal number (GUP) for a given beamline and date

    Returns users."""
    runScheduleServiceClient, beamlineScheduleServiceClient = setup_connection()
    if not date:
        date = datetime.datetime.now()
    run_name = findRunName(date, date)
    schedule = findBeamlineSchedule(beamline, run_name)

    events = schedule.activities.activity
    users = defaultdict(dict)
    for event in events:
        try:
            if event.activityType.activityTypeName in ['GUP', 'PUP', 'rapid-access', 'sector staff']:
                if date >= event.startTime and date <= event.endTime:
                        proposal_id = event.beamtimeRequest.proposal.id
        except:
            ipdb.set_trace()
            raise

    return proposal_id

def get_proposal_title(beamline='2-BM-A,B', date=None):
    """Find the proposal title for a given beamline and date

    Returns users."""
    runScheduleServiceClient, beamlineScheduleServiceClient = setup_connection()
    if not date:
        date = datetime.datetime.now()
    run_name = findRunName(date, date)
    schedule = findBeamlineSchedule(beamline, run_name)

    events = schedule.activities.activity
    users = defaultdict(dict)
    for event in events:
        try:
            if event.activityType.activityTypeName in ['GUP', 'PUP', 'rapid-access', 'sector staff']:
                if date >= event.startTime and date <= event.endTime:
                        proposal_title = event.beamtimeRequest.proposal.proposalTitle
        except:
            ipdb.set_trace()
            raise

    return proposal_title

if __name__ == "__main__":

# Global settings
    hdf5_file_name = '/local/dataraid/databank/tmp/test/ALS.h5'

    #beamline = '2-BM-A,B'
    #instrument_name = 'microCT'
    beamline = '32-ID-B,C'
    instrument_name = 'TXM'

    sample_name = "sample_name"

    #now = datetime.datetime(year, month, day, hour, min, s)
    #now = datetime.datetime(2014, 10, 13, 10, 10, 30).replace(tzinfo=pytz.timezone('US/Central'))
    #now = datetime.datetime(2014, 10, 19, 10, 10, 30).replace(tzinfo=pytz.timezone('US/Central'))
    #now = datetime.datetime(2014, 10, 27, 10, 10, 30).replace(tzinfo=pytz.timezone('US/Central'))
    now = datetime.datetime(2014, 11, 03, 10, 10, 30).replace(tzinfo=pytz.timezone('US/Central'))
    #now = datetime.datetime(2014, 11, 10, 10, 10, 30).replace(tzinfo=pytz.timezone('US/Central'))
    #now = datetime.datetime(2014, 11, 17, 10, 10, 30).replace(tzinfo=pytz.timezone('US/Central'))
    #now = datetime.datetime(2014, 11, 24, 10, 10, 30).replace(tzinfo=pytz.timezone('US/Central'))
    #now = datetime.datetime(2014, 12, 01, 10, 10, 30).replace(tzinfo=pytz.timezone('US/Central'))
    #now = datetime.datetime(2014, 12, 8, 10, 10, 30).replace(tzinfo=pytz.timezone('US/Central'))
    #now = datetime.datetime.now(pytz.timezone('US/Central'))

    datetime_format = '%Y-%m-%dT%H:%M:%S%z'
    print "Time of Day: ", now.strftime(datetime_format)

# PV settings
    current  = pv.current.get()
    top_up_status = pv.top_up_status.get()
    aps_mode = 'Regular Fill'    
    if top_up_status == 1:
        aps_mode = 'Top-up'
    
    energy = pv.undulator_energy.get()
    energy_dcm = pv.energy_dcm.get()
    undulator_gap = pv.undulator_gap.get()
    mirror_x = pv.mirror_x.get()
    mirror_y = pv.mirror_y.get()
    ccd_camera_objective_mode = pv.ccd_camera_objective_mode.get()
    if ccd_camera_objective_mode == 0:
        ccd_camera_objective = pv.ccd_camera_objective_label_0.get() 
        ccd_camera_objective_manufacturer = 'Zeiss' 
        ccd_camera_objective_model = 'Epiplan-Neofluar HD (422310-9900-000)'
        ccd_camera_objective_magnification = '1.25'
        ccd_camera_objective_numerical_aperture = '0.03'
    if ccd_camera_objective_mode == 1:
        ccd_camera_objective = pv.ccd_camera_objective_label_1.get()
        ccd_camera_objective_manufacturer = 'Zeiss' 
        ccd_camera_objective_model = 'Fluar (420130-9900-000)'
        ccd_camera_objective_magnification = '5'
        ccd_camera_objective_numerical_aperture = '0.25'
    if ccd_camera_objective_mode == 2:
        ccd_camera_objective = pv.ccd_camera_objective_label_2.get()
        ccd_camera_objective_manufacturer = 'Zeiss' 
        ccd_camera_objective_model = 'EC Epiplan-Neofluar HD (000000-1156-524)'
        ccd_camera_objective_magnification = '20'
        ccd_camera_objective_numerical_aperture = '0.5'
        
    
    print "Current: ", current
    print "Top Up Status: ", top_up_status
    print "Undulator Energy: ", energy
    print "Undulator Gap:", undulator_gap
    print "Energy DCM: ", energy_dcm
    print "Mirror X", mirror_x
    print "Mirror Y", mirror_y
    print "CCD camera objective number", ccd_camera_objective_mode
    print "CCD camera objective label", ccd_camera_objective

# scheduling system settings
    runScheduleServiceClient, beamlineScheduleServiceClient = setup_connection()
    run_name = findRunName(now.replace(tzinfo=None), now.replace(tzinfo=None))
    schedule = findBeamlineSchedule(beamline, run_name)
    beamline_request = findBeamtimeRequestsByBeamline(beamline, run_name)
    users = get_users(beamline, now.replace(tzinfo=None))
    proposal_id = get_proposal_id(beamline, now.replace(tzinfo=None))
    proposal_title = str(get_proposal_title(beamline, now.replace(tzinfo=None)))
    
    print "Proposal Title: ", proposal_title
    #print beamline_request

    # find the Principal Investigator
    for tag in users:
        if users[tag].get('piFlag') != None:
            name = str(users[tag]['firstName'] + ' ' + users[tag]['lastName'])            
            role = "Project PI"
            affiliation = str(users[tag]['institution'])
            facility_user_id = str(users[tag]['badge'])
            email = str(users[tag]['email'])
            print role, name, affiliation, facility_user_id, email
        else:            
            print users[tag]['badge'], users[tag]['firstName'], users[tag]['lastName'], users[tag]['institution']

# adding tags to the Data Exchange file
    if (hdf5_file_name != None):
        if os.path.isfile(hdf5_file_name):
            print "Data Exchange file: [%s] already exists", hdf5_file_name
            # Open DataExchange file
            f = DataExchangeFile(hdf5_file_name, mode='a') 

        else:
            print "Creating Data Exchange File [%s]", hdf5_file_name
            # Create new folder.
            dirPath = os.path.dirname(hdf5_file_name)
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)

            # Get the file_name in lower case.
            lFn = hdf5_file_name.lower()

            # Split the string with the delimeter '.'
            end = lFn.split('.')

            # Open DataExchange file
            f = DataExchangeFile(hdf5_file_name, mode='w') 

        # Create HDF5 subgroup
        # /measurement/instrument
        f.add_entry(DataExchangeEntry.instrument(name={'value': instrument_name}) )

        f.add_entry(DataExchangeEntry.source(name={'value': 'Advanced Photon Source'},
                                            date_time={'value': now.strftime(datetime_format)},
                                            beamline={'value': beamline},
                                            current={'value': current, 'units': 'mA', 'dataset_opts': {'dtype': 'd'}},
                                            energy={'value': 7.0, 'units':'GeV', 'dataset_opts': {'dtype': 'd'}},
                                            mode={'value':aps_mode}
                                            )
        )
        # Create HDF5 subgroup
        # /measurement/instrument/attenuator
        f.add_entry(DataExchangeEntry.attenuator(thickness={'value': 1e-3, 'units': 'm', 'dataset_opts': {'dtype': 'd'}},
                                                type={'value': 'Al'}
                                                )
            )

        # Create HDF5 subgroup
        # Create HDF5 subgroup
        # /measurement/instrument/monochromator
        f.add_entry(DataExchangeEntry.monochromator(type={'value': 'double crystal monochromator'},
                                                    energy={'value': energy_dcm, 'units': 'keV', 'dataset_opts': {'dtype': 'd'}},
                                                    energy_error={'value': 1e-5, 'units': 'keV', 'dataset_opts': {'dtype': 'd'}},
                                                    mono_stripe={'value': 'Si (1,1,1)'},
                                                    )
            )

        # Create HDF5 subgroup
        # /measurement/experimenter
        #print role, name, affiliation, facility_user_id, email

        f.add_entry(DataExchangeEntry.experimenter(name={'value':name},
                                                    role={'value':role},
                                                    affiliation={'value':affiliation},
                                                    facility_user_id={'value':facility_user_id},
                                                    email={'value':email}
                        )
            )

        f.add_entry(DataExchangeEntry.objective(manufacturer={'value':ccd_camera_objective_manufacturer},
                                                    model={'value':ccd_camera_objective_model},
                                                    magnification={'value':ccd_camera_objective_magnification},
                                                    numerical_aperture={'value':ccd_camera_objective_numerical_aperture}
                        )
            )

        f.add_entry(DataExchangeEntry.scintillator(manufacturer={'value':'Crytur'},
                                                    serial_number={'value':'12'},
                                                    name={'value':'LuAg'},
                                                    type={'value':'LuAg'},
                                                    scintillating_thickness={'value':50e-6, 'dataset_opts': {'dtype': 'd'}},
                                                    substrate_thickness={'value':50e-6, 'dataset_opts': {'dtype': 'd'}},
                )
            )

        # Create HDF5 subgroup
        # /measurement/experiment
        f.add_entry( DataExchangeEntry.experiment(proposal={'value':proposal_id}
                    )
            )

        if (sample_name != None):
            f.add_entry(DataExchangeEntry.sample(root='/measurement', 
                                                    name={'value':sample_name}, 
                                                    description={'value':proposal_title}),
                                                    overwrite=True
                                                )

    f.close()

