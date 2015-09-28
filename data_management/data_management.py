# -*- coding: utf-8 -*-
"""
.. module:: data_management.py
   :platform: Unix
   :synopsis:   Finds users running at specific date, 
                creates and share top data folder 
                share top data folder with the users liste in the proposal 
   :INPUT
      Date of the experiments 

.. moduleauthor:: Francesco De Carlo <decarlof@gmail.com>

This module is largely John Hammonds work to which I'll be adding
some scripts as needed.
""" 

import os
import pytz
import datetime
from globusonline.transfer.api_client import Transfer, create_client_from_args
from suds.wsse import Security, UsernameToken
from suds.client import Client
from suds.transport.https import HttpAuthenticated
import logging
import sys
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
#base = cf.get('hosts', 'internal')
base = cf.get('hosts', 'external')

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

def get_experiment_start(beamline='2-BM-A,B', date=None):
    """Find the experiment start date for a given beamline and date

    Returns experiment_start."""
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
                        experiment_start = event.startTime
        except:
            ipdb.set_trace()
            raise

    return experiment_start

def get_experiment_end(beamline='2-BM-A,B', date=None):
    """Find the experiment end date for a given beamline and date

    Returns experiment_end."""
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
                        experiment_end = event.endTime
        except:
            ipdb.set_trace()
            raise

    return experiment_end

def get_beamtime_request(beamline='2-BM-A,B', date=None):
    """Find the proposal beamtime request id for a given beamline and date

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
                        beamtime_request = event.beamtimeRequest.id
        except:
            ipdb.set_trace()
            raise

    return beamtime_request
    
if __name__ == "__main__":

    # Global settings
    beamline = '32-ID-B,C'
    instrument_name = 'TXM'

    #now = datetime.datetime.now(pytz.timezone('US/Central'))
    now = datetime.datetime(2014, 10, 18, 10, 10, 30).replace(tzinfo=pytz.timezone('US/Central'))

    datetime_format = '%Y-%m-%dT%H:%M:%S%z'
    print "Time of Day: ", now.strftime(datetime_format)

    # scheduling system settings
    runScheduleServiceClient, beamlineScheduleServiceClient = setup_connection()
    run_name = findRunName(now.replace(tzinfo=None), now.replace(tzinfo=None))
    run_schedule = findBeamlineSchedule(beamline, run_name)
    #run_beamline_request = findBeamtimeRequestsByBeamline(beamline, run_name)

    proposal_id = get_proposal_id(beamline, now.replace(tzinfo=None))
    beamtime_request = get_beamtime_request(beamline, now.replace(tzinfo=None))
    proposal_title = str(get_proposal_title(beamline, now.replace(tzinfo=None)))
    experiment_start = str(get_experiment_start(beamline, now.replace(tzinfo=None)))
    experiment_end = str(get_experiment_end(beamline, now.replace(tzinfo=None)))
    users = get_users(beamline, now.replace(tzinfo=None))

    print "Run Name: ", run_name 
    #print "Run Schedule: ", run_schedule 
    #print run_beamline_request 
    #print users 
    print "GUP: ", proposal_id 
    print "Proposal Title: ", proposal_title
    print "Experiment Start: ", experiment_start
    print "Experiment End: ", experiment_end
    print "Beamtime Request: ",  beamtime_request 
    
    # find the Principal Investigator
    for tag in users:
        if users[tag].get('piFlag') != None:
            name = str(users[tag]['firstName'] + ' ' + users[tag]['lastName'])            
            role = "*"
            institution = str(users[tag]['institution'])
            badge = str(users[tag]['badge'])
            email = str(users[tag]['email'])
            print role, badge, name, institution, email
        else:            
            print users[tag]['badge'], users[tag]['firstName'], users[tag]['lastName'], users[tag]['institution'], users[tag]['email']
            
    # find user emails
    print "Data owner emails: "
    for tag in users:
        if users[tag].get('email') != None:
            email = str(users[tag]['email'])
            print email
        else:            
            print "Missing e-mail for:", users[tag]['badge'], users[tag]['firstName'], users[tag]['lastName'], users[tag]['institution']

    data_directory = 'd' + str(proposal_id) + str(beamtime_request)
    os.system('mkdir '+ data_directory)
    print "Created directory: ", data_directory

        
    