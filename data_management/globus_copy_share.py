# -*- coding: utf-8 -*-
"""
.. module:: globus_share.py
   :platform: Unix
   :synopsis:   
        Share via Globus a data directory with the users email 
   :INPUT
        folder, email 

.. moduleauthor:: Francesco De Carlo <decarlof@gmail.com>

""" 

import os
import sys, getopt
import ConfigParser
from validate_email import validate_email


# see README.txt to set a globus personal shared folder
cf = ConfigParser.ConfigParser()
cf.read('globus.ini')
globus_address = cf.get('settings', 'cli_address')
globus_user = cf.get('settings', 'cli_user')

local_server = cf.get('globus connect personal', 'server') 
local_share = cf.get('globus connect personal', 'share') 
local_shared_folder = cf.get('globus connect personal', 'shared_directory')  

remote_server = cf.get('globus remote server', 'server') 
remote_share = cf.get('globus remote server', 'share') 
remote_shared_folder = cf.get('globus remote server', 'shared_directory')  

def main(argv):
    input_folder = ''
    input_email = ''

    try:
        opts, args = getopt.getopt(argv,"hf:e:",["ffolder=","eemail="])
    except getopt.GetoptError:
        print 'test.py -f <folder> -e <email>'
        #print 'test.py -i <inputfile> -o <outputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'globus_copy_share.py -f <folder> -e <email>'
            print 'copy data from globus connect personal ', local_server + local_share + os.sep + '<folder> to ' + remote_server + remote_share + os.sep + remote_shared_folder
            print 'share data from', remote_server + remote_share + os.sep + remote_shared_folder + "<folder>", ' with ' + "<email>"

            sys.exit()
        elif opt in ("-f", "--ffolder"):
            input_folder = arg
        elif opt in ("-e", "--eemail"):
            input_email = arg
    
    input_folder = os.path.normpath(input_folder) + os.sep # will add the trailing slash if it's not already there.

    globus_scp = "scp -r " + local_server + local_share + ":" + os.sep + input_folder + " " + remote_server + remote_share + remote_shared_folder
    globus_add = "acl-add " + remote_server + remote_share + os.sep + remote_shared_folder + input_folder + " --perm r --email " + input_email

    if validate_email(input_email) and os.path.isdir(local_shared_folder + input_folder):
        cmd_1 = "ssh " + globus_user + globus_address + " " + globus_scp
        cmd_2 = "ssh " + globus_user + globus_address + " " + globus_add
        print cmd_1
        #os.system(cmd1)
        print "Done data trasfer to: ", remote_server
        #os.system(cmd2)
        print cmd_2
        print "Download link sent to: ", input_email
    else:
        print "ERROR: "
        if not validate_email(input_email):
            print "email is not valid ..."
        else:
            print local_shared_folder + input_folder, "does not exists on the local server", 

    
if __name__ == "__main__":
    main(sys.argv[1:])

