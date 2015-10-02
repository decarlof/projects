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

# see README.txt to set a globus personal shared folder
globus_user = 'usr32idc'
globus_share_folder = "/local/dataraid/"
globus_share = "#dataraid"
globus_ssh = 'ssh ' + globus_user + '@cli.globusonline.org'

def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hf:e:",["ffolder=","eemail="])
    except getopt.GetoptError:
        print 'test.py -f <folder> -e <email>'
        #print 'test.py -i <inputfile> -o <outputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'globus_share.py -f <folder> -e <email>'
            sys.exit()
        elif opt in ("-f", "--ffolder"):
            inputfolder = arg
        elif opt in ("-e", "--eemail"):
            inputemail = arg
    print 'Shared Folder is: ', inputfolder
    print 'User e-mail is: ', inputemail
    
    #folder = "dm" + os.sep
    #email = "decarlo@aps.anl.gov"
    
    globus_add = "acl-add " + globus_user + globus_share + os.sep + inputfolder  + " --perm r --email " + inputemail

    cmd = globus_ssh + " " + globus_add
    print cmd
    #os.system(cmd)
    print "Download link sent to: ", inputemail


if __name__ == "__main__":
    main(sys.argv[1:])

