# -*- coding: utf-8 -*-
"""
.. module:: globus_share.py
   :platform: Unix
   :synopsis:   
                Share via Globus a data directory with the users listed in the proposal 
   :INPUT
      Date of the experiments 

.. moduleauthor:: Francesco De Carlo <decarlof@gmail.com>

""" 

import os

globus_user = 'decarlo'
globus_share_folder = "/local/dataraid/"
globus_ssh = 'ssh ' + globus_user + '@cli.globusonline.org'

      
if __name__ == "__main__":

    folder = "dm"
    email = "decarlo@aps.anl.gov"
    
    globus_add = "acl-add " + globus_user + "#databank" + os.sep + folder + os.sep + " --perm r --email " + email

    cmd = globus_ssh + " " + globus_add
    print cmd
    #os.system(cmd)
    print "\n\n======================================"
    print "Check your email to download the data "
    print "======================================"

