# -*- coding: utf-8 -*-
"""
.. module:: data_management.py
   :platform: Unix
   :synopsis:   Finds users running at specific date, 
                Creates top data directory 
                Share top data directory with the users listed in the proposal 
   :INPUT
      Date of the experiments 

.. moduleauthor:: Francesco De Carlo <decarlof@gmail.com>

This module is largely John Hammonds work to which I'll be adding
some scripts as needed.
""" 

import os

from subprocess import call


call(["ssh", "usr32idc@cli.globusonline.org"])
