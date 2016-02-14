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
import sys
import datetime
import process_variables_32ID as pv

if __name__ == "__main__":


    beamline = '32-ID-B,C'
    instrument_name = 'TXM'


# PV settings  
    print "date time:", pv.date_time.get()
    print "current:", pv.current.get()
    print "top_up_status:", pv.top_up_status.get()
    print "source_energy:", pv.source_energy.get()
    print "source_gap:", pv.source_gap.get()
    print "energy_dcm:", pv.energy_dcm.get()
    print "mirror_x:", pv.mirror_x.get()
    print "mirror_y:", pv.mirror_y.get()

    # User Status
    print "user_name:", pv.user_name.get()
    print "user_affiliation:", pv.user_affiliation.get()
    print "user_email:", pv.user_email.get()
    print "user_badge:", pv.user_badge.get()
    print "proposal_number:", pv.proposal_number.get()
    print "proposal_title:", pv.proposal_title.get()

    # Beam Monitor
    print "beam_monitor_x:", pv.beam_monitor_x.get()
    print "beam_monitor_y:", pv.beam_monitor_y.get()

    # Filter
    print "filter_x:", pv.filter_x.get()

    # Diffuser
    print "diffuser:", pv.diffuser.get()

    # Beam Stop
    print "beam_stop_x:", pv.beam_stop_x.get()
    print "beam_stop_y:", pv.beam_stop_y.get()

    # CRL
    print "crl_x:", pv.crl_x.get()
    print "crl_y:", pv.crl_y.get()
    print "crl_pitch:", pv.crl_pitch.get()
    print "crl_yaw:", pv.crl_yaw.get()

    # Condenser
    print "condenser_x:", pv.condenser_x.get()
    print "condenser_y:", pv.condenser_y.get()
    print "condenser_z:", pv.condenser_z.get()

    # Pin Hole
    print "pin_hole_x:", pv.pin_hole_x.get()
    print "pin_hole_y:", pv.pin_hole_y.get()
    print "pin_hole_z:", pv.pin_hole_z.get()

    # Sample
    print "sample_x:", pv.sample_x.get()
    print "sample_y:", pv.sample_y.get()
    print "sample_rotary:", pv.sample_rotary.get()

    print "sample_top_x:", pv.sample_top_x.get()
    print "sample_top_z:", pv.sample_top_z.get()

    # Zone Plate
    print "zone_plate_x:", pv.zone_plate_x.get()
    print "zone_plate_y:", pv.zone_plate_y.get()
    print "zone_plate_z:", pv.zone_plate_z.get()

    # 2nd Zone Plate
    print "zone_plate_2nd_x:", pv.zone_plate_2nd_x.get()
    print "zone_plate_2nd_y:", pv.zone_plate_2nd_y.get()
    print "zone_plate_2nd_z:", pv.zone_plate_2nd_z.get()

    # Bertrand Lens
    print "bertrand_x:", pv.bertrand_x.get()
    print "bertrand_y:", pv.bertrand_y.get()

    # Flight Tube
    print "flight_tube_z:", pv.flight_tube_z.get()

    # CCD camera
    print "ccd_camera_x:", pv.ccd_camera_x.get()
    print "ccd_camera_y:", pv.ccd_camera_y.get()
    print "ccd_camera_z:", pv.ccd_camera_z.get()
    print "ccd_rotation:", pv.ccd_rotation.get()
    print "ccd_objective:", pv.ccd_objective.get()


    # Beam Monitor
    print "beam_monitor_x_dial:", pv.beam_monitor_x_dial.get()
    print "beam_monitor_y_dial:", pv.beam_monitor_y_dial.get()

    # Filter
    print "filter_x_dial:", pv.filter_x_dial.get()

    # Diffuser
    print "diffuser_dial:", pv.diffuser_dial.get()

    # Beam Stop
    print "beam_stop_x_dial:", pv.beam_stop_x_dial.get()
    print "beam_stop_y_dial:", pv.beam_stop_y_dial.get()

    # CRL
    print "crl_x_dial:", pv.crl_x_dial.get()
    print "crl_y_dial:", pv.crl_y_dial.get()
    print "crl_pitch_dial:", pv.crl_pitch_dial.get()
    print "crl_yaw_dial:", pv.crl_yaw_dial.get()

    # Condenser
    print "condenser_x_dial:", pv.condenser_x_dial.get()
    print "condenser_y_dial:", pv.condenser_y_dial.get()
    print "condenser_z_dial:", pv.condenser_z_dial.get()

    # Pin Hole
    print "pin_hole_x_dial:", pv.pin_hole_x_dial.get()
    print "pin_hole_y_dial:", pv.pin_hole_y_dial.get()
    print "pin_hole_z_dial:", pv.pin_hole_z_dial.get()

    # Sample
    print "sample_x_dial:", pv.sample_x_dial.get()
    print "sample_y_dial:", pv.sample_y_dial.get()
    print "sample_rotary_dial:", pv.sample_rotary_dial.get()

    print "sample_top_x_dial:", pv.sample_top_x_dial.get()
    print "sample_top_z_dial:", pv.sample_top_z_dial.get()

    # Zone Plate
    print "zone_plate_x_dial:", pv.zone_plate_x_dial.get()
    print "zone_plate_y_dial:", pv.zone_plate_y_dial.get()
    print "zone_plate_z_dial:", pv.zone_plate_z_dial.get()

    # 2nd Zone Plate
    print "zone_plate_2nd_x_dial:", pv.zone_plate_2nd_x_dial.get()
    print "zone_plate_2nd_y_dial:", pv.zone_plate_2nd_y_dial.get()
    print "zone_plate_2nd_z_dial:", pv.zone_plate_2nd_z_dial.get()

    # Bertrand Lens
    print "bertrand_x_dial:", pv.bertrand_x_dial.get()
    print "bertrand_y_dial:", pv.bertrand_y_dial.get()

    # Flight Tube
    print "flight_tube_z_dial:", pv.flight_tube_z_dial.get()

    # CCD camera
    print "ccd_camera_x_dial:", pv.ccd_camera_x_dial.get()
    print "ccd_camera_y_dial:", pv.ccd_camera_y_dial.get()
    print "ccd_camera_z_dial:", pv.ccd_camera_z_dial.get()
    print "ccd_rotation_dial:", pv.ccd_rotation_dial.get()
    print "ccd_objective_dial:", pv.ccd_objective_dial.get()

