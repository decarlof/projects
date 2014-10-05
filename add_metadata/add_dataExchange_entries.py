# -*- coding: utf-8 -*-
"""
.. module:: add_dataExchange_entries.py
   :platform: Unix
   :synopsis: add Data Exchange entries to an HDF5 file
   :INPUT
      Data Exchange file name 

.. moduleauthor:: Francesco De Carlo <decarlof@gmail.com>


""" 

from data_exchange import DataExchangeFile, DataExchangeEntry
import os

def main():
    #****************************************************************************
    hdf5_file_name = '/local/dataraid/databank/dataExchange/tmp/test_01.h5'
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
        f.add_entry( DataExchangeEntry.instrument(name={'value': 'APS 2-BM'}) )

        f.add_entry( DataExchangeEntry.source(name={'value': 'Advanced Photon Source'},
                                            date_time={'value': "2012-07-31T21:15:23+0600"},
                                            beamline={'value': "2-BM"},
                                            current={'value': 101.199, 'units': 'mA', 'dataset_opts': {'dtype': 'd'}},
                                            energy={'value': 7.0, 'units':'GeV', 'dataset_opts': {'dtype': 'd'}},
                                            mode={'value':'TOPUP'}
                                            )
        )
        # Create HDF5 subgroup
        # /measurement/instrument/attenuator
        f.add_entry( DataExchangeEntry.attenuator(thickness={'value': 1e-3, 'units': 'm', 'dataset_opts': {'dtype': 'd'}},
                                                type={'value': 'Al'}
                                                )
            )

        # Create HDF5 subgroup
        # Create HDF5 subgroup
        # /measurement/instrument/monochromator
        f.add_entry( DataExchangeEntry.monochromator(type={'value': 'Multilayer'},
                                                    energy={'value': 19.26, 'units': 'keV', 'dataset_opts': {'dtype': 'd'}},
                                                    energy_error={'value': 1e-3, 'units': 'keV', 'dataset_opts': {'dtype': 'd'}},
                                                    mono_stripe={'value': 'Ru/C'},
                                                    )
            )


        # Create HDF5 subgroup
        # /measurement/experimenter
        f.add_entry( DataExchangeEntry.experimenter(name={'value':"Jane Waruntorn"},
                                                    role={'value':"Project PI"},
                                                    affiliation={'value':"University of California"},
                                                    facility_user_id={'value':"64924"},

                        )
            )

        f.add_entry(DataExchangeEntry.objective(manufacturer={'value':'Zeiss'},
                                                model={'value':'Plan-NEOFLUAR 1004-072'},
                                                magnification={'value':5, 'dataset_opts': {'dtype': 'd'}},
                                                numerical_aperture={'value':0.5, 'dataset_opts': {'dtype': 'd'}},
                                            )
            )

        f.add_entry(DataExchangeEntry.scintillator(manufacturer={'value':'Crytur'},
                                                    serial_number={'value':'12'},
                                                    name={'value':'LuAg '},
                                                    type={'value':'LuAg'},
                                                    scintillating_thickness={'value':50e-6, 'dataset_opts': {'dtype': 'd'}},
                                                    substrate_thickness={'value':50e-6, 'dataset_opts': {'dtype': 'd'}},
                )
            )

        # Create HDF5 subgroup
        # /measurement/experiment
        f.add_entry( DataExchangeEntry.experiment( proposal={'value':"GUP-34353"},
                                                    activity={'value':"32-IDBC-2013-106491"},
                                                    safety={'value':"106491-49734"},
                    )
            )
        sample_name = "test_of_sample_name"

        if (sample_name != None):
            f.add_entry( DataExchangeEntry.sample(root='/measurement', name={'value':sample_name}, description={'value':'added by AddEntry.py'}), overwrite=True)

    f.close()
 
if __name__ == "__main__":
    main()

