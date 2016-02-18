import sys
import epics
import h5py
import traceback

area_detector_prefix = '32idcPG3:'
ioc_prefix = '32idcTXM:'

hdf_writer_full_filename_pv_str = area_detector_prefix + 'HDF1:FullFileName_RBV'
theta_pv_str = ioc_prefix + 'recPV:PV1'
theta_cnt_pv_str = ioc_prefix + 'recPV:PV1_nUse'

def get_full_file_path():
	hdf_w_pv = epics.PV(hdf_writer_full_filename_pv_str)
	return hdf_w_pv.get(as_string=True)

def get_theta_arr():
	theta_arr = []
	theta_cnt_pv = epics.PV(theta_cnt_pv_str)
	theta_pv = epics.PV(theta_pv_str)
	theta_cnt = theta_cnt_pv.get()
	if theta_cnt > 0:
		print 'theta_cnt ', theta_cnt
		theta_arr = theta_pv.get(count=theta_cnt)
	return theta_arr

def add_theta_to_hdf(full_path, theta_arr):
	try:
		hdf_f = h5py.File(full_path)
		theta_ds = hdf_f.create_dataset('/exchange/theta',
		(len(theta_arr),))
		theta_ds[:] = theta_arr[:]
		hdf_f.close()
	except:
		traceback.print_exc(file=sys.stdout)
	
def main():
	full_path = get_full_file_path()
	print 'File Name:', full_path

	theta_arr = get_theta_arr()
	print 'theta_arr:', theta_arr
	#add_theta_to_hdf(full_path, theta_arr)

if __name__ == '__main__':
	main()
