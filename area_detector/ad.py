# make sure that the hdf plugin layout xml file contains the following layout:

#  <global name="detector_data_destination" ndattribute="SaveDest"></global>
#    <group name="exchange">
#      <dataset name="data" source="detector">
#        <attribute name="description" source="constant" value="ImageData" type="string"></attribute>
#        <attribute name="axes" source="constant" value="theta:y:x" type="string"></attribute>
#        <attribute name="units" source="constant" value="counts" type="string"></attribute>
#      </dataset>
#      <dataset name="data_white" source="detector">
#        <attribute name="description" source="constant" value="WhiteData" type="string"></attribute>
#        <attribute name="axes" source="constant" value="theta:y:x" type="string"></attribute>
#        <attribute name="units" source="constant" value="counts" type="string"></attribute>
#      </dataset>
#      <dataset name="data_dark" source="detector">
#        <attribute name="description" source="constant" value="DarkData" type="string"></attribute>
#        <attribute name="axes" source="constant" value="theta:y:x" type="string"></attribute>
#        <attribute name="units" source="constant" value="counts" type="string"></attribute>
#      </dataset>
#

from epics import PV
import time 

prefix = '32idcPG3:'
area_detector = prefix + 'cam1:'
plugin = 'HDF1:'

n_white = 10
n_dark = 11
n_proj = 100

total_images = n_white + n_dark + n_proj

frame_0 = PV(area_detector + 'FrameType.ZRST')
frame_1 = PV(area_detector + 'FrameType.ONST')
frame_2 = PV(area_detector + 'FrameType.TWST')

frame_type = PV(area_detector + 'FrameType')
frame_acquire = PV(area_detector + 'Acquire')
image_mode = PV(area_detector + 'ImageMode')
num_capture = PV(prefix + plugin + 'NumCapture')
capture = PV(prefix + plugin + 'Capture')

#wait on a pv to be a value until max_timeout (default forever)
def wait_pv(pv, wait_val, max_timeout_sec=-1):
    print 'wait_pv(', pv.pvname, wait_val, max_timeout_sec, ')'
    #delay for pv to change
    time.sleep(.05)
    startTime = time.time()
    while(True):
        pv_val = pv.get()
        if (pv_val != wait_val):
            if max_timeout_sec > -1:
                curTime = time.time()
                diffTime = curTime - startTime
                if diffTime >= max_timeout_sec:
                    return False
            time.sleep(.1)
        else:
            return True

def init():
    frame_0.put('/exchange/data')
    frame_1.put('/exchange/data_dark')
    frame_2.put('/exchange/data_white')
    image_mode.put(0) # single 
    num_capture.put(total_images)
    capture.put(1)
    wait_pv(capture, 1)

def take_dark():
    #Image is saved in /exchange/data_dark
    frame_type.put(1, wait=True)
    frame_acquire.put(1)
    #wait for capture to finish
    wait_pv(frame_acquire, 0)

def take_white():
    #Image is saved in /exchange/data_white
    frame_type.put(2, wait=True)
    frame_acquire.put(1)
    #wait for capture to finish
    wait_pv(frame_acquire, 0)
 
def take_data():
    #Image is saved in /exchange/data
    frame_type.put(0, wait=True)
    frame_acquire.put(1)
    #wait for capture to finish
    wait_pv(frame_acquire, 0)

def main():
	init()
	for num in range(0, n_dark):
	    take_dark()
	for num in range(0, n_white):
	    take_white()
	for num in range(0, n_proj):
	    take_data()
    
if __name__ == '__main__':
	main()
