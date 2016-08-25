from epics import PV

prefix = '32idcPG3:'
area_detector = prefix + 'cam1:'
plugin = 'HDF1:'

n_white = 10
n_dark = 10
n_proj = 100

total_images = n_white + n_dark + n_proj

frame_0 = PV(area_detector + 'FrameType.ZRST')
frame_1 = PV(area_detector + 'FrameType.ONST')
frame_2 = PV(area_detector + 'FrameType.TWST')

frame_type = PV(area_detector + 'FrameType')
frame_capture = PV(area_detector + 'Acquire')
image_mode = PV(area_detector + 'ImageMode')
num_capture = PV(prefix + plugin + 'NumCapture')
capture = PV(prefix + plugin + 'Capture')

print "########################################"
print prefix + plugin + 'NumCapture'
print "########################################"

def init():
    frame_0.put('/exchange/mydata')
    frame_1.put('/exchange/mydata_dark')
    frame_2.put('/exchange/mydata_white')
    image_mode.put(0) # single 
    num_capture.put(total_images)
    capture.put(1)
    
def take_dark():
    #Image is saved in /exchange/data_dark
    frame_type.put(1)
    frame_capture.put(1)
    
def take_white():
    #Image is saved in /exchange/data_white
    frame_type.put(2)
    frame_capture.put(1)
    
def take_data():
    #Image is saved in /exchange/data
    frame_type.put(0)
    frame_capture.put(1)
    
	
def main():
	init()
	take_dark()
	#take_white()
	#take_data()
    
if __name__ == '__main__':
	main()
