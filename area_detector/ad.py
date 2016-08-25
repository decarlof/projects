from epics import PV

area_detector = '32idcPG3:cam1:'

frame_0 = PV(area_detector + 'FrameType.ZRST')
frame_1 = PV(area_detector + 'FrameType.ONST')
frame_2 = PV(area_detector + 'FrameType.TWST')

frame_type = PV(area_detector + 'FrameType')
frame_capture = PV(area_detector + 'Acquire')

def init_tags():
    frame_0.put('/exchange/mydata')
    frame_1.put('/exchange/mydata_dark')
    frame_2.put('/exchange/mydata_white')


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
	init_tags()
	take_dark()
	#take_white()
	#take_data()
    
if __name__ == '__main__':
	main()
