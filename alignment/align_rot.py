import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from scipy import misc

from imreg import translation, similarity

def read_tiff(file_name, dtype='uint16'):
    """
    Read TIFF files.
    Parameters
    ----------
    file_name : str
        Name of the input TIFF file.
    dtype : str, optional   
        Corresponding numpy data type of the TIFF file.
    Returns
    -------
    out : ndarray
    Output 2-D matrix as numpy array.
    """
#    PIL.Image fails on little endian tiff ...
#    im = Image.open(file_name)
#    out = np.fromstring(im.tostring(), dtype).reshape(tuple(list(im.size[::-1])))
##    im.close()

    out = misc.imread(file_name)

    return out

def normalize(image, image_white):

    c = image / ((image_white.astype('float') + 1) / 65535)
    d = c * (c < 65535) + 65535 * np.ones(np.shape(c)) * (c > 65535)
    image = d.astype('uint16')

    return image


def main():

    image_file_name_0 = '/local/decarlo/projects/data/nanoCT/0_180/Pin_0deg.tif'
    image_file_name_180 = '/local/decarlo/projects/data/nanoCT/0_180/Pin_180deg.tif'

    image_0 = read_tiff(image_file_name_0)
    image_180 = read_tiff(image_file_name_180)

    plt.imshow(image_0+image_180, cmap=plt.cm.hot)
    plt.colorbar()
    plt.show()

    image_180 = np.fliplr(image_180)

    plt.imshow(image_0+image_180, cmap=plt.cm.hot)
    plt.colorbar()
    plt.show()

    im2, scale, angle, t = similarity(image_0, image_180)
    print "Scale: ", scale, "Angle: ", angle, "Transforamtion Matrix: ", t

    rot_axis_shift_x = -t[1]/1.0
    rot_axis_shift_y = -t[0]/2.0
    
    print "Rotation Axis Shift (x, y):", "(", rot_axis_shift_x, ",", rot_axis_shift_y,")"
    

if __name__ == "__main__":
    main()
