import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from scipy import misc

from pyhdf import SD
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
#    im = Image.open(file_name)
#    out = np.fromstring(im.tostring(), dtype).reshape(tuple(list(im.size[::-1])))
##    im.close()
    out = misc.imread(file_name)
    print "TYPE", type(out)
    print "SHAPE", out.shape, "DTYPE", out.dtype

    return out


def read_hdf4(file_name, array_name):
    """
    Read 2-D tomographic data from hdf4 file.
    Opens ``file_name`` and reads the contents
    of the array specified by ``array_name`` in
    the specified group of the HDF file.
    Parameters
    ----------
    file_name : str
    Input HDF file.
    array_name : str
    Name of the array to be read at exchange group.
    x_start, x_end, x_step : scalar, optional
    Values of the start, end and step of the
    slicing for the whole ndarray.
    y_start, y_end, y_step : scalar, optional
    Values of the start, end and step of the
    slicing for the whole ndarray.
    Returns
    -------
    out : ndarray
    Returns the data as a matrix.
    """
    # Read data from file.
    f = SD.SD(file_name)
    sds = f.select(array_name)
    hdfdata = sds.get()
    sds.endaccess()
    f.end()

    return hdfdata

def normalize(image, image_white):

    c = image / ((image_white.astype('float') + 1) / 65535)
    d = c * (c < 65535) + 65535 * np.ones(np.shape(c)) * (c > 65535)
    image = d.astype('uint16')

    return image


def main():

    image_file_name_0 = '/local/decarlo/projects/alignment_data/nanoCT/0_180/Pin_0deg.tif'
    image_file_name_180 = '/local/decarlo/projects/alignment_data/nanoCT/0_180/Pin_180deg.tif'

    image_0 = read_tiff(image_file_name_0)
    image_180 = read_tiff(image_file_name_180)

#    print image_0.dtype

    plt.imshow(image_0-image_180, cmap=plt.cm.hot)
#    plt.imshow(image_0+image_180, cmap=plt.cm.hot)
    plt.colorbar()
    plt.show()

    image_180 = np.fliplr(image_180)
    print image_180.shape

    im2, scale, angle, t = similarity(image_0, image_180)
    print "Scale: ", scale, "Angle: ", angle, "Transforamtion Matrix: ", t

    rot_axis_shift_x = -t[0]/2.0
    rot_axis_tilt = -t[1]/1.0
    
    print "Rotation Axis Shift (x, y):", "(", rot_axis_shift_x, ",", rot_axis_tilt,")"


if __name__ == "__main__":
    main()
