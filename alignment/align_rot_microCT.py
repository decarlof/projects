import numpy as np
import matplotlib.pyplot as plt
from pyhdf import SD

from imreg import translation, similarity
from skimage import data, io, filter
from skimage.feature import match_template
from skimage.measure import structural_similarity
from skimage import transform as tf

def tiff(file_name, dtype='uint16'):
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
    im = Image.open(file_name)
    out = np.fromstring(im.tostring(), dtype).reshape(tuple(list(im.size[::-1])))
    #im.close()

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

#    image = data.coins() # or any NumPy array!
#    edges = filter.sobel(image)
#    io.imshow(edges)
#    io.show()

    image_file_name_0 = '/local/decarlo/projects/alignment_data/0_180/hdf4/tilt_020_020_0001.hdf'
    image_file_name_180 = '/local/decarlo/projects/alignment_data/0_180/hdf4/tilt_020_020_0002.hdf'
    image_file_name_white = '/local/decarlo/projects/alignment_data/0_180/hdf4/tilt_020_020_0003.hdf'

    image_0 = read_hdf4(image_file_name_0, 'data')
    image_180 = read_hdf4(image_file_name_180, 'data')
    image_white = read_hdf4(image_file_name_white, 'data')

    image_0 = normalize (image_0, image_white)
    image_180 = normalize (image_180, image_white)
    
    plt.imshow(image_0+image_180, cmap=plt.cm.hot)
    plt.colorbar()
    plt.show()

    image_180 = np.fliplr(image_180)

    tform = tf.estimate_transform('similarity', image_0, image_180)

    a, grad = structural_similarity(image_0, image_180, gradient=True)
    print a
    print "grad shape", grad.shape

#    print grad
    plt.imshow(grad, cmap=plt.cm.hot)
    plt.colorbar()
    plt.show()

    result = match_template(image_0, image_180)
    print result.shape
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]
    print x, y

    im2, scale, angle, t = similarity(image_0, image_180)
    print "Scale: ", scale, "Angle: ", angle, "Transforamtion Matrix: ", t

    rot_axis_shift_x = -t[0]/2.0
    rot_axis_tilt = -t[1]/1.0
    
    print "Rotation Axis Shift (x, y):", "(", rot_axis_shift_x, ",", rot_axis_tilt,")"


if __name__ == "__main__":
    main()
