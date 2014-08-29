import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import PIL.Image as Image
from scipy import misc

execfile('crosscorr_fun.py')

FileName_Im1 = '/local/decarlo/projects/alignment_data/nanoCT/0_180/Pin_0deg.tif'
FileName_Im2 = '/local/decarlo/projects/alignment_data/nanoCT/0_180/Pin_180deg.tif'

Im1 = misc.imread(FileName_Im1)
Im2 = misc.imread(FileName_Im2)
Im2 = np.fliplr(Im2)

amax = np.argmax(phase_corr(Im1,Im2))
shift = np.unravel_index(amax, Im1.shape)

print amax, shift

plt.subplot(1,2,1)
plt.imshow(Im1, cmap=plt.cm.hot)
plt.title('0 deg image'), plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(Im2, cmap=plt.cm.hot)
plt.title('180 deg image flipped left - right'), plt.colorbar()
plt.show()

