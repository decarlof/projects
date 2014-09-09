import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

from scipy.misc import lena
from scipy import ndimage

from imreg import model, register, sampler, metric

def main():

    image = ndimage.zoom(lena(), 0.50)

    plt.imshow(image, cmap=plt.cm.hot)
    plt.colorbar()
    plt.show()

    r, c = image.shape
    template = ndimage.rotate(image, 20, reshape=False)[(r/2)-20:(r/2)+20, (c/2)-20:(c/2)+20]

    plt.imshow(template, cmap=plt.cm.hot)
    plt.colorbar()
    plt.show()

    fix, ax1 = plt.subplots(1, 1, figsize=(8, 8))
    ax1.imshow(image, cmap='gray')
    _ = ax1.set_title('image')

    fix, ax2 = plt.subplots(1, 1, figsize=(2, 2))
    ax2.imshow(template, cmap='gray')
    _ = ax2.set_title('template')

    template = register.RegisterData(template)
    image = register.RegisterData(image)

    sampler.nearest(image.data, template.coords.tensor)

    tform = model.Affine()
    registrator = register.Register()

    # First perform a shift
    p = np.array([100, 100])
    step, search = registrator.register(image, template, model.Shift(), sampler=sampler.bilinear, p=p)

    # Second perform an affine shift
    p = np.array([0,0,0,0,step.p[0],step.p[1]])
    step, search = registrator.register(image, template, tform, sampler=sampler.bilinear, p=p)

    fix, ax1 = plt.subplots(1, 1, figsize=(8, 8))
    ax1.imshow(image.data, cmap='gray')
    ax1.axis('tight')

    coords = tform(step.p, template.coords)

    plt.plot(coords.xy[0], coords.xy[1], '.r')
    plt.plot(coords.xy[0], coords.xy[1], '.r')

    _ = ax1.set_title('image')
    plt.plot([x.error for x in search])
    plt.show()


if __name__ == "__main__":
    main()