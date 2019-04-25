from __future__ import division

import cv2
import numpy
import scipy.misc

import matplotlib.pyplot as plt
from scipy import ndimage

import odemis.dataio.hdf5 as h5
from odemis.acq import align

WSize = 15  # Size of local window (only some operators)

INDICES = [
    0, 1,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3,
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4,
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 5,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 6,
    60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 7,
    70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 8,
    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 9,
    90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
]


def measure_vola_focus(image):
    I1 = numpy.copy(image)
    I1[1:-1, :] = image[2:, :]
    I2 = numpy.copy(image)
    I2[1:-2, :] = image[3:, :]
    image = numpy.multiply(image, I1 - I2)
    return numpy.mean(image)


def measure_bren_focus(image):
    m, n = image.shape
    im_shift_hor = numpy.zeros((m, n))
    im_shift_vert = numpy.zeros((m, n))
    im_shift_vert[1:m - 2, :] = image[3:, :] - image[1:- 2, :]
    im_shift_hor[:, 1:n - 2] = image[:, 3:] - image[:, 1:-2]
    focus = numpy.maximum(im_shift_hor, im_shift_vert)
    focus = numpy.multiply(focus, focus)
    return numpy.mean(focus)


def measure_tenv_focus(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_image = sobelx ** 2 + sobely ** 2
    return sobel_image.var()


def measure_teng_focus(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_image = sobelx ** 2 + sobely ** 2
    return numpy.mean(sobel_image)


def measure_helm_focus(image):
    kernel = numpy.ones((WSize, WSize), numpy.float32) / (WSize * WSize)
    U = cv2.filter2D(image, -1, kernel)
    R1 = numpy.multiply(U, 1 / image)
    R1[image == 0] = 1
    index = (U > image)
    FM = 1 / R1
    FM[index] = R1[index]
    return numpy.mean(FM)


def measure_grae_focus(image):
    Ix = numpy.copy(image)
    Iy = numpy.copy(image)
    Iy[1:, :] = numpy.diff(image, 1, axis=0)
    Ix[:, 1:] = numpy.diff(image, 1, axis=1)
    return numpy.mean(Ix ** 2 + Iy ** 2)


def measure_gras_focus(image):
    Ix = numpy.diff(image, axis=1)
    return numpy.mean(Ix ** 2)


def measure_bren_focus(image):
    M, N = image.shape
    DH = numpy.zeros((M, N))
    DV = numpy.zeros((M, N))
    DV[1:M - 2, :] = image[3:, :] - image[1:- 2, :]
    DH[:, 1:N - 2] = image[:, 3:] - image[:, 1:-2]
    FM = numpy.maximum(DH, DV)
    FM = numpy.multiply(FM, FM)
    return numpy.mean(FM)


def measure_sfrq_focus(image):
    Ix = numpy.copy(image)
    Iy = numpy.copy(image)
    Ix[:, 1:] = numpy.diff(image, axis=1)
    Iy[1:, :] = numpy.diff(image, axis=0)
    return numpy.mean(numpy.sqrt(Iy ** 2 + Ix ** 2))


def measure_opt_focus(image):
    """
    Given an image, focus measure is calculated using the variance of Laplacian
    of the raw data.
    image (model.DataArray): Optical image
    returns (float): The focus level of the optical image (higher is better)
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()


def measure_sem_focus(image):
    """
    Given an image, focus measure is calculated using the standard deviation of
    the raw data.
    image (model.DataArray): SEM image
    returns (float): The focus level of the SEM image (higher is better)
    """
    return ndimage.standard_deviation(image)


def test_focus_method():
    # 5.14330601692
    images = h5.read_data("/home/pals/Documents/mb autofocus images/autofocus_testimages.h5")
    measure = measure_tenv_focus
    focus_levels = []
    import time
    the_time = time.time()
    for idx, image in zip(INDICES, images):
        image = scipy.misc.imresize(image, (int(image.shape[0] / 2), int(image.shape[1] / 2)))
        focus = measure(image)
        focus_levels.append(focus)
        print(idx)
    print(time.time() - the_time)
    sorted_focus = [x for y, x in sorted(zip(INDICES, focus_levels))]
    sorted_focus_blue = [k / max(sorted_focus) for k in sorted_focus]
    sorted_focus_orange = [(k - min(sorted_focus)) / (max(sorted_focus) - min(sorted_focus)) for k
                           in sorted_focus]

    fig, axes = plt.subplots()

    axes.plot(sorted_focus_blue, color='blue')
    axes.plot(sorted_focus_orange, color='orange')
    axes.axvline(43, color='green', linestyle='--')
    axes.set_xlabel('Z position stage [um]')
    axes.set_ylabel('Focus level')
    axes.set_title(measure.__name__.split('_')[1].upper())

    # fig.savefig(
    #     "/home/pals/Documents/mb autofocus images/focus_levels_{}_proc.png".format(measure.__name__.split('_')[1]),
    #     dpi=300)
    plt.show()


def test_with_images():
    images = h5.read_data("/home/pals/Documents/mb autofocus images/autofocus_testimages.h5")
    measurements = [measure_vola_focus,
                    measure_bren_focus,
                    measure_tenv_focus,
                    measure_teng_focus,
                    measure_helm_focus,
                    measure_grae_focus,
                    measure_gras_focus,
                    # measure_sfrq_focus,
                    measure_sem_focus,
                    measure_opt_focus]
    focus_levels = {m.__name__.split('_')[1]: [] for m in measurements}

    for image in images:
        for measure in measurements:
            focus = measure(image)
            focus_levels[measure.__name__.split('_')[1]].append(focus)

    fig, axes = plt.subplots(3, 3, figsize=(10, 8))
    i = 0
    j = 0
    for measure in measurements:
        focus_level = focus_levels[measure.__name__.split('_')[1]]
        sorted_focus = [x for y, x in sorted(zip(INDICES, focus_level))]
        # normalized to 1: y/y_max
        normalized_focus_blue = [k / max(sorted_focus) for k in sorted_focus]
        # normalized between 0 and 1: (y - y_min) / (y_max - y_min)
        normalized_focus_orange = [(k - min(sorted_focus)) / (max(sorted_focus) - min(sorted_focus)) for k in
                                   sorted_focus]
        # All for plotting:
        if j == 2:
            i += 1
            j = 0
        else:
            j += 1
        if i == 3:
            i = 0
        axes[i, j].plot(normalized_focus_blue, color='blue')
        axes[i, j].plot(normalized_focus_orange, color='orange')
        axes[i, j].axvline(43, color='green', linestyle='--')
        axes[i, j].set_xlabel('Z position stage [um]')
        axes[i, j].set_ylabel('Focus level')
        axes[i, j].set_title(measure.__name__.split('_')[1].upper())
    plt.tight_layout()
    fig.savefig("/home/pals/Documents/mb autofocus images/focus_levels_norm_subplot1234.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    test_focus_method()
