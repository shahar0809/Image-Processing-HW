import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

MAX_GRAY_VAL = 255


def print_IDs():
    print("213991029+213996549\n")


def contrastEnhance(im, range):
    a = (np.max(range) - np.min(range)) / (im.max() - im.min())
    b = np.min(range) - im.min() * a
    nim = np.array(im * a + b)
    return np.array(nim, dtype=np.uint8), a, b


def showMapping(old_range, a, b):
    imMin = np.min(old_range)
    imMax = np.max(old_range)
    x = np.arange(imMin, imMax + 1, dtype=np.float)
    y = a * x + b
    plt.figure()
    plt.plot(x, y)
    plt.xlim([0, MAX_GRAY_VAL])
    plt.ylim([0, MAX_GRAY_VAL])
    plt.title('contrast enhance mapping')


def minkowski2Dist(im1, im2):
    d = 0

    hist1 = np.histogram(im1, bins=MAX_GRAY_VAL + 1, range=(0, MAX_GRAY_VAL))[0]
    hist2 = np.histogram(im2, bins=MAX_GRAY_VAL + 1, range=(0, MAX_GRAY_VAL))[0]

    for k in range(MAX_GRAY_VAL + 1):
        d += np.power(float(np.abs(hist1[k] - hist2[k])), 2)

    return np.power(d, 0.5)


def meanSqrDist(im1, im2):
    return np.sum(np.power(im2.astype(float).ravel() - im1.astype(float).ravel(), 2)) / len(im1.ravel())


def sliceMat(im):
    original_img = im.ravel()
    slices = np.full((np.size(im.ravel()), MAX_GRAY_VAL + 1), -1)

    for gray_val in range(0, MAX_GRAY_VAL + 1):
        slices[:, gray_val] = (original_img == gray_val).transpose()

    return slices


def SLTmap(im1, im2):
    TM = np.zeros(MAX_GRAY_VAL + 1)
    slice_mat = sliceMat(im1)

    for gray_val in range(0, MAX_GRAY_VAL + 1):
        bits_count = float(np.sum(slice_mat[:, gray_val]))
        if bits_count == 0:
            TM[gray_val] = 0
        else:
            TM[gray_val] = np.sum(slice_mat[:, gray_val] * im2.ravel()) / bits_count

    return mapImage(im1, TM), TM

def mapImage(im, tm):
    slice_mat = sliceMat(im)
    return np.matmul(slice_mat, tm.transpose()).reshape(im.shape)


def sltNegative(im):
    return mapImage(im, MAX_GRAY_VAL - np.arange(0, MAX_GRAY_VAL + 1))


def sltThreshold(im, thresh):
    tone_mapping = np.zeros(MAX_GRAY_VAL + 1)
    tone_mapping[thresh:] = MAX_GRAY_VAL
    return mapImage(im, tone_mapping)
