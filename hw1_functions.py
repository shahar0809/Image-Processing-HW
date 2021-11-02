import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

MAX_GRAY_VAL = 255


def print_IDs():
    # print("123456789")
    print("123456789+987654321\n")


def contrastEnhance(im, range):
    min_gray, max_gray = np.min(im), np.max(im)

    a = (np.max(range) - np.min(range)) / (max_gray - min_gray)
    b = np.max(range) - a * max_gray

    nim = a * im + b
    return nim, a, b


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

    hist1 = np.histogram(im1, bins=MAX_GRAY_VAL + 1, range=(0, MAX_GRAY_VAL), density=True)[0]
    hist2 = np.histogram(im2, bins=MAX_GRAY_VAL + 1, range=(0, MAX_GRAY_VAL), density=True)[0]

    for k in range(MAX_GRAY_VAL + 1):
        d += np.power(float(np.abs(hist1[k] - hist2[k])), 2)

    return np.power(d, 0.5)


def meanSqrDist(im1, im2):
    return np.sum(np.power(im2.ravel() - im1.ravel(), 2)) / len(im1.ravel())


def sliceMat(im):
    original_img = im.ravel()
    slices = np.full((np.size(im.ravel()), MAX_GRAY_VAL + 1), -1)

    # Special case for 0
    slices[:, 0] = original_img
    slices[:, 0][slices[:, 0] == 0] = -1
    slices[:, 0][slices[:, 0] != -1] = 0
    slices[:, 0][slices[:, 0] == -1] = 1

    for gray_val in range(1, MAX_GRAY_VAL + 1):
        slices[:, gray_val] = original_img
        slices[:, gray_val][slices[:, gray_val] != gray_val] = 0
        slices[:, gray_val][slices[:, gray_val] == gray_val] = 1

    return slices


def SLTmap(im1, im2):
    TM = np.zeros(MAX_GRAY_VAL + 1)
    slice_mat = sliceMat(im1)

    for gray_val in range(0, MAX_GRAY_VAL + 1):
        bits_count = float(np.sum(slice_mat[:, gray_val]))
        if bits_count == 0:
            TM[gray_val] = 0
        else:
            TM[gray_val] = np.matmul(slice_mat[:, gray_val], im2.ravel()) / bits_count

    return mapImage(im1, TM), TM


def mapImage(im, tm):
    slice_mat = sliceMat(im)
    new_slice_mat = np.zeros(slice_mat.shape)

    for gray_val in range(0, MAX_GRAY_VAL + 1):
        new_slice_mat[:, int(tm[gray_val])] = slice_mat[:, gray_val]

    return np.matmul(new_slice_mat, np.arange(MAX_GRAY_VAL + 1)).reshape(im.shape)


def sltNegative(im):
    return mapImage(im, MAX_GRAY_VAL - np.arange(MAX_GRAY_VAL + 1))


def sltThreshold(im, thresh):
    tone_mapping = np.arange(MAX_GRAY_VAL + 1)
    tone_mapping[tone_mapping <= thresh] = 0
    tone_mapping[tone_mapping > thresh] = MAX_GRAY_VAL
    return mapImage(im, tone_mapping)
