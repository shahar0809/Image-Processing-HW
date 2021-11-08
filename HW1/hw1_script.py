import numpy as np

from hw1_functions import *

if __name__ == "__main__":

    # Read image from folder
    path_image = r'Images\darkimage.tif'
    dark_img = cv2.imread(path_image)
    dark_img_gray = cv2.cvtColor(dark_img, cv2.COLOR_BGR2GRAY)

    print("Start running script  ------------------------------------\n")
    print_IDs()

    print("a ------------------------------------\n")
    enhanced_img, a, b = contrastEnhance(dark_img_gray, np.arange(MAX_GRAY_VAL + 1))

    # Display images
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(dark_img)
    plt.title('original')

    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_img, cmap='gray', vmin=0, vmax=255)
    plt.title('enhanced contrast')

    # Print results from contrast enhancement
    print("a = {}, b = {}\n".format(a, b))

    # Display mapping
    showMapping([np.min(dark_img), np.max(dark_img)], a, b)

    print("b ------------------------------------\n")
    enhanced2_img, a, b = contrastEnhance(enhanced_img, [np.min(enhanced_img), np.max(enhanced_img)])
    # Print results from second contrast enhancement
    print("enhancing an already enhanced image")
    print("a = {}, b = {}\n".format(a, b))

    plt.figure()
    plt.imshow(enhanced2_img - enhanced_img, cmap='gray', vmin=0, vmax=255)
    plt.title("Difference between enhancement images")

    print("c ------------------------------------\n")
    minkowski_dist = minkowski2Dist(dark_img_gray, dark_img_gray)
    print("Minkowski dist between image and itself")
    print("d = {}\n".format(minkowski_dist))

    Max = np.max(dark_img_gray)
    Min = np.min(dark_img_gray)
    step = (Max - Min) / 20
    contrast = np.array([])
    dists = np.array([])
    for k in np.arange(1, 21):
        EIm, a, b = contrastEnhance(dark_img_gray, [Min, Min + k * step])
        n_contrast = np.max(EIm) - np.min(EIm)
        contrast = np.append(contrast, np.array(n_contrast))
        dists = np.append(dists, np.array(minkowski2Dist(dark_img_gray, EIm)))

    plt.figure()
    plt.plot(contrast, dists)
    plt.xlabel("contrast")
    plt.ylabel("distance")
    plt.title("Minkowski distance as function of contrast")

    print("d ------------------------------------\n")
    path_image = r'Images\fruit.tif'
    fruit_img = cv2.imread(path_image)
    fruit_img_gray = cv2.cvtColor(fruit_img, cv2.COLOR_BGR2GRAY)

    # Extract min and max values of image
    max_gray_val, min_gray_val = np.max(fruit_img_gray), np.min(fruit_img_gray)

    d = np.dot(sliceMat(fruit_img_gray), np.arange(MAX_GRAY_VAL + 1)).reshape(fruit_img_gray.shape) - fruit_img_gray
    f = np.equal(d, np.zeros((np.shape(d)[0], np.shape(d)[1]))).all()
    print("Is sliceMat(im) * [0:255] == im? {}\n".format(f))

    print("e ------------------------------------\n")
    enhanced_fruit_img, a, b = contrastEnhance(fruit_img_gray, [0, MAX_GRAY_VAL])
    tone_mapped_img = SLTmap(fruit_img_gray, enhanced_fruit_img)[0]
    np_diff = tone_mapped_img - enhanced_fruit_img
    d = np.sum(np_diff.astype(float))
    print("sum of diff between image and slices*[0..255] = {}\n".format(d))

    # then display
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(fruit_img)
    plt.title("original image")
    plt.subplot(1, 2, 2)
    plt.imshow(tone_mapped_img, cmap='gray', vmin=0, vmax=255)
    plt.title("tone mapped")

    print("f ------------------------------------\n")
    negative_im = sltNegative(dark_img_gray)
    plt.figure()
    plt.imshow(negative_im, cmap='gray', vmin=0, vmax=255)
    plt.title("negative image using SLT")

    print("g ------------------------------------\n")
    thresh = 120
    lena = cv2.imread(r"Images\\RealLena.tif")
    lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    thresh_im = sltThreshold(lena_gray, thresh)

    plt.figure()
    plt.imshow(thresh_im, cmap='gray', vmin=0, vmax=255)
    plt.title("thresh image using SLT")

    print("h ------------------------------------\n")
    im1 = lena_gray
    im2 = dark_img_gray
    SLTim, TM = SLTmap(im1, im2)

    # then print
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(im1, cmap='gray', vmin=0, vmax=255)
    plt.title("original image")
    plt.subplot(1, 3, 2)
    plt.imshow(SLTim, cmap='gray', vmin=0, vmax=255)
    plt.title("tone mapped")
    plt.subplot(1, 3, 3)
    plt.imshow(im2, cmap='gray', vmin=0, vmax=255)
    plt.title("tone mapped")

    d1 = meanSqrDist(im1, im2)  # mean sqr dist between im1 and im2
    d2 = meanSqrDist(SLTim, im2)  # mean sqr dist between mapped image and im2
    print("mean sqr dist between im1 and im2 = {}\n".format(d1))
    print("mean sqr dist between mapped image and im2 = {}\n".format(d2))

    print("i ------------------------------------\n")
    # prove comutationally
    tm_img, tone_mapping = SLTmap(dark_img_gray, lena_gray)
    tm_img2, tone_mapping2 = SLTmap(lena_gray, dark_img_gray)
    d = True if np.array_equal(tm_img, tm_img2) else False
    print("Is SLTmap symmetric? {}".format(d))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(tm_img, cmap='gray', vmin=0, vmax=255)
    plt.title("tone mapping img 1")
    plt.subplot(1, 2, 2)
    plt.imshow(tm_img2, cmap='gray', vmin=0, vmax=255)
    plt.title("tone mapping img 2")

    plt.show()
