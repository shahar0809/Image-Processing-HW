import matplotlib.pyplot as plt

from hw1_functions import *

if __name__ == "__main__":
    # Read image from folder
    path_image = r'Images\darkimage.tif'
    dark_img = cv2.imread(path_image)
    dark_img_gray = cv2.cvtColor(dark_img, cv2.COLOR_BGR2GRAY)

    print("Start running script  ------------------------------------\n")
    print_IDs()

    print("a ------------------------------------\n")
    enhanced_img, a, b = contrastEnhance(dark_img_gray, [0, MAX_GRAY_VAL])

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
    enhanced2_img, a, b = contrastEnhance(dark_img_gray, [0, MAX_GRAY_VAL])
    # Print results from second contrast enhancement
    print("enhancing an already enhanced image\n")
    print("a = {}, b = {}\n".format(a, b))

    # TODO: display the difference between the two image (Do not simply display both images)
    plt.figure()
    plt.imshow(enhanced2_img - enhanced_img, cmap='gray', vmin=0, vmax=255)
    plt.title("Difference between enhancement images")
    plt.show()

    print("c ------------------------------------\n")
    minkowski_dist = minkowski2Dist(dark_img_gray, dark_img_gray)
    print("Minkowski dist between image and itself\n")
    print("d = {}\n".format(minkowski_dist))

    path_image = r'Images\fruit.tif'
    fruit_img = cv2.imread(path_image)
    fruit_img_gray = cv2.cvtColor(dark_img, cv2.COLOR_BGR2GRAY)

    # Extract min and max values of image
    max_gray_val, min_gray_val = np.max(fruit_img_gray), np.min(fruit_img_gray)

    contrasts = []
    dists = []

    k = max_gray_val / 20
    for step in range(0, 20):
        curr_max_val = max_gray_val + k * step
        contrasts += [curr_max_val - min_gray_val]

        enhanced_img = contrastEnhance(fruit_img_gray, [min_gray_val, curr_max_val])[0]
        dists += [minkowski2Dist(enhanced_img, fruit_img_gray)]

    plt.figure()
    plt.plot(contrasts, dists)
    plt.xlabel("contrast")
    plt.ylabel("distance")
    plt.title("Minkowski distance as function of contrast")
    plt.show()

    print("d ------------------------------------\n")

    d = np.dot(sliceMat(fruit_img_gray), np.arange(MAX_GRAY_VAL + 1)).reshape(fruit_img_gray.shape) - fruit_img_gray
    print("{}".format(d))

    # print("e ------------------------------------\n")
    #
    # d = # computationally compare
    # print("sum of diff between image and slices*[0..255] = {}".format(d))
    #
    # # then display
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(dark_img)
    # plt.title("original image")
    # plt.subplot(1, 2, 2)
    # plt.imshow(TMim, cmap='gray', vmin=0, vmax=255)
    # plt.title("tone mapped")
    #
    #
    #
    # print("f ------------------------------------\n")
    # negative_im = sltNegative(dark_img_gray)
    # plt.figure()
    # plt.imshow(negative_im, cmap='gray', vmin=0, vmax=255)
    # plt.title("negative image using SLT")
    #
    #
    #
    # print("g ------------------------------------\n")
    # thresh = 120 # play with it to see changes
    # lena = cv2.imread(r"Images\\RealLena.tif")
    # lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    # thresh_im = sltThreshold()#add parameters
    #
    # plt.figure()
    # plt.imshow(thresh_im, cmap='gray', vmin=0, vmax=255)
    # plt.title("thresh image using SLT")
    #
    #
    #
    # print("h ------------------------------------\n")
    # im1 = lena_gray
    # im2 = darkimage
    # SLTim = #TODO
    #
    # # then print
    # plt.figure()
    # plt.subplot(1, 3, 1)
    # plt.imshow(im1)
    # plt.title("original image")
    # plt.subplot(1, 3, 2)
    # plt.imshow(SLTim, cmap='gray', vmin=0, vmax=255)
    # plt.title("tone mapped")
    # plt.subplot(1, 3, 3)
    # plt.imshow(im2, cmap='gray', vmin=0, vmax=255)
    # plt.title("tone mapped")
    #
    #
    #
    # d1 = # mean sqr dist between im1 and im2
    # d2 = # mean sqr dist between mapped image and im2
    # print("mean sqr dist between im1 and im2 = {}\n".format(d1))
    # print("mean sqr dist between mapped image and im2 = {}\n".format(d2))
    #
    # print("i ------------------------------------\n")
    # # prove comutationally
    # d = # TODO:
    # print(" {}".format(d))
    #
    #
    # plt.show()
