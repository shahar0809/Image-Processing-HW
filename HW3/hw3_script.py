import cv2
import matplotlib.pyplot as plt

from hw3_functions import *

if __name__ == "__main__":
    # feel free to load different image than lena
    lena = cv2.imread(r"Images\lena.tif")
    lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)

    # 1 ----------------------------------------------------------
    # add salt and pepper noise - low
    lena_sp_low = addSPnoise(lena_gray, 0.1)

    # cv2.imwrite("add_low_noise_1_image.jpg", lena_sp_low)
    # cv2.imwrite("clean_median_1_image.jpg", cleanImageMedian(lena_sp_low, 1))
    # cv2.imwrite("clean_mean_1_image.jpg", cleanImageMean(lena_sp_low, 2, 1))
    # cv2.imwrite("clean_bilateral_1_image.jpg", bilateralFilt(lena_sp_low, 2, 20, 100))

    # add parameters to functions cleanImageMedian, cleanImageMean, bilateralFilt
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(lena_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("original")
    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(cv2.imread(r"add_low_noise_1_image.jpg"), cv2.COLOR_BGR2GRAY), cmap='gray', vmin=0,
               vmax=255)
    plt.title("salt and pepper - low")
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(cv2.imread(r"clean_median_1_image.jpg"), cv2.COLOR_BGR2GRAY), cmap='gray', vmin=0, vmax=255)
    plt.title("median")
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(cv2.imread(r"clean_mean_1_image.jpg"), cv2.COLOR_BGR2GRAY), cmap='gray', vmin=0, vmax=255)
    plt.title("mean")
    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(cv2.imread(r"clean_bilateral_1_image.jpg"), cv2.COLOR_BGR2GRAY), cmap='gray', vmin=0,
               vmax=255)
    plt.title("bilateral")

    print("Conclusions for ----- salt and pepper low noise")
    print("The best cleaner is: median.")
    print("In second place is: mean")
    print("The last one is: bilateral \n")

    # 2 ----------------------------------------------------------
    # add salt and pepper noise - high
    lena_sp_high = addSPnoise(lena_gray, 0.3)

    # cv2.imwrite("add_high_noise_2_image.jpg", lena_sp_high)
    # cv2.imwrite("clean_median_2_image.jpg", cleanImageMedian(lena_sp_high, 1))
    # cv2.imwrite("clean_mean_2_image.jpg", cleanImageMean(lena_sp_high, 2, 1))
    # cv2.imwrite("clean_bilateral_2_image.jpg", bilateralFilt(lena_sp_high, 1, 20, 100))

    # add parameters to functions cleanImageMedian, cleanImageMean, bilateralFilt
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(lena_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("original")
    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(cv2.imread(r"add_high_noise_2_image.jpg"), cv2.COLOR_BGR2GRAY), cmap='gray', vmin=0,
               vmax=255)
    plt.title("salt and pepper - high")
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(cv2.imread(r"clean_median_2_image.jpg"), cv2.COLOR_BGR2GRAY), cmap='gray', vmin=0, vmax=255)
    plt.title("median")
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(cv2.imread(r"clean_mean_2_image.jpg"), cv2.COLOR_BGR2GRAY), cmap='gray', vmin=0, vmax=255)
    plt.title("mean")
    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(cv2.imread(r"clean_bilateral_2_image.jpg"), cv2.COLOR_BGR2GRAY), cmap='gray', vmin=0,
               vmax=255)
    plt.title("bilateral")

    print("Conclusions for ----- salt and pepper high noise")
    print("The best cleaner is: median")
    print("In second place is: mean")
    print("The last one is: bilateral \n")

    # 3 ----------------------------------------------------------
    # add gaussian noise - low
    lena_gaussian = addGaussianNoise(lena_gray, 20)

    # cv2.imwrite("add_low_noise_3_image.jpg", lena_gaussian)
    # cv2.imwrite("clean_median_3_image.jpg", cleanImageMedian(lena_gaussian, 1))
    # cv2.imwrite("clean_mean_3_image.jpg", cleanImageMean(lena_gaussian, 2, 20))
    # cv2.imwrite("clean_bilateral_3_image.jpg", bilateralFilt(lena_gaussian, 1, 20, 40))

    # add parameters to functions cleanImageMedian, cleanImageMean, bilateralFilt
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(lena_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("original")
    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(cv2.imread(r"add_low_noise_3_image.jpg"), cv2.COLOR_BGR2GRAY), cmap='gray', vmin=0,
               vmax=255)
    plt.title("gaussian noise - low")
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(cv2.imread(r"clean_median_3_image.jpg"), cv2.COLOR_BGR2GRAY), cmap='gray', vmin=0, vmax=255)
    plt.title("median")
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(cv2.imread(r"clean_mean_3_image.jpg"), cv2.COLOR_BGR2GRAY), cmap='gray', vmin=0, vmax=255)
    plt.title("mean")
    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(cv2.imread(r"clean_bilateral_3_image.jpg"), cv2.COLOR_BGR2GRAY), cmap='gray', vmin=0,
               vmax=255)
    plt.title("bilateral")

    print("Conclusions for ----- gaussian low noise")
    print("The best cleaner is: bilateral")
    print("In second place is: mean")
    print("The last one is: median \n")

    # 4 ----------------------------------------------------------
    # add gaussian noise - high
    lena_gaussian = addGaussianNoise(lena_gray, 50)

    # cv2.imwrite("add_low_noise_4_image.jpg", lena_gaussian)
    # cv2.imwrite("clean_median_4_image.jpg", cleanImageMedian(lena_gaussian, 1))
    # cv2.imwrite("clean_mean_4_image.jpg", cleanImageMean(lena_gaussian, 2, 50))
    # cv2.imwrite("clean_bilateral_4_image.jpg", bilateralFilt(lena_gaussian, 1, 50, 80))

    # add parameters to functions cleanImageMedian, cleanImageMean, bilateralFilt
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(lena_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("original")
    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(cv2.imread(r"add_low_noise_4_image.jpg"), cv2.COLOR_BGR2GRAY), cmap='gray', vmin=0,
               vmax=255)
    plt.title("gaussian noise - high")
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(cv2.imread(r"clean_median_4_image.jpg"), cv2.COLOR_BGR2GRAY), cmap='gray', vmin=0, vmax=255)
    plt.title("median")
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(cv2.imread(r"clean_mean_4_image.jpg"), cv2.COLOR_BGR2GRAY), cmap='gray', vmin=0, vmax=255)
    plt.title("mean")
    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(cv2.imread(r"clean_bilateral_4_image.jpg"), cv2.COLOR_BGR2GRAY), cmap='gray', vmin=0,
               vmax=255)
    plt.title("bilateral")

    print("Conclusions for ----- gaussian high noise")
    print("The best cleaner is: mean")
    print("In second place is: bilateral")
    print("The last one is: median \n")

    plt.show()
