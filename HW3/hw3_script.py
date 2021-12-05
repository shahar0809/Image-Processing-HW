from hw3_functions import *

if __name__ == "__main__":
    # feel free to load different image than lena
    lena = cv2.imread(r"Images\lena.tif")
    lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)


    # 1 ----------------------------------------------------------
    # add salt and pepper noise - low
    lena_sp_low = addSPnoise(lena_gray, 0.05)  # add low noise
    lena_sp_low_median = cleanImageMedian(lena_sp_low, 1)
    np.save("lena_sp_low_median",lena_sp_low_median)
    lena_sp_low_mean = cleanImageMean(lena_sp_low, 2, 1.5)
    np.save("lena_sp_low_mean",lena_sp_low_mean)
    lena_sp_low_bilateral = bilateralFilt(lena_sp_low, 2, 20, 100)
    np.save("lena_sp_low_bilateral",lena_sp_low_bilateral)

    lena_sp_low_median = np.load("lena_sp_low_median.npy")
    lena_sp_low_mean = np.load("lena_sp_low_mean.npy")
    lena_sp_low_bilateral = np.load("lena_sp_low_bilateral.npy")

    # add parameters to functions cleanImageMedian, cleanImageMean, bilateralFilt
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(lena_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("original")
    plt.subplot(2, 3, 2)
    plt.imshow(lena_sp_low, cmap='gray', vmin=0, vmax=255)
    plt.title("salt and pepper - low")
    plt.subplot(2, 3, 4)
    plt.imshow(lena_sp_low_median, cmap='gray', vmin=0, vmax=255)
    plt.title("median")
    plt.subplot(2, 3, 5)
    plt.imshow(lena_sp_low_mean, cmap='gray', vmin=0, vmax=255)
    plt.title("mean")
    plt.subplot(2, 3, 6)
    plt.imshow(lena_sp_low_bilateral, cmap='gray', vmin=0, vmax=255)
    plt.title("bilateral")

    print("Conclusions ----- sp low")
    print("The best cleaning is median. After that is mean and finally bilateral \n")


    # 2 ----------------------------------------------------------
    # add salt and pepper noise - high
    lena_sp_high = addSPnoise(lena_gray, 0.3)  # add high noise
    lena_sp_high_median = cleanImageMedian(lena_sp_high,2)
    np.save("lena_sp_high_median", lena_sp_high_median)
    lena_sp_high_mean = cleanImageMean(lena_sp_high,2,1)
    np.save("lena_sp_high_mean", lena_sp_high_mean)
    lena_sp_high_bilateral = bilateralFilt(lena_sp_high,2,20,100)
    np.save("lena_sp_high_bilateral", lena_sp_high_bilateral)

    lena_sp_high_median = np.load("lena_sp_high_median.npy")
    lena_sp_high_mean = np.load("lena_sp_high_mean.npy")
    lena_sp_high_bilateral = np.load("lena_sp_high_bilateral.npy")

    # add parameters to functions cleanImageMedian, cleanImageMean, bilateralFilt
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(lena_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("original")
    plt.subplot(2, 3, 2)
    plt.imshow(lena_sp_high, cmap='gray', vmin=0, vmax=255)
    plt.title("salt and pepper - high")
    plt.subplot(2, 3, 4)
    plt.imshow(lena_sp_high_median, cmap='gray', vmin=0, vmax=255)
    plt.title("median")
    plt.subplot(2, 3, 5)
    plt.imshow(lena_sp_high_mean, cmap='gray', vmin=0, vmax=255)
    plt.title("mean")
    plt.subplot(2, 3, 6)
    plt.imshow(lena_sp_high_bilateral, cmap='gray', vmin=0, vmax=255)
    plt.title("bilateral")

    print("Conclusions ----- sp high")
    print("The best cleaning is median. After that is mean and finally bilateral \n")


    # 3 ----------------------------------------------------------
    # add gaussian noise - low
    lena_gaussian_low = addGaussianNoise(lena_gray, 20)  # add low noise
    # lena_gaussian_low_median = cleanImageMedian(lena_gaussian_low, 1)
    # np.save("lena_gaussian_low_median", lena_gaussian_low_median)
    # lena_gaussian_low_mean = cleanImageMean(lena_gaussian_low, 2, 5)
    # np.save("lena_gaussian_low_mean", lena_gaussian_low_mean)
    # lena_gaussian_low_bilateral = bilateralFilt(lena_gaussian_low, 2, 20, 40)
    # np.save("lena_gaussian_low_bilateral", lena_gaussian_low_bilateral)

    lena_gaussian_low_bilateral = np.load("lena_gaussian_low_bilateral.npy")
    lena_gaussian_low_mean = np.load("lena_gaussian_low_mean.npy")
    lena_gaussian_low_median = np.load("lena_gaussian_low_median.npy")

    # add parameters to functions cleanImageMedian, cleanImageMean, bilateralFilt
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(lena_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("original")
    plt.subplot(2, 3, 2)
    plt.imshow(lena_gaussian_low, cmap='gray', vmin=0, vmax=255)
    plt.title("gaussian noise - low")
    plt.subplot(2, 3, 4)
    plt.imshow(lena_gaussian_low_median , cmap='gray', vmin=0, vmax=255)
    plt.title("median")
    plt.subplot(2, 3, 5)
    plt.imshow(lena_gaussian_low_mean, cmap='gray', vmin=0, vmax=255)
    plt.title("mean")
    plt.subplot(2, 3, 6)
    plt.imshow(lena_gaussian_low_bilateral, cmap='gray', vmin=0, vmax=255)
    plt.title("bilateral")

    print("Conclusions ----- gaussian noise - low")
    print("The best cleaning is bilateral. After that is mean and finally median  \n")

    # 4 ----------------------------------------------------------
    # add gaussian noise - high
    lena_gaussian_high = addGaussianNoise(lena_gray,50) # add high noise
    # lena_gaussian_high_median = cleanImageMedian(lena_gaussian_high,1)
    # np.save("lena_gaussian_high_median", lena_gaussian_high_median)
    # lena_gaussian_high_mean = cleanImageMean(lena_gaussian_high,2,5)
    # np.save("lena_gaussian_high_mean", lena_gaussian_high_mean)
    # lena_gaussian_high_bilateral = bilateralFilt(lena_gaussian_high,2,50,80)
    # np.save("lena_gaussian_high_bilateral", lena_gaussian_high_bilateral)

    lena_gaussian_high_bilateral = np.load("lena_gaussian_high_bilateral.npy")
    lena_gaussian_high_mean = np.load("lena_gaussian_high_mean.npy")
    lena_gaussian_high_median = np.load("lena_gaussian_high_median.npy")

    # add parameters to functions cleanImageMedian, cleanImageMean, bilateralFilt
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(lena_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("original")
    plt.subplot(2, 3, 2)
    plt.imshow(lena_gaussian_high, cmap='gray', vmin=0, vmax=255)
    plt.title("gaussian noise - high")
    plt.subplot(2, 3, 4)
    plt.imshow(lena_gaussian_high_median, cmap='gray', vmin=0, vmax=255)
    plt.title("median")
    plt.subplot(2, 3, 5)
    plt.imshow(lena_gaussian_high_mean, cmap='gray', vmin=0, vmax=255)
    plt.title("mean")
    plt.subplot(2, 3, 6)
    plt.imshow(lena_gaussian_high_bilateral, cmap='gray', vmin=0, vmax=255)
    plt.title("bilateral")

    print("Conclusions ----- gaussian noise - high")
    print("The best cleaning is mean. After that is bilateral and finally median  \n")

plt.show()