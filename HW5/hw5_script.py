from hw5_functions import *
import matplotlib.pyplot as plt


def main_script():
    # Applying Sobel Edge Detection on balls image
    im1 = cv2.imread("balls1.tif")
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(im1, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.title("Sobel")
    im1_edge_det = sobel_edge_detection(im1)
    plt.imshow(im1_edge_det, cmap='gray', vmin=0, vmax=255)

    plt.show()


if __name__ == '__main__':
    main_script()
