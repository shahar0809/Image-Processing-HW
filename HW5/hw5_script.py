from hw5_functions import *
import matplotlib.pyplot as plt


def main_script():
    # Applying Sobel Edge Detection on balls image
    im1 = cv2.imread("balls1.tif")
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(im1, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 3, 2)
    plt.title("Sobel Magnitude")
    im1_edge_det, edges_directions = sobel_edge_detection(im1)
    img1_thresh = threshold_filter(im1_edge_det, 24)
    plt.imshow(img1_thresh, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 3, 3)
    plt.title("Sobel Directions")
    plt.imshow(edges_directions)
    plt.show()


if __name__ == '__main__':
    main_script()
