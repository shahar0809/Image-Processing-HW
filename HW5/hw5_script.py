from hw5_functions import *
import matplotlib.pyplot as plt

from hw5_functions import *


def main_script():
    # section 1 - Applying Sobel Edge Detection on balls image
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
    plt.imshow(edges_directions, cmap='gray', vmin=0, vmax=255)

    # section 2
    im2 = cv2.cvtColor(cv2.imread("coins1.tif"), cv2.COLOR_BGR2GRAY)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(im2, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.title("Canny")
    im2_edge_detection = canny_edge_detection1(im2)
    plt.imshow(im2_edge_detection, cmap='gray', vmin=0, vmax=255)

    # section 3
    im3 = cv2.cvtColor(cv2.imread("balls2.tif"), cv2.COLOR_BGR2GRAY)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(im3, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.title("Canny")
    im3_edge_detection = canny_edge_detection2(im3)
    plt.imshow(im3_edge_detection, cmap='gray', vmin=0, vmax=255)

    # section 4 - Applying Hough Transform for circles on coins image
    im4 = cv2.cvtColor(cv2.imread("coins3.tif"), cv2.COLOR_BGR2GRAY)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(im4, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.title("Hough Transform - Circles")
    im4_hough_transform = hough_transform_circles(im4)
    plt.imshow(im4_hough_transform, cmap='gray', vmin=0, vmax=255)

    # section 5 - Applying Hough Transform for lines on boxOfChocolates images
    # boxOfChocolates1
    im5 = cv2.cvtColor(cv2.imread("boxOfchocolates1.tif"), cv2.COLOR_BGR2GRAY)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(im5, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.title("Hough Transform - Lines, boxOfChocolates1")
    im5_hough_transform = hough_transform_lines1(im5)
    plt.imshow(im5_hough_transform)

    # boxOfChocolates2
    im5 = cv2.cvtColor(cv2.imread("boxOfchocolates2.tif"), cv2.COLOR_BGR2GRAY)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(im5, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.title("Hough Transform - Lines, boxOfChocolates2")
    im5_hough_transform = hough_transform_lines2(im5)
    plt.imshow(im5_hough_transform)

    # boxOfChocolates2rot
    im5 = cv2.cvtColor(cv2.imread("boxOfchocolates2rot.tif"), cv2.COLOR_BGR2GRAY)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(im5, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.title("Hough Transform - Lines, boxOfChocolates2rot")
    im5_hough_transform = hough_transform_lines3(im5)
    plt.imshow(im5_hough_transform, cmap='gray', vmin=0, vmax=255)

    plt.show()


if __name__ == '__main__':
    main_script()
