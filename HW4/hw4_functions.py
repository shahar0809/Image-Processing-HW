import cv2 as cv2
import numpy as np
import scipy.signal

from matplotlib import pyplot as plt


def print_IDs():
    print("213991029 + 213996549 \n")


# baby
def clean_im1(img):
    transform = find_projective_transform(np.load("baby_points1.npy"), np.load("baby_points2.npy"))
    clean_img = cv2.medianBlur(img, 3)
    return map_image(clean_img, transform, clean_img.shape)


# windmill
def clean_im2(img):
    img_fourier = np.fft.fftshift(np.fft.fft2(img))

    rows, cols = img.shape
    center_rows, center_cols = rows // 2, cols // 2
    magnitude = 15 * np.log(abs(img_fourier) + 1)
    magnitude_peak = np.flip(np.dstack(np.unravel_index(np.argsort(magnitude.ravel()), magnitude.shape)))
    magnitude_peak = magnitude_peak.reshape((img.ravel().shape[0], 2))
    peak_points = magnitude_peak[np.logical_and(np.abs(magnitude_peak[:, 0] - center_rows) > 2,
                                                np.abs(magnitude_peak[:, 1] - center_rows) > 2)]
    peak_points[:, [0, 1]] = peak_points[:, [1, 0]]
    peak_points = peak_points[:2]
    row, col = peak_points[:, 0], peak_points[:, 1]
    img_fourier[row, col] = 0
    f_ishift = np.fft.ifftshift(img_fourier)
    img_back = np.fft.ifft2(f_ishift)
    clean_img = np.real(img_back)
    return clean_img


# watermelon
def clean_im3(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    return contrast_enhance(image_sharp, [0, 255])[0]


# umbrella
def clean_im4(img):
    mask = np.zeros(img.shape)
    mask[0][0] = 1
    mask[4][79] = 1
    mask = mask / np.sum(mask)

    return fft_moved_img(img, mask)


# USA flag
def clean_im5(im):
    clean_im = 0
    return clean_im


# cups
def clean_im6(img):
    clean_im = 0
    return clean_im


# house
def clean_im7(img):
    mask = np.zeros(img.shape)
    mask[0:1, 0:10] = np.array([1] * 10)
    mask = mask / np.sum(mask)

    return fft_moved_img(img, mask)


# bears
def clean_im8(img):
    return contrast_enhance(img, [0, 255])[0]


""" Auxiliary functions for clean_im1 """


def get_image_pts(im, var_name, n_points):
    fig = plt.figure()
    fig.set_label("Select {} points in the first image".format(n_points))
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    image_list = plt.ginput(n=n_points, show_clicks=True)
    plt.close()

    image_points = np.round(np.array([[first_in_tuple[0] for first_in_tuple in image_list],
                                      [second_in_tuple[1] for second_in_tuple in image_list], [1] * n_points]))

    np.save(var_name + ".npy", image_points)


def map_image(im, transform, size_out_image):
    im_new = np.zeros(size_out_image)

    # create mesh grid of all coordinates in new image [x,y]
    xx, yy = np.meshgrid(list(range(size_out_image[1])), list(range(size_out_image[0])))
    xy = np.vstack([xx.ravel(), yy.ravel()])

    # add homogenous coord [x,y,1]
    homogeneous_xy = np.vstack([xy, np.array([1] * xx.ravel().size)])

    # calculate source coordinates that correspond to [x,y,1] in new image
    source_coordinates = np.matmul(np.linalg.inv(transform), homogeneous_xy)
    source_coordinates[0] = source_coordinates[0] / source_coordinates[2]
    source_coordinates[1] = source_coordinates[1] / source_coordinates[2]
    source_coordinates = np.delete(source_coordinates, 2, 0)

    # find coordinates outside range and delete (in source and target)
    outside_range_bool_array = np.vstack(
        [np.any([source_coordinates[0] < 0, source_coordinates[0] > im.shape[1] - 1], axis=0),
         np.any([source_coordinates[1] < 0, source_coordinates[1] > im.shape[0] - 1], axis=0)])

    only_inside_range = np.delete(source_coordinates, np.argwhere(np.any(outside_range_bool_array, axis=0)), 1)

    x_left = np.floor(only_inside_range[1])
    x_right = np.ceil(only_inside_range[1])
    y_top = np.floor(only_inside_range[0])
    y_bottom = np.ceil(only_inside_range[0])

    # interpolate - bilinear
    deltaX = only_inside_range[1] - x_left
    deltaY = only_inside_range[0] - y_top

    upper_x = np.multiply(deltaX, im[np.uint8(x_right), np.uint8(y_bottom)]) + np.multiply((1 - deltaX), im[
        np.uint8(x_left), np.uint8(y_bottom)])
    bottom_x = np.multiply(deltaX, im[np.uint8(x_right), np.uint8(y_top)]) + np.multiply((1 - deltaX), im[
        np.uint8(x_left), np.uint8(y_top)])
    temp_im = np.multiply(deltaY, upper_x) + np.multiply((1 - deltaY), bottom_x)

    # apply corresponding coordinates
    flat_new_im = im_new.ravel()
    flat_new_im[(np.argwhere(np.logical_not(np.any(outside_range_bool_array, axis=0)))).transpose()] = temp_im[
        list(range(temp_im.shape[0]))]

    return flat_new_im.reshape(size_out_image)


def find_projective_transform(points_set1, points_set2):
    N = points_set1.shape[1]
    new_points_list_x1 = list()
    new_points_list_x2 = list()

    # iterate iver points to create x , x'
    for i in range(0, N):
        point_x_set1 = points_set1[0][i]
        point_y_set1 = points_set1[1][i]
        point_x_set2 = points_set2[0][i]
        point_y_set2 = points_set2[1][i]
        new_points_list_x1.append(
            [point_x_set1, point_y_set1, 0, 0, 1, 0, -point_x_set1 * point_x_set2, -point_y_set1 * point_x_set2])
        new_points_list_x1.append(
            [0, 0, point_x_set1, point_y_set1, 0, 1, -point_x_set1 * point_y_set2, -point_y_set1 * point_y_set2])

        new_points_list_x2.append([point_x_set2])
        new_points_list_x2.append([point_y_set2])

    x1_matrix = np.array(new_points_list_x1)
    x2_matrix = np.array(new_points_list_x2)

    projective_parameters = np.vstack([np.matmul(np.linalg.pinv(x1_matrix), x2_matrix), [1]]).reshape((3, 3))
    c = projective_parameters[0][2]
    projective_parameters[0][2] = projective_parameters[1][1]
    projective_parameters[1][1] = projective_parameters[1][0]
    projective_parameters[1][0] = c
    return projective_parameters


""" Auxiliary functions for clean_im4 and clean_im7 """


def fft_moved_img(img, mask):
    mask_fft = np.fft.fftshift(np.fft.fft2(mask))
    for i in range(mask_fft.shape[0]):
        for j in range(mask_fft.shape[1]):
            if abs(mask_fft[i][j]) <= 0.01:
                mask_fft[i][j] = 1
    img_fft = np.fft.fftshift(np.fft.fft2(img))
    clean_im = img_fft / mask_fft
    return abs(np.fft.ifft2(clean_im))


""" Auxiliary functions for clean_im3 and clean_im8 """


def contrast_enhance(im, gray_range):
    a = (np.max(gray_range) - np.min(gray_range)) / (im.max() - im.min())
    b = np.min(gray_range) - im.min() * a
    nim = np.array(im * a + b)
    return np.array(nim, dtype=np.uint8), a, b


'''
    # an example of how to use fourier transform:
    img = cv2.imread(r'Images\windmill.tif')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_fourier = np.fft.fftshift(np.fft.fft2(img))
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.title('original image')
    
    plt.subplot(1,3,2)
    plt.imshow(np.log(abs(img_fourier)), cmap='gray')
    plt.title('fourier transform of image')

    img_inv = np.fft.ifft2(img_fourier)
    plt.subplot(1,3,3)
    plt.imshow(abs(img_inv), cmap='gray')
    plt.title('inverse fourier of the fourier transform')

'''
