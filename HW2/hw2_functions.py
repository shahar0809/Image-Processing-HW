import cv2
import matplotlib.pyplot as plt
import numpy as np


def writeMorphingVideo(image_list, video_name):
    out = cv2.VideoWriter(video_name + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20.0, image_list[0].shape, 0)
    for im in image_list:
        out.write(im)
    out.release()


def createMorphSequence(im1, im1_pts, im2, im2_pts, t_list, transformType):
    if transformType:
        T12 = findProjectiveTransform(im1_pts, im2_pts)
        T21 = findProjectiveTransform(im2_pts, im1_pts)
    else:
        T12 = findAffineTransform(im1_pts, im2_pts)
        T21 = findAffineTransform(im2_pts, im1_pts)
    ims = []
    for t in t_list:
        T12_t = np.multiply(1 - t, np.identity(3)) + np.multiply(t, T12)
        T21_1_t = np.multiply(1 - t, T21) + np.multiply(t, np.identity(3))
        newIm1 = mapImage(im1, T12_t, im1.shape)
        newIm2 = mapImage(im2, T21_1_t, im1.shape)
        nim = (np.multiply(1 - t, newIm1) + np.multiply(t, newIm2)).reshape(im1.shape)
        ims.append(np.uint8(nim))

    # for image in ims:
    #     plt.figure()
    #     plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    #     plt.show()
    return ims


def mapImage(im, T, sizeOutIm):
    im_new = np.zeros(sizeOutIm)

    # create mesh grid of all coordinates in new image [x,y]
    xx, yy = np.meshgrid(list(range(sizeOutIm[1])), list(range(sizeOutIm[0])))
    xy = np.vstack([xx.ravel(), yy.ravel()])

    # add homogenous coord [x,y,1]
    homogeneous_xy = np.vstack([xy, np.array([1] * xx.ravel().size)])

    # calculate source coordinates that correspond to [x,y,1] in new image
    source_coordinates = np.matmul(np.linalg.inv(T), homogeneous_xy)
    source_coordinates[0] = source_coordinates[0] / source_coordinates[2]
    source_coordinates[1] = source_coordinates[1] / source_coordinates[2]
    source_coordinates = np.delete(source_coordinates, 2, 0)
    print(source_coordinates)

    # find coordinates outside range and delete (in source and target)
    outside_range_bool_array = np.vstack(
        [np.any([source_coordinates[0] < 0, source_coordinates[0] > im.shape[1] - 1], axis=0),
         np.any([source_coordinates[1] < 0, source_coordinates[1] > im.shape[0] - 1], axis=0)])

    only_inside_range = np.delete(source_coordinates, np.argwhere(np.any(outside_range_bool_array, axis=0)), 1)
    #print(only_inside_range)

    x_left = np.floor(only_inside_range[1])
    x_right = np.ceil(only_inside_range[1])
    y_top = np.floor(only_inside_range[0])
    y_bottom = np.ceil(only_inside_range[0])

    # interpolate - bilinear
    # upper_left = np.vstack([x_left, y_top])
    # upper_right = np.vstack([x_right, y_top])
    # bottom_left = np.vstack([x_left, y_bottom])
    # bottom_right = np.vstack([x_right, y_bottom])

    deltaX = only_inside_range[1] - x_left
    deltaY = only_inside_range[0] - y_top

    upper_x = np.multiply(deltaX, im[np.uint8(x_right), np.uint8(y_top)]) + np.multiply((1 - deltaX), im[
        np.uint8(x_left), np.uint8(y_bottom)])
    bottom_x = np.multiply(deltaX, im[np.uint8(x_right), np.uint8(y_bottom)]) + np.multiply((1 - deltaX), im[
        np.uint8(x_left), np.uint8(y_top)])
    temp_im = np.multiply(deltaY, upper_x) + np.multiply((1 - deltaY), bottom_x)

    # upper_x = np.multiply(deltaX, im[x_right.astype(int), y_top.astype(int)]) + np.multiply((1 - deltaX), im[
    #     x_left.astype(int), y_bottom.astype(int)])
    # bottom_x = np.multiply(deltaX, im[x_right.astype(int), y_bottom.astype(int)]) + np.multiply((1 - deltaX), im[
    #     x_left.astype(int), y_top.astype(int)])
    # temp_im = np.multiply(deltaY, upper_x) + np.multiply((1 - deltaY), bottom_x)

    # upper_x = np.multiply(deltaY, im[x_left.astype(int), y_top.astype(int)]) + np.multiply((1 - deltaY), im[
    #     x_right.astype(int), y_top.astype(int)])
    # bottom_x = np.multiply(deltaY, im[x_left.astype(int), y_bottom.astype(int)]) + np.multiply((1 - deltaY), im[
    #     x_right.astype(int), y_bottom.astype(int)])
    # temp_im = np.multiply(deltaX, upper_x) + np.multiply((1 - deltaX), bottom_x)


    # upper_x = np.multiply(deltaX, im[np.uint8(x_left), np.uint8(y_top)]) + np.multiply((1 - deltaX), im[
    #     np.uint8(x_right), np.uint8(y_top)])
    # bottom_x = np.multiply(deltaX, im[np.uint8(x_left), np.uint8(y_bottom)]) + np.multiply((1 - deltaX), im[
    #     np.uint8(x_right), np.uint8(y_bottom)])
    # temp_im = np.multiply(deltaY, upper_x) + np.multiply((1 - deltaY), bottom_x)

    # apply corresponding coordinates
    flat_new_im = im_new.ravel()
    # for position, index in enumerate(list(np.argwhere(np.logical_not(np.any(outside_range_bool_array, axis=0))))):
    #     flat_new_im[index] = temp_im[position]
    flat_new_im[(np.argwhere(np.logical_not(np.any(outside_range_bool_array, axis=0)))).transpose()] = temp_im[
        list(range(temp_im.shape[0]))]

    return flat_new_im.reshape(sizeOutIm)


def findProjectiveTransform(pointsSet1, pointsSet2):
    N = pointsSet1.shape[1]
    new_points_list_x1 = list()
    new_points_list_x2 = list()

    # iterate iver points to create x , x'
    for i in range(0, N):
        point_x_set1 = pointsSet1[0][i]
        point_y_set1 = pointsSet1[1][i]
        point_x_set2 = pointsSet2[0][i]
        point_y_set2 = pointsSet2[1][i]
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


def findAffineTransform(pointsSet1, pointsSet2):
    N = pointsSet1.shape[1]
    new_points_list_x1 = list()
    new_points_list_x2 = list()

    # iterate over points to create x , x'
    for i in range(0, N):
        new_points_list_x1.append([pointsSet1[0][i], pointsSet1[1][i], 0, 0, 1, 0])
        new_points_list_x1.append([0, 0, pointsSet1[0][i], pointsSet1[1][i], 0, 1])

        new_points_list_x2.append([pointsSet2[0][i]])
        new_points_list_x2.append([pointsSet2[1][i]])

    x1_matrix = np.array(new_points_list_x1)
    x2_matrix = np.array(new_points_list_x2)

    affine_parameters = np.matmul(np.linalg.pinv(x1_matrix), x2_matrix).reshape((2, 3))
    c = affine_parameters[0][2]
    affine_parameters[0][2] = affine_parameters[1][1]
    affine_parameters[1][1] = affine_parameters[1][0]
    affine_parameters[1][0] = c
    T = np.vstack([affine_parameters, [0, 0, 1]])
    return T


def getImagePts(im1, im2, varName1, varName2, nPoints):
    fig = plt.figure()
    fig.set_label("Select {} points in the first image".format(nPoints))
    plt.imshow(im1, cmap='gray', vmin=0, vmax=255)
    image1_list = plt.ginput(n=nPoints, show_clicks=True)
    plt.close()

    fig = plt.figure()
    fig.set_label("Select {} points in the second image".format(nPoints))
    plt.imshow(im2, cmap='gray', vmin=0, vmax=255)
    image2_list = plt.ginput(n=nPoints, show_clicks=True)
    plt.close()

    imagePts1 = np.round(np.array([[first_in_tuple[0] for first_in_tuple in image1_list],
                                   [second_in_tuple[1] for second_in_tuple in image1_list], [1] * nPoints]))
    imagePts2 = np.round(np.array([[first_in_tuple[0] for first_in_tuple in image2_list],
                                   [second_in_tuple[1] for second_in_tuple in image2_list], [1] * nPoints]))

    np.save(varName1 + ".npy", imagePts1)
    np.save(varName2 + ".npy", imagePts2)
