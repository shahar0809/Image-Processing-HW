import cv2
import matplotlib.pyplot as plt
import numpy as np


def writeMorphingVideo(image_list, video_name):
    out = cv2.VideoWriter(video_name + ".mp4", cv2.VideoWriter_fourcc(*'MP4V'), 20.0, image_list[0].shape, 0)
    for im in image_list:
        out.write(im)
    out.release()


# def createMorphSequence(im1, im1_pts, im2, im2_pts, t_list, transformType):
#     if transformType:
#     # TODO: projective transforms
#     else:
#     # TODO: affine transforms
#     ims = []
#     for t in t_list:
#         # TODO: calculate nim for each t
#         ims.append(nim)
#     return ims


def mapImage(im, T, sizeOutIm):
    im_new = np.zeros(sizeOutIm)

    # create mesh grid of all coordinates in new image [x,y]
    xx, yy = np.meshgrid(list(range(sizeOutIm[0])), list(range(sizeOutIm[1])))
    xy = np.vstack([xx.ravel(), yy.ravel()])
    print(xy)

    # add homogenous coord [x,y,1]
    homogeneous_xy = np.vstack([xy, np.array([1] * xx.ravel().size)])
    print(homogeneous_xy)

    # calculate source coordinates that correspond to [x,y,1] in new image
    print(T)
    print(np.linalg.inv(T))
    source_coordinates = np.matmul(np.linalg.inv(T), homogeneous_xy)
    print(source_coordinates)

    """np.vstack([source_coordinates[0][:] < 0 && source_coordinates[0][:] > 255, my_2d_array[1][:] > 2])"""

    # find coordinates outside range and delete (in source and target)
    inside_range = np.delete(source_coordinates, 2, 1)

    # interpolate - bilinear

    # apply corresponding coordinates
    # new_im [ target coordinates ] = old_im [ source coordinates ]


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

    affine_parameters = np.vstack([np.matmul(np.linalg.pinv(x1_matrix), x2_matrix), [1]])
    T = affine_parameters.reshape((3, 3))
    return T


def findAffineTransform(pointsSet1, pointsSet2):
    N = pointsSet1.shape[1]
    new_points_list_x1 = list()
    new_points_list_x2 = list()

    # iterate iver points to create x , x'
    for i in range(0, N):
        new_points_list_x1.append([pointsSet1[0][i], pointsSet1[1][i], 0, 0, 1, 0])
        new_points_list_x1.append([0, 0, pointsSet1[0][i], pointsSet1[1][i], 0, 1])

        new_points_list_x2.append([pointsSet2[0][i]])
        new_points_list_x2.append([pointsSet2[1][i]])

    x1_matrix = np.array(new_points_list_x1)
    x2_matrix = np.array(new_points_list_x2)

    affine_parameters = np.matmul(np.linalg.pinv(x1_matrix), x2_matrix).reshape((2, 3))
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
