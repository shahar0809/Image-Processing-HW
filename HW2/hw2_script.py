import os

from hw2_functions import *

if __name__ == "__main__":
    # # Read images from folder
    # path_image = r'C:\dev\Image-Processing-HW\HW2\FaceImages\Face1.tif'
    # face_img1 = cv2.imread(path_image)
    # face_img1_gray = cv2.cvtColor(face_img1, cv2.COLOR_BGR2GRAY)
    #
    # path_image = r'C:\dev\Image-Processing-HW\HW2\FaceImages\Face2.tif'
    # face_img2 = cv2.imread(path_image)
    # face_img2_gray = cv2.cvtColor(face_img2, cv2.COLOR_BGR2GRAY)
    #
    # getImagePts(face_img1, face_img2, "varName1", "varName2", 12)
    #
    # imagePts1 = np.load("varName1.npy")
    # imagePts2 = np.load("varName2.npy")
    #
    # t = findAffineTransform(imagePts1, imagePts2)
    # findProjectiveTransform(imagePts1, imagePts2)
    #
    # mapImage(imagePts1, t, (2, 3))

    # Read images from folder
    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "FaceImages")
    print(dir_path)

    path_image = os.path.join(dir_path, "Face1.tif")
    print(path_image)
    face_img1 = cv2.imread("FaceImages\Face5.tif")
    face_img1_gray = cv2.cvtColor(face_img1, cv2.COLOR_BGR2GRAY)

    path_image = os.path.join(dir_path, "Face2.tif")
    face_img2 = cv2.imread("FaceImages\Face6.tif")
    face_img2_gray = cv2.cvtColor(face_img2, cv2.COLOR_BGR2GRAY)

    path_image = os.path.join(dir_path, "Face3.tif")
    section_b_img1 = cv2.imread("FaceImages\Face3.tif")
    section_b_img1_gray = cv2.cvtColor(section_b_img1, cv2.COLOR_BGR2GRAY)

    path_image = os.path.join(dir_path, "Face4.tif")
    section_b_img2 = cv2.imread("FaceImages\Face4.tif")
    section_b_img2_gray = cv2.cvtColor(section_b_img2, cv2.COLOR_BGR2GRAY)

    # getImagePts(face_img1, face_img2, "section_a1", "section_a2", 12)
    # getImagePts(face_img1, face_img2, "section_b1", "section_b2", 4)
    # getImagePts(face_img1, face_img2, "section_c1_1_small", "section_c2_1_small", 6)
    # getImagePts(face_img1, face_img2, "section_c1_1_large", "section_c2_1_large", 12)
    # getImagePts(face_img1, face_img2, "section_c1_2_distributed", "section_c2_2_distributed", 12)
    # getImagePts(face_img1, face_img2, "section_c1_2_focused", "section_c2_2_focused", 12)

    """section a"""

    # Load the 12 selected points
    section_a1_pts = np.load("section_a1.npy")
    section_a2_pts = np.load("section_a2.npy")

    image_list = createMorphSequence(face_img1_gray, section_a1_pts, face_img2_gray, section_a2_pts,
                                     np.linspace(0, 1, 100), 1)
    writeMorphingVideo(image_list, "transform_face_to_another")

    """section b"""

    # load point for the picture we choose
    section_b1_pts = np.load("section_b1.npy")
    section_b2_pts = np.load("section_b2.npy")

    # affine transformation
    affine_transform = findAffineTransform(section_b1_pts, section_b2_pts)
    affine_image = mapImage(section_b_img1_gray, affine_transform, section_b_img1_gray.shape)

    # projective transformation
    projective_transform = findProjectiveTransform(section_b1_pts, section_b2_pts)
    projective_image = mapImage(section_b_img2_gray, projective_transform, section_b_img2_gray.shape)

    # then display
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(affine_image, cmap='gray', vmin=0, vmax=255)
    plt.title("Image after affine transformation")
    plt.subplot(1, 2, 2)
    plt.imshow(projective_image, cmap='gray', vmin=0, vmax=255)
    plt.title("Image after projective transformation")

    """section c"""

    # display images with different amount of points - load point for the picture we choose
    section_c1_1_small_pts = np.load("section_c1_1_small.npy")
    section_c2_1_small_pts = np.load("section_c2_1_small.npy")
    section_c1_1_large_pts = np.load("section_c1_1_large.npy")
    section_c2_1_large_pts = np.load("section_c2_1_large.npy")

    small_img = createMorphSequence(face_img1_gray, section_c1_1_small_pts, face_img2_gray, section_c2_1_small_pts,
                                    np.linspace(0, 1, 2), 1)[-1]
    large_img = createMorphSequence(face_img1_gray, section_c1_1_large_pts, face_img2_gray, section_c2_1_large_pts,
                                    np.linspace(0, 1, 2), 1)[-1]

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(small_img, cmap='gray', vmin=0, vmax=255)
    plt.title("Morph result with 6 points")
    plt.subplot(1, 2, 2)
    plt.imshow(large_img, cmap='gray', vmin=0, vmax=255)
    plt.title("Morph result with 12 points")

    # display images with different location of points - load point for the picture we choose
    section_c1_2_distributed_pts = np.load("section_c1_2_distributed.npy")
    section_c2_2_distributed_pts = np.load("section_c2_2_distributed.npy")
    section_c1_2_focused_pts = np.load("section_c1_2_focused.npy")
    section_c2_2_focused_pts = np.load("section_c2_2_focused.npy")

    distributed_img = \
        createMorphSequence(face_img1_gray, section_c1_2_distributed_pts, face_img2_gray, section_c2_2_distributed_pts,
                            np.linspace(0, 1, 2), 1)[-1]
    focused_img = \
        createMorphSequence(face_img1_gray, section_c1_2_focused_pts, face_img2_gray, section_c2_2_focused_pts,
                            np.linspace(0, 1, 2), 1)[-1]

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(small_img, cmap='gray', vmin=0, vmax=255)
    plt.title("Morph result with distributed points")
    plt.subplot(1, 2, 2)
    plt.imshow(large_img, cmap='gray', vmin=0, vmax=255)
    plt.title("Morph result with focused points")
