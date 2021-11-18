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
    path_image = r'C:\dev\Image-Processing-HW\HW2\FaceImages\Face1.tif'
    face_img1 = cv2.imread(path_image)
    face_img1_gray = cv2.cvtColor(face_img1, cv2.COLOR_BGR2GRAY)

    path_image = r'C:\dev\Image-Processing-HW\HW2\FaceImages\Face2.tif'
    face_img2 = cv2.imread(path_image)
    face_img2_gray = cv2.cvtColor(face_img2, cv2.COLOR_BGR2GRAY)

    path_image = r'C:\dev\Image-Processing-HW\HW2\FaceImages\Face3.tif'
    section_b_img1 = cv2.imread(path_image)
    section_b_img1_gray = cv2.cvtColor(section_b_img1, cv2.COLOR_BGR2GRAY)

    path_image = r'C:\dev\Image-Processing-HW\HW2\FaceImages\Face4.tif'
    section_b_img2 = cv2.imread(path_image)
    section_b_img2_gray = cv2.cvtColor(section_b_img2, cv2.COLOR_BGR2GRAY)

    getImagePts(face_img1, face_img2, "varName1", "varName2", 12)

    """section a"""

    # Load the 12 selected points
    im1_pts = np.load("varName1.npy")
    im2_pts = np.load("varName2.npy")

    image_list = createMorphSequence(face_img1_gray, im1_pts, face_img2_gray, im2_pts, np.linspace(0, 1, 100), 1)
    writeMorphingVideo(image_list, "transform_face_to_another")

    # """section b"""
    #
    # # load point for the picture we choose
    # im3_pts = np.load("varName3.npy")
    # im4_pts = np.load("varName4.npy")
    #
    # # affine transformation
    # affine_transform = findAffineTransform(im3_pts, im4_pts)
    # mapImage(section_b_img1_gray, T, sizeOutIm)
    #
    # # projective transformation
    # projective_transform = findProjectiveTransform(im3_pts, im4_pts)
    # mapImage(section_b_img2_gray, T, sizeOutIm)
