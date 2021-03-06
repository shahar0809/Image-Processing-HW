from hw2_functions import *

if __name__ == "__main__":
    # Read images from folder
    face_img1 = cv2.imread("FaceImages\Face5.tif")
    face_img1_gray = cv2.cvtColor(face_img1, cv2.COLOR_BGR2GRAY)

    face_img2 = cv2.imread("FaceImages\Face6.tif")
    face_img2_gray = cv2.cvtColor(face_img2, cv2.COLOR_BGR2GRAY)

    section_b_img1 = cv2.imread("our_images\WhatsApp Image 2021-11-20 at 22.11.13.jpeg")
    section_b_img1_gray = cv2.cvtColor(section_b_img1, cv2.COLOR_BGR2GRAY)

    section_b_img2 = cv2.imread("our_images\WhatsApp Image 2021-11-20 at 22.11.15.jpeg")
    section_b_img2_gray = cv2.cvtColor(section_b_img2, cv2.COLOR_BGR2GRAY)

    section_d_img1 = cv2.imread("our_images\d1.jpg")
    section_d_img1_gray = cv2.cvtColor(section_d_img1, cv2.COLOR_BGR2GRAY)

    section_d_img2 = cv2.imread("our_images\d2.jpg")
    section_d_img2_gray = cv2.cvtColor(section_d_img2, cv2.COLOR_BGR2GRAY)

    # getImagePts(face_img1_gray, face_img2_gray, "section_a1", "section_a2", 12)
    # getImagePts(section_b_img1_gray, section_b_img2, "section_b1", "section_b2", 12)
    # getImagePts(face_img1, face_img2, "section_c1_1_small", "section_c2_1_small", 6)
    # getImagePts(face_img1, face_img2, "section_c1_1_large", "section_c2_1_large", 12)
    # getImagePts(face_img1, face_img2, "section_c1_2_distributed", "section_c2_2_distributed", 12)
    # getImagePts(face_img1, face_img2, "section_c1_2_focused", "section_c2_2_focused", 12)
    # getImagePts(section_d_img1_gray, section_d_img2_gray, "section_d1", "section_d2", 24)

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
    affine_transformation = findAffineTransform(section_b1_pts, section_b2_pts)
    affine_image = mapImage(section_b_img1_gray, affine_transformation, section_b_img1_gray.shape)

    # projective transformation
    projective_transformation = findProjectiveTransform(section_b1_pts, section_b2_pts)
    projective_image = mapImage(section_b_img1_gray, projective_transformation, section_b_img1_gray.shape)

    # then display
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(section_b_img1_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("Original image")
    plt.subplot(1, 3, 2)
    plt.imshow(affine_image, cmap='gray', vmin=0, vmax=255)
    plt.title("Image after affine transformation")
    plt.subplot(1, 3, 3)
    plt.imshow(projective_image, cmap='gray', vmin=0, vmax=255)
    plt.title("Image after projective transformation")

    """section c"""

    # display images with different amount of points - load point for the picture we choose
    section_c1_1_small_pts = np.load("section_c1_1_small.npy")
    section_c2_1_small_pts = np.load("section_c2_1_small.npy")
    section_c1_1_large_pts = np.load("section_c1_1_large.npy")
    section_c2_1_large_pts = np.load("section_c2_1_large.npy")

    small_img = createMorphSequence(face_img1_gray, section_c1_1_small_pts, face_img2_gray, section_c2_1_small_pts,
                                    np.linspace(0, 1, 51), 1)
    large_img = createMorphSequence(face_img1_gray, section_c1_1_large_pts, face_img2_gray, section_c2_1_large_pts,
                                    np.linspace(0, 1, 51), 1)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(small_img[25], cmap='gray', vmin=0, vmax=255)
    plt.title("Morph result with 6 points")
    plt.subplot(1, 2, 2)
    plt.imshow(large_img[25], cmap='gray', vmin=0, vmax=255)
    plt.title("Morph result with 12 points")

    # display images with different location of points - load point for the picture we choose
    section_c1_2_distributed_pts = np.load("section_c1_2_distributed.npy")
    section_c2_2_distributed_pts = np.load("section_c2_2_distributed.npy")
    section_c1_2_focused_pts = np.load("section_c1_2_focused.npy")
    section_c2_2_focused_pts = np.load("section_c2_2_focused.npy")

    distributed_img = createMorphSequence(face_img1_gray, section_c1_2_distributed_pts, face_img2_gray,
                                          section_c2_2_distributed_pts,
                                          np.linspace(0, 1, 51), 1)
    focused_img = createMorphSequence(face_img1_gray, section_c1_2_focused_pts, face_img2_gray,
                                      section_c2_2_focused_pts,
                                      np.linspace(0, 1, 51), 1)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(small_img[25], cmap='gray', vmin=0, vmax=255)
    plt.title("Morph result with distributed points")
    plt.subplot(1, 2, 2)
    plt.imshow(large_img[25], cmap='gray', vmin=0, vmax=255)
    plt.title("Morph result with focused points")
    plt.show()

    """section d"""
    # Load the 12 selected points
    section_d1_pts = np.load("section_d1.npy")
    section_d2_pts = np.load("section_d2.npy")

    image_list_d = createMorphSequence(section_d_img1_gray, section_d1_pts, section_d_img2_gray, section_d2_pts,
                                       np.linspace(0, 1, 100), 1)

    writeMorphingVideo(image_list_d, "vid")
