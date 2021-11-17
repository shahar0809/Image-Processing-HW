from hw2_functions import *

if __name__ == "__main__":
    # Read images from folder
    path_image = r'C:\dev\Image-Processing-HW\HW2\FaceImages\Face1.tif'
    face_img1 = cv2.imread(path_image)
    face_img1_gray = cv2.cvtColor(face_img1, cv2.COLOR_BGR2GRAY)

    path_image = r'C:\dev\Image-Processing-HW\HW2\FaceImages\Face2.tif'
    face_img2 = cv2.imread(path_image)
    face_img2 = cv2.cvtColor(face_img2, cv2.COLOR_BGR2GRAY)

    getImagePts(face_img1, face_img2, "varName1", "varName2", 4)

    imagePts1 = np.load("varName1.npy")
    imagePts2 = np.load("varName2.npy")

    t = findAffineTransform(imagePts1, imagePts2)
    findProjectiveTransform(imagePts1, imagePts2)

    mapImage(imagePts1, t, (2, 3))
