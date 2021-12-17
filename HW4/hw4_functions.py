import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt


def print_IDs():
    print("213991029 + 213996549 \n")


def get_dst_points(points):
    # rect = np.zeros((4, 2), dtype="float32")
    # s = points.sum(axis=1)
    # rect[0] = points[np.argmin(s)]
    # rect[2] = points[np.argmax(s)]
    # diff = np.diff(points, axis=1)
    # rect[1] = points[np.argmin(diff)]
    # rect[3] = points[np.argmax(diff)]

    return np.transpose(points).astype('float32')


def get_rect_width(points):
    bottom_left = points[3], bottom_right = points[2]
    return np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))


def align_img(img, points, isPerspective):
    rect = get_dst_points(points)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [tl[0], tl[1]],
        [tl[0] + maxWidth - 1, tl[1]],
        [tl[0] + maxWidth - 1, tl[1] + maxHeight - 1],
        [tl[0], tl[1] + maxHeight - 1]], dtype="float32")

    if isPerspective:
        M = cv2.getPerspectiveTransform(rect[:, [0, 1]].astype("float32"), dst)
        return cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    else:
        M = cv2.getAffineTransform(rect[:, [0, 1]].astype("float32"), dst)
        return cv2.warpAffine(img, M, (maxWidth, maxHeight))


# baby
def clean_im1(img, points1, points2, points3):
    rect = get_dst_points(points1)
    (tl, tr, br, bl) = rect.astype(int)
    height = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    width = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    aligned_img1 = img[tl[1]:bl[1] + 1, tl[0]:tr[0] + 1]

    aligned_img2 = align_img(img, points2, True)
    aligned_img3 = align_img(img, points3, False)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(aligned_img1, cmap='gray', vmin=0, vmax=255)
    plt.title("im1")
    plt.subplot(1, 3, 2)
    plt.imshow(aligned_img2, cmap='gray', vmin=0, vmax=255)
    plt.title("im2")
    plt.subplot(1, 3, 3)
    plt.imshow(aligned_img3, cmap='gray', vmin=0, vmax=255)
    plt.title("im3")
    plt.show()

    return img


# windmill
def clean_im2(img):
    img_fourier = np.fft.fftshift(np.fft.fft2(img))
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift


# watermelon
def clean_im3(img):
    img_fourier = np.fft.fftshift(np.fft.fft2(img))

    return clean_im


# umbrella
def clean_im4(im):
    clean_im = 0
    return clean_im


# USA flag
def clean_im5(im):
    clean_im = 0
    return clean_im


# cups
def clean_im6(im):
    clean_im = 0
    return clean_im


# house
def clean_im7(im):
    clean_im = 0
    return clean_im


# bears
def clean_im8(img):
    return contrast_enhance(img, [0, 255])[0]


def highpass(img, sigma):
    return img - cv2.GaussianBlur(img, (0, 0), sigma) + 127


def fft(img):
    return np.fft.fftshift(np.fft.fft2(img))


def contrast_enhance(im, gray_range):
    a = (np.max(gray_range) - np.min(gray_range)) / (im.max() - im.min())
    b = np.min(gray_range) - im.min() * a
    nim = np.array(im * a + b)
    return np.array(nim, dtype=np.uint8), a, b


def getImagePts(im, varName, nPoints):
    fig = plt.figure()
    fig.set_label("Select {} points in the first image".format(nPoints))
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    image_list = plt.ginput(n=nPoints, show_clicks=True)
    plt.close()

    imagePts = np.round(np.array([[first_in_tuple[0] for first_in_tuple in image_list],
                                  [second_in_tuple[1] for second_in_tuple in image_list], [1] * nPoints]))

    np.save(varName + ".npy", imagePts)


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
