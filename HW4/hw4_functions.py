import numpy as np
import cv2 as cv2

def print_IDs():
    print("213991029 + 213996549 \n")

# baby
def clean_im1(im):
    clean_im = 0
    return clean_im

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
    # alpha = 1.95  # Contrast control (1.0-3.0)
    # beta = 0  # Brightness control (0-100)
    #
    # return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return contrast_enhance(img, [0, 255])[0]

def highpass(img, sigma):
    return img - cv2.GaussianBlur(img, (0, 0), sigma) + 127

def fft(img):
    return np.fft.fftshift(np.fft.fft2(img))

def contrast_enhance(im, range):
    a = (np.max(range) - np.min(range)) / (im.max() - im.min())
    b = np.min(range) - im.min() * a
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
