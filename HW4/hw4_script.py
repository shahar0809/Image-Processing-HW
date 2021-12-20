from hw4_functions import *

if __name__ == "__main__":
    print("----------------------------------------------------\n")
    print_IDs()

    print("-----------------------image 1----------------------\n")
    im1 = cv2.imread(r'Images\baby.tif')
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

    # get_image_pts(im1, "baby_points1", 4)
    # get_image_pts(im1, "baby_points2", 4)

    im1_clean = clean_im1(im1)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im1, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im1_clean, cmap='gray', vmin=0, vmax=255)

    print("Describe the problem with the image and your method/solution:")
    print("   TODO: add explanation    \n")

    print("-----------------------image 2----------------------\n")
    im2 = cv2.imread(r'Images\windmill.tif')
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    im2_clean = clean_im2(im2)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im2, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im2_clean, cmap='gray', vmin=0, vmax=255)

    print("Describe the problem with the image and your method/solution:")
    print("We noticed periodic noise in the image.\n"
          "Then, we transferred the image to the frequency space, and zeroed out the first 2 peaks that weren't close the origin of the image.\n"
          "lastly, we transferred the image back using inverse fft.\n")

    print("-----------------------image 3----------------------\n")
    im3 = cv2.imread(r'Images\watermelon.tif')
    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)
    im3_clean = clean_im3(im3)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im3, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im3_clean, cmap='gray', vmin=0, vmax=255)

    print("Describe the problem with the image and your method/solution:")
    print("We noticed that the image wasn't sharp and bright enough.\n"
          "So, we enhanced the contrast to be maximal, and used sharpening filter [kernel taken from Wikipedia].")

    print("-----------------------image 4----------------------\n")
    im4 = cv2.imread(r'Images\umbrella.tif')
    im4 = cv2.cvtColor(im4, cv2.COLOR_BGR2GRAY)
    im4_clean = clean_im4(im4)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im4, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im4_clean, cmap='gray', vmin=0, vmax=255)

    print("Describe the problem with the image and your method/solution:")
    print(
        "We noticed the echo effect because the image wasn't smooth in color (there were double objects) .\n"
        "We calculated the point in which the picture was moved in both axes [(4, 79), (0, 0)].\n"
        "Then, we created translation mask with the points above, and moved to the frequency space.\n"
        "In order to reduce noise and avoid division by 0, we set small values to 1.\n"
        "In order to restore the image, we divided the fft of the image by the fft of the mask.\n"
        "Lastly, we converted the image back using inverse fft."
        " \n")

    print("-----------------------image 5----------------------\n")
    im5 = cv2.imread(r'Images\USAflag.tif')
    im5 = cv2.cvtColor(im5, cv2.COLOR_BGR2GRAY)
    im5_clean = clean_im5(im5)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im5, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im5_clean, cmap='gray', vmin=0, vmax=255)

    print("Describe the problem with the image and your method/solution:")
    print("The flag had letters that we wanted to remove.\n"
          "We used a median filter (after playing around with the parameters), "
          "and put the original image over the part of the stars (to avoid median artifacts)\n")

    print("-----------------------image 6----------------------\n")
    im6 = cv2.imread(r'Images\cups.tif')
    im6 = cv2.cvtColor(im6, cv2.COLOR_BGR2GRAY)
    im6_clean = clean_im6(im6)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im6, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im6_clean, cmap='gray', vmin=0, vmax=255)

    print("Describe the problem with the image and your method/solution:")
    print("We saw that the image has ringing artifact.\n"
          "In order to remove it, we looked at the spectrum of the image, and noticed a rectangle.\n"
          "We wanted to increase the color of the rectangle in the spectrum, so that it wouldn't stand out - we want increased it\n"
          "After trying a lot of combinations, and found the one.\n"
          "Lastly, we used inverse fft.\n")

    print("-----------------------image 7----------------------\n")
    im7 = cv2.imread(r'Images\house.tif')
    im7 = cv2.cvtColor(im7, cv2.COLOR_BGR2GRAY)
    im7_clean = clean_im7(im7)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im7, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im7_clean, cmap='gray', vmin=0, vmax=255)

    print("Describe the problem with the image and your method/solution:")
    print(
        "We noticed the motion blur effect because the white point in the original image was smeared over 10 pixels.\n"
        "Then, we created translation mask and moved to the frequency space.\n"
        "In order to reduce noise and avoid division by 0, we set small values to 1.\n"
        "In order to restore the image, we divided the fft of the image by the fft of the mask.\n"
        "Lastly, we converted the image back using inverse fft."
        " \n")

    print("-----------------------image 8----------------------\n")
    im8 = cv2.imread(r'Images\bears.tif')
    im8 = cv2.cvtColor(im8, cv2.COLOR_BGR2GRAY)
    im8_clean = clean_im8(im8)

    print("Describe the problem with the image and your method/solution:")
    print("The image was dull, so we enhanced the contrast to be maximal.\n")

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im8, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im8_clean, cmap='gray', vmin=0, vmax=255)
    plt.show()

