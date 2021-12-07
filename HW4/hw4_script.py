from hw4_functions import *


if __name__ == "__main__":
    print("----------------------------------------------------\n")
    print_IDs

    print("-----------------------image 1----------------------\n")
    im1 = cv2.imread(r'Images\baby.tif')
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im1_clean = clean_im1(im1)

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(im1, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im1_clean, cmap='gray', vmin=0, vmax=255)

    print("Describe the problem with the image and your method/solution: \n")
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

    print("Describe the problem with the image and your method/solution: \n")
    print("   TODO: add explanation    \n")

    print("-----------------------image 3----------------------\n")
    im3 = cv2.imread(r'Images\watermelon.tif')
    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)
    im3_clean = clean_im3(im3)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im3, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im3_clean, cmap='gray', vmin=0, vmax=255)

    print("Describe the problem with the image and your method/solution: \n")
    print("   TODO: add explanation    \n")

    print("-----------------------image 4----------------------\n")
    im4 = cv2.imread(r'Images\umbrella.tif')
    im4 = cv2.cvtColor(im4, cv2.COLOR_BGR2GRAY)
    im4_clean = clean_im2(im4)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im4, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im4_clean, cmap='gray', vmin=0, vmax=255)

    print("Describe the problem with the image and your method/solution: \n")
    print("   TODO: add explanation    \n")

    print("-----------------------image 5----------------------\n")
    im5 = cv2.imread(r'Images\USAflag.tif')
    im5 = cv2.cvtColor(im5, cv2.COLOR_BGR2GRAY)
    im5_clean = clean_im2(im5)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im5, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im5_clean, cmap='gray', vmin=0, vmax=255)

    print("Describe the problem with the image and your method/solution: \n")
    print("   TODO: add explanation    \n")

    print("-----------------------image 6----------------------\n")
    im6 = cv2.imread(r'Images\cups.tif')
    im6 = cv2.cvtColor(im6, cv2.COLOR_BGR2GRAY)
    im6_clean = clean_im2(im6)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im6, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im6_clean, cmap='gray', vmin=0, vmax=255)

    print("Describe the problem with the image and your method/solution: \n")
    print("   TODO: add explanation    \n")

    print("-----------------------image 7----------------------\n")
    im7 = cv2.imread(r'Images\house.tif')
    im7 = cv2.cvtColor(im7, cv2.COLOR_BGR2GRAY)
    im7_clean = clean_im2(im7)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im7, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im7_clean, cmap='gray', vmin=0, vmax=255)

    print("Describe the problem with the image and your method/solution: \n")
    print("   TODO: add explanation    \n")

    print("-----------------------image 8----------------------\n")
    im8 = cv2.imread(r'Images\bears.tif')
    im8 = cv2.cvtColor(im8, cv2.COLOR_BGR2GRAY)
    im8_clean = clean_im2(im8)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im8, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im8_clean, cmap='gray', vmin=0, vmax=255)

    print("Describe the problem with the image and your method/solution: \n")
    print("   TODO: add explanation    \n")

