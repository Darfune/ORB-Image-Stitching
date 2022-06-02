import cv2 as cv
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

def fast_corners(image, height, width):
    thres = 30
    pixels_of_interest = ()
    for x in range(30,height-30):
        for y in range(30,width-30):
            above_thres = 0
            below_thres = 0
            if image[x-3, y+1] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x-3, y+1] < image[x, y] - thres: below_thres = below_thres + 1
            if image[x-3, y] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x-3, y] < image[x, y] - thres: below_thres = below_thres + 1
            if image[x-3, y-1] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x-3, y-1] < image[x, y] - thres: below_thres = below_thres + 1
            ##########################################################################
            if image[x-2, y+2] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x-2, y+2] < image[x, y] - thres: below_thres = below_thres + 1
            if image[x-2, y-2] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x-2, y-2] < image[x, y] - thres: below_thres = below_thres + 1
            ##########################################################################
            if image[x-1, y+3] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x-1, y+3] < image[x, y] - thres: below_thres = below_thres + 1
            if image[x, y+3] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x, y+3] < image[x, y] - thres: below_thres = below_thres + 1
            if image[x+1, y+3] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x+1, y+3] < image[x, y] - thres: below_thres = below_thres + 1
            ##########################################################################
            if image[x-1, y-3] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x-1, y-3] < image[x, y] - thres: below_thres = below_thres + 1
            if image[x, y-3] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x, y-3] < image[x, y] - thres: below_thres = below_thres + 1
            if image[x+1, y-3] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x+1, y-3] < image[x, y] - thres: below_thres = below_thres + 1
            ##########################################################################
            if image[x+2, y+2] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x+2, y+2] < image[x, y] - thres: below_thres = below_thres + 1
            if image[x+2, y-2] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x+2, y-2] < image[x, y] - thres: below_thres = below_thres + 1
            ##########################################################################
            if image[x+3, y+1] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x+3, y+1] < image[x, y] - thres: below_thres = below_thres + 1
            if image[x+3, y] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x+3, y] < image[x, y] - thres: below_thres = below_thres + 1
            if image[x+3, y-1] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x+3, y-1] < image[x, y] - thres: below_thres = below_thres + 1
            ##########################################################################
            if above_thres > 11 or below_thres > 11:
                pixels_of_interest += ([x,y],)

    return pixels_of_interest

def breif_descriptor(image, points):
    descriptors = []
    for point in points:
        descriptor = []
        for i in range(1, 16):
            for j in range(1, 16):
                if image[point[0],point[1]] < image[i,j]:
                    descriptor.append(1)
                else:
                    descriptor.append(0)

        descriptors.append(descriptor)
    return descriptors


if __name__ == '__main__':
    image = cv.imread("images/beach.jpg")
    image_2 = cv.imread("images/beach2.jpg")
    # image3 = cv.imread("images/beach3.jpg")
    image_grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_grayscale_2 = cv.cvtColor(image_2, cv.COLOR_BGR2GRAY)
    # cv.imshow("Result",image_grayscale)

    height, width = image_grayscale.shape
    # new_height = height/2
    # new_width = width/2
    # image = cv.resize(image,(new_height,new_width))

    # cv.imshow("Result",image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    pixels_of_interest_1 = fast_corners(image_grayscale, height, width)
    # print(pixels_of_interest_1)
    # for pixel in pixels_of_interest_1:
    #     print(f"{image_grayscale[pixel[0],pixel[1]]}\n")
    # corners_2 = fast_corners(image_grayscale_2, height, width)
    # print(corners_1)

    descriptors = breif_descriptor(image_grayscale, pixels_of_interest_1)
    for descriptor in descriptors:
        print(descriptor)
