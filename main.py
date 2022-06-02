import cv2 as cv
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

def fast_corners(image, height, width):
    thres = 30
    corners = np.zeros((height, width))
    # print(corners)
    for x in range(20,height-20):
        for y in range(20,width-20):
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
            if above_thres > 11 : corners[x, y] = 1
            elif below_thres > 11 : corners[x, y] = -1
    return corners

if __name__ == '__main__':
    image = cv.imread("images/beach.jpg")
    # image2 = cv.imread("images/beach2.jpg")
    # image3 = cv.imread("images/beach3.jpg")
    image_grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # cv.imshow("Result",image_grayscale)

    height, width = image_grayscale.shape
    # new_height = height/2
    # new_width = width/2
    # image = cv.resize(image,(new_height,new_width))

    # cv.imshow("Result",image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    corners = fast_corners(image_grayscale, height, width)
    print(corners)
