import cv2 as cv

#open the images for the /images/set_1 folder using opencv
img1 = cv.imread('images/pada_images/palm_tree_1.jpg')
# img2 = cv.imread('images/set_1/right.jpg')


#convert the images to grayscale
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
# gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)


#create FAST detector 
fast = cv.FastFeatureDetector_create(threshold=80)
#find 200 and draw the keypoints
kp1 = fast.detect(gray1, 50)


img1 = cv.drawKeypoints(img1, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#create orb detector
orb = cv.ORB_create(nfeatures = 10,patchSize=10)


#display the images
cv.imshow('img1', img1)
cv.waitKey()
cv.destroyAllWindows()

#save the image
cv.imwrite('fast_true.png',img1)