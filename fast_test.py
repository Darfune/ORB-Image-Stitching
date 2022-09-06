import cv2 as cv

#open the images for the /images/set_1 folder using opencv
img1 = cv.imread('images/set_1/left.jpg')
img2 = cv.imread('images/set_1/right.jpg')


#convert the images to grayscale
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

#create orb detector
orb = cv.ORB_create(nfeatures = 10,patchSize=10)

#find the keypoints and descriptors with orb
kp1 = orb.detect(gray1, None)
kp2 = orb.detect(gray2, None)

#draw keypoints on the images
img1 = cv.drawKeypoints(gray1, kp1, None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2 = cv.drawKeypoints(gray2, kp2, None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

for kp in kp1:
    print(kp.pt[0], " ", kp.size, " ", kp.octave)
    print('----------------')

#display the images
cv.imshow('img1', img1)
cv.imshow('img2', img2)
cv.waitKey(10000000)
cv.destroyAllWindows()