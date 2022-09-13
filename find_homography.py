import numpy as np
import cv2

def homography_stitching(keypoints_train_img, keypoints_query_img, matches, reprojThresh):
 
    keypoints_train_img = np.float32([keypoint.pt for keypoint in keypoints_train_img])
    keypoints_query_img = np.float32([keypoint.pt for keypoint in keypoints_query_img])
    
 
    if len(matches) > 4:
        # construct the two sets of points
        points_train = np.float32([keypoints_train_img[m.queryIdx] for m in matches])
        points_query = np.float32([keypoints_query_img[m.trainIdx] for m in matches])
        
        # Calculate the homography between the sets of points
        (H, mask) = cv2.findHomography( points_query, points_train, cv2.RANSAC, 5)

        return (matches, H, mask)
    else:
        return None
