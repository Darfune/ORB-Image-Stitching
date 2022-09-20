import numpy as np
import cv2
import os
from fast_detector import fast_detect
from orientation import corner_orientations
from brief import brief_descriptor_function
from find_matches import match
from find_keypoints import topN_kepoints, find_keypoints_across_octaves
from harris_score import find_harris_corners
from find_homography import homography_stitching
import concurrent.futures
from pattern import generate_pattern

def sortScore(val):
    return val[2]
all_keypoints = []

def keypoint_details_processing(pyramid_details):
    octave = pyramid_details[0]
    threshold = pyramid_details[1]
    image = pyramid_details[2]
    
    keypoints_of_image = fast_detect(image, threshold, octave)
    keypoints_of_image = find_harris_corners(image,1000.0,keypoints_of_image)
    
    keypoints_of_image = corner_orientations(image,keypoints_of_image)
    print("Finished keypoint detection")
    return keypoints_of_image
       

def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop top
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop top
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame



if __name__ == '__main__':
    os.system('clear')
    path_pada = "images/pada_images/"
    DOWNSCALE = 2
    N_LAYERS = 4
    threshold = 20
    images = []
    num_of_images = 0
    pattern = generate_pattern()
    
    all_descriptors = []
    all_keypoints = []
    for i in os.listdir(path_pada):
        print("Processing image {}".format(i))
        image_path = path_pada + i
        img = cv2.imread(image_path)
        images.append(img)
        num_of_images += 1
    keypoints_and_octaves = []
    
    for image in images:
        image_keypoints = ()
        
        gaussian_pyramid = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gaussian_pyramid.append([0,threshold,gray])
        layer = gray
        for i in range(1, N_LAYERS):
            downscale = gray
            for j in range(i):
                downscale = cv2.pyrDown(downscale)
            for j in range(i):
                downscale = cv2.pyrUp(downscale)
            
            gaussian_pyramid.append([i, threshold,downscale])


        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(keypoint_details_processing, gp) for gp in gaussian_pyramid]
            for future in concurrent.futures.as_completed(results):
                
                
                for kp in future.result():
                    
                    image_keypoints = image_keypoints + (cv2.KeyPoint(
                        x = float(kp.x),
                        y = float(kp.y),
                        size = 7 * (2**kp.octave),
                        angle = kp.orientation,
                        response = kp.score,
                        octave = kp.octave,
                        class_id = -1),)
                
        image_keypoints = topN_kepoints(image_keypoints, 300)
        image_with_keypoints = cv2.drawKeypoints(image, image_keypoints, None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('image_1', image_with_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('image_1.jpg', image_with_keypoints)

        all_keypoints.append(image_keypoints)
        all_descriptors.append(brief_descriptor_function(gray, image_keypoints, pattern = pattern))

    
    good_matches = match(images[0],images[1],all_keypoints[0],all_keypoints[1],all_descriptors[0],all_descriptors[1])
    M = homography_stitching(all_keypoints[0], all_keypoints[1], good_matches, reprojThresh=4)

    if M is None:
        print("Error!")
            
    else:
        (matches, Homography_Matrix, mask) = M
    


        print(Homography_Matrix)


        width = images[1].shape[1] + images[0].shape[1]
        
        h, w = images[0].shape[:2]
        print(h, w)

        height = max(images[1].shape[0], images[0].shape[0])

        result = cv2.warpPerspective(images[1], Homography_Matrix,  (width, height))
        #save the transformed image
        cv2.imwrite('transformed.jpg', result)

        i1x , i1y = images[0].shape[:2]
        i2x , i2y = result.shape[:2]
        for i in range(0, i1x):
            for j in range(0, i1y):
                try:
                    if(np.array_equal(images[0][j,i],np.array([0,0,0])) and  \
                        np.array_equal(result[j,i],np.array([0,0,0]))):
                        result[j,i] = [0, 0, 0]
                    else:
                        if(np.array_equal(result[j,i],[0,0,0])):
                            result[j,i] = images[0][j,i]
                        else:
                            if not np.array_equal(images[0][j,i], [0,0,0]):
                                bl,gl,rl = images[0][j,i]                               
                                result[j, i] = [bl,gl,rl]
                except:
                    pass

        cv2.imshow("stitched images", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("stitched_images.jpg", result)

        crop_result = trim(result)
        cv2.imshow("cropped image mix", crop_result)
        cv2.imwrite("finished.jpg", crop_result)
        cv2.waitKey()


    
        


