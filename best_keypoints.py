from ast import If
import cv2
import time


def find_best_keypoints(keypoints, amount=50):

    best_keypoints = ()

    if len(keypoints) > amount:
        while len(best_keypoints) < amount:
            max_score = -1
            for kp in keypoints:
                if kp.response > max_score:
                    max_score = kp.response
                    best_current_kp = kp
            best_keypoints = best_keypoints + (best_current_kp,)
            index = keypoints.index(best_current_kp)
            keypoints = keypoints[:index] + keypoints[index + 1:]
        return best_keypoints
    else:
        return keypoints
