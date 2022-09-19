import numpy as np
from scipy.signal import convolve2d


def brief_descriptor_function(img , keypoints_object, pattern, n=256, mode='uniform'):

    
    
    kernel = np.array([[1, 4,  7,  4,  1],
                       [4, 16, 26, 16, 4],
                       [7, 26, 41, 26, 7],
                       [4, 16, 26, 16, 4],
                       [1, 4,  7,  4,  1]])/273      # 5x5 Gaussian Window
    
    img = convolve2d(img, kernel, mode='same')

    descriptors = np.zeros((len(keypoints_object), n), dtype=int)
    for ek, kp in enumerate(keypoints_object):
        cosangle = np.cos(kp.angle)
        sinangle = np.sin(kp.angle)
        for i in range(0, len(pattern)):
            y1 = pattern[i][0]
            x1 = pattern[i][1]
            y2 = pattern[i][2]
            x2 = pattern[i][3]

            spy0 = round(sinangle*y1 + cosangle*x1)
            spx0 = round(cosangle*y1 - sinangle*x1)
            spy1 = round(sinangle*y2 + cosangle*x2)
            spx1 = round(cosangle*y2 - sinangle*x2)
            
            if img[int(kp.pt[1]) + spy0, int(kp.pt[0]) + spx0] < img[int(kp.pt[1]) + spy1, int(kp.pt[0]) + spx1]:
                descriptors[ek][i] = 1
    end_descriptors =  bits_to_bytes(descriptors)

    return end_descriptors

def bits_to_bytes(desc):
    descriptors = ()
    for byte in desc:
        descriptor = []
        for bit_pair in range(0,32):
            desc_byte = 0
            for bit in range(0,8):
                desc_byte += (byte[bit + bit_pair] * (2 ** bit))
            descriptor.append(desc_byte)
        descriptors = descriptors + (descriptor,)
        
    return np.array(descriptors).astype(np.uint8)

