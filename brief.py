import numpy as np
from scipy.signal import convolve2d

def brief_descriptor_function(img, keypoints_object, n=256, patch_size=9, sigma=1, mode='uniform', sample_seed=42):

    orientations = []
    keypoints = []
    for kp in keypoints_object:
        orientations.append(kp.angle)
        keypoints.append((kp.pt))

    keypoints = np.array(keypoints)
    orientations = np.array(orientations)
    random = np.random.RandomState(seed=sample_seed)

    # kernel = np.array([[1,2,1],
    #                    [2,4,2],
    #                    [1,2,1]])/16      # 3x3 Gaussian Window

    kernel = np.array([[1, 4,  7,  4,  1],
                       [4, 16, 26, 16, 4],
                       [7, 26, 41, 26, 7],
                       [4, 16, 26, 16, 4],
                       [1, 4,  7,  4,  1]])/273      # 5x5 Gaussian Window
    
    img = convolve2d(img, kernel, mode='same')

    if mode == 'normal':
        samples = (patch_size / 5.0) * random.randn(n*8)
        samples = np.array(samples, dtype=np.int32)
        samples = samples[(samples < (patch_size // 2)) & (samples > - (patch_size - 2) // 2)]
        pos1 = samples[:n * 2].reshape(n, 2)
        pos2 = samples[n * 2:n * 4].reshape(n, 2)
    elif mode == 'uniform':
        samples = random.randint(-(patch_size - 2) // 2 +1, (patch_size // 2), (n * 2, 2))
        samples = np.array(samples, dtype=np.int32)
        pos1, pos2 = np.split(samples, 2)

    rows, cols = img.shape

    
    # Using orientations

    # masking the keypoints with a safe distance from borders
    # instead of the patch_size//2 distance used in case of no rotations.
    distance = int((patch_size//2)*1.5)
    mask = (  ((distance - 1) < keypoints[:, 0])
            & (keypoints[:, 0] < (cols - distance + 1))
            & ((distance - 1) < keypoints[:, 1])
            & (keypoints[:, 1] < (rows - distance + 1)))

    keypoints = np.array(keypoints[mask], dtype=np.intp, copy=0)
    orientations = np.array(orientations[mask], copy=0)
    descriptors = np.zeros((keypoints.shape[0], n), dtype=int)

    for i in range(descriptors.shape[0]):
        angle = orientations[i]
        sin_theta = np.sin(angle)
        cos_theta = np.cos(angle)
        
        kr = keypoints[i, 1]
        kc = keypoints[i, 0]
        for p in range(pos1.shape[0]):
            pr0 = pos1[p, 0]
            pc0 = pos1[p, 1]
            pr1 = pos2[p, 0]
            pc1 = pos2[p, 1]
                
            # Rotation is based on the idea that:
            # x` = x*cos(th) - y*sin(th)
            # y` = x*sin(th) + y*cos(th)
            # c -> x & r -> y
            spr0 = round(sin_theta*pr0 + cos_theta*pc0)
            spc0 = round(cos_theta*pr0 - sin_theta*pc0)
            spr1 = round(sin_theta*pr1 + cos_theta*pc1)
            spc1 = round(cos_theta*pr1 - sin_theta*pc1)

            if img[kr + spr0, kc + spc0] < img[kr + spr1, kc + spc1]:
                descriptors[i, p] = 1
    
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
        
    return np.array(descriptors)
    