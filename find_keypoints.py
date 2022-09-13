

def find_keypoints_across_octaves(keypoints, octaves):
    print("Finding keypoints across octaves")
    to_return_keypoints = ()
    for kp_one in keypoints:
        if kp_one.octave == 0:
            count = 0
            kp_family = []
            best_of_octave = kp_one
            for kp_two in keypoints:
                kp_family.append(kp_one)
                if kp_one.pt == kp_two.pt and kp_one.octave != kp_two.octave and kp_one.angle == kp_two.angle:
                    kp_family.append(kp_two)
                    count += 1
                    if kp_one.response < kp_two.response:
                        best_of_octave = kp_two
            if count == octaves:
                to_return_keypoints += (best_of_octave,)
                kp_temp = list(keypoints)
                for kp in kp_family:
                    kp_temp.remove(kp)
                kp_family.clear()
                keypoints = tuple(kp_temp)
    print("Finished finding keypoints across octaves")
    return to_return_keypoints

def topN_kepoints(keypoints, n):
    print("Finding top {} keypoints".format(n))
    size = len(keypoints)
    keypoints = list(keypoints)
    for index in range(size):
        min_index = index

        for j in range(index + 1, size):
            if keypoints[j].response > keypoints[min_index].response:
                min_index = j

        keypoints[index], keypoints[min_index] = keypoints[min_index], keypoints[index]
    print("Finished finding top {} keypoints".format(n))
    return tuple(keypoints[:n])