class keypoint:
    def __init__(self, x, y, score = None, orientation = None, octave = None):
        self.x = x
        self.y = y
        self.score = score
        self.orientation = orientation
        self.octave = octave
