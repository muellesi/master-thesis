import cv2


class SimpleHandSegmenter:
    def __init__(self):
        pass

    def remove_background(self, depth_image):
        min, max, minloc, maxloc = cv2.minMaxLoc(depth_image)
        #_, depth_new = cv2.threshold(depth_image, thresh=(max-min)/2, maxval=0, type=cv2.THRESH_TOZERO_INV)
        _, depth_mask = cv2.threshold(depth_image, thresh=1000, maxval=(2**16 -1), type=cv2.THRESH_BINARY_INV)
        return depth_mask