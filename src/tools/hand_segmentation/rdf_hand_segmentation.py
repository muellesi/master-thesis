from sklearn.tree import DecisionTreeClassifier
import random
import numpy as np

from .. import loggingutil

sample_pixels_per_image = 2000
training_data_location = r"E:\MasterDaten\Datasets\HOI\HOI_training_data\training_data"

class RdfHandSegmenter:
    def __init__(self):
        self.tree = None
        self.feature_number = 50
        self.offset_list = []
        self.logger = loggingutil.get_logger(__name__, do_file_logging=False)

    def load_tree(self):
        # load feature list and tree model
        pass

    def make_tree(self):
        self.tree = DecisionTreeClassifier()
        
    def load_training_data(self):
        import os
        import cv2
        
        depth_path = os.path.join(training_data_location, "depth")
        label_path = os.path.join(training_data_location, "label")
        for i in range(19000):
            depth_map = cv2.imread(os.path.join(depth_path, "{:06d}.png".format(i)), cv2.IMREAD_ANYDEPTH)
            label_map = cv2.imread(os.path.join(label_path, "{:06d}.png".format(i)), cv2.IMREAD_ANYDEPTH)
            self.logger.info("loading image files {} and {}".format(os.path.join(depth_path, "{:06d}.png".format(i)), os.path.join(label_path, "{:06d}.png".format(i))))

            yield (depth_map, label_map)

    def fit_tree(self):

        self.offset_list = self.generate_random_offset_list(self.feature_number, 50000, 50000)

        # load depth images
        # load label images
        # get random samples from depth image
        # apply all feature offsets to all samples
        
        samples = []
        labels = []
        img_index = 0
        for (depth, label) in self.load_training_data():

            for sample_num in range(sample_pixels_per_image):
                samples.append([]) # append new feature vector for current sample
                current_sample_num = len(samples) - 1
                sample_pos = self.random_coordinate(depth.shape[1] - 1,  depth.shape[0] - 1)

                for offset in self.offset_list:
                    feature = self.calculate_feature_value(depth, sample_pos, offset)
                    samples[current_sample_num].append(feature)
                

                labels.append(label[sample_pos[0]][sample_pos[1]] != 0)


        if self.tree is not None:
            self.tree.fit()
        else:
            raise TypeError("Tree must be initialized first!")

    def random_coordinate(self, max_x, max_y, min_x=0, min_y=0):
        return np.array([random.randint(min_y, max_y), random.randint(min_x, max_x)])

    def generate_random_offset_list(self, num, max_x, max_y):
        return np.array([[self.random_coordinate(max_x, max_y), self.random_coordinate(max_x, max_y)] for i in range(num)])
        
    def calculate_feature_value(self, depth_image, pos, theta, error_val=(2**16 - 1)):
        try:
            pos_depth = depth_image[pos[0]][pos[1]]
            offset1 = pos + (theta[0] / pos_depth)
            offset2 = pos + (theta[1] / pos_depth)
            depth_u_offset = depth_image[int(offset1[0])][int(offset1[1])]
            depth_v_offset = depth_image[int(offset2[0])][int(offset2[1])]
            return depth_u_offset - depth_v_offset

        except (IndexError, ZeroDivisionError, OverflowError, ValueError): # feature out of bounds or background/occluded pixel at pos
            return error_val