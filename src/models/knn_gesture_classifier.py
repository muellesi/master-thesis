import copy
from collections import deque

import numpy as np
from sklearn.neighbors import KNeighborsClassifier



class KNNClassifier():

    def __init__(self, k, batch_size, sample_shape):
        self.k = k
        self.batch_size = batch_size

        self.train_data = []
        self.train_data_labels = []
        self.class_names = []

        self.sample_shape = sample_shape

        self.knn = KNeighborsClassifier(n_neighbors = 3)

        self.current_data_frame = deque(maxlen = batch_size)
        for i in range(self.batch_size):
            self.current_data_frame.append(np.zeros(self.sample_shape))


    def push_sample(self, sample):
        self.current_data_frame.appendleft(copy.copy(sample))


    def save_current_as_train_data(self, label):
        self.add_class(label)
        self.train_data.append(np.array(self.current_data_frame).copy())
        self.train_data_labels.append(label)
        self.__fit()


    def add_class(self, class_name):
        if class_name not in self.class_names:
            self.class_names.append(class_name)


    def set_train_data(self, X, Y):
        self.train_data = X
        self.train_data_labels = Y
        self.__fit()


    def predict(self):
        return self.knn.predict([np.stack(self.current_data_frame).reshape(-1)])


    def kneighors_graph(self):
        return self.knn.kneighbors_graph([np.stack(self.current_data_frame).reshape(-1)], mode = 'distance')


    def predict_proba(self):
        return zip(self.train_data_labels, self.knn.predict_proba([np.stack(self.current_data_frame).reshape(-1)]))


    def __fit(self):
        self.knn.fit(self.train_data, self.train_data_labels)


    def reset_queue(self):
        for i in range(self.batch_size):
            self.current_data_frame.append(np.zeros(self.sample_shape))
