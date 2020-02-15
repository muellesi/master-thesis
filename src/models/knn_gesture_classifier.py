import copy
from collections import deque

import numpy as np
from sklearn.neighbors import KNeighborsClassifier



class KNNClassifier():

    def __init__(self, k, batch_size, sample_shape, metric = None):
        self.k = k
        self.batch_size = batch_size

        self.train_data = []
        self.train_data_labels = []
        self.class_names = []

        self.sample_shape = sample_shape

        self.metric = metric
        self.knn = None

        self.current_data_frame = deque(maxlen = batch_size)
        for i in range(self.batch_size):
            self.current_data_frame.append(np.zeros(self.sample_shape))


    def push_sample(self, sample):
        self.current_data_frame.appendleft(copy.copy(sample))

    def push_samples(self, samples):
        for sample in samples:
            # since we want to maintain the order of the data, we use append instead of appendleft!
            self.current_data_frame.append(sample)

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
        for classname in self.train_data_labels:
            self.add_class(classname)
        self.__fit()


    def predict(self):
        assert self.knn is not None, "Please add training data before trying to predict!"
        return self.knn.predict([np.stack(self.current_data_frame).reshape(-1)])


    def kneighors_graph(self):
        assert self.knn is not None, "Please add training data before trying to predict kneighors_graph!"
        return self.knn.kneighbors_graph([np.stack(self.current_data_frame).reshape(-1)], mode = 'distance')


    def predict_proba(self):
        assert self.knn is not None, "Please add training data before trying to predict_proba!"
        return zip(self.train_data_labels, self.knn.predict_proba([np.stack(self.current_data_frame).reshape(-1)]))


    def __fit(self):
        if self.knn is not None:
            del self.knn

        add_settings = { }

        if self.metric is not None:
            add_settings["metric"] = self.metric

        self.knn = KNeighborsClassifier(n_neighbors = 3, **add_settings)

        self.knn.fit(self.train_data, self.train_data_labels)


    def reset_queue(self):
        for i in range(self.batch_size):
            self.current_data_frame.append(np.zeros(self.sample_shape))