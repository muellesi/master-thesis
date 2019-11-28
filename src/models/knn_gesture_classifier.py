import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from collections import deque
import json
import os



class KNNClassifier():

    def __init__(self, k, batch_size, sample_shape, save_dir="data/", save_name='knn.json'):
        self.k = k
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.save_name = save_name

        self.train_data = []
        self.train_data_labels = []
        self.class_names = []

        if self.load_state():
            self.__fit()

        self.knn = KNeighborsClassifier(n_neighbors=3)

        self.current_data_frame = deque(maxlen=batch_size)
        while len(self.current_data_frame) < batch_size:
            self.current_data_frame.append(np.zeros(sample_shape))


    def push_sample(self, sample):
        self.current_data_frame.append(list(sample).copy())


    def save_current_as_train_data(self, label):
        self.add_class(label)
        self.train_data.append(np.array(self.current_data_frame).copy())
        self.train_data_labels.append(label)
        self.__fit()


    def add_class(self, class_name):
        if class_name not in self.class_names:
            self.class_names.append(class_name)


    def predict(self):
        return self.knn.predict(self.current_data_frame)


    def kneighors_graph(self):
        return self.knn.kneighbors_graph(self.current_data_frame, mode='distance')


    def predict_proba(self):
        return self.knn.predict_proba(self.current_data_frame)


    def __fit(self):
        self.knn.fit(self.train_data, self.train_data_labels)


    def save_state(self):
        save_dir = os.path.abspath(self.save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_data = {
                "data": zip(self.train_data, self.train_data_labels),
                "classes": self.class_names
                }

        save_file = os.path.join(self.save_dir, self.save_name)
        with open(save_file, "w") as f:
            json.dump(save_data, f)


    def load_state(self, save_file=None):
        save_file = os.path.join(self.save_dir, self.save_name) if save_file is None else save_file
        if os.path.exists(save_file):
            with open(save_file, "r") as f:
                save_data = json.load(f)
            if save_data:
                self.class_names = save_data['classes']
                train_data = save_data['data']
                train_data = list(zip(*train_data))
                self.train_data = train_data[0]
                self.train_data_labels = train_data[1]
                return True
        return False


    def __del__(self):
        self.save_state()
