import numpy as np
from scipy.signal import butter, filtfilt, lfilter, freqz
import collections

class LowpassRefiner:
    def __init__(self, f_cutoff, sampling_frequency, max_sample_history = 30, order=5):
        self._maximum_sample_history = max_sample_history
        self._cutoff_freq = f_cutoff
        self._samplig_frequency = sampling_frequency
        self._order = order

        self._filter_params = self._butter_lowpass()
        self._sample_list = collections.deque()

    def _butter_lowpass(self):
        nyq = 0.5 * self._samplig_frequency
        normal_cutoff = self._cutoff_freq / nyq
        b, a = butter(self._order, normal_cutoff, btype='low', analog=False, output = 'ba')
        return b, a

    def filter(self, sample):
        original_shape = sample.shape
        sample = sample.reshape(-1)

        if len(self._sample_list) < self._maximum_sample_history:
            self._sample_list.extend([np.zeros(sample.shape)] * (self._maximum_sample_history - 1))

        self._sample_list.appendleft(sample)
        self._sample_list.pop()

        x = np.stack(self._sample_list, axis = 1)
        y = filtfilt(self._filter_params[0], self._filter_params[1], x)
        y = y[:, 0]
        return y.reshape(original_shape)

