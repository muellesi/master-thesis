import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.distance as dst
import seaborn as sn
from sklearn.decomposition import PCA

from app_framework.gesture_save_file import deserialize_to_gesture_collection




np.set_printoptions(suppress = True)


def element_diff(samples):
    if len(samples.shape) == 2:
        diff = np.diff(samples, axis = 0, append = 0)
        diff_diff = np.diff(diff, axis = 0, append = 0)
        diff_gtzero = diff[diff > 0]

        gt_m = np.median(diff_gtzero)
        mask_gt = np.logical_or(diff > 3 * gt_m, diff > 30)

        diff_ltzero = diff[diff < 0]
        lt_m = np.median(diff_ltzero)
        mask_lt = np.logical_or(diff < 3 * lt_m, diff < -30)

        diff[mask_gt] = 0
        diff[mask_lt] = 0

        return diff
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    save_file_name = "gesture_data_airdraw_7sample.json"

    gesture_data = []
    if os.path.exists(save_file_name):
        gesture_data = deserialize_to_gesture_collection(save_file_name)

    labels = ["Bolt", "Circle", "Rectangle", "Swipe Right", "Swipe Left", "Swipe Up", "Swipe Down", "Cross", "Play1", "Play2"]
    X = []
    Y = []
    for idx, gesture in enumerate(gesture_data):
        for sample in gesture.samples:
            sample_diff = element_diff(sample)
            X.append(sample_diff.reshape(-1))

            # X.append(sample.reshape(-1))
            Y.append(labels[idx])

    pre_pca = PCA(svd_solver = 'arpack')
    pre_pca.fit(np.stack(X))
    pca_factors = pre_pca.explained_variance_ratio_
    pca_factors_cumsum = pca_factors.cumsum()
    sn.set_context("talk")
    with sn.axes_style("darkgrid"):
        fig = plt.figure(figsize = (12, 6))
        ax = fig.add_subplot(111)
        sn.scatterplot(range(0, len(pca_factors)), pca_factors, ax = ax)
        sn.scatterplot(range(0, len(pca_factors_cumsum)), pca_factors_cumsum, ax = ax)
        fig.show()

    pca = PCA(svd_solver = 'arpack', n_components = 15)
    x_hat = pca.fit_transform(np.stack(X))

    Z = pca.inverse_transform(x_hat)

    print(pca.explained_variance_ratio_.cumsum())

    # print(np.array(X) - Z)

    distances = np.zeros((len(Y), len(Y)))

    dist_metrics = [dst.correlation, dst.cosine, dst.euclidean, dst.cityblock, dst.chebyshev, dst.minkowski,
                    dst.braycurtis, dst.canberra]
    pca_toggle = [True, False]

    for use_pca in pca_toggle:
        for dst_metric in dist_metrics:

            if not use_pca:
                x_hat = np.stack(X)
            for i in range(len(Y)):
                for j in range(len(Y)):
                    ref_sample = x_hat[i]
                    dist = dst_metric(ref_sample, x_hat[j])
                    distances[i, j] = dist

            df = pd.DataFrame(distances, index = Y, columns = Y)

            fig = plt.figure(figsize = (24, 24))
            ax = fig.add_subplot(111)
            sn.heatmap(df, annot = False, ax = ax, fmt = 'g', cbar = False)
            basename = "auswertung_rel"
            os.makedirs(basename, exist_ok = True)
            plt.savefig(os.path.join(basename, "pca_{}_dist_{}.png".format(str(use_pca), str(dst_metric.__name__))))
