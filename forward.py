"""
Functions for forward prediction

Author(s): Wei Chen (wchen459@gmail.com)
"""

import numpy as np


def combine_des_fre(designs, frequencies):
    n_designs = designs.shape[0]
    designs = np.tile(designs[:, np.newaxis], [1, len(frequencies), 1]) # n_designs x n_frequencies x n_des_var
    frequencies = np.tile(frequencies[np.newaxis, :, np.newaxis], [n_designs, 1, 1]) # n_designs x n_frequencies x 1
    x = np.concatenate([designs, frequencies], axis=-1) # n_designs x n_frequencies x 4
    return x


def predict_frequencies(clf, designs, frequencies):
    n_designs, n_des_vars = designs.shape
    x = combine_des_fre(designs, frequencies)
    x = x.reshape(-1, n_des_vars+1)
    y_pred = clf.predict(x)
    y_pred_proba = clf.predict_proba(x)[:, 1]
    indicators_pred = y_pred.reshape(n_designs, -1)
    indicators_pred_proba = y_pred_proba.reshape(n_designs, -1)
    return indicators_pred, indicators_pred_proba