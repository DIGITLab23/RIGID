"""
Author(s): Wei Chen (wchen459@gmail.com)
"""

import os
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})


def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        
    
def safe_remove(pathname):
    if os.path.isdir(pathname):
        shutil.rmtree(pathname)
    elif os.path.isfile(pathname):
        os.remove(pathname)


def show_wall_time(func):
    def wrapper(*args, **kwargs):
        begin = time.time()
        func(*args, **kwargs)
        end = time.time()
        print("Wall-clock time for {}: {:.1f}s".format(func.__name__, end-begin))
    return wrapper


def get_overlap(fre_ranges, frequencies):
    n_designs, n_fre_ranges = fre_ranges.shape[:2]
    n_fre = frequencies.shape[0]
    overlap = np.zeros((n_designs, n_fre, n_fre_ranges))
    for i in range(n_fre_ranges):
        if frequencies.ndim == 1:
            overlap[:, :, i] = np.logical_and(fre_ranges[:, i, :1] <= frequencies[np.newaxis, :], 
                                              fre_ranges[:, i, 1:] >= frequencies[np.newaxis, :])
        else:
            overlap[:, :, i] = np.logical_and(fre_ranges[:, i, :1] <= frequencies[:, :1].T, 
                                              fre_ranges[:, i, 1:] >= frequencies[:, 1:].T)
    overlap = np.any(overlap, axis=-1)
    return overlap
