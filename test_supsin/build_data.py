"""
Build datasets using test functions

Author(s): Wei Chen (wchen459@gmail.com)
"""

import json
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

import sys
sys.path.insert(0, '..')
from utils import create_dir
np.set_printoptions(threshold=sys.maxsize, edgeitems=30, linewidth=100000, precision=2)


def test_func(x, a, b):
    return np.sin(np.pi * 2 * (x+a)) + np.sin(np.pi * 3 * (x+b))


if __name__ == "__main__":
    
    # Load configuration
    with open('config.json', 'r') as file:
        config = json.load(file)
    data_size = config['data_size']
    fre_rez = config['frequency_resolution']
    
    # Create a dataset
    x = np.linspace(0, 1, fre_rez)
    designs = []
    responses = []
    for i in range(data_size):
        a = np.random.rand()
        b = np.random.rand()
        y = test_func(x, a, b)
        indicators = y >= config['absorbance_threshold']
        designs.append([a, b])
        responses.append(indicators)
    designs = np.array(designs)
    responses = np.array(responses)
    
    # Save the dataset
    create_dir('data')
    sio.savemat(config['data_path'], {'Xs': designs, 'Ys': responses})
    
    # Plot
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(3+data_size*.1, 5), gridspec_kw={'width_ratios': [3, data_size*.1]})
    for i in range(data_size):
        a = designs[i, 0]
        b = designs[i, 1]
        indicators = responses[i]
        y = test_func(x, a, b)
        axs[0].plot(y, x)
        axs[0].axvline(config['absorbance_threshold'], ls=':', color='gray')
        axs[1].plot([i]*sum(indicators), x[indicators], '|')
    axs[0].set_xlabel('Response-y')
    axs[0].set_ylabel('Response-x')
    axs[1].set_xlabel('Design ID')
    plt.tight_layout()
    plt.savefig(config['data_path'].replace('mat', 'svg'))
    plt.close()
    
    
    
