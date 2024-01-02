"""
Check if generated designs meet targets

Author(s): Wei Chen (wchen459@gmail.com)
"""

import glob
import json
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

from build_data import test_func

import sys
sys.path.insert(0, '..')
from inverse import evaluate_satisfaction_discrete
from utils import create_dir

EPSILON = 1e-6

    
if __name__ == "__main__":
    
    # Load configuration
    with open('config.json', 'r') as file:
        config = json.load(file)
    
    # Load discretized frequencies
    frequencies = np.load('results_rf/frequencies.npy')
    
    # Get paths of experiment configurations and generated designs
    exp_config_paths = glob.glob('results_rf/exp_*/config.json')
    gen_design_paths = glob.glob('results_rf/exp_*/generated_designs.mat')
    n_exp = len(exp_config_paths)
    
    assert n_exp == len(gen_design_paths)
    
    # Define sample thresholds
    list_sample_threshold = np.linspace(0, 0.8, 5)
    # Define metrics to be plotted
    list_select_rate = [] # n_selected / n_total
    list_sat_rate = [] # n_satisfied / n_selected
    list_avg_score = []
    
    for sample_threshold in list_sample_threshold:
        print('Sample threshold:', sample_threshold)
        n_total = 0
        n_selected = 0
        n_satisfied = 0
        avg_score = 0
        for i in range(n_exp):
            # Get target frequency ranges from experiment configuration
            exp_config_path = exp_config_paths[i]
            with open(exp_config_path, 'r') as file:
                exp_config = json.load(file)
            target_fre_ranges = exp_config['target_frequency_ranges']
            # Get generated designs and their likelihood
            gen_design_path = gen_design_paths[i]
            designs = sio.loadmat(gen_design_path)['Xs']
            n_total += designs.shape[0]
            probabilities = sio.loadmat(gen_design_path)['Ps'].flatten()
            # Test if generated designs satisfy the target
            indicators_true = []
            for (a, b) in designs:
                y = test_func(frequencies, a, b)
                indicators_true.append(y >= config['absorbance_threshold'])
            indicators_true = np.array(indicators_true)
            recalls, _ = evaluate_satisfaction_discrete(target_fre_ranges, indicators_true, frequencies)
            # Calculate how many designs satisfy the target
            n_selected += np.sum(probabilities>=sample_threshold)
            n_satisfied += np.sum(np.logical_and(probabilities>=sample_threshold, recalls==1))
            avg_score += np.sum(recalls[probabilities>=sample_threshold])
        print('Satisfied designs: {}/{}'.format(n_satisfied, n_selected)) # based on hard satisfaction
        print('Average satisfaction: {}'.format(avg_score/n_selected)) # based on soft satisfaction
        
        list_select_rate.append(n_selected/n_total)
        list_sat_rate.append(n_satisfied/n_selected)
        list_avg_score.append(avg_score/n_selected)
        
    # Plot
    create_dir('validation')
    plt.figure(figsize=(5, 4))
    plt.plot(list_sample_threshold, list_select_rate, '^-.', label='Selection rate')
    plt.plot(list_sample_threshold, list_sat_rate, 'o--', label='Satisfaction rate')
    plt.plot(list_sample_threshold, list_avg_score, 's-', label='Average score')
    plt.axhline(y=1, ls=':')
    plt.legend()
    plt.xlabel('Sampling threshold')
    plt.ylabel('Metrics')
    plt.tight_layout()
    plt.savefig('validation/metrics.svg')
    plt.close()
            