"""
Check if generated designs meet targets

Author(s): Wei Chen (wchen459@gmail.com)
"""

import glob
import json
from joblib import load
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import seaborn as sns

from rf_inverse import plot_frequencies

import sys
np.set_printoptions(threshold=sys.maxsize, edgeitems=30, linewidth=100000, precision=2)
sys.path.insert(0, '..')
from forward import predict_frequencies
from inverse import evaluate_satisfaction_discrete


def get_val_data(exp_dir):
    exp_config_path = exp_dir + '/config.json'
    gen_design_path = exp_dir + '/generated_designs_with_responses.mat'
    # Get target frequency ranges from experiment configuration
    with open(exp_config_path, 'r') as file:
        exp_config = json.load(file)
    target_fre_ranges = exp_config['target_frequency_ranges']
    t = exp_config['absorbance_threshold']
    # Get generated designs and their likelihood
    designs = sio.loadmat(gen_design_path)['Xs']
    probabilities = sio.loadmat(gen_design_path)['Ps'].flatten()
    responses = sio.loadmat(gen_design_path)['Ys']
    # Test if generated designs satisfy the target
    indicators_true = responses >= config['absorbance_threshold']
    recalls, _ = evaluate_satisfaction_discrete(target_fre_ranges, indicators_true, frequencies)
    return target_fre_ranges, designs, probabilities, responses, recalls, t
    
    
if __name__ == "__main__":
    
    # Load configuration
    with open('config.json', 'r') as file:
        config = json.load(file)
    
    # Load discretized frequencies
    frequencies = np.load('results_rf/frequencies.npy')
    
    # Get paths of experiment configurations and generated designs
    exp_dirs = glob.glob('validation/exp_*')
    n_exp = len(exp_dirs)
    
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
            _, designs, probabilities, _, recalls, _ = get_val_data(exp_dirs[i])
            n_total += designs.shape[0]
            # Calculate how many designs satisfy the target
            n_selected += np.sum(probabilities>=sample_threshold)
            n_satisfied += np.sum(np.logical_and(probabilities>=sample_threshold, recalls==1))
            avg_score += np.sum(recalls[probabilities>=sample_threshold])
        print('Satisfied designs: {}/{}'.format(n_satisfied, n_selected)) # based on hard satisfaction
        print('Average satisfaction: {}'.format(avg_score/n_selected)) # based on soft satisfaction
        
        list_select_rate.append(n_selected/n_total)
        list_sat_rate.append(n_satisfied/n_selected)
        list_avg_score.append(avg_score/n_selected)
        
    # Plot metrics
    plt.figure(figsize=(6, 4))
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
    
    # Plot overall info
    all_recalls = np.array([])
    all_probs = np.array([])
    for i in range(n_exp):
        target_fre_ranges, designs, probabilities, responses, recalls, t = get_val_data(exp_dirs[i])
        all_recalls = np.append(all_recalls, recalls)
        all_probs = np.append(all_probs, probabilities)
        
        plt.figure(figsize=(6, 4))
        wl_lb, wl_ub = 380, 700
        wavelength = np.linspace(wl_lb, wl_ub, responses.shape[1])
        for j, response in enumerate(responses[np.argsort(probabilities)[-5:]]):
            plt.plot(wavelength, response, 'k-', alpha=0.5)
        for (start, end) in target_fre_ranges:
            start = start * (wl_ub-wl_lb) + wl_lb
            end = end * (wl_ub-wl_lb) + wl_lb
            plt.axvspan(start, end, facecolor='gray', alpha=0.5) # plot target bandgaps
        plt.axhline(t, ls=':', color='gray')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Absorbance')
        plt.tight_layout()
        plt.savefig('{}/generated_responses.svg'.format(exp_dirs[i]))
        plt.close()
        
        data = np.concatenate([designs, probabilities.reshape(-1,1), recalls.reshape(-1,1)], axis=1)
        np.savetxt('{}/generated_designs.csv'.format(exp_dirs[i]), data, delimiter=',')
    
    plt.figure(figsize=(6, 4))
    plt.scatter(all_probs, all_recalls, alpha=0.3)
    plt.xlabel('Likelihood estimation')
    plt.ylabel('Percentage target overlap')
    plt.tight_layout()
    plt.savefig('validation/likelihood_vs_overlap.svg')
    plt.close()
    
    probabilities_0 = all_probs[all_recalls<1]
    probabilities_1 = all_probs[all_recalls==1]
    data = {'Estimated likelihood': np.concatenate([probabilities_0, probabilities_1]),
            'Target satisfaction': ['No'] * len(probabilities_0) + ['Yes'] * len(probabilities_1)}
    df = pd.DataFrame(data)
    plt.figure(figsize=(6, 4))
    sns.kdeplot(
               data=df, x='Estimated likelihood', hue='Target satisfaction',
               fill=True, common_norm=False, palette='crest',
               alpha=.5, linewidth=0,
            )
    sns.kdeplot(data=all_probs, color='gray')
    plt.tight_layout()
    plt.savefig('validation/distribution.svg')
    plt.close()
    