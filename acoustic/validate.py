"""
Check if generated designs meet targets

Author(s): Wei Chen (wchen459@gmail.com)
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import seaborn as sns

import sys
np.set_printoptions(threshold=sys.maxsize, edgeitems=30, linewidth=100000, precision=2)
sys.path.insert(0, '..')
from inverse import evaluate_satisfaction

EPSILON = 1e-6


def get_val_data(exp_dir):
    exp_config_path = exp_dir + '/config.json'
    gen_design_path = exp_dir + '/generated_designs_{}.xlsx'.format(os.path.basename(exp_dir))
    # Get target bandgaps from experiment configuration
    with open(exp_config_path, 'r') as file:
        exp_config = json.load(file)
    target_bandgap = exp_config['target_bandgap']
    # Get generated designs and their likelihood
    designs = pd.read_excel(gen_design_path)
    designs.drop(designs[pd.isnull(designs['Design Type'])].index, inplace=True)
    # Get the actual bandgaps of generated designs
    bandgaps = []
    for _, (w1, c1, w2, c2, w3, c3) in designs[bandgap_column_names].iterrows():
        bandgaps.append([[c1-w1/2, c1+w1/2], [c2-w2/2, c2+w2/2], [c3-w3/2, c3+w3/2]])
    bandgaps = np.array(bandgaps)
    # Get generated designs and their likelihood
    probabilities = designs['Probability'].to_numpy()
    # Test if generated designs satisfy the target
    recalls, _ = evaluate_satisfaction(target_bandgap, bandgaps)
    return target_bandgap, designs, probabilities, bandgaps, recalls

    
if __name__ == "__main__":
    
    # Load configuration
    with open('config.json', 'r') as file:
        config = json.load(file)
        
    bandgap_column_names = ['Bandgap 1 Width (MHz) ', 'Bandgap 1 Center (MHz)', 
                            'Bandgap 2 Width (MHz)', 'Bandgap 2 Center (MHz)', 
                            'Bandgap 3 Width (MHz)', 'Bandgap 3 Center (MHz)']
    
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
            _, designs, probabilities, _, recalls = get_val_data(exp_dirs[i])
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
        target_bandgap, designs, probabilities, bandgaps, recalls = get_val_data(exp_dirs[i])
        all_recalls = np.append(all_recalls, recalls)
        all_probs = np.append(all_probs, probabilities)
        
        plt.figure(figsize=(6, 4))
        for j, bg in enumerate(bandgaps[np.argsort(probabilities)[-5:]]):
            for (start, end) in bg:
                plt.plot([start, end], [j, j], c='k')
        for (start, end) in target_bandgap:
            plt.axvspan(start, end, facecolor='gray', alpha=0.5) # plot target bandgaps
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Generated Design ID')
        plt.tight_layout()
        plt.savefig('{}/generated_bandgaps.svg'.format(exp_dirs[i]))
        plt.close()
    
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
    
        
        
        