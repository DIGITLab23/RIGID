"""
Inverse design: target frequency range --> design variables?

Author(s): Wei Chen (wchen459@gmail.com)
"""

import json
import uuid
from tqdm import tqdm
import numpy as np
import scipy.io as sio
from joblib import load
import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

import sys
sys.path.insert(0, '..')
from forward import predict_frequencies
from inverse import *
from build_data import test_func
from utils import create_dir, get_overlap
np.set_printoptions(threshold=sys.maxsize, edgeitems=30, linewidth=100000, precision=2)
  
    
def plot_frequencies(target_fre_ranges, indicators, frequencies, figname, proba=None, ref_indicators=None):
    n_designs = indicators.shape[0]
    cmap = plt.cm.jet
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    plt.figure(figsize=(1*n_designs, 6))
    for (start, end) in target_fre_ranges:
        plt.axhspan(start, end, facecolor='gray', alpha=0.5) # plot target bandgaps
    for i in range(n_designs):
        ys = frequencies[indicators[i]]
        if proba is None:
            plt.scatter([i]*len(ys), ys, color='k', marker='o')
        else:
            for j in range(len(ys)):
                p = proba[i][indicators[i]][j]
                plt.scatter(i, ys[j], color=cmap(norm(p)), marker='o')
    if ref_indicators is not None:
        assert ref_indicators.shape[0] == n_designs
        for i in range(n_designs):
            ys = frequencies[ref_indicators[i]]
            plt.scatter([i+.1]*len(ys), ys, color='k', marker='o')
    if proba is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label('Probability')
    plt.xlabel('Design ID')
    plt.ylabel('Frequency (MHz)')
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()


if __name__ == "__main__":
    
    # Load configuration
    with open('config.json', 'r') as file:
        config = json.load(file)
    
    # Load the forest, frequency intervals, and design variable bounds
    clf = load('results_rf/forest.joblib')
    frequencies = np.load('results_rf/frequencies.npy')
    des_var_bounds = np.load('results_rf/des_var_bounds.npy')
    
    # Define target high-absorbance frequencies
    target_fre_ranges = config['target_frequency_ranges']
    if target_fre_ranges == 'auto':
        fre_lb, fre_ub = frequencies.min(), frequencies.max()
        min_width = (fre_ub - fre_lb) / 50
        max_width = (fre_ub - fre_lb) / 20
        target_fre_ranges = random_target_fre_range(fre_lb, fre_ub, min_width, max_width)
    print('Target frequency ranges:', target_fre_ranges)
    
    # Check data satisfaction
    data = sio.loadmat(config['data_path'])
    indicators_data = data['Ys'] >= config['absorbance_threshold']
    recalls_data, _ = evaluate_satisfaction_discrete(target_fre_ranges, indicators_data, frequencies)
    print('{}/{} feasible designs in data.'.format(sum(recalls_data==1), len(recalls_data)))
    
    # Features
    des_var_names = config['design_variable_names']
    n_des_vars = len(des_var_names)
    
    # Get trees from the forest
    estimators = clf.estimators_
    n_estimators = len(estimators)
    
    print('Getting feasible regions for all trees ...')
    list_des_var_ranges = []
    for i, dt in enumerate(tqdm(estimators)):
        
        # Get the design variable ranges that satisfy the target bandgap
        des_var_ranges = inverse_design(dt, target_fre_ranges, frequencies, des_var_bounds, verbose=0)
        list_des_var_ranges.append(des_var_ranges)
             
    # Generate designs from the obtained design variable ranges
    print('Generating designs ...')
    n_designs = config['num_generated_designs']
    cat_var_indices = config['categorical_variable_indices']
    sample_threshold = config['sample_threshold']
    designs, probabilities = generate(n_designs, list_des_var_ranges, des_var_bounds, cat_var_indices)
    
    if len(designs) > 0:
    
        # Create a folder for this experiment
        exp_dir = 'results_rf/exp_' + str(uuid.uuid4())
        create_dir(exp_dir)
        with open('{}/config.json'.format(exp_dir), 'w') as fout:
            config['target_frequency_ranges'] = target_fre_ranges
            json.dump(config, fout)
        
        # Save feasible design variable ranges for all the estimators
        with open('{}/list_des_var_ranges.json'.format(exp_dir), 'w') as fout:
            json.dump(list_des_var_ranges, fout)
            
        # Save designs and their probabilities
        sio.savemat('{}/generated_designs.mat'.format(exp_dir), {'Xs': designs, 'Ps': probabilities})
    
        # Plot high-absorbance frequency ranges of data
        plot_frequencies(target_fre_ranges, indicators_data, frequencies, 
                         '{}/data_fre_ranges.svg'.format(exp_dir))
        
        # Plot predicted and true high-absorbance frequency ranges of generated designs
        indicators_pred, indicators_pred_proba = predict_frequencies(clf, designs, frequencies)
        indicators_true = []
        for (a, b) in designs:
            y = test_func(frequencies, a, b)
            indicators_true.append(y >= config['absorbance_threshold'])
        indicators_true = np.array(indicators_true)
        plot_frequencies(target_fre_ranges, indicators_pred, frequencies, 
                         '{}/generated_prediction.svg'.format(exp_dir), proba=indicators_pred_proba, ref_indicators=indicators_true)
        
        # Evaluate target satisfaction
        recalls, percent_sat = evaluate_satisfaction_discrete(target_fre_ranges, indicators_true, frequencies)
        msg = 'Probability: {}'.format(probabilities)
        msg += '\nRecall: {}'.format(recalls)
        msg += '\nPercentage target satisfaction: {}'.format(percent_sat)
        with open('{}/score.txt'.format(exp_dir), 'w') as f:
            f.write(msg)
        print(msg)
        
        # # Plot feasible data and generated designs
        # feasible_data = data['Xs'][recalls_data==1]
        # infeasible_data = data['Xs'][recalls_data!=1]
        # feasible_generated = designs[recalls==1]
        # infeasible_generated = designs[recalls!=1]
        # plt.figure(figsize=(8, 8))
        # plt.scatter(feasible_data[:,0], feasible_data[:,1], color='b', alpha=.5, marker='.', label='Feasible data')
        # plt.scatter(infeasible_data[:,0], infeasible_data[:,1], color='r', alpha=.5, marker='.', label='Infeasible data')
        # plt.scatter(feasible_generated[:,0], feasible_generated[:,1], color='b', alpha=probabilities[recalls==1], marker='D', label='Feasible generated')
        # plt.scatter(infeasible_generated[:,0], infeasible_generated[:,1], color='r', alpha=probabilities[recalls!=1], marker='D', label='Infeasible generated')
        # plt.legend()
        # plt.xlabel('a')
        # plt.ylabel('b')
        # plt.tight_layout()
        # plt.savefig('{}/design_space.svg'.format(exp_dir))
        # plt.close()
        
        # Plot estimated density function and feasible regions
        print('Plotting design space ...')
        rez = 100
        a = np.linspace(0, 1, rez)
        b = np.linspace(0, 1, rez)
        av, bv = np.meshgrid(a, b)
        designs = np.stack((av, bv), axis=-1).reshape(-1, 2)
        indicators_true = []
        probabilities = []
        for (a, b) in tqdm(designs):
            y = test_func(frequencies, a, b)
            indicators_true.append(y >= config['absorbance_threshold'])
            proba = evaluate_proba(np.array([a, b]), list_des_var_ranges)
            probabilities.append(proba)
        indicators_true = np.array(indicators_true)
        recalls, _ = evaluate_satisfaction_discrete(target_fre_ranges, indicators_true, frequencies)
        labels = (recalls == 1).reshape(rez, rez)
        probabilities = np.array(probabilities).reshape(rez, rez)
        plt.figure(figsize=(5, 4))
        cnt = plt.contourf(av, bv, probabilities, levels=np.linspace(0, 1, 1000), cmap='Greys')
        for c in cnt.collections:
            c.set_edgecolor('face')
        plt.colorbar(label='Likelihood function value', ticks=np.linspace(0, 1, 6, endpoint=True))
        plt.contour(av, bv, labels, levels=[0.5], colors='#c00000')
        plt.xlabel('a')
        plt.ylabel('b')
        formatted_target = np.around(np.array(target_fre_ranges), 2).tolist()
        plt.title('Target = {}'.format(formatted_target))
        plt.tight_layout()
        plt.savefig('{}/design_space.svg'.format(exp_dir))
        plt.savefig('{}/design_space.png'.format(exp_dir))
        plt.close()
        
        # import plotly.graph_objects as go
    	
        # fig = go.Figure(data=[go.Surface(z=probabilities)])
        # fig.update_layout(title='', autosize=False,
    		  #         width=500, height=500,
    		  #         margin=dict(l=65, r=50, b=65, t=90))
        # fig.update_layout(
    	   #  scene = dict(
        # 		xaxis = dict(visible=False),
        # 		yaxis = dict(visible=False),
        # 		zaxis =dict(visible=False)
        # 		)
    	   #  )
        # fig.update(layout_coloraxis_showscale=False)
        # fig.update_traces(showlegend=False)
        # fig.update(layout_showlegend=False)
        # fig.update_traces(showscale=False)
        # fig.show()
        # fig.write_image('likelihood3d.svg', scale=6)
    
    else:
        print('Cannot find designs that meet the target!')