"""
Inverse design: target bandgap --> design variables?

Author(s): Wei Chen (wchen459@gmail.com)
"""

import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from joblib import load
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import plotly.graph_objects as go

from dt_inverse import *

import sys
np.set_printoptions(threshold=sys.maxsize, edgeitems=30, linewidth=100000, precision=2)

EPSILON = 1e-6


if __name__ == "__main__":
    
    # Load configuration
    with open('config.json', 'r') as file:
        config = json.load(file)
    
    # Load the forest, frequency intervals, and design variable bounds
    clf = load('results_rf/forest.joblib')
    fre_intervals = np.load('results_rf/fre_intervals.npy')
    des_var_bounds = np.load('results_rf/des_var_bounds.npy')
    
    # Define target bandgap
    target_bandgap = config['target_bandgap']
    if target_bandgap == 'auto':
        fre_lb, fre_ub = fre_intervals.min(), fre_intervals.max()
        min_width = (fre_intervals[0, 1] - fre_intervals[0, 0]) * 2
        max_width = min_width * 5
        target_bandgap = random_target_fre_range(fre_lb, fre_ub, min_width, max_width)
    print('Target bandgap:', target_bandgap)
    
    # Check data satisfaction
    raw_data = pd.read_csv(config['data_path'], sep=',')
    bandgaps_data = get_bandgap_ranges(raw_data)
    recalls, _ = evaluate_satisfaction(target_bandgap, bandgaps_data)
    print('{}/{} feasible designs in data.'.format(sum(recalls==1), len(recalls)))
    
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
        des_var_ranges = inverse_design(dt, target_bandgap, fre_intervals[:,1], des_var_bounds, verbose=0)
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
            config['target_bandgap'] = target_bandgap
            json.dump(config, fout)
        
        # Save feasible design variable ranges for all the estimators
        with open('{}/list_des_var_ranges.json'.format(exp_dir), 'w') as fout:
            json.dump(list_des_var_ranges, fout)
            
        # Save designs and their probabilities
        gen_des_df = pd.DataFrame(designs, columns=des_var_names)
        gen_des_df['Probability'] = probabilities
        gen_des_path = '{}/generated_designs.csv'.format(exp_dir)
        gen_des_df.to_csv(gen_des_path, index=False)
        
        # Plot bandgaps of data
        plot_bandgaps(target_bandgap, None, None, '{}/data_bandgaps.svg'.format(exp_dir), ref_bandgaps=bandgaps_data)
            
        # Plot predicted bandgaps of generated designs
        indicators_pred, indicators_pred_proba = predict_frequencies(clf, designs, fre_intervals[:,1])
        plot_bandgaps(target_bandgap, indicators_pred, fre_intervals, 
                      '{}/generated_prediction.svg'.format(exp_dir), proba=indicators_pred_proba)
        
        # Get designs from data with satisfied target bandgaps
        satisfactions = (recalls == 1)
        designs_data = raw_data[des_var_names].to_numpy()
        designs_data = np.nan_to_num(designs_data)
        satisfied_designs_data = designs_data[satisfactions]
        unsatisfied_designs_data = designs_data[np.logical_not(satisfactions)]
                
        # # Plot design variable distribution (represented as a histogram)
        # print('Visualizing design space ...')
        # x, y, z, val = discretize_des_space(des_var_bounds, list_des_var_ranges)
        # fig = go.Figure(data=go.Volume(x=x, y=y, z=z, value=val,
        #                                 isomin=1e-4, isomax=1,
        #                                 opacity=0.1, # needs to be small to see through all surfaces
        #                                 surface_count=17, # needs to be a large number for good volume rendering
        #                                 ))
        # fig.add_trace(go.Scatter3d(x=satisfied_designs_data[:,0], y=satisfied_designs_data[:,1], z=satisfied_designs_data[:,2],
        #                             mode='markers', marker=dict(size=5, color='blue', opacity=0.8))
        #               )
        # fig.add_trace(go.Scatter3d(x=unsatisfied_designs_data[:,0], y=unsatisfied_designs_data[:,1], z=unsatisfied_designs_data[:,2],
        #                             mode='markers', marker=dict(size=5, color='red', opacity=0.8))
        #               )
        # fig.add_trace(go.Scatter3d(x=designs[:,0], y=designs[:,1], z=designs[:,2],
        #                             mode='markers', marker=dict(size=5, symbol='diamond', color='blue', opacity=0.8))
        #               )
        # fig.update_traces(showlegend=False)
        # fig.update_layout(
        #                 title='Design Space', 
        #                 autosize=False,
        #                 width=800, 
        #                 height=800,
        #                 margin=dict(l=65, r=50, b=65, t=90),
        #                 scene=dict(xaxis_title=des_var_names[0],
        #                             yaxis_title=des_var_names[1],
        #                             zaxis_title=des_var_names[2]),
        #                 )
        # fig.write_image('{}/design_space.png'.format(exp_dir))
        # fig.show()
    
    else:
        print('Cannot find designs that meet the target!')