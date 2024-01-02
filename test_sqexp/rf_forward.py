"""
Forward prediction: design variables, frequency range --> is there a bandgap?

Author(s): Wei Chen (wchen459@gmail.com)
"""

import json
from tqdm import tqdm
import numpy as np
import scipy.io as sio
from joblib import dump
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import export_text
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import graphviz

import sys
sys.path.insert(0, '..')
from forward import combine_des_fre
from utils import create_dir
np.set_printoptions(threshold=sys.maxsize, edgeitems=30, linewidth=100000, precision=2)

EPSILON = 1e-6


def process_data(raw_data, des_var_names, absorb_threshold):
    
    data_x = raw_data['Xs'] # n_designs x n_des_var
    data_y = raw_data['Ys'] # n_designs x n_frequencies
    
    # Add frequencies to input data
    n_frequencies = data_y.shape[1]
    frequencies = np.linspace(0, 1, n_frequencies) # assuming frequencies are from 0 to 1
    data_x = combine_des_fre(data_x, frequencies) # n_designs x n_frequencies x (n_des_var+1)
    
    # Threshold the output
    data_y = data_y >= absorb_threshold
    
    return data_x, data_y, frequencies


def plot_prediction(y_test, y_pred, frequencies, figname):
    n_frequencies = frequencies.shape[0]
    indicators_pred = y_pred.reshape(-1, n_frequencies)
    indicators_test = y_test.reshape(-1, n_frequencies)
    n_designs = indicators_test.shape[0]
    plt.figure(figsize=(1*n_designs, 6))
    for i in range(indicators_test.shape[0]):
        for j, f in enumerate(frequencies):
            if indicators_test[i, j]:
                line1, = plt.plot(i, f, 'bo') # plot true high-absorbance frequencies
            if indicators_pred[i, j]:
                line2, = plt.plot(i+.1, f, 'ro') # plot predicted high-absorbance frequencies
    plt.xlabel('Test Design ID')
    plt.ylabel('Frequency (MHz)')
    line1.set_label('True High-Absorbance Frequencies')
    line2.set_label('Predicted High-Absorbance Frequencies')
    plt.legend(loc='upper left', bbox_to_anchor=(-0.015, 1.2), ncol=2, fancybox=True)
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()


if __name__ == "__main__":
    
    # Load configuration
    with open('config.json', 'r') as file:
        config = json.load(file)
    
    # Prepare data
    raw_data = sio.loadmat(config['data_path'])
    des_var_names = config['design_variable_names']
    feature_names = des_var_names + ['frequency']
    n_des_vars = len(des_var_names)
    absorb_threshold = config['absorbance_threshold']
    data_x, data_y, frequencies = process_data(raw_data, des_var_names, absorb_threshold)
    create_dir('results_rf')
    np.save('results_rf/frequencies.npy', frequencies)
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
    print('# designs for training:', x_train.shape[0])
    print('# designs for testing:', x_test.shape[0])
    n_features = len(feature_names)
    x_train = x_train.reshape(-1, n_features)
    x_test = x_test.reshape(-1, n_features)
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    # Get ranges of design variables
    des_var_bounds = np.vstack([np.min(x_train[:,:-1], axis=0), np.max(x_train[:,:-1], axis=0)]).T
    np.save('results_rf/des_var_bounds.npy', des_var_bounds)
    
    # # Over-sampling using SMOTE
    # sm = SMOTE(sampling_strategy=1, random_state=42)
    # x_res, y_res = sm.fit_resample(x_train, y_train)
    # print('Resampled class sizes:', sum(y_res), len(y_res) - sum(y_res))
    # x_train, y_train = x_res, y_res
    
    # Train a random forest
    clf = RandomForestClassifier(n_estimators=config['num_estimators'], criterion='gini', 
                                 min_samples_split=2, min_samples_leaf=1, random_state=0)
    # rf = RandomForestClassifier(n_estimators=config['num_estimators'], random_state=0)
    # parameters = {'criterion': ['gini', 'entropy', 'log_loss'],
    #               'min_samples_split': [2, 4, 8, 16, 32],
    #               'min_samples_leaf': [1, 2, 4, 8, 16]}
    # clf = GridSearchCV(rf, parameters, scoring='f1', n_jobs=-1, verbose=3)
    clf = clf.fit(x_train, y_train)
    
    # Evaluate the model on training data
    y_pred = clf.predict(x_train)
    f1 = f1_score(y_train, y_pred)
    print('F1 score on train:', f1)
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    print('Confusion matrix on train:', tn, fp, fn, tp)
    
    # Evaluate the model on test data
    y_pred = clf.predict(x_test)
    f1 = f1_score(y_test, y_pred)
    print('F1 score on test:', f1)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print('Confusion matrix on test:', tn, fp, fn, tp)
    lines = ['tn\tfp\tfn\ttp', '{}\t{}\t{}\t{}'.format(tn, fp, fn, tp), 'F1 score on test: {:.4f}'.format(f1)]
    with open('results_rf/forest_acc.txt', 'w') as f:
        f.write('\n'.join(lines))
        
    # Predict the bandgaps of test designs, and compare them with ground truth
    plot_prediction(y_test, y_pred, frequencies, 'results_rf/prediction.svg')
    
    # Save the forest
    dump(clf, 'results_rf/forest.joblib')
    
    # # Get trees from the forest
    # estimators = clf.estimators_
    # print('Plotting trees in the forest ...')
    # for i, dt in enumerate(tqdm(estimators)):
        
    #     # Plot tree
    #     dot_data = tree.export_graphviz(dt, out_file=None, 
    #                      feature_names=feature_names,
    #                      filled=True, rounded=True,  
    #                      special_characters=True)  
    #     graph = graphviz.Source(dot_data)
    #     graph.format = 'svg'
    #     graph.render('results_rf/tree_{}'.format(i), view=False)
        
    #     # Print tree in textual format
    #     tree_text = export_text(dt, feature_names=feature_names)
    #     with open('results_rf/tree_{}.txt'.format(i), 'w') as file:
    #         file.write(tree_text)
    
