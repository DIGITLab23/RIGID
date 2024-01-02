"""
Functions for inverse design

Author(s): Wei Chen (wchen459@gmail.com)
"""

from tqdm import tqdm
from functools import reduce
import numpy as np

from utils import get_overlap

EPSILON = 1e-8


def random_target_fre_range(fre_lb, fre_ub, min_width, max_width):
    n_fre_ranges = np.random.choice([1, 2])
    while True:
        pts = np.random.uniform(fre_lb, fre_ub, size=n_fre_ranges*2)
        pts = np.sort(pts)
        widths = [pts[2*i+1] - pts[2*i] for i in range(n_fre_ranges)]
        if min(widths) > min_width and max(widths) < max_width:
            break
    target_fre_ranges = [[pts[2*i], pts[2*i+1]] for i in range(n_fre_ranges)]
    return target_fre_ranges


def satisfy_at_node(clf, node, sample):
    feature = clf.tree_.feature[node]
    if not np.isnan(sample[feature]) and sample[feature] > clf.tree_.threshold[node]:
        return 0
    else:
        return 1
        

def leaves_satisfying_condition(clf, sample, condition_feature):
    
    def dfs_binary_tree(clf, node):
        
        if clf.tree_.feature[node] < 0:
            relevant_leaves.append(node)
            return
    
        elif clf.tree_.feature[node] == condition_feature:
            if satisfy_at_node(clf, node, sample):
                dfs_binary_tree(clf, clf.tree_.children_left[node])
            else:
                dfs_binary_tree(clf, clf.tree_.children_right[node])
                
        else:
            dfs_binary_tree(clf, clf.tree_.children_left[node])
            dfs_binary_tree(clf, clf.tree_.children_right[node])
    
    relevant_leaves = []
    dfs_binary_tree(clf, 0)
    
    return relevant_leaves


def get_path(clf, leaf, verbose=0):
    
    if verbose > 1:
        print('Leaf split:', clf.tree_.value[leaf])
    
    def trace_up(clf, leaf):
        if leaf in clf.tree_.children_left:
            parent = np.where(clf.tree_.children_left == leaf)[0][0]
            if verbose > 1:
                print('x_{} <= {}'.format(clf.tree_.feature[parent], clf.tree_.threshold[parent]))
            path.append(parent)
            satisfactions.append(1)
            trace_up(clf, parent)
        elif leaf in clf.tree_.children_right:
            parent = np.where(clf.tree_.children_right == leaf)[0][0]
            if verbose > 1:
                print('x_{} > {}'.format(clf.tree_.feature[parent], clf.tree_.threshold[parent]))
            path.append(parent)
            satisfactions.append(0)
            trace_up(clf, parent)
        else:
            return
        
    path = []
    satisfactions = [] # <= (1) or > (0)
    trace_up(clf, leaf)
    
    return path, satisfactions


def intersect(range1, range2):
    left = max(range1[0], range2[0])
    right = min(range1[1], range2[1])
    if left < right:
        return [left, right]
    else:
        return 0


def get_feature_ranges(clf, path_list, satisfactions_list, initial_ranges, ignore_feature):
    n_paths = len(path_list)
    n_features = clf.tree_.n_features
    features = [feature for feature in range(n_features) if feature != ignore_feature]
    initial_ranges = initial_ranges.tolist()
    feature_ranges = [{feature: initial_ranges[feature] for feature in features} for i in range(n_paths)]
    for i, path in enumerate(path_list):
        satisfactions = satisfactions_list[i]
        for j, node in enumerate(path):
            feature = clf.tree_.feature[node]
            if feature != ignore_feature:
                satisfied = satisfactions[j]
                threshold = clf.tree_.threshold[node]
                if satisfied:
                    criterion = [-np.inf, threshold]
                else:
                    criterion = [threshold, np.inf]
                feature_ranges[i][feature] = intersect(feature_ranges[i][feature], criterion)
    return feature_ranges


def inverse_design(clf, target_fre_ranges, frequencies, des_var_bounds, verbose=0):
    
    # Overlap between target frequency ranges and frequencies
    overlap = get_overlap(np.array(target_fre_ranges, ndmin=3), frequencies).flatten()
    
    # Prepare inputs for tracing down the tree
    n_des_vars = des_var_bounds.shape[0]
    target_fre = frequencies[overlap][:, np.newaxis]
    inputs = np.concatenate([np.nan*np.ones((target_fre.shape[0], n_des_vars)), target_fre], axis=-1)
    
    # Trace down (using DFS) by checking criteria satisfaction given the target frequency ranges as the condition
    condition_feature = n_des_vars
    relevant_leaves_list = []
    for sample in inputs:
        relevant_leaves = leaves_satisfying_condition(clf, sample, condition_feature)
        relevant_leaves_list.append(relevant_leaves)
    if len(relevant_leaves_list) > 0:
        relevant_leaves_set = reduce(np.intersect1d, relevant_leaves_list)
        relevant_leaves_set = np.array(relevant_leaves_set)
    
        # Trace up from the positive leaves
        leaf_values = clf.tree_.value[relevant_leaves_set]
        mask = leaf_values[:,0,1] > 0 # consider all leaf nodes containing positive samples
        positive_leaves_set = relevant_leaves_set[mask]
        positive_leaves_proba = leaf_values[mask,0,1]/np.sum(leaf_values[mask,0,:], axis=-1)
        n_positive_leaves = len(positive_leaves_set)
        if n_positive_leaves > 0:
            path_list = []
            satisfactions_list = []
            for leaf in positive_leaves_set:
                path, satisfactions = get_path(clf, leaf, verbose=verbose)
                path_list.append(path)
                satisfactions_list.append(satisfactions)
            # Get design variable ranges based on paths
            des_var_ranges = get_feature_ranges(clf, path_list, satisfactions_list, des_var_bounds, condition_feature)
            # Add probability of each positive leaf
            for i in range(n_positive_leaves):
                des_var_ranges[i]['proba'] = positive_leaves_proba[i]
            if verbose > 1:
                print(des_var_ranges)
            return des_var_ranges
        else:
            if verbose:
                print('No leaves with positive samples!')
            return 0
    else:
        if verbose:
            print('No leaves satisfying target frequency ranges!')
        return 0
    
    
def check_feasibility(design, des_var_ranges):
    n_des_vars = len(design)
    n_ranges = len(des_var_ranges)
    feasibility = np.ones((n_ranges, n_des_vars))
    for i, r in enumerate(des_var_ranges):
        for feature in range(n_des_vars):
            if design[feature] < r[feature][0] or design[feature] > r[feature][1]:
                feasibility[i, feature] = False
                break
    feasibility = np.all(feasibility, axis=1)
    feas_idx = np.argmax(feasibility)
    weighted_feasibility = feasibility[feas_idx].astype(float) * des_var_ranges[feas_idx]['proba']
    return weighted_feasibility


def uniform_designs(des_var_bounds, cat_var_indices, n_designs=1):
    n_des_vars = des_var_bounds.shape[0]
    con_var_indices = [i for i in range(n_des_vars) if i not in cat_var_indices]
    designs = np.zeros((n_designs, n_des_vars))
    for i in cat_var_indices:
        designs[:, i] = np.random.choice(np.arange(des_var_bounds[i, 0], des_var_bounds[i, 1]+1), size=n_designs)
    designs[:, con_var_indices] = np.random.uniform(des_var_bounds[con_var_indices, 0], des_var_bounds[con_var_indices, 1],
                                                    size=(n_designs, len(con_var_indices)))
    return np.squeeze(designs)


def evaluate_proba(design, list_des_var_ranges):
    proba = 0
    for des_var_ranges in list_des_var_ranges:
        if des_var_ranges:
            weighted_feasibility = check_feasibility(design, des_var_ranges)
            proba += weighted_feasibility
    n_estimators = len(list_des_var_ranges)
    proba /= n_estimators
    return proba


def is_in_bounds(design, des_var_bounds):
    flag = np.logical_and(design >= des_var_bounds[:, 0], design <= des_var_bounds[:, 1])
    return np.all(flag)


def mcmc_updater(curr_state, curr_likeli,  likelihood, cat_var_indices, des_var_bounds):
    """ 
    Reference: 
        https://exowanderer.medium.com/metropolis-hastings-mcmc-from-scratch-in-python-c21e53c485b7
        
    Propose a new state and compare the likelihoods
    
    Given the current state (initially random), 
      current likelihood, the likelihood function, and 
      the transition (proposal) distribution, `mcmc_updater` generates 
      a new proposal, evaluate its likelihood, compares that to the current 
      likelihood with a uniformly samples threshold, 
    then it returns new or current state in the MCMC chain.

    Args:
        curr_state (float): the current parameter/state value
        curr_likeli (float): the current likelihood estimate
        likelihood (function): a function handle to compute the likelihood
        proposal_distribution (function): a function handle to compute the 
          next proposal state

    Returns:
        (tuple): either the current state or the new state
          and its corresponding likelihood
    """
    # Set step size (Reference: Automatic Step Size Selection in Random Walk Metropolis Algorithms, Todd L. Graves, 2011.)
    n_des_vars = des_var_bounds.shape[0]
    con_var_indices = [i for i in range(n_des_vars) if i not in cat_var_indices]
    proposal_state = curr_state.copy()
    stepsize = 2.4 * len(con_var_indices)**(-0.5) * (des_var_bounds[con_var_indices,1] - des_var_bounds[con_var_indices,0])
    # Generate a proposal state using the proposal distribution
    # Proposal state == new guess state to be compared to current
    proposal_state[con_var_indices] = np.random.normal(curr_state[con_var_indices], stepsize)
    if len(cat_var_indices) > 0:
        rand_cat_idx = np.random.choice(cat_var_indices)
        proposal_state[rand_cat_idx] = np.random.choice(np.arange(des_var_bounds[rand_cat_idx, 0], des_var_bounds[rand_cat_idx, 1]+1))

    # Calculate the acceptance criterion
    prop_likeli = likelihood(proposal_state)
    accept_crit = prop_likeli / (curr_likeli + EPSILON)

    # Generate a random number between 0 and 1
    accept_threshold = np.random.uniform(0, 1)

    # If the acceptance criterion is greater than the random number,
    # accept the proposal state as the current state
    if accept_crit > accept_threshold and is_in_bounds(proposal_state, des_var_bounds):
        return proposal_state, prop_likeli, 1
    
    # Else
    return curr_state, curr_likeli, 0


def generate(n_designs, list_des_var_ranges, des_var_bounds, cat_var_indices,
             initial_design=None, burnin=0.2):
    n_des_vars = des_var_bounds.shape[0]
    if initial_design is None:
        print('Deciding the initial design for MCMC ...')
        # Uniformly sample designs
        designs = uniform_designs(des_var_bounds, cat_var_indices, n_designs=10*n_des_vars)
        # Select the one with the highest likelihood as the initial design
        max_proba = 0
        for design in tqdm(designs):
            proba = evaluate_proba(design, list_des_var_ranges)
            if proba >= max_proba:
                initial_design = design
    print('Generating designs using MCMC ...')
    # The number of samples in the burn in phase
    n_burnin = int(burnin * n_designs)
    # Set the current state to the initial state
    curr_state = initial_design
    curr_likeli = evaluate_proba(curr_state, list_des_var_ranges)
    # Metropolis-Hastings with unique samples
    designs = []
    probabilities = []
    count = 0
    accepted_count = 0
    with tqdm(total=n_designs) as pbar:
        while accepted_count < n_designs:
            # The proposal distribution sampling and comparison
            #   occur within the mcmc_updater routine
            curr_state, curr_likeli, flag = mcmc_updater(
                curr_state=curr_state,
                curr_likeli=curr_likeli,
                likelihood=lambda x: evaluate_proba(x, list_des_var_ranges),
                cat_var_indices=cat_var_indices,
                des_var_bounds=des_var_bounds
            )
            count += 1
    
            # Append the current state to the list of samples
            if count > n_burnin and flag:
                # Only append after the burnin to avoid including
                #   parts of the chain that are prior-dominated
                designs.append(curr_state)
                probabilities.append(curr_likeli)
                accepted_count += 1
                pbar.update(1)
    return np.array(designs), np.array(probabilities)


def evaluate_satisfaction(target_fre_ranges, fre_ranges):
    n_designs, n_ranges_per_design = fre_ranges.shape[:2]
    n_target_fre_ranges = len(target_fre_ranges)
    target_total = 0
    for k, (start, end) in enumerate(target_fre_ranges):
        target_total += (end - start)
    recalls = []
    for i in range(n_designs):
        s = np.zeros((n_ranges_per_design, n_target_fre_ranges))
        for j, r in enumerate(fre_ranges[i]):
            for k, rt in enumerate(target_fre_ranges):
                r_inter = intersect(r, rt)
                if r_inter:
                    s[j, k] = r_inter[1] - r_inter[0]
        recalls.append(np.nansum(s)/target_total)
    recalls = np.array(recalls)
    percent_sat = (recalls==1).sum() / n_designs
    return recalls, percent_sat


def evaluate_satisfaction_discrete(target_fre_ranges, indicators, frequencies):
    n_designs = indicators.shape[0]
    overlap = get_overlap(np.array(target_fre_ranges, ndmin=3), frequencies)
    tp = np.logical_and(indicators, overlap)
    recalls = np.sum(tp, axis=1)/overlap.sum()
    percent_sat = (recalls==1).sum() / n_designs # percentage of designs satisfying the target
    return recalls, percent_sat
    
    