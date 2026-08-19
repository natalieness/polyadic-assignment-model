''' Functions to examine how well prediction fits data. '''

import numpy as np
from scipy.stats import spearmanr


def compare_order_correlation(predicted_test_neurons, og_key='site_orders', predicted_key='predicted_orders', verbose=True):
    ''' Per-neuron Spearman rank correlation between observed and predicted
    site orders - a softer metric than exact-match accuracy for a weak,
    continuous-valued signal. Neurons with no variation in either array
    (undefined correlation) score 0.0 rather than nan. '''
    u_test_neurons = [neuron['pre_id'] for neuron in predicted_test_neurons]

    correlation = []
    for neuron in predicted_test_neurons:
        og_orders = neuron[og_key]
        predicted_orders = neuron[predicted_key]
        if len(og_orders) != len(predicted_orders):
            raise ValueError("Observed and predicted orders must have the same length.")
        if len(og_orders) < 2 or np.std(og_orders) == 0 or np.std(predicted_orders) == 0:
            corr = 0.0
        else:
            corr = spearmanr(og_orders, predicted_orders).correlation
        correlation.append(corr)
    if verbose:
        print(f"Mean Spearman correlation across {len(predicted_test_neurons)} neurons: {np.mean(correlation):.4f}")
    return correlation, u_test_neurons


def compare_assigned_orders(predicted_test_neurons, og_key='site_orders', predicted_key='predicted_orders', verbose=True):
    u_test_neurons = [neuron['pre_id'] for neuron in predicted_test_neurons]

    accuracy = []
    for neuron in predicted_test_neurons:
        og_orders = neuron[og_key]
        predicted_orders = neuron[predicted_key]
        if len(og_orders) != len(predicted_orders):
            raise ValueError("Observed and predicted orders must have the same length.")
        correct = np.sum(og_orders == predicted_orders)
        total = len(og_orders)
        acc = correct / total
        accuracy.append(acc)
    if verbose:
        print(f"Mean accuracy across {len(predicted_test_neurons)} neurons: {np.mean(accuracy):.4f}")
    return accuracy, u_test_neurons


def get_target_co_occurrence(X):
    ''' X: assignment matrix of shape [sites, targets] '''
    cooc = X.T @ X
    # diagonal self pairs 
    self_pairs = (X * (X - 1) / 2).sum(axis=0)
    np.fill_diagonal(cooc, self_pairs)
    return cooc


# polyadic co-adj cosine similarity
def compare_co_targeting_similarity(predicted_test_neurons, og_key='assignment', predicted_key='predicted_assignments', verbose=True):
    u_test_neurons = [neuron['pre_id'] for neuron in predicted_test_neurons]

    similarity = []
    for neuron in predicted_test_neurons:
        og_assignment = neuron[og_key]
        predicted_assignment = neuron[predicted_key]
        if og_assignment.shape != predicted_assignment.shape:
            raise ValueError("Observed and predicted assignments must have the same shape.")
        og_cooc = get_target_co_occurrence(og_assignment)
        predicted_cooc = get_target_co_occurrence(predicted_assignment)
        # cosine similarity
        dot_product = np.sum(og_cooc * predicted_cooc)
        norm_og = np.sqrt(np.sum(og_cooc ** 2))
        norm_predicted = np.sqrt(np.sum(predicted_cooc ** 2))
        if norm_og == 0 or norm_predicted == 0:
            sim = 0.0
        else:
            sim = dot_product / (norm_og * norm_predicted)
        similarity.append(sim)
    if verbose:
        print(f"Mean co-targeting similarity across {len(predicted_test_neurons)} neurons: {np.mean(similarity):.4f}")
    return similarity