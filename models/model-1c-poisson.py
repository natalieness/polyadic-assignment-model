''' Model version 1C (Poisson variant): Predict polyadic degree of a synapse based on location features.

E[k | compartment, normalized cumulative distance from axo/den split, n_sites], fit as a Poisson ordinal regression.

Inputs for each presynaptic neuron: # presynapses, location of presynapses, distribution of polyadic degrees, postsynaptic partners and frequency

Learned parameters: polyadic degree

'''
import numpy as np
import ast
import pandas as pd
from collections import Counter
from itertools import chain

from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import lil_matrix
from scipy.stats import poisson
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split

from validate_assignment import validate_assignment
from validate_predictions_functions import compare_assigned_orders, compare_co_targeting_similarity, compare_order_correlation

#%%
def fit_order_location_model(training_locations, training_orders, training_n_sites):
    """
    Learn E[polyadic order | active-zone location, neuron size] via
    Poisson regression (order is a count, not an unordered class label).

    training_n_sites must be aligned row-for-row with training_locations/
    training_orders: each row holds the total number of presynaptic sites
    belonging to the neuron that row came from.

    Training data should come from held-out neurons, hemispheres,
    or brains rather than the test example itself.
    """
    training_locations = np.asarray(training_locations, dtype=float)
    training_n_sites = np.asarray(training_n_sites, dtype=float).reshape(-1, 1)
    # column 0 = compartment_id (categorical: axon/dendrite/linker),
    # column 1 = compartment_relative_dist, column 2 = n_sites.
    training_features = np.hstack([training_locations, training_n_sites])

    # compartment_id is categorical, not an ordinal float, and standardizing
    # its one-hot dummy would blow the rare category up into a huge z-score
    # (it's ~90/10 axon/dendrite here) - so it bypasses scaling entirely and
    # only the genuinely continuous columns get standardized, before the
    # polynomial expansion builds interaction terms across all of them.
    preprocessing = ColumnTransformer([
        ("compartment", OneHotEncoder(drop="first", handle_unknown="ignore"), [0]),
        ("continuous", StandardScaler(), [1, 2]),
    ])

    model = Pipeline([
        ("preprocess", preprocessing),
        ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
        ("regressor", PoissonRegressor(alpha=1.0, max_iter=1000)), # alpha og 1.0
    ])

    model.fit(
        training_features,
        np.asarray(training_orders, dtype=float),
    )

    return model

def fit_order_model_from_neurons(train_neurons):
    training_locations = np.concatenate(
        [neuron["site_locations"]for neuron in train_neurons],
        axis=0,
    )

    training_orders = np.concatenate(
        [neuron["site_orders"]for neuron in train_neurons],
        axis=0,
    )

    training_n_sites = np.concatenate(
        [
            np.full(neuron["site_locations"].shape[0], neuron["site_locations"].shape[0], dtype=float)
            for neuron in train_neurons
        ],
        axis=0,
    )

    return fit_order_location_model(
        training_locations=training_locations,
        training_orders=training_orders,
        training_n_sites=training_n_sites,
    )


def allocate_orders_using_location(site_locations, order_histogram, order_model):
    """
    Assign a fixed order histogram to sites using a Poisson model of
    order ~ location + neuron size.

    The histogram is preserved exactly. Because scores come from
    poisson.logpmf evaluated at the requested order against each site's
    predicted rate, any nonnegative integer order can be scored, even
    one that never appeared in the training data.
    """
    site_locations = np.asarray(site_locations, dtype=float)
    number_of_sites = site_locations.shape[0]

    order_slots = []
    for order, count in order_histogram.items():
        order_slots.extend([int(order)] * int(count))

    if len(order_slots) != number_of_sites:
        raise ValueError("Order histogram must contain one order per site.")

    order_slots = np.asarray(order_slots, dtype=int)

    n_sites_column = np.full((number_of_sites, 1), number_of_sites, dtype=float)
    features = np.hstack([site_locations, n_sites_column])

    predicted_mu = order_model.predict(features)
    predicted_mu = np.clip(predicted_mu, a_min=1e-10, a_max=None)

    # score_matrix[site, slot] = log P(order_slots[slot] | site's predicted mu)
    score_matrix = poisson.logpmf(order_slots[np.newaxis, :], predicted_mu[:, np.newaxis])

    site_indices, slot_indices = linear_sum_assignment(-score_matrix)

    assigned_orders = np.empty(number_of_sites, dtype=int)
    assigned_orders[site_indices] = order_slots[slot_indices]

    return assigned_orders


def construct_random_feasible_assignment(
    site_orders,
    target_counts,
    random_seed=None,
    allowed=None,
):
    """
    Construct one randomized feasible assignment.

    This preserves:
    - assigned site orders;
    - target frequencies;
    - optional spatial possibility constraints.

    Random objective coefficients select a feasible solution.
    """
    site_orders = np.asarray(site_orders, dtype=int)
    target_counts = np.asarray(target_counts, dtype=int)

    number_of_sites = len(site_orders)
    number_of_targets = len(target_counts)

    if site_orders.sum() != target_counts.sum():
        raise ValueError(
            "Site orders and target counts imply different numbers "
            "of total contacts."
        )

    if allowed is None:
        allowed = np.ones((number_of_sites, number_of_targets), dtype=bool)
    else:
        allowed = np.asarray(allowed, dtype=bool)

    if allowed.shape != (number_of_sites, number_of_targets):
        raise ValueError("allowed has the wrong shape.")

    number_of_variables = number_of_sites * number_of_targets
    constraints = lil_matrix(
        (number_of_sites + number_of_targets, number_of_variables),
        dtype=float,
    )

    # Preserve each site's order.
    for site in range(number_of_sites):
        start = site * number_of_targets
        stop = (site + 1) * number_of_targets
        constraints[site, start:stop] = 1.0

    # Preserve each target's contact count.
    for target in range(number_of_targets):
        constraints[
            number_of_sites + target,
            target::number_of_targets,
        ] = 1.0

    required_sums = np.concatenate([site_orders, target_counts]).astype(float)
    rng = np.random.default_rng(random_seed)

    # No biological score: random costs choose among feasible assignments.
    objective = rng.normal(size=number_of_variables)

    # A site may contact the same target more than once, so each cell is
    # bounded by the smaller of its site's order and its target's count
    # (the tightest bound that never excludes a feasible assignment),
    # not capped at 1.
    upper_bounds = np.minimum.outer(site_orders, target_counts) * allowed

    result = milp(
        c=objective,
        integrality=np.ones(number_of_variables, dtype=int),
        bounds=Bounds(
            np.zeros(number_of_variables),
            upper_bounds.astype(float).ravel(),
        ),
        constraints=LinearConstraint(
            constraints.tocsr(),
            required_sums,
            required_sums,
        ),
    )

    if not result.success:
        raise RuntimeError(f"No feasible assignment found: {result.message}")

    assignment = (
        np.rint(result.x)
        .astype(int)
        .reshape(number_of_sites, number_of_targets)
    )

    validate_assignment(assignment, site_orders, target_counts)

    return assignment

# prepare neurons

def prepare_neuron(pre_id, site_locations, assignment, target_ids=None):
    site_locations = np.asarray(site_locations, dtype=float)
    assignment = np.asarray(assignment, dtype=int)

    if site_locations.ndim != 2 or assignment.ndim != 2:
        raise ValueError("Inputs must be 2D arrays.")

    if site_locations.shape[0] != assignment.shape[0]:
        raise ValueError("Site counts do not match.")

    target_ids = (
        np.arange(assignment.shape[1])
        if target_ids is None
        else np.asarray(target_ids)
    )

    if len(target_ids) != assignment.shape[1]:
        raise ValueError("Need one target ID per assignment column.")

    return {
        "pre_id": pre_id,
        "site_locations": site_locations,
        "assignment": assignment,
        "site_orders": assignment.sum(axis=1),
        "target_counts": assignment.sum(axis=0),
        "target_ids": target_ids,
    }



# get site locations for each presynaptic site in the connector table

def prep_neurons_from_connector_table(con, nodes, all_neurons):
    # filter nodes to just presynaptic sites
    presyn_nodes = con['presynaptic_to_node'].unique()
    nodes_presyn = nodes[nodes['node_id'].isin(presyn_nodes)]

    # to get node features - get one hot encoded axon/dendrite/other compartment and scale cumulative distance by total length of either compartment
    axon_length_per_neuron = nodes[nodes['compartment'] == 'axon'].groupby('skeleton_id')['cum_dist'].max().to_dict()
    dendrite_length_per_neuron = nodes[nodes['compartment'] == 'dendrite'].groupby('skeleton_id')['cum_dist'].max().to_dict()

    comp_to_comp_id = {'axon': 0, 'dendrite': 1, 'linker': 0.5}
    node_to_comp_id = nodes_presyn.set_index('node_id')['compartment'].map(comp_to_comp_id).to_dict()
    nodes_presyn['compartment_id'] = nodes_presyn['node_id'].map(node_to_comp_id)
    nodes_presyn['compartment_relative_dist'] = nodes_presyn.apply(lambda row: row['cum_dist'] / axon_length_per_neuron[row['skeleton_id']] if row['compartment'] == 'axon' else row['cum_dist'] / dendrite_length_per_neuron[row['skeleton_id']], axis=1)

    node_to_cumdist = nodes_presyn.set_index('node_id')['compartment_relative_dist'].to_dict()

    node_to_features = dict(zip(nodes_presyn['node_id'], nodes_presyn[['compartment_id', 'compartment_relative_dist']].values.tolist()))


    con['site_locations'] = con['presynaptic_to_node'].map(node_to_features)

    # get a vector of target_ids per presynaptic site
    target_ids = con.groupby('presynaptic_to')['postsynaptic_to'].apply(lambda x: list(set(chain.from_iterable(x)))).to_dict()
    con['target_ids'] = con['presynaptic_to'].map(target_ids)

    # get assignments for each presynaptic site - needs to be of length target_ids, with each entry being the number of times it is postsynaptic at this site
    con['assignment'] = con.apply(lambda row: np.array([row['postsynaptic_to'].count(t) for t in row['target_ids']]), axis=1)


    # for all neurons
    neurons = []
    index_to_neuron = {}
    for e, (pre_id, group) in enumerate(con.groupby('presynaptic_to')):
        site_locations = np.stack(group['site_locations'].values)
        assignment = np.stack(group['assignment'].values)
        target_ids = group['target_ids'].values[0]
        neuron = prepare_neuron(pre_id, site_locations, assignment, target_ids)
        neurons.append(neuron)
        index_to_neuron[e] = pre_id
    return neurons, index_to_neuron, node_to_features

def predict(test_neurons, order_model, seed=42):
    predicted_test_neurons = test_neurons.copy()
    for e, test_neuron in enumerate(test_neurons):
        observed_orders = test_neuron["site_orders"]
        test_order_histogram = dict(Counter(observed_orders.tolist()))
        assigned_orders = allocate_orders_using_location(
            site_locations=test_neuron['site_locations'],
            order_histogram=test_order_histogram,
            order_model=order_model,
        )
        predicted_test_neurons[e]['predicted_orders'] = assigned_orders

        random_assignment = construct_random_feasible_assignment(
            site_orders=assigned_orders,
            target_counts=test_neuron['target_counts'],
            random_seed=seed,
        )
        predicted_test_neurons[e]['predicted_assignments'] = random_assignment
    return predicted_test_neurons

def get_null(test_neurons, seed=42):
    ''' Works like predict but returns a random site order assignment '''
    rng = np.random.default_rng(seed)
    null_test_neurons = test_neurons.copy()
    for e, test_neuron in enumerate(null_test_neurons):
        observed_orders = test_neuron["site_orders"]
        assigned_orders = rng.permutation(observed_orders)
        null_test_neurons[e]['null_orders'] = assigned_orders

        random_assignment = construct_random_feasible_assignment(
            site_orders=assigned_orders,
            target_counts=test_neuron['target_counts'],
            random_seed=seed,
        )
        null_test_neurons[e]['null_assignments'] = random_assignment
    return null_test_neurons



#%% run 
# load connector table
con = pd.read_csv('../data/con_short_2026.csv')
con['postsynaptic_to'] = con['postsynaptic_to'].apply(ast.literal_eval)
con['presynaptic_to'] = con['presynaptic_to'].astype(int)
con['presynaptic_to_node'] = con['presynaptic_to_node'].astype(int)
nodes = pd.read_csv('../data/nodes_manual_wdendrites_euclid.csv')
all_neurons = pd.read_csv('../data/all_neurons.csv') # get all neurons of interest
all_neurons = all_neurons['skeleton_id'].tolist()

neurons, index_to_neuron, node_to_features = prep_neurons_from_connector_table(con, nodes, all_neurons)

#%% RUN ON RANDOM TRAIN/TEST SPLIT OF NEURONS

# neus = neurons.copy()
# neus = neus[:100]

# # split neurons into training and test sets
# # either by percentage
# train_neurons, test_neurons = train_test_split(
#     neus,
#     test_size=0.20,
#     random_state=42,
# )
# print(f'Training neurons: {len(train_neurons)}, Test neurons: {len(test_neurons)}')
# order_model = fit_order_model_from_neurons(train_neurons)

# # apply to all test neurons
# predicted_test_neurons = predict(test_neurons, order_model, seed=42)

# accuracy, u_test_neurons = compare_assigned_orders(predicted_test_neurons, og_key='site_orders', predicted_key='predicted_orders', verbose=True)
# similarity = compare_co_targeting_similarity(predicted_test_neurons, og_key='assignment', predicted_key='predicted_assignments', verbose=True)
# %% RUN ON LEFT, TEST ON RIGHT 

skid_hemi = pd.read_csv('../data/skid_hemisphere.csv')
skid_hemi['skid'] = skid_hemi['skid'].astype(int)
left_neurons = skid_hemi[skid_hemi['hemisphere'] == 'left']['skid'].tolist()
right_neurons = skid_hemi[skid_hemi['hemisphere'] == 'right']['skid'].tolist()
center_neurons = skid_hemi[skid_hemi['hemisphere'] == 'center']['skid'].tolist()

train_neurons = [neuron for neuron in neurons if neuron['pre_id'] in left_neurons]
test_neurons = [neuron for neuron in neurons if neuron['pre_id'] in right_neurons]

print(f'Training neurons (left hemi): {len(train_neurons)}, Test neurons (right hemi): {len(test_neurons)}')

order_model = fit_order_model_from_neurons(train_neurons)

seed = 34 # 42
# apply to all test neurons
predicted_test_neurons = predict(test_neurons, order_model, seed=seed)
accuracy, u_test_neurons = compare_assigned_orders(predicted_test_neurons, og_key='site_orders', predicted_key='predicted_orders', verbose=True)
similarity = compare_co_targeting_similarity(predicted_test_neurons, og_key='assignment', predicted_key='predicted_assignments', verbose=True)
correlation = compare_order_correlation(predicted_test_neurons, og_key='site_orders', predicted_key='predicted_orders', verbose=True)[0]

# apply to train neurons to see how well it does on the training set
predicted_train_neurons = predict(train_neurons, order_model, seed=seed)
train_accuracy, u_train_neurons = compare_assigned_orders(predicted_train_neurons, og_key='site_orders', predicted_key='predicted_orders', verbose=True)
train_similarity = compare_co_targeting_similarity(predicted_train_neurons, og_key='assignment', predicted_key='predicted_assignments', verbose=True)
train_correlation = compare_order_correlation(predicted_train_neurons, og_key='site_orders', predicted_key='predicted_orders', verbose=True)[0]

# null just shuffled
null_test_neurons = get_null(test_neurons, seed=seed)
null_accuracy, null_u_test_neurons = compare_assigned_orders(null_test_neurons, og_key='site_orders', predicted_key='null_orders', verbose=True)
null_similarity = compare_co_targeting_similarity(null_test_neurons, og_key='assignment', predicted_key='null_assignments', verbose=True)
null_correlation = compare_order_correlation(null_test_neurons, og_key='site_orders', predicted_key='null_orders', verbose=True)[0]
print('alpha =1.0, max_iter=1000')
print(f'Null mean accuracy: {np.mean(null_accuracy):.4f}, mean similarity: {np.mean(null_similarity):.4f}, mean correlation: {np.mean(null_correlation):.4f}')
print(f'Train set mean accuracy: {np.mean(train_accuracy):.4f}, mean similarity: {np.mean(train_similarity):.4f}, mean correlation: {np.mean(train_correlation):.4f}')
print(f'Test set mean accuracy: {np.mean(accuracy):.4f}, mean similarity: {np.mean(similarity):.4f}, mean correlation: {np.mean(correlation):.4f}')

#%% for reference maybe compare how similar pairs are 
pairs = pd.read_csv('../data/pairs-2022-02-14.csv')
pairs_no_duplicates = pairs.drop_duplicates(subset=['leftid', 'rightid'])
left_ordered = pairs_no_duplicates['leftid'].tolist()
right_ordered = pairs_no_duplicates['rightid'].tolist()

# need to think about this as it requires 1-to-1 matching of the left and right neuron pair's synapses

# %% TRY RUNNING ON CELLTYPES INDIVIDUALLY left and right 

celltype_df = pd.read_csv('../data/celltype_df.csv')
celltype_df['skids'] = celltype_df['skids'].apply(ast.literal_eval)
celltype_df['skids'] = celltype_df['skids'].apply(lambda x: [int(i) for i in x])

ct_to_skids = dict(zip(celltype_df['name'], celltype_df['skids']))

train_neurons = [neuron for neuron in neurons if neuron['pre_id'] in left_neurons]
test_neurons = [neuron for neuron in neurons if neuron['pre_id'] in right_neurons]

ct_results = {}
for e, (ct, skids) in enumerate(ct_to_skids.items()):
    ct_train_neurons = [neuron for neuron in train_neurons if neuron['pre_id'] in skids]
    ct_test_neurons = [neuron for neuron in test_neurons if neuron['pre_id'] in skids]
    if len(ct_train_neurons) == 0 or len(ct_test_neurons) == 0:
        print(f"Skipping cell type {ct} due to insufficient neurons.")
        continue
    print(f"Cell type {ct}: Training neurons: {len(ct_train_neurons)}, Test neurons: {len(ct_test_neurons)}")
    ct_order_model = fit_order_model_from_neurons(ct_train_neurons)
    ct_predicted_test_neurons = predict(ct_test_neurons, ct_order_model, seed=42)
    ct_accuracy, ct_u_test_neurons = compare_assigned_orders(ct_predicted_test_neurons, og_key='site_orders', predicted_key='predicted_orders', verbose=True)
    ct_similarity = compare_co_targeting_similarity(ct_predicted_test_neurons, og_key='assignment', predicted_key='predicted_assignments', verbose=True)
    ct_correlation = compare_order_correlation(ct_predicted_test_neurons, og_key='site_orders', predicted_key='predicted_orders', verbose=True)[0]
    # also do train set for comparison
    ct_predicted_train_neurons = predict(ct_train_neurons, ct_order_model, seed=42)
    ct_train_accuracy, ct_u_train_neurons = compare_assigned_orders(ct_predicted_train_neurons, og_key='site_orders', predicted_key='predicted_orders', verbose=True)    
    ct_train_similarity = compare_co_targeting_similarity(ct_predicted_train_neurons, og_key='assignment', predicted_key='predicted_assignments', verbose=True)
    ct_train_correlation = compare_order_correlation(ct_predicted_train_neurons, og_key='site_orders', predicted_key='predicted_orders', verbose=True)[0]
    # get null comparison on test set 
    ct_null_test_neurons = get_null(ct_test_neurons, seed=42)
    ct_null_accuracy, ct_null_u_test_neurons = compare_assigned_orders(ct_null_test_neurons, og_key='site_orders', predicted_key='null_orders', verbose=True)
    ct_null_similarity = compare_co_targeting_similarity(ct_null_test_neurons, og_key='assignment', predicted_key='null_assignments', verbose=True)
    ct_null_correlation = compare_order_correlation(ct_null_test_neurons, og_key='site_orders', predicted_key='null_orders', verbose=True)[0]
    ct_results[ct] = {
        "accuracy": ct_accuracy,
        "similarity": ct_similarity,
        "train_accuracy": ct_train_accuracy,
        "train_similarity": ct_train_similarity,
        "rightid": ct_u_test_neurons,
        "num_train_neurons": len(ct_train_neurons),
        "num_test_neurons": len(ct_test_neurons),
        "null_accuracy": ct_null_accuracy,
        "null_similarity": ct_null_similarity,
        "correlation": ct_correlation,
        "train_correlation": ct_train_correlation,
        "null_correlation": ct_null_correlation,
    }
    
ct_results_df = pd.DataFrame.from_dict(ct_results, orient='index')
ct_results_df['mean_accuracy'] = ct_results_df['accuracy'].apply(np.mean)
ct_results_df['mean_similarity'] = ct_results_df['similarity'].apply(np.mean)
ct_results_df['mean_train_accuracy'] = ct_results_df['train_accuracy'].apply(np.mean)
ct_results_df['mean_train_similarity'] = ct_results_df['train_similarity'].apply(np.mean)
ct_results_df['mean_null_accuracy'] = ct_results_df['null_accuracy'].apply(np.mean)
ct_results_df['mean_null_similarity'] = ct_results_df['null_similarity'].apply(np.mean)
ct_results_df['mean_correlation'] = ct_results_df['correlation'].apply(np.mean)
ct_results_df['mean_train_correlation'] = ct_results_df['train_correlation'].apply(np.mean)
ct_results_df['mean_null_correlation'] = ct_results_df['null_correlation'].apply(np.mean)
# print
ct_results_df[['mean_accuracy', 'mean_similarity', 'mean_train_accuracy', 'mean_train_similarity', 'mean_null_accuracy', 'mean_null_similarity']].sort_values(by='mean_accuracy', ascending=False)
ct_results_df[['mean_correlation', 'mean_train_correlation', 'mean_null_correlation']].sort_values(by='mean_correlation', ascending=False)
# %% predict per-site orders independently (no histogram / assignment constraint)

def predict_orders_using_location(site_locations, order_model, total_order=None):
    """
    Predict a polyadic order for each site independently from its own
    location (and neuron size) - no histogram constraint, no assignment
    problem. Each site just gets the model's own predicted rate, rounded
    to the nearest integer.

    If total_order is given, the per-site predictions are rescaled and
    rounded (largest-remainder method) so they sum to exactly total_order
    while staying as close as possible to the model's relative
    site-to-site pattern. Assumes total_order >= number_of_sites (every
    site gets at least order 1), matching the invariant real order
    histograms have.
    """
    site_locations = np.asarray(site_locations, dtype=float)
    number_of_sites = site_locations.shape[0]

    n_sites_column = np.full((number_of_sites, 1), number_of_sites, dtype=float)
    features = np.hstack([site_locations, n_sites_column])

    predicted_mu = order_model.predict(features)
    predicted_mu = np.clip(predicted_mu, a_min=1e-10, a_max=None)

    if total_order is None:
        return np.rint(predicted_mu).astype(int)

    # Largest-remainder method: rescale predictions to sum to total_order
    # (preserving each site's relative share), floor, then hand out the
    # leftover units to the sites with the largest fractional remainder.
    scaled_mu = predicted_mu * (total_order / predicted_mu.sum())
    site_order = np.floor(scaled_mu).astype(int)
    leftover = int(total_order) - site_order.sum()

    remainder = scaled_mu - site_order
    top_up_indices = np.argsort(-remainder)[:leftover]
    site_order[top_up_indices] += 1

    return site_order


# example: independent per-site prediction on one test neuron
example_neuron = test_neurons[4]

independent_orders = predict_orders_using_location(
    site_locations=example_neuron['site_locations'],
    order_model=order_model,
)
print(f"Independent predicted orders (neuron {example_neuron['pre_id']}): {independent_orders}")
print(f"Observed orders: {example_neuron['site_orders']}")

matched_total_orders = predict_orders_using_location(
    site_locations=example_neuron['site_locations'],
    order_model=order_model,
    total_order=int(example_neuron['site_orders'].sum()),
)
print(f"Total-matched predicted orders (sum={matched_total_orders.sum()}): {matched_total_orders}")

def error_metrics(predicted_orders, observed_orders):
    predicted_orders = np.asarray(predicted_orders, dtype=int)
    observed_orders = np.asarray(observed_orders, dtype=int)

    if predicted_orders.shape != observed_orders.shape:
        raise ValueError("Predicted and observed orders must have the same shape.")

    absolute_error = np.abs(predicted_orders - observed_orders)
    mean_absolute_error = absolute_error.mean()
    root_mean_squared_error = np.sqrt((absolute_error ** 2).mean())
    print(f"Mean absolute error: {mean_absolute_error:.4f}, Root mean squared error: {root_mean_squared_error:.4f}")
    return {
        "mean_absolute_error": mean_absolute_error,
        "root_mean_squared_error": root_mean_squared_error,
    }

error = error_metrics(independent_orders, example_neuron['site_orders'])
error_matched = error_metrics(matched_total_orders, example_neuron['site_orders'])
# %% what if i train on single left neuron 
idx = 16
left, right = left_neurons[idx], right_neurons[idx]

left_neuron = [neuron for neuron in neurons if neuron['pre_id'] == left]
right_neuron = [neuron for neuron in neurons if neuron['pre_id'] == right]

single_order_model = fit_order_model_from_neurons(left_neuron)


# apply to all test neurons
predicted_right = predict(right_neuron, single_order_model, seed=42)
null_right = get_null(right_neuron, seed=42)
acc, _ = compare_assigned_orders(predicted_right, og_key='site_orders', predicted_key='predicted_orders', verbose=True)
sim = compare_co_targeting_similarity(predicted_right, og_key='assignment', predicted_key='predicted_assignments', verbose=True)
null_acc, _ = compare_assigned_orders(null_right, og_key='site_orders', predicted_key='null_orders', verbose=True)
null_sim = compare_co_targeting_similarity(null_right, og_key='assignment', predicted_key='null_assignments', verbose=True)
print(f"Null mean accuracy: {np.mean(null_acc):.4f}, mean similarity: {np.mean(null_sim):.4f}")
print(f"Single left neuron trained model: mean accuracy: {np.mean(acc):.4f}, mean similarity: {np.mean(sim):.4f}")

print(f"Observed orders: {right_neuron[0]['site_orders']}")
print(f"Independent predicted orders: {predicted_right[0]['predicted_orders']}")
print(f"Matched predicted orders: {predict_orders_using_location(right_neuron[0]['site_locations'], single_order_model, total_order=int(right_neuron[0]['site_orders'].sum()))}")
err_independent = error_metrics(predicted_right[0]['predicted_orders'], right_neuron[0]['site_orders'])
err_matched = error_metrics(predict_orders_using_location(right_neuron[0]['site_locations'], single_order_model, total_order=int(right_neuron[0]['site_orders'].sum())), right_neuron[0]['site_orders'])
print(f'Null predicted orders: {null_right[0]["null_orders"]}')
err_null_independent = error_metrics(null_right[0]['null_orders'], right_neuron[0]['site_orders'])
err_null_matched = error_metrics(predict_orders_using_location(right_neuron[0]['site_locations'], single_order_model, total_order=int(right_neuron[0]['site_orders'].sum())), right_neuron[0]['site_orders'])

# %%
