''' Model version 1C: Predict polyadic degree of a synapse based on location.

P(k | x,y,z). 

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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split

from validate_assignment import validate_assignment

#%%
def fit_order_location_model(training_locations, training_orders):
    """
    Learn P(polyadic order | active-zone location).

    Training data should come from held-out neurons, hemispheres,
    or brains rather than the test example itself.
    """
    model = Pipeline([
        ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
        ("standardize", StandardScaler()),
        ("classifier", LogisticRegression(C=1.0, max_iter=5000)),
    ])

    model.fit(
        np.asarray(training_locations, dtype=float),
        np.asarray(training_orders, dtype=int),
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

    return fit_order_location_model(
        training_locations=training_locations,
        training_orders=training_orders,
    )


def allocate_orders_using_location(site_locations, order_histogram, order_model):
    """
    Assign a fixed order histogram to sites using P(k | location).

    The histogram is preserved exactly.
    """
    site_locations = np.asarray(site_locations, dtype=float)
    number_of_sites = site_locations.shape[0]

    order_slots = []
    for order, count in order_histogram.items():
        order_slots.extend([int(order)] * int(count))

    if len(order_slots) != number_of_sites:
        raise ValueError("Order histogram must contain one order per site.")

    order_slots = np.asarray(order_slots, dtype=int)
    learned_order_to_column = {
        int(order): column
        for column, order in enumerate(order_model.classes_)
    }

    log_probabilities = order_model.predict_log_proba(site_locations)
    score_matrix = np.empty((number_of_sites, number_of_sites), dtype=float)

    for slot_column, order in enumerate(order_slots):
        if int(order) not in learned_order_to_column:
            raise ValueError(f"Order {order} was absent from the training data.")

        probability_column = learned_order_to_column[int(order)]
        score_matrix[:, slot_column] = log_probabilities[:, probability_column]

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

#%% prepare neurons 

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

# load connector table 
con = pd.read_csv('../data/con_short_2026.csv')
con['postsynaptic_to'] = con['postsynaptic_to'].apply(ast.literal_eval)
con['presynaptic_to'] = con['presynaptic_to'].astype(int)
con['presynaptic_to_node'] = con['presynaptic_to_node'].astype(int)
nodes = pd.read_csv('../data/nodes_manual_wdendrites_euclid.csv')
all_neurons = pd.read_csv('../data/all_neurons.csv') # get all neurons of interest
all_neurons = all_neurons['skeleton_id'].tolist()


#%% get site locations for each presynaptic site in the connector table

def prep_neurons_from_connector_table(con, nodes, all_neurons):
    # filter nodes to just presynaptic sites 
    presyn_nodes = con['presynaptic_to_node'].unique()
    nodes_presyn = nodes[nodes['node_id'].isin(presyn_nodes)]
    node_to_xyz = dict(zip(nodes_presyn['node_id'], nodes_presyn[['x', 'y', 'z']].values.tolist()))


    con['site_locations'] = con['presynaptic_to_node'].map(node_to_xyz)

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
    return neurons, index_to_neuron, node_to_xyz

neurons, index_to_neuron, node_to_xyz = prep_neurons_from_connector_table(con, nodes, all_neurons)


#%% run 


##### NOTE: strong polyadic degree class imbalance - certain degrees are essentially unlearnable 
#### model still converges but might not be a good sign 

# Maybe Merge or drop ultra-rare classes (e.g. lump orders ≥10 into an "order≥10" bucket, or drop singleton classes like 14/21 entirely) before fitting — this is the actual fix for both the warnings and the meaningless per-class predictions.
# or use class_weight="balanced" and expect the warnings to persist but be more informative about a real modeling limitation rather than a bug.

# use a subset for now 
neus = neurons.copy()
neurons = neurons[:100]



# split neurons into training and test sets
# either by percentage 
train_neurons, test_neurons = train_test_split(
    neurons,
    test_size=0.20,
    random_state=42,
)

# or hemisphere

print(f'Training neurons: {len(train_neurons)}, Test neurons: {len(test_neurons)}')


order_model = fit_order_model_from_neurons(train_neurons)


#%% apply to a test neurons
test_neuron = test_neurons[0]
# construct order histogram for the test neuron
observed_orders = test_neuron["site_orders"]

test_order_histogram = dict(Counter(observed_orders.tolist()))

assigned_orders = allocate_orders_using_location(
    site_locations=test_neuron['site_locations'],
    order_histogram=test_order_histogram,
    order_model=order_model,
)

random_assignment = construct_random_feasible_assignment(
    site_orders=assigned_orders,
    target_counts=test_neuron['target_counts'],
    random_seed=42,
)

#%% apply to all test neurons 

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

predicted_test_neurons = predict(test_neurons, order_model, seed=42)



#%% validate predictions 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# mean-site wise jaccard similarity 

### ignore below - uses histogram so they have to be the same 
# distribution of polyadic degrees 
# def plot_polyadic_degree_dist_comp(neuron_site_orders, assigned_orders):
#     fig, ax = plt.subplots(figsize=(6, 4))
#     bin_max = max(max(neuron_site_orders), max(assigned_orders)) + 1
#     bins = np.arange(0, bin_max + 1) - 0.5
#     ax.hist(neuron_site_orders, bins=bins, alpha=0.5, label='Observed', color='purple')
#     ax.hist(assigned_orders, bins=bins, alpha=0.5, label='Predicted', color='green')
#     ax.set_xlabel('Polyadic Degree')
#     ax.set_ylabel('Number of presynaptic sites')

# plot_polyadic_degree_dist_comp(
#     neuron_site_orders=test_neuron['site_orders'],
#     assigned_orders=assigned_orders,
# )

def compare_assigned_orders(neuron_site_orders, assigned_orders, verbose=False):
    # Calculate the number of sites with matching orders
    matching_sites = np.sum(neuron_site_orders == assigned_orders)
    total_sites = len(neuron_site_orders)
    similarity_score = matching_sites / total_sites
    if verbose:
        print(f"Matching Sites: {matching_sites}/{total_sites}")
        print(f"Similarity Score: {similarity_score:.2f}")
    return similarity_score

sim_score = compare_assigned_orders(
    neuron_site_orders=test_neuron['site_orders'],
    assigned_orders=assigned_orders,
)

# polyadic co-adj cosine similarity 
# %%
