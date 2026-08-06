from collections import Counter
from collections.abc import Mapping

import numpy as np

def validate_assignment(
    assignment,
    site_orders,
    target_counts,
):
    """
    Validate active-zone-by-target assignment.
    assignment[r, j] == 1 means target j occurs at active zone r.
    Targets can occur at same active zone multiple times.
    """

    assignment = np.asarray(assignment, dtype=int)
    site_orders = np.asarray(site_orders, dtype=int)
    target_counts = np.asarray(target_counts, dtype=int)

    if assignment.ndim != 2:
        raise ValueError("assignment must have shape [sites, targets].")

    if not np.all(assignment >= 0):
        raise ValueError("Assignments have to be >= 0 at each site.")

    if not np.array_equal(
        assignment.sum(axis=1),
        site_orders,
    ):
        raise ValueError(
            "Assignments do not match the site orders. Note in learnt site order models, this is checked against the allocated/predicted site orders."
        )

    if not np.array_equal(
        assignment.sum(axis=0),
        target_counts,
    ):
        raise ValueError(
            "Assignment column sums do not match target counts."
        )