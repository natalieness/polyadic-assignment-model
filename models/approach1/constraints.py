"""Exact-count site, degree, and partner-assignment decoders."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .config import DegreeSummary, ModelConfig


def poisson_log_probability(order: int | np.ndarray, mean: np.ndarray) -> np.ndarray:
    order_array = np.asarray(order, dtype=float)
    mean = np.clip(np.asarray(mean, dtype=float), 1e-12, None)
    log_factorial = np.vectorize(math.lgamma)(order_array + 1.0)
    return order_array * np.log(mean) - mean - log_factorial


def select_exact_sites(intensity: np.ndarray, n_sites: int, config: ModelConfig) -> np.ndarray:
    """Return candidate indices, repeating a candidate only through explicit capacity slots."""
    intensity = np.clip(np.asarray(intensity, dtype=float), 1e-12, None)
    if n_sites < 0:
        raise ValueError("n_sites cannot be negative")
    if n_sites == 0:
        return np.zeros(0, dtype=int)
    capacity = config.site.max_sites_per_candidate
    if n_sites > len(intensity) * capacity:
        raise ValueError(
            f"Cannot place {n_sites} sites on {len(intensity)} candidates with capacity {capacity}"
        )
    candidate_indices = np.repeat(np.arange(len(intensity)), capacity)
    duplicate_index = np.tile(np.arange(capacity), len(intensity))
    score = np.log(np.repeat(intensity, capacity)) - config.site.duplicate_site_penalty * duplicate_index
    rng = np.random.default_rng(config.site.random_seed)
    if config.site.selection == "sample":
        uniform = np.clip(rng.random(len(score)), 1e-12, 1.0 - 1e-12)
        score = score - np.log(-np.log(uniform))
    chosen_slots = np.argpartition(score, -n_sites)[-n_sites:]
    chosen_slots = chosen_slots[np.argsort(-score[chosen_slots])]
    return candidate_indices[chosen_slots]


def degrees_from_histogram(histogram: Mapping[int, int], n_sites: int, total_contacts: int) -> np.ndarray:
    degrees = np.asarray(
        [int(degree) for degree, count in sorted(histogram.items()) for _ in range(int(count))],
        dtype=int,
    )
    if len(degrees) != n_sites:
        raise ValueError(f"Degree histogram has {len(degrees)} sites; expected {n_sites}")
    if int(degrees.sum()) != total_contacts:
        raise ValueError(
            f"Degree histogram sums to {degrees.sum()} contacts; target counts sum to {total_contacts}"
        )
    if np.any(degrees < 1):
        raise ValueError("Every presynaptic site must have degree at least one")
    return degrees


def degrees_from_summary(
    summary: DegreeSummary,
    n_sites: int,
    total_contacts: int,
    predicted_means: np.ndarray,
) -> np.ndarray:
    """Construct a summary-shaped degree multiset, then reconcile its exact contact total."""
    summary.validate()
    if not summary.minimum * n_sites <= total_contacts <= summary.maximum * n_sites:
        raise ValueError("Degree summary range cannot satisfy the supplied total contact count")
    probabilities = (np.arange(n_sites) + 0.5) / max(n_sites, 1)
    anchors_p = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    anchors_k = np.asarray(
        [summary.minimum, summary.q1, summary.median, summary.q3, summary.maximum], dtype=float
    )
    degrees = np.rint(np.interp(probabilities, anchors_p, anchors_k)).astype(int)
    degrees = np.clip(degrees, summary.minimum, summary.maximum)
    predicted = np.asarray(predicted_means, dtype=float)
    difference = int(total_contacts - degrees.sum())
    while difference:
        direction = 1 if difference > 0 else -1
        allowed = degrees < summary.maximum if direction > 0 else degrees > summary.minimum
        indices = np.flatnonzero(allowed)
        if not len(indices):
            raise RuntimeError("Unable to reconcile degree summary with total contacts")
        before = (degrees[indices] - predicted[indices]) ** 2
        after = (degrees[indices] + direction - predicted[indices]) ** 2
        chosen = indices[int(np.argmin(after - before))]
        degrees[chosen] += direction
        difference -= direction
    return degrees


def assign_degree_multiset(degrees: np.ndarray, predicted_means: np.ndarray) -> np.ndarray:
    """Optimal assignment for Poisson scores by the rearrangement inequality."""
    degrees = np.asarray(degrees, dtype=int)
    predicted_means = np.asarray(predicted_means, dtype=float)
    if len(degrees) != len(predicted_means):
        raise ValueError("Degree and site arrays differ in length")
    assigned = np.empty_like(degrees)
    assigned[np.argsort(predicted_means)] = np.sort(degrees)
    return assigned


def allocate_learned_degrees(
    predicted_means: np.ndarray,
    total_contacts: int,
    minimum: int,
    maximum: int,
) -> np.ndarray:
    """Dynamic program maximizing independent Poisson likelihood under an exact sum."""
    means = np.asarray(predicted_means, dtype=float)
    n_sites = len(means)
    if not minimum * n_sites <= total_contacts <= maximum * n_sites:
        raise ValueError(
            f"Cannot allocate {total_contacts} contacts over {n_sites} sites in [{minimum}, {maximum}]"
        )
    negative_infinity = -np.inf
    previous = np.full(total_contacts + 1, negative_infinity)
    previous[0] = 0.0
    back = np.full((n_sites, total_contacts + 1), -1, dtype=np.int16)
    degree_values = np.arange(minimum, maximum + 1)
    for site_index, mean in enumerate(means):
        current = np.full(total_contacts + 1, negative_infinity)
        scores = poisson_log_probability(degree_values, np.full(len(degree_values), mean))
        for prior_total in np.flatnonzero(np.isfinite(previous)):
            for degree, score in zip(degree_values, scores):
                new_total = prior_total + int(degree)
                if new_total <= total_contacts and previous[prior_total] + score > current[new_total]:
                    current[new_total] = previous[prior_total] + score
                    back[site_index, new_total] = int(degree)
        previous = current
    if not np.isfinite(previous[total_contacts]):
        raise RuntimeError("No feasible degree allocation found")
    result = np.empty(n_sites, dtype=int)
    running_total = total_contacts
    for site_index in range(n_sites - 1, -1, -1):
        degree = int(back[site_index, running_total])
        if degree < 0:
            raise RuntimeError("Broken degree-allocation backpointer")
        result[site_index] = degree
        running_total -= degree
    return result


@dataclass
class _Edge:
    target: int
    reverse: int
    capacity: int
    cost: float
    initial_capacity: int


class _MinCostFlow:
    def __init__(self, n_vertices: int):
        self.graph: list[list[_Edge]] = [[] for _ in range(n_vertices)]

    def add_edge(self, source: int, target: int, capacity: int, cost: float) -> tuple[int, int]:
        forward = _Edge(target, len(self.graph[target]), int(capacity), float(cost), int(capacity))
        reverse = _Edge(source, len(self.graph[source]), 0, -float(cost), 0)
        self.graph[source].append(forward)
        self.graph[target].append(reverse)
        return source, len(self.graph[source]) - 1

    def solve(self, source: int, sink: int, required_flow: int) -> tuple[int, float]:
        n = len(self.graph)
        potential = np.zeros(n, dtype=float)
        flow = 0
        total_cost = 0.0
        while flow < required_flow:
            distance = np.full(n, np.inf)
            parent_vertex = np.full(n, -1, dtype=int)
            parent_edge = np.full(n, -1, dtype=int)
            distance[source] = 0.0
            queue: list[tuple[float, int]] = [(0.0, source)]
            while queue:
                current_distance, vertex = heapq.heappop(queue)
                if current_distance > distance[vertex] + 1e-12:
                    continue
                for edge_index, edge in enumerate(self.graph[vertex]):
                    if edge.capacity <= 0:
                        continue
                    candidate = current_distance + edge.cost + potential[vertex] - potential[edge.target]
                    if candidate + 1e-12 < distance[edge.target]:
                        distance[edge.target] = candidate
                        parent_vertex[edge.target] = vertex
                        parent_edge[edge.target] = edge_index
                        heapq.heappush(queue, (candidate, edge.target))
            if not np.isfinite(distance[sink]):
                raise RuntimeError("Partner assignment is infeasible under the configured capacities")
            reachable = np.isfinite(distance)
            potential[reachable] += distance[reachable]
            addition = required_flow - flow
            vertex = sink
            while vertex != source:
                previous = int(parent_vertex[vertex])
                edge = self.graph[previous][int(parent_edge[vertex])]
                addition = min(addition, edge.capacity)
                vertex = previous
            vertex = sink
            while vertex != source:
                previous = int(parent_vertex[vertex])
                edge_index = int(parent_edge[vertex])
                edge = self.graph[previous][edge_index]
                edge.capacity -= addition
                self.graph[vertex][edge.reverse].capacity += addition
                total_cost += addition * edge.cost
                vertex = previous
            flow += addition
        return flow, total_cost


def solve_partner_transport(
    site_degrees: np.ndarray,
    target_counts: np.ndarray,
    score_matrix: np.ndarray,
    max_same_target_per_site: int | None = None,
) -> np.ndarray:
    """Maximum-score integer transportation with exact row and column sums."""
    site_degrees = np.asarray(site_degrees, dtype=int)
    target_counts = np.asarray(target_counts, dtype=int)
    scores = np.asarray(score_matrix, dtype=float)
    if scores.shape != (len(site_degrees), len(target_counts)):
        raise ValueError("Partner score matrix has the wrong shape")
    if int(site_degrees.sum()) != int(target_counts.sum()):
        raise ValueError("Site degrees and target counts have different contact totals")
    n_sites, n_targets = scores.shape
    source = 0
    first_site = 1
    first_target = first_site + n_sites
    sink = first_target + n_targets
    network = _MinCostFlow(sink + 1)
    for site, degree in enumerate(site_degrees):
        network.add_edge(source, first_site + site, int(degree), 0.0)
    for target, count in enumerate(target_counts):
        network.add_edge(first_target + target, sink, int(count), 0.0)
    finite_scores = scores[np.isfinite(scores)]
    if not len(finite_scores):
        raise ValueError("No finite partner-assignment scores")
    costs = -scores
    minimum_cost = float(np.nanmin(costs))
    if minimum_cost < 0:
        costs = costs - minimum_cost
    references: dict[tuple[int, int], tuple[int, int]] = {}
    for site, degree in enumerate(site_degrees):
        for target, count in enumerate(target_counts):
            cap = min(int(degree), int(count))
            if max_same_target_per_site is not None:
                cap = min(cap, int(max_same_target_per_site))
            if cap > 0 and np.isfinite(costs[site, target]):
                references[(site, target)] = network.add_edge(
                    first_site + site, first_target + target, cap, float(costs[site, target])
                )
    network.solve(source, sink, int(site_degrees.sum()))
    assignment = np.zeros((n_sites, n_targets), dtype=int)
    for (site, target), (vertex, edge_index) in references.items():
        edge = network.graph[vertex][edge_index]
        assignment[site, target] = edge.initial_capacity - edge.capacity
    if not np.array_equal(assignment.sum(axis=1), site_degrees):
        raise RuntimeError("Partner solver failed site-degree validation")
    if not np.array_equal(assignment.sum(axis=0), target_counts):
        raise RuntimeError("Partner solver failed target-count validation")
    return assignment


def residualize_partner_scores(
    score_matrix: np.ndarray,
    *,
    standardize: bool = True,
    clip: float | None = 2.0,
    uninformative_threshold: float = 1e-6,
) -> tuple[np.ndarray, float]:
    """Remove row/column main effects, leaving site-target interaction evidence.

    Exact row and column totals already determine pure site and target main
    effects. Keeping them in the assignment score can therefore create
    arbitrary packing without adding identifiable placement information.
    """
    scores = np.asarray(score_matrix, dtype=float)
    if scores.ndim != 2 or not np.all(np.isfinite(scores)):
        raise ValueError("Partner scores must be a finite two-dimensional matrix")
    residual = (
        scores
        - scores.mean(axis=1, keepdims=True)
        - scores.mean(axis=0, keepdims=True)
        + scores.mean()
    )
    strength = float(np.std(residual))
    if strength <= uninformative_threshold:
        return np.zeros_like(residual), strength
    if standardize:
        residual = residual / strength
    if clip is not None:
        residual = np.clip(residual, -float(clip), float(clip))
    return residual, strength


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits)
    weights = np.exp(shifted)
    return weights / weights.sum()


def _co_target_bonus(
    assignment_row: np.ndarray,
    co_target_scores: np.ndarray | None,
    *,
    apply_to: str,
    aggregation: str,
    clip: float | None,
) -> np.ndarray:
    n_targets = len(assignment_row)
    bonus = np.zeros(n_targets, dtype=float)
    if co_target_scores is None:
        return bonus
    present = np.flatnonzero(assignment_row > 0)
    if not len(present):
        return bonus
    for target in range(n_targets):
        if apply_to == "first_copy_only" and assignment_row[target] > 0:
            continue
        others = present[present != target]
        if not len(others):
            continue
        values = co_target_scores[target, others]
        bonus[target] = float(np.mean(values) if aggregation == "mean" else np.sum(values))
    if clip is not None:
        bonus = np.clip(bonus, -float(clip), float(clip))
    return bonus


def _row_multiplicity_profile(row: np.ndarray) -> tuple[int, ...]:
    return tuple(sorted(int(value) for value in row[row > 0]))


def _assignment_row_objective(
    row: np.ndarray,
    information_scores: np.ndarray,
    repeat_information_discount: float,
    co_target_scores: np.ndarray | None,
    co_targeting_weight: float,
    co_targeting_apply_to: str,
    co_targeting_aggregation: str,
    co_targeting_score_clip: float | None,
) -> float:
    effective_contacts = np.where(
        row > 0,
        1.0 + repeat_information_discount * (row - 1),
        0.0,
    )
    value = float(effective_contacts @ information_scores)
    if co_target_scores is None or not co_targeting_weight:
        return value
    present = np.flatnonzero(row > 0)
    pair_values: list[float] = []
    pair_weights: list[float] = []
    for left_number, left in enumerate(present):
        for right in present[left_number + 1:]:
            pair_values.append(float(co_target_scores[left, right]))
            pair_weights.append(
                1.0
                if co_targeting_apply_to == "first_copy_only"
                else float(row[left] * row[right])
            )
    if not pair_values:
        return value
    values = np.asarray(pair_values)
    weights = np.asarray(pair_weights)
    if co_targeting_aggregation == "mean":
        co_value = float(np.average(values, weights=weights))
    else:
        co_value = float(values @ weights)
    if co_targeting_score_clip is not None:
        co_value = float(np.clip(
            co_value, -co_targeting_score_clip, co_targeting_score_clip
        ))
    return value + co_targeting_weight * co_value


def _tilt_assignment_with_locked_multiplicity(
    assignment: np.ndarray,
    information_scores: np.ndarray,
    rng: np.random.Generator,
    *,
    information_mix: float,
    temperature: float,
    repeat_information_discount: float,
    co_target_scores: np.ndarray | None,
    co_targeting_weight: float,
    co_targeting_apply_to: str,
    co_targeting_aggregation: str,
    co_targeting_score_clip: float | None,
    steps: int,
) -> np.ndarray:
    """Metropolis target-label swaps that preserve every site's repeat profile."""
    result = assignment.copy()
    active_sites = np.flatnonzero(result.sum(axis=1) > 0)
    if len(active_sites) < 2 or steps <= 0 or information_mix <= 0:
        return result
    co_is_informative = (
        co_target_scores is not None
        and co_targeting_weight != 0
        and np.any(np.abs(co_target_scores) > 0)
    )
    if not np.any(np.abs(information_scores) > 0) and not co_is_informative:
        return result
    profiles = [_row_multiplicity_profile(row) for row in result]
    objectives = np.asarray([
        _assignment_row_objective(
            result[site], information_scores[site], repeat_information_discount,
            co_target_scores, co_targeting_weight, co_targeting_apply_to,
            co_targeting_aggregation, co_targeting_score_clip,
        )
        for site in range(len(result))
    ])
    inverse_temperature = information_mix / temperature
    for _ in range(steps):
        left_site, right_site = rng.choice(active_sites, size=2, replace=False)
        left_present = np.flatnonzero(result[left_site] > 0)
        right_present = np.flatnonzero(result[right_site] > 0)
        left_target = int(rng.choice(left_present))
        right_target = int(rng.choice(right_present))
        if left_target == right_target:
            continue
        proposed_left = result[left_site].copy()
        proposed_right = result[right_site].copy()
        proposed_left[left_target] -= 1
        proposed_left[right_target] += 1
        proposed_right[right_target] -= 1
        proposed_right[left_target] += 1
        if (
            _row_multiplicity_profile(proposed_left) != profiles[left_site]
            or _row_multiplicity_profile(proposed_right) != profiles[right_site]
        ):
            continue
        proposed_left_objective = _assignment_row_objective(
            proposed_left, information_scores[left_site], repeat_information_discount,
            co_target_scores, co_targeting_weight, co_targeting_apply_to,
            co_targeting_aggregation, co_targeting_score_clip,
        )
        proposed_right_objective = _assignment_row_objective(
            proposed_right, information_scores[right_site], repeat_information_discount,
            co_target_scores, co_targeting_weight, co_targeting_apply_to,
            co_targeting_aggregation, co_targeting_score_clip,
        )
        delta = (
            proposed_left_objective + proposed_right_objective
            - objectives[left_site] - objectives[right_site]
        )
        if delta >= 0 or np.log(max(rng.random(), 1e-300)) < inverse_temperature * delta:
            result[left_site] = proposed_left
            result[right_site] = proposed_right
            objectives[left_site] = proposed_left_objective
            objectives[right_site] = proposed_right_objective
    return result


def sample_partner_assignment(
    site_degrees: np.ndarray,
    target_counts: np.ndarray,
    information_scores: np.ndarray,
    rng: np.random.Generator,
    *,
    information_mix: float = 0.25,
    temperature: float = 2.0,
    repeat_information_discount: float = 0.0,
    self_repeat_penalty: float = 0.0,
    preserve_null_repeat_mass: bool = True,
    max_same_target_per_site: int | None = None,
    co_target_scores: np.ndarray | None = None,
    co_targeting_weight: float = 0.0,
    co_targeting_apply_to: str = "first_copy_only",
    co_targeting_aggregation: str = "mean",
    co_targeting_score_clip: float | None = 1.0,
    max_restarts: int = 100,
    mcmc_steps_per_contact: int = 10,
) -> np.ndarray:
    """Sample an integer assignment with exact row and column totals.

    With ``information_mix=0`` this is exactly a random shuffle of labelled
    target-contact tokens into shuffled site slots. With the repeat-profile
    safeguard, learned information only tilts target-label swaps that preserve
    the null sample's per-site multiplicities. Without it, information is mixed
    directly with each sequential null probability.
    """
    degrees = np.asarray(site_degrees, dtype=int)
    counts = np.asarray(target_counts, dtype=int)
    scores = np.asarray(information_scores, dtype=float)
    if scores.shape != (len(degrees), len(counts)):
        raise ValueError("Partner information score matrix has the wrong shape")
    if int(degrees.sum()) != int(counts.sum()):
        raise ValueError("Site degrees and target counts have different contact totals")
    if np.any(degrees < 0) or np.any(counts < 0):
        raise ValueError("Degrees and counts cannot be negative")
    if max_same_target_per_site is not None:
        capacity = int(max_same_target_per_site)
        if np.any(degrees > capacity * len(counts)):
            raise ValueError("A site degree exceeds the configured per-target capacity")
        if np.any(counts > capacity * len(degrees)):
            raise ValueError("A target count exceeds the configured per-site capacity")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Partner information scores must be finite")
    if co_target_scores is not None:
        co_target_scores = np.asarray(co_target_scores, dtype=float)
        if co_target_scores.shape != (len(counts), len(counts)):
            raise ValueError("Co-target score matrix has the wrong shape")

    if preserve_null_repeat_mass:
        null_assignment = sample_partner_assignment(
            degrees,
            counts,
            np.zeros_like(scores),
            rng,
            information_mix=0.0,
            temperature=temperature,
            repeat_information_discount=0.0,
            self_repeat_penalty=0.0,
            preserve_null_repeat_mass=False,
            max_same_target_per_site=max_same_target_per_site,
            co_target_scores=None,
            co_targeting_weight=0.0,
            max_restarts=max_restarts,
            mcmc_steps_per_contact=0,
        )
        return _tilt_assignment_with_locked_multiplicity(
            null_assignment,
            scores,
            rng,
            information_mix=information_mix,
            temperature=temperature,
            repeat_information_discount=repeat_information_discount,
            co_target_scores=co_target_scores,
            co_targeting_weight=co_targeting_weight,
            co_targeting_apply_to=co_targeting_apply_to,
            co_targeting_aggregation=co_targeting_aggregation,
            co_targeting_score_clip=co_targeting_score_clip,
            steps=int(mcmc_steps_per_contact * counts.sum()),
        )

    base_slots = np.repeat(np.arange(len(degrees), dtype=int), degrees)
    last_failure = ""
    for _ in range(max_restarts):
        slots = rng.permutation(base_slots)
        remaining = counts.copy()
        assignment = np.zeros(scores.shape, dtype=int)
        failed = False
        for site in slots:
            feasible = remaining > 0
            if max_same_target_per_site is not None:
                feasible &= assignment[site] < int(max_same_target_per_site)
            target_indices = np.flatnonzero(feasible)
            if not len(target_indices):
                failed = True
                last_failure = f"no feasible target remained for site {site}"
                break

            null_probability = remaining[target_indices].astype(float)
            null_probability /= null_probability.sum()

            repeat_scale = np.where(
                assignment[site, target_indices] > 0,
                repeat_information_discount,
                1.0,
            )
            logits = scores[site, target_indices] * repeat_scale
            logits -= self_repeat_penalty * assignment[site, target_indices]
            if co_target_scores is not None and co_targeting_weight:
                bonus = _co_target_bonus(
                    assignment[site],
                    co_target_scores,
                    apply_to=co_targeting_apply_to,
                    aggregation=co_targeting_aggregation,
                    clip=co_targeting_score_clip,
                )
                logits += co_targeting_weight * bonus[target_indices]
            information_probability = _softmax(
                np.log(remaining[target_indices].astype(float)) + logits / temperature
            )
            probability = (
                (1.0 - information_mix) * null_probability
                + information_mix * information_probability
            )
            probability /= probability.sum()
            target = int(rng.choice(target_indices, p=probability))
            assignment[site, target] += 1
            remaining[target] -= 1

        if not failed:
            if not np.array_equal(assignment.sum(axis=1), degrees):
                raise RuntimeError("Stochastic assignment failed site-degree validation")
            if not np.array_equal(assignment.sum(axis=0), counts):
                raise RuntimeError("Stochastic assignment failed target-count validation")
            return assignment
    raise RuntimeError(
        "Unable to sample a feasible partner assignment after "
        f"{max_restarts} restarts ({last_failure}). Increase or remove "
        "max_same_target_per_site if it is too restrictive."
    )
