"""Tests for whether a categorical neuron label (e.g. axon_structure) predicts geo_abs.

geo_abs is zero-and-one inflated: it is clamped to exactly 0 when the null is
sufficient or beats geometry, and equals exactly 1.0 whenever geometry is the best
performing sample. So it gets split into two questions that a single rank test
cannot separate:

    part A  does geometry contribute at all      (geo_abs > 0)
    part B  among contributors, how much         (geo_abs | geo_abs > 0)

Two properties of the neuron table make the textbook p-values wrong:

    left/right pairs   each pairid contributes two near-identical neurons, so rows
                       are not independent and n is effectively halved
    celltype confound  axon laterality is close to deterministic within some cell
                       types, so a raw association may be celltype in disguise

Both are handled by the same device: permute labels between *blocks* (a pairid moves
as one unit) and only *within* a stratum (celltype), then read the p-value off the
resulting null. Nothing is assumed about the shape of the outcome.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu, rankdata

__all__ = [
    'make_blocks', 'prepare_blocks', 'stratified_block_permutation',
    'kruskal_factory', 'chi2_factory', 'cramers_v', 'epsilon_squared',
    'rank_biserial', 'holm', 'point_mass_summary', 'analyse_structure',
    'print_report',
]

NA_STRATUM = '__na__'


# ---------------------------------------------------------------- blocking


def make_blocks(df, pair_col='pairid'):
    """One block id per row: the pairid where present, else a block of the neuron's own.

    pairid is NaN for neurons missing from the pairs table, so this cannot be a cast.
    Unpaired neurons become singleton blocks, which permute freely as themselves.
    """
    pair = df[pair_col] if pair_col in df.columns else pd.Series(np.nan, index=df.index)
    return pd.Series(
        [f'pair_{int(p)}' if pd.notna(p) else f'solo_{i}' for p, i in zip(pair, df.index)],
        index=df.index, dtype=object,
    )


def prepare_blocks(df, group_col, stratum_col=None, pair_col='pairid', block_col='_block'):
    """Attach block ids, then drop blocks whose members disagree on label or stratum.

    A block is the unit that gets permuted, so it has to carry a single label. The two
    hemispheres of a pairid nearly always share both annotations; the few that do not
    cannot move as one unit and are dropped (and counted, so the loss stays visible).
    """
    out = df.copy()
    out[block_col] = make_blocks(out, pair_col=pair_col)
    if stratum_col is not None:
        out[stratum_col] = out[stratum_col].fillna(NA_STRATUM)

    check_cols = [group_col] + ([stratum_col] if stratum_col is not None else [])
    nun = out.groupby(block_col)[check_cols].nunique()
    bad = nun.index[(nun > 1).any(axis=1)]
    dropped = out[block_col].isin(bad)
    return out.loc[~dropped].copy(), list(bad), int(dropped.sum())


# ------------------------------------------------------- statistic factories
# Both factories close over the outcome and take integer label codes, because the
# outcome never changes across permutations - only the labels do. For Kruskal-Wallis
# that means the ranks and the tie correction are computed once, and each of the
# 10k permutations is a single bincount rather than a full scipy call.


def kruskal_factory(values):
    """Return stat(codes, k) -> Kruskal-Wallis H, with ranks precomputed."""
    values = np.asarray(values, dtype=float)
    n = values.size
    ranks = rankdata(values)
    _, counts = np.unique(values, return_counts=True)
    tie_corr = 1.0 - (counts ** 3 - counts).sum() / (n ** 3 - n) if n > 1 else 1.0

    def stat(codes, k):
        group_n = np.bincount(codes, minlength=k).astype(float)
        rank_sums = np.bincount(codes, weights=ranks, minlength=k)
        nz = group_n > 0
        h = 12.0 / (n * (n + 1)) * np.sum(rank_sums[nz] ** 2 / group_n[nz]) - 3 * (n + 1)
        return h / tie_corr if tie_corr > 0 else np.nan

    return stat


def chi2_factory(binary):
    """Return stat(codes, k) -> Pearson chi-square of the k x 2 table, no continuity fix."""
    b = np.asarray(binary).astype(float)
    n = b.size

    def stat(codes, k):
        obs = np.empty((k, 2))
        obs[:, 1] = np.bincount(codes, weights=b, minlength=k)
        obs[:, 0] = np.bincount(codes, minlength=k) - obs[:, 1]
        exp = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / n
        good = exp > 0
        return float(np.sum((obs[good] - exp[good]) ** 2 / exp[good]))

    return stat


# ----------------------------------------------------------- permutation test


def stratified_block_permutation(df, group_col, stat_fn, stratum_col=None,
                                 block_col='_block', n_perm=10000, seed=0):
    """p-value for stat_fn under labels shuffled between blocks within each stratum.

    Shuffling within a stratum holds the stratum's label counts fixed, so the null
    asks the narrower question 'is the statistic large given celltype?'. Pass
    stratum_col=None for the unadjusted version; the difference between the two
    p-values is a direct readout of how much of the effect was celltype.
    """
    codes_rows, block_index = pd.factorize(df[block_col])
    first = df.drop_duplicates(subset=block_col).set_index(block_col)

    labels = first.loc[block_index, group_col].to_numpy()
    label_codes, label_levels = pd.factorize(pd.Series(labels))
    k = len(label_levels)

    if stratum_col is None:
        strata = np.zeros(len(block_index), dtype=int)
    else:
        strata = first.loc[block_index, stratum_col].fillna(NA_STRATUM).to_numpy()

    stratum_pos = [np.flatnonzero(strata == s) for s in pd.unique(strata)]

    # A stratum with one block, or one where every block shares a label, offers nothing
    # to shuffle. That is the correct behaviour once celltype is held fixed, but it
    # shrinks the effective sample, so it is counted and reported rather than hidden.
    informative = np.zeros(len(block_index), dtype=bool)
    for pos in stratum_pos:
        if pos.size > 1 and np.unique(label_codes[pos]).size > 1:
            informative[pos] = True

    observed = stat_fn(label_codes[codes_rows], k)

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    perm = label_codes.copy()
    shuffleable = [pos for pos in stratum_pos if pos.size > 1]
    for i in range(n_perm):
        for pos in shuffleable:
            perm[pos] = rng.permutation(label_codes[pos])
        null[i] = stat_fn(perm[codes_rows], k)

    # the +1 on both sides keeps p away from exactly 0, which n_perm cannot justify
    p = (1.0 + np.sum(null >= observed)) / (1.0 + n_perm)

    return {
        'observed': float(observed),
        'p': float(p),
        'null': null,
        'n_perm': n_perm,
        'levels': list(label_levels),
        'n_rows': int(len(df)),
        'n_blocks': int(len(block_index)),
        'n_informative_blocks': int(informative.sum()),
        'n_informative_rows': int(informative[codes_rows].sum()),
        'n_strata': len(stratum_pos),
    }


# ------------------------------------------------------------- effect sizes


def cramers_v(a, b, bias_correction=True):
    """Cramer's V between two categorical Series, with optional Bergsma correction."""
    tab = pd.crosstab(a, b)
    tab = tab.loc[tab.sum(axis=1) > 0, tab.sum(axis=0) > 0]
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan
    chi2 = chi2_contingency(tab.to_numpy(), correction=False).statistic
    n = tab.to_numpy().sum()
    phi2 = chi2 / n
    r, k = tab.shape
    if not bias_correction:
        return np.sqrt(phi2 / (min(r, k) - 1))
    phi2 = max(0.0, phi2 - (r - 1) * (k - 1) / (n - 1))
    r = r - (r - 1) ** 2 / (n - 1)
    k = k - (k - 1) ** 2 / (n - 1)
    return np.sqrt(phi2 / (min(r, k) - 1))


def epsilon_squared(h, n, k):
    """Kruskal-Wallis effect size: share of rank variance explained, 0 to 1."""
    if n <= k:
        return np.nan
    return (h - k + 1) / (n - k)


def rank_biserial(a, b):
    """Rank-biserial r for two samples: +1 if every a exceeds every b, -1 if reversed."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return np.nan
    u = mannwhitneyu(a, b, alternative='two-sided').statistic
    return 2.0 * u / (a.size * b.size) - 1.0


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, order preserved."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


# -------------------------------------------------------------- descriptives


def point_mass_summary(df, group_col, value_col='geo_abs'):
    """Per group: n, the two point masses, and the spread of the interior only.

    A histogram of geo_abs buries the spikes at 0 and 1 inside their bins, which is
    exactly the information the two-part split turns on, so it gets its own table.
    """
    rows = []
    for label, sub in df.groupby(group_col):
        v = sub[value_col].to_numpy(dtype=float)
        interior = v[(v > 0) & (v < 1)]
        rows.append({
            group_col: label,
            'n': v.size,
            'frac_zero': float(np.mean(v == 0)) if v.size else np.nan,
            'frac_one': float(np.mean(v == 1)) if v.size else np.nan,
            'n_interior': interior.size,
            'interior_median': float(np.median(interior)) if interior.size else np.nan,
            'interior_q25': float(np.percentile(interior, 25)) if interior.size else np.nan,
            'interior_q75': float(np.percentile(interior, 75)) if interior.size else np.nan,
        })
    return pd.DataFrame(rows).sort_values('n', ascending=False).reset_index(drop=True)


# ------------------------------------------------------------ the full recipe


def analyse_structure(sims, group_col, value_col='geo_abs', stratum_col='celltype',
                      pair_col='pairid', exclude=('other',), n_perm=10000, seed=0,
                      pairwise_alpha=0.05):
    """Run the whole two-part, block-permuted, celltype-stratified comparison.

    Returns a dict of results; hand it to print_report for a readable summary.
    """
    excluded = sims[sims[group_col].isin(exclude)]
    df_raw = sims[~sims[group_col].isin(exclude)].dropna(subset=[value_col])

    df, bad_blocks, n_dropped = prepare_blocks(
        df_raw, group_col, stratum_col=stratum_col, pair_col=pair_col)

    res = {
        'group_col': group_col,
        'value_col': value_col,
        'stratum_col': stratum_col,
        'n_total': int(len(sims)),
        'n_excluded': int(len(excluded)),
        'excluded_labels': list(exclude),
        'n_dropped_inconsistent': n_dropped,
        'inconsistent_blocks': bad_blocks,
        'confound_cramers_v': cramers_v(df[stratum_col], df[group_col]) if stratum_col else np.nan,
        'summary': point_mass_summary(df, group_col, value_col),
        'excluded_summary': (point_mass_summary(excluded, group_col, value_col)
                             if len(excluded) else None),
        'excluded_by_stratum': (pd.crosstab(excluded[stratum_col], excluded[group_col])
                                if len(excluded) and stratum_col else None),
    }

    # --- part A: does geometry contribute at all
    contributes = (df[value_col].to_numpy(dtype=float) > 0)
    stat_a = chi2_factory(contributes)
    res['partA'] = stratified_block_permutation(
        df, group_col, stat_a, stratum_col=stratum_col, n_perm=n_perm, seed=seed)
    res['partA_unstratified'] = stratified_block_permutation(
        df, group_col, stat_a, stratum_col=None, n_perm=n_perm, seed=seed)
    res['partA_table'] = pd.crosstab(df[group_col], pd.Series(contributes, index=df.index,
                                                              name='contributes'))
    res['partA_cramers_v'] = cramers_v(df[group_col], contributes)

    # --- part B: among contributors, how much
    sub = df[contributes].copy()
    res['partB_n'] = int(len(sub))
    if sub[group_col].nunique() > 1 and len(sub) > sub[group_col].nunique():
        stat_b = kruskal_factory(sub[value_col].to_numpy(dtype=float))
        res['partB'] = stratified_block_permutation(
            sub, group_col, stat_b, stratum_col=stratum_col, n_perm=n_perm, seed=seed)
        res['partB_unstratified'] = stratified_block_permutation(
            sub, group_col, stat_b, stratum_col=None, n_perm=n_perm, seed=seed)
        res['partB_epsilon_sq'] = epsilon_squared(
            res['partB']['observed'], len(sub), sub[group_col].nunique())
    else:
        res['partB'] = res['partB_unstratified'] = None
        res['partB_epsilon_sq'] = np.nan

    # --- pairwise follow-up, only where the omnibus earned it
    res['pairwise'] = {}
    for part, frame, binary in (('partA', df, True), ('partB', sub, False)):
        omnibus = res.get(part)
        if omnibus is None or omnibus['p'] > pairwise_alpha:
            continue
        levels = sorted(frame[group_col].unique())
        rows, pvals = [], []
        for i in range(len(levels)):
            for j in range(i + 1, len(levels)):
                a, b = levels[i], levels[j]
                pair_df = frame[frame[group_col].isin([a, b])]
                if pair_df[group_col].nunique() < 2:
                    continue
                if binary:
                    vals = (pair_df[value_col].to_numpy(dtype=float) > 0)
                    stat_fn = chi2_factory(vals)
                    effect = (vals[pair_df[group_col].to_numpy() == a].mean()
                              - vals[pair_df[group_col].to_numpy() == b].mean())
                    effect_name = 'diff_frac_contributing'
                else:
                    stat_fn = kruskal_factory(pair_df[value_col].to_numpy(dtype=float))
                    effect = rank_biserial(
                        pair_df.loc[pair_df[group_col] == a, value_col],
                        pair_df.loc[pair_df[group_col] == b, value_col])
                    effect_name = 'rank_biserial'
                pr = stratified_block_permutation(
                    pair_df, group_col, stat_fn, stratum_col=stratum_col,
                    n_perm=n_perm, seed=seed)
                rows.append({'a': a, 'b': b, effect_name: effect, 'p_raw': pr['p'],
                             'n': pr['n_rows'], 'n_informative': pr['n_informative_rows']})
                pvals.append(pr['p'])
        if rows:
            out = pd.DataFrame(rows)
            out['p_holm'] = holm(pvals)
            res['pairwise'][part] = out

    return res


def print_report(res):
    """Readable summary of analyse_structure, in the order the numbers should be read."""
    g, v, s = res['group_col'], res['value_col'], res['stratum_col']
    line = '=' * 78
    print(f'\n{line}\n{g} vs {v}   (stratified by {s}, blocked by pairid)\n{line}')

    print(f"\n[confound] Cramer's V({s}, {g}) = {res['confound_cramers_v']:.3f}"
          '   (high means the two are near-redundant, so read the stratified p)')
    print(f"[excluded] {res['n_excluded']} rows labelled {res['excluded_labels']}; "
          f"{res['n_dropped_inconsistent']} rows dropped for L/R label disagreement")

    print(f'\n[distribution] point masses are where the two-part split lives\n')
    print(res['summary'].to_string(index=False))

    for part, title in (('partA', 'PART A  does geometry contribute at all (geo_abs > 0)'),
                        ('partB', 'PART B  among contributors, how much')):
        r = res.get(part)
        print(f'\n{"-" * 78}\n{title}\n{"-" * 78}')
        if r is None:
            print('  not enough data to test')
            continue
        u = res.get(f'{part}_unstratified')
        print(f"  statistic          {r['observed']:.3f}")
        print(f"  p (stratified)     {r['p']:.4f}   <- the one to report")
        print(f"  p (unstratified)   {u['p']:.4f}   <- gap vs above = the {s} confound")
        print(f"  rows {r['n_rows']}  blocks {r['n_blocks']}  "
              f"informative blocks {r['n_informative_blocks']} "
              f"({r['n_informative_rows']} rows) across {r['n_strata']} strata")
        if part == 'partA':
            print(f"  Cramer's V         {res['partA_cramers_v']:.3f}")
            print('\n' + res['partA_table'].to_string())
        else:
            print(f"  epsilon squared    {res['partB_epsilon_sq']:.4f}")
        if part in res['pairwise']:
            print('\n  pairwise (Holm-corrected):')
            print(res['pairwise'][part].to_string(index=False))

    # the excluded bucket is a catch-all for missing annotations, not a class, so it is
    # described rather than tested - the crosstab says whether it is coherent or junk
    if res['excluded_summary'] is not None:
        print(f'\n{"-" * 78}\nEXCLUDED  {res["excluded_labels"]} (not tested)\n{"-" * 78}')
        print(res['excluded_summary'].to_string(index=False))
        if res['excluded_by_stratum'] is not None:
            print(f'\n  by {s}:')
            print(res['excluded_by_stratum'].to_string())

    a, b = res.get('partA'), res.get('partB')
    if a is not None and b is not None:
        sig_a, sig_b = a['p'] < 0.05, b['p'] < 0.05
        print(f'\n{"-" * 78}\nreading')
        if sig_a and not sig_b:
            print(f'  {g} governs WHETHER geometry matters, not how much.')
        elif sig_b and not sig_a:
            print(f'  {g} does not change whether geometry matters, but does change how much.')
        elif sig_a and sig_b:
            print(f'  {g} relates to both whether geometry matters and how much.')
        else:
            print(f'  no evidence {g} adds anything beyond {s}.')
        print('  a null stratified result next to a strong unstratified one is a finding')
        print(f'  ("{g} adds nothing beyond {s}"), not a failed test.')
