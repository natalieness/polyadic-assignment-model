from collections import Counter
from itertools import chain

import numpy as np
import pandas as pd
from numba import njit, prange

@njit(cache=True, parallel=True)
def _shuffle_within_groups(codes, group_pool_ptr, u):
    ''' Permute each presynaptic neuron's partner pool independently.

    The pool is already group-contiguous, so this is a Fisher-Yates pass per group - O(pool)
    against the O(pool log pool) of sorting on random keys, and the groups are independent so
    it runs in parallel. The uniforms are drawn by the caller, so the result depends only on
    the seed and not on how the threads happen to be scheduled.
    '''
    out = codes.copy()
    for g in prange(group_pool_ptr.shape[0] - 1):
        s, e = group_pool_ptr[g], group_pool_ptr[g + 1]
        for i in range(e - 1, s, -1):
            j = s + int(u[i] * (i - s + 1))  # u is in [0, 1), so j stays within [s, i]
            tmp = out[i]
            out[i] = out[j]
            out[j] = tmp
    return out


@njit(cache=True, parallel=True)
def _assign_no_self_cotarget(pool, group_pool_ptr, group_row_ptr, sizes, row_ptr):
    ''' Greedy assignment of shuffled partners to connectors without repeats within a connector.

    Per presynaptic group: connectors are filled largest first, each taking the n
    most-frequent still-available partners. Ties are broken by first occurrence in the
    (already shuffled) pool, which is what randomises the choice.

    Presynaptic neurons are independent - every scratch array is allocated inside the loop,
    and a group only ever writes the out_flat range and out_len entries of its own rows - so
    the group loop is a prange. Output does not depend on the thread count.
    '''
    n_rows = sizes.shape[0]
    out_flat = np.zeros(pool.shape[0], dtype=np.int64)  # zeros, not empty: gaps stay valid indices
    out_len = np.zeros(n_rows, dtype=np.int64)

    for g in prange(group_pool_ptr.shape[0] - 1):
        ps, pe = group_pool_ptr[g], group_pool_ptr[g + 1]
        rs, re = group_row_ptr[g], group_row_ptr[g + 1]
        m = pe - ps
        if m == 0:
            continue
        seg = pool[ps:pe]

        # unique partners + counts, kept in order of first occurrence in the pool
        sorted_idx = np.argsort(seg, kind='mergesort')  # stable -> runs stay in pool order
        vals = np.empty(m, dtype=np.int64)
        counts = np.empty(m, dtype=np.int64)
        first_pos = np.empty(m, dtype=np.int64)
        n_uniq = 0
        i = 0
        while i < m:
            v = seg[sorted_idx[i]]
            j = i
            while j < m and seg[sorted_idx[j]] == v:
                j += 1
            vals[n_uniq] = v
            counts[n_uniq] = j - i
            first_pos[n_uniq] = sorted_idx[i]  # stable sort -> smallest index of the run
            n_uniq += 1
            i = j

        occ_order = np.argsort(first_pos[:n_uniq])
        uniq = np.empty(n_uniq, dtype=np.int64)
        cnt = np.empty(n_uniq, dtype=np.int64)
        for k in range(n_uniq):
            uniq[k] = vals[occ_order[k]]
            cnt[k] = counts[occ_order[k]]
        n_avail = n_uniq  # partners with a count left

        # partners bucketed by remaining count, so picking the top n is a linear scan
        # rather than a re-sort for every connector
        cmax = 0
        for k in range(n_uniq):
            if cnt[k] > cmax:
                cmax = cnt[k]
        bucket = np.zeros(cmax + 2, dtype=np.int64)
        for k in range(n_uniq):
            bucket[cnt[k]] += 1
        take = np.zeros(cmax + 2, dtype=np.int64)
        wptr = np.zeros(cmax + 2, dtype=np.int64)

        # largest connectors first: most constrained get first pick
        conn_order = np.argsort(-sizes[rs:re], kind='mergesort')
        for jj in range(conn_order.shape[0]):
            r = rs + conn_order[jj]
            n = sizes[r]
            if n == 0:
                continue
            n_sel = n if n <= n_avail else n_avail
            out_len[r] = n_sel
            if n_sel == 0:
                continue
            while cmax > 0 and bucket[cmax] == 0:
                cmax -= 1

            # how many to take from each count level, highest first
            left = n_sel
            c_low = cmax
            for c in range(cmax, 0, -1):
                take[c] = bucket[c] if bucket[c] < left else left
                left -= take[c]
                c_low = c
                if left == 0:
                    break
            # write offsets: count descending, so higher counts land first
            acc = 0
            for c in range(cmax, c_low - 1, -1):
                wptr[c] = acc
                acc += take[c]

            # one pass in first-occurrence order -> ties broken exactly like the python version
            o = row_ptr[r]
            partial_left = take[c_low]
            for k in range(n_uniq):
                c = cnt[k]
                if c < c_low or c > cmax:
                    continue
                if c == c_low:
                    if partial_left == 0:
                        continue
                    partial_left -= 1
                out_flat[o + wptr[c]] = uniq[k]
                wptr[c] += 1
                cnt[k] = c - 1
                bucket[c] -= 1
                bucket[c - 1] += 1
                if c == 1:
                    n_avail -= 1
            for c in range(c_low, cmax + 1):
                take[c] = 0
    return out_flat, out_len


def polyadic_edge_permutation_numba(connectors, rng=None, connector_col='connector_id', pre_col='presynaptic_to', post_col='postsynaptic_to', allow_self_cotarget=True, max_warnings=20):
    ''' Numba-accelerated version of polyadic_edge_permutation.

    Same model: postsynaptic partners are permuted within each presynaptic neuron, keeping
    every connector's number of partners fixed, and rows come out grouped by presynaptic
    neuron. Expects a dataframe similar to connector_details from catmaid, one row per
    connector. With allow_self_cotarget=False the partner choice (including how ties in
    partner frequency are broken) is identical to the pure-python version.

    Differences from the pure-python version:
    - one output row per input row (the python version collapses duplicated connector ids
      within a presynaptic neuron); a warning is printed if such duplicates exist
    - the shuffle is vectorised, so the random draw differs from the python version for the
      same rng seed even though the null model is identical
    - infeasible-connector warnings are capped at max_warnings, then summarised
    '''
    if rng is None:
        rng = np.random.default_rng(42)

    cols = [connector_col, pre_col, post_col]
    if len(connectors) == 0:
        return pd.DataFrame(columns=cols)
    if connectors.duplicated([pre_col, connector_col]).any():
        print('Warning: duplicated connector ids within a presynaptic neuron are kept as separate rows')

    # group rows by presynaptic neuron, first-appearance order, rows within a group in table order
    pre_codes, _ = pd.factorize(connectors[pre_col], sort=False)
    row_order = np.argsort(pre_codes, kind='stable')
    conn_sorted = connectors.iloc[row_order]
    pre_codes = pre_codes[row_order]

    posts = conn_sorted[post_col].to_numpy()
    sizes = np.fromiter((len(p) for p in posts), dtype=np.int64, count=len(posts))
    row_ptr = np.zeros(len(sizes) + 1, dtype=np.int64)
    np.cumsum(sizes, out=row_ptr[1:])

    # concatenate rather than chain: keeps the pool in numpy instead of 6M python objects
    non_empty = [np.asarray(p) for p in posts if len(p)]
    flat_raw = np.concatenate(non_empty) if non_empty else np.asarray([])
    all_ids, codes = np.unique(flat_raw, return_inverse=True)
    codes = codes.astype(np.int64).ravel()

    # row -> group offsets, and the matching pool offsets
    group_row_ptr = np.concatenate((
        np.array([0], dtype=np.int64),
        np.flatnonzero(np.diff(pre_codes)) + 1,
        np.array([len(pre_codes)], dtype=np.int64),
    )).astype(np.int64)
    group_pool_ptr = row_ptr[group_row_ptr]

    # shuffle the partner pool within each presynaptic group (pool is already group-contiguous)
    shuffled = _shuffle_within_groups(codes, group_pool_ptr, rng.random(codes.shape[0]))

    if allow_self_cotarget:
        out_flat, out_len = shuffled, sizes
    else:
        out_flat, out_len = _assign_no_self_cotarget(shuffled, group_pool_ptr, group_row_ptr, sizes, row_ptr[:-1])
        short = np.flatnonzero(out_len < sizes)
        c_ids, p_ids = conn_sorted[connector_col].to_numpy(), conn_sorted[pre_col].to_numpy()
        for r in short[:max_warnings]:
            print(f'Warning: connector {c_ids[r]} (pre {p_ids[r]}): only {out_len[r]} unique '
                  f'partners available, needed {sizes[r]} — numerically infeasible without self-cotarget')
        if len(short) > max_warnings:
            print(f'Warning: {len(short)} connectors in total fell short of their number of '
                  f'partners ({len(short) - max_warnings} more not listed)')

    out_ids = all_ids[out_flat]
    # views rather than .tolist(): saves building a python list per connector, and matches the
    # arrays the real connector table already holds, so con_to_hyperedges takes them as they are
    new_post = [out_ids[row_ptr[r]:row_ptr[r] + out_len[r]] for r in range(len(sizes))]
    return pd.DataFrame({
        connector_col: conn_sorted[connector_col].to_numpy(),
        pre_col: conn_sorted[pre_col].to_numpy(),
        post_col: new_post,
    })

def get_perms(con_sub, n_perms=10, rng=None, connector_col='connector_id', pre_col='presynaptic_to', post_col='postsynaptic_to', allow_self_cotarget=True):
    if rng is None:
        rng = np.random.default_rng(seed=48)
    perms = []
    for i in range(n_perms):
        con_perm = polyadic_edge_permutation_numba(con_sub, rng=rng, connector_col=connector_col, pre_col=pre_col, post_col=post_col, allow_self_cotarget=allow_self_cotarget)
        perms.append(con_perm)
    return perms