import math
from functools import partial
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dython.nominal import associations, correlation_ratio
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import issparse
from scipy.stats import norm
from sklearn.cluster import (
    AgglomerativeClustering, DBSCAN, OPTICS, cluster_optics_dbscan)
from sklearn.metrics import (
    adjusted_mutual_info_score, adjusted_rand_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def get_cat_features(X):
    x_n_cols = X.shape[1]
    if isinstance(X, pd.DataFrame):
        is_number = np.vectorize(lambda x: not np.issubdtype(x, np.number))
        cat_features = is_number(X.dtypes)
    else:
        cat_features = np.zeros(x_n_cols, dtype=bool)
        for col in range(x_n_cols):
            if not np.issubdtype(type(X[0, col]), np.number):
                cat_features[col] = True
    return cat_features


def get_percentile_range(X, q):
    out = np.nanpercentile(X, 100 - q, axis=0) - np.nanpercentile(X, q, axis=0)
    out[out == 0] = 1
    return out


def get_num_weight(x):
    """
    This value is always between 1 and sqrt(len(x)).
    It represents the "resolution" of the column in terms of perplexity.
    Binary variables get the lowest weight of 1 due to no perplexity.
    """
    assert 0 <= np.nanmin(x) <= np.nanmax(x) <= 1, x
    x = np.array([i for i in x if i is not None])
    x = x[~np.isnan(x)] * 1.0
    x = np.diff(np.sort(x))  # a pmf of ordered categories
    return math.sqrt(np.prod(x ** -x))  # perplexity


def gower_matrix(data_x, data_y=None, p=1.0, cat_features=None, weight_cat=None,
                 weight_num=None, lower_q=0.0, c_t=0.0, knn=False, use_mp=True,
                 **tqdm_kwargs):
    # function checks
    X = data_x
    if data_y is None:
        Y = data_x
    else:
        Y = data_y
    if isinstance(X, pd.DataFrame):
        if not np.array_equal(X.columns, Y.columns):
            raise TypeError("X and Y must have same columns!")
    else:
        if not X.shape[1] == Y.shape[1]:
            raise TypeError("X and Y must have same y-dim!")

    if issparse(X) or issparse(Y):
        raise TypeError("Sparse matrices are not supported!")

    x_n_rows, x_n_cols = X.shape
    y_n_rows, y_n_cols = Y.shape

    if cat_features is None:
        cat_features = get_cat_features(X)
    else:
        cat_features = np.array(cat_features)

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(Y, pd.DataFrame):
        Y = pd.DataFrame(Y)

    Z = pd.concat((X, Y))

    x_index = range(0, x_n_rows)
    y_index = range(x_n_rows, x_n_rows + y_n_rows)

    Z_num = Z.loc[:, np.logical_not(cat_features)].astype(float)
    Z_num -= Z_num.min()
    Z_num /= Z_num.max()
    Z_num.fillna(1, inplace=True)

    num_cols = Z_num.shape[1]

    g_t = get_percentile_range(Z_num, 100 * lower_q)
    knn_models = []
    if np.any(knn):
        if isinstance(knn, bool):
            knn = np.ones(num_cols, dtype=bool)
        n_knn = math.ceil(math.sqrt(x_n_rows))
        for col in np.where(knn)[0]:
            values = Z_num.iloc[:, col].dropna().to_numpy()
            knn_models.append((values, NearestNeighbors(n_neighbors=n_knn).fit(
                values.reshape(-1, 1))))

    Z_cat = Z.loc[:, cat_features]
    cat_cols = Z_cat.shape[1]

    # weights

    if isinstance(weight_cat, str):
        if weight_cat == "uniform":
            weight_cat = np.ones(cat_cols)
        else:
            raise ValueError("Unknown weight_cat: {}".format(weight_cat))
    elif weight_cat is None:
        # if use_mp:
        #     weight_cat = process_map(get_cat_weight, Z_cat.T.to_numpy(),
        #                              **tqdm_kwargs)
        # else:
        #     weight_cat = [get_cat_weight(Z_cat[:, col]) for col in
        #                   tqdm(range(cat_cols))]
        weight_cat = np.ones(cat_cols)
    weight_cat = np.array(weight_cat)

    if isinstance(weight_num, str):
        if weight_num == "uniform":
            weight_num = np.ones(num_cols)
        else:
            raise ValueError("Unknown weight_num: {}".format(weight_num))
    elif weight_num is None:
        if use_mp:
            weight_num = process_map(get_num_weight, Z_num.T.to_numpy(),
                                     **tqdm_kwargs)
        else:
            weight_num = [get_num_weight(Z_num.iloc[:, col]) for col in
                          tqdm(range(num_cols))]
    weight_num = np.array(weight_num)

    print(weight_cat, weight_num)
    weight_sum = weight_cat.sum() + weight_num.sum()

    # distance matrix

    out = np.zeros((x_n_rows, y_n_rows))

    X_cat = Z_cat.iloc[x_index, ]
    X_num = Z_num.iloc[x_index, ]
    Y_cat = Z_cat.iloc[y_index, ]
    Y_num = Z_num.iloc[y_index, ]

    h_t = np.zeros(num_cols)
    if np.any(c_t > 0):
        dist = norm(0, 1)
        h_t = c_t * x_n_rows ** -0.2 * np.minimum(
            Z_num.std(),
            g_t / (dist.ppf(1 - lower_q) - dist.ppf(lower_q)))
        print("h_t:", h_t)
    f = partial(call_gower_get, x_n_rows=x_n_rows, y_n_rows=y_n_rows,
                X_cat=X_cat, X_num=X_num, p=p, Y_cat=Y_cat, Y_num=Y_num,
                weight_cat=weight_cat, weight_num=weight_num,
                weight_sum=weight_sum, g_t=g_t, h_t=h_t, knn=knn,
                knn_models=knn_models.copy())
    if use_mp:
        processed = process_map(f, range(x_n_rows), **tqdm_kwargs)
    else:
        processed = list(map(f, tqdm(range(x_n_rows))))
    for i, res in enumerate(processed):
        j_start = i if x_n_rows == y_n_rows else 0
        out[i, j_start:] = res
        if x_n_rows == y_n_rows:
            out[i:, j_start] = res

    max_distance = np.nanmax(out)
    assert math.isclose(max_distance, 1) or (max_distance < 1), max_distance

    return out


def gower_get(xi_cat, xi_num, xj_cat, xj_num, p, feature_weight_cat,
              feature_weight_num, feature_weight_sum, g_t, h_t, knn,
              knn_models):
    # categorical columns
    sij_cat = np.where(xi_cat == xj_cat,
                       np.zeros_like(xi_cat),
                       np.ones_like(xi_cat))
    sum_cat = np.multiply(feature_weight_cat, sij_cat).sum(axis=1)

    # numerical columns
    abs_delta = np.absolute(xi_num - xj_num) ** p
    abs_delta = np.maximum(abs_delta - h_t, np.zeros_like(abs_delta))
    xi_num = xi_num.to_numpy()
    if knn_models:
        for i in np.where(knn)[0]:
            values, knn_model = knn_models.pop(0)
            xi = xi_num[i]
            if np.isnan(xi).any():
                continue
            neighbors = knn_model.kneighbors(xi.reshape(-1, 1),
                                             return_distance=False)
            neighbors = values[neighbors]
            for j, x in enumerate(xj_num.iloc[:, i]):
                if x in neighbors:
                    abs_delta.iloc[j, i] = 0.0
    sij_num = abs_delta.to_numpy() / g_t
    sij_num = np.minimum(sij_num, np.ones_like(sij_num))

    sum_num = np.multiply(feature_weight_num, sij_num).sum(axis=1)
    sums = np.add(sum_cat, sum_num)
    sum_sij = np.divide(sums, feature_weight_sum)
    sum_sij **= 1 / p

    return sum_sij


def call_gower_get(i, x_n_rows, y_n_rows, X_cat, X_num, p, Y_cat, Y_num,
                   weight_cat, weight_num, weight_sum, g_t, h_t, knn,
                   knn_models):
    j_start = i if x_n_rows == y_n_rows else 0
    # call the main function
    res = gower_get(X_cat.iloc[i, :],
                    X_num.iloc[i, :],
                    Y_cat.iloc[j_start:y_n_rows, :],
                    Y_num.iloc[j_start:y_n_rows, :],
                    p, weight_cat, weight_num, weight_sum, g_t, h_t, knn,
                    knn_models)
    return res


def smallest_indices(ary, n):
    """Returns the n smallest indices from a numpy array."""
    flat = np.nan_to_num(ary.flatten(), nan=999)
    indices = np.argpartition(-flat, -n)[-n:]
    indices = indices[np.argsort(flat[indices])]
    values = flat[indices]
    return {'index': indices, 'values': values}


def gower_topn(data_x, n=5, **kwargs):
    if data_x.shape[0] >= 2:
        TypeError("Only support `data_x` of 1 row. ")
    dm = gower_matrix(data_x, **kwargs)

    return smallest_indices(np.nan_to_num(dm[0], nan=1), n)


def hamming_similarity(df):
    df = df.astype(str)
    return df.values[:, None] == df.values


def fix_classes(x):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    x = [i for i in x if i is not None]
    if "-1" in x:
        x = [int(i) for i in x]
    if -1 in x:
        assert not any(i < -1 for i in x), x
        x = [i for i in x if i != -1] + \
            list(range(-1, -1 - list(x).count(-1), -1))
    return x


def all_possible_clusters(n, memo=None):
    """
    *** This function was (mostly) written by GitHub Copilot and ChatGPT. ***
    *** See also s.o. 24582741/5295786 ***

    This function returns a list of all possible clusterings given a number of
    elements to cluster. The clusterings are returned as a list of lists of
    integers. The integers are the element counts per cluster.

    For example, if there are 3 elements, then the possible clusterings are:
    [[1, 1, 1], [1, 2], [3]]

    This function is not recommended for n > 20.
    """
    if n == 1:
        return [[1]]
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    out = []
    for i in range(1, n):
        for j in all_possible_clusters(n - i, memo):
            out.append(sorted([i] + j))
    out.append([n])
    out = [c for i, c in enumerate(out) if c not in out[:i]]
    memo[n] = out
    return out


def tidy_clusters(n):
    r = math.sqrt(n)
    r_floor = int(r)
    r_ceil = r_floor + 1
    diff_floor = n - r_floor ** 2
    diff_ceil = r_ceil ** 2 - n
    if r > r_floor:
        if diff_ceil < diff_floor:
            return [r_floor] * diff_ceil + \
                [r_ceil] * (r_ceil - diff_ceil)
        else:
            return [r_ceil] * diff_floor + \
                [r_floor] * (r_floor - diff_floor)
    else:
        return [r_floor] * r_floor


def cluster_niceness(cluster_sizes: Union[np.ndarray[int], list[int]],
                     normalize=True) -> float:
    """
    This value tells you to what extent clusters are "nice". It is not a measure
    of the separation between clusters such as the Davies-Bouldin index, but
    rather a measure of how well clusters are distributed.

    Note that only a square number of elements permits a return value of 1.0,
    but any number of elements could return a value of 0.0 (besides 1, which
    returns NaN).

    If there are fewer than 2 elements, the value is undefined.
    If the elements comprise one cluster, or the clusters are all singletons,
    the value is 0 as useless clusters are not "nice".
    Otherwise, if the elements are evenly distributed, and the number of
    clusters equals the number of elements per cluster, the value is 1.
    In all other cases the value is on the open interval (0, 1).

    One additional property of this function is that it is symmetric with
    respect to the number of clusters and the number of elements per cluster
    when the elements are evenly distributed:

    cluster_niceness([a]*b) == cluster_niceness([b]*a)

    This function is designed to be used in conjunction with grid search and
    DBSCAN to find the best value for the "eps" parameter.

    Example 1: Clusterings of 1-6 elements
    ---------------------------------------
    >>> from gower.gower_dist import *
    >>> C = [x for i in range(1, 7) for x in all_possible_clusters(i)]
    >>> pairs = [(str(tuple(x)), cluster_niceness(x, False), cluster_niceness(x)
    ...           ) for x in C]
    >>> for k, v1, v2 in sorted(pairs, key=lambda x: (x[2], x[1])):
    ...     print(f"{k:25}{v1:25}{v2:25}")
    (1,)                                           nan                      nan
    (1, 1)                                         0.0                      nan
    (2,)                                           0.0                      nan
    (1, 1, 1)                                      0.0                      0.0
    (3,)                                           0.0                      0.0
    (1, 1, 1, 1)                                   0.0                      0.0
    (4,)                                           0.0                      0.0
    (1, 1, 1, 1, 1)                                0.0                      0.0
    (5,)                                           0.0                      0.0
    (1, 1, 1, 1, 1, 1)                             0.0                      0.0
    (6,)                                           0.0                      0.0
    (1, 5)                         0.27967548206998305        0.308641975308642
    (1, 1, 1, 1, 2)                 0.3034661711693837       0.3219516401426915
    (1, 1, 1, 2)                    0.3904448798314012      0.42044820762685725
    (1, 4)                          0.3947962732559819       0.4444444444444444
    (1, 1, 2)                       0.5343674833364678       0.5343674833364678
    (1, 1, 1, 3)                    0.5055483130871623       0.5443310539518174
    (1, 1, 4)                       0.5097085660725442                   0.5625
    (1, 3)                                      0.5625                   0.5625
    (1, 1, 2, 2)                    0.5700406478222112       0.6137708673769092
    (1, 1, 3)                       0.5959669640956297       0.6577980034215026
    (2, 4)                          0.7159692340991565       0.7901234567901234
    (1, 2, 2)                       0.7445183931563902       0.8217615103414929
    (1, 2, 3)                       0.7614164999355287       0.8402777777777777
    (1, 2)                          0.7552705498301204                      1.0
    (2, 3)                          0.8882916148259593                      1.0
    (2, 2, 2)                       0.9061485619067451                      1.0
    (3, 3)                          0.9061485619067451                      1.0
    (2, 2)                                         1.0                      1.0

    Parameters
    ----------
    cluster_sizes : Union[np.ndarray[int], list[int]]
        A 1D array of cluster sizes.
    normalize : bool, optional
        Whether to normalize the intermediate result by the maximum possible
        value.

    Returns
    -------
    float
        A float on the closed interval [0, 1].

    Raises
    ------
    ValueError
        If the number of elements in any cluster is not a natural number.
    """
    def f(x):
        # check inputs, convert to numpy array
        if any(x < 1 or x != int(x) for x in x):
            raise ValueError("Every count must be a positive integer.")
        x = np.array(x, dtype=float)

        _k = len(x)
        _n = x.sum()
        _n_2 = _n ** 2

        # compute factors
        gi = _n_2 - np.square(x).sum()
        dof = _n - _k
        _out = gi * dof

        return _k, _n, _n_2, _out

    k, n, n_2, out = f(cluster_sizes)

    if normalize:
        # get the nicest possible clustering for the given number of elements
        _, _, _, out_ = f(tidy_clusters(int(n)))
        out /= out_
    else:  # n is square <==> nicest possible == 1
        # divide by the denoms that would have cancelled had we normalized
        out /= n_2
        out /= n - 2 * math.sqrt(n) + 1  # compute (sqrt(n)-1)^2

    return out ** min(k, n / k)  # "gamma" correction


def gini_coefficient(cluster_sizes: Union[np.ndarray[int], list[int]],
                     normalize=True, return_factors=False):
    """
    Examples:
    ---------
    >>> from gower.gower_dist import *
    >>> C = [x for i in range(1, 7) for x in all_possible_clusters(i)]
    >>> pairs = [(str(tuple(x)), gini_coefficient(x, False),
    ...           gini_coefficient(x)) for x in C]
    >>> for k, v1, v2 in sorted(pairs, key=lambda x: (x[2], x[1])):
    ...     print(f"{k:25}{v1:25}{v2:25}")
    (1,)                                           0.0                      0.0
    (1, 1)                                         0.0                      0.0
    (2,)                                           0.0                      0.0
    (1, 1, 1)                                      0.0                      0.0
    (3,)                                           0.0                      0.0
    (1, 1, 1, 1)                                   0.0                      0.0
    (2, 2)                                         0.0                      0.0
    (4,)                                           0.0                      0.0
    (1, 1, 1, 1, 1)                                0.0                      0.0
    (5,)                                           0.0                      0.0
    (1, 1, 1, 1, 1, 1)                             0.0                      0.0
    (2, 2, 2)                                      0.0                      0.0
    (3, 3)                                         0.0                      0.0
    (6,)                                           0.0                      0.0
    (2, 3)                                         0.1       0.3333333333333333
    (1, 1, 1, 1, 2)                0.13333333333333333                      0.4
    (1, 2, 2)                      0.13333333333333333       0.4444444444444444
    (1, 1, 1, 2)                                  0.15                      0.5
    (1, 1, 2, 2)                   0.16666666666666666                      0.5
    (2, 4)                         0.16666666666666666                      0.5
    (1, 1, 2)                      0.16666666666666666       0.6666666666666666
    (1, 2, 3)                       0.2222222222222222       0.6666666666666666
    (1, 1, 1, 3)                                  0.25                     0.75
    (1, 1, 3)                      0.26666666666666666       0.8888888888888888
    (1, 2)                         0.16666666666666666                      1.0
    (1, 3)                                        0.25                      1.0
    (1, 4)                                         0.3                      1.0
    (1, 1, 4)                       0.3333333333333333                      1.0
    (1, 5)                          0.3333333333333333                      1.0
    """
    def f(x):
        x = sorted(x)
        n = len(x)
        s = sum(x)
        d = n * s
        G = sum(xi * (n - i) for i, xi in enumerate(x))
        return d + s - 2 * G, d, s
    num, den, total = f(cluster_sizes)
    if normalize:
        n_ones = int(-0.5 + math.sqrt(0.25 + total))  # solve n(n+1) = total
        num1, den1, _ = f([1] * n_ones + [total - n_ones])
        if num1:
            if return_factors:
                return num * den1, den * num1
            num *= den1
            num /= num1 * den
            den = 1
    if return_factors:
        return num, den
    return num / den


def cluster_neatness(cluster_sizes, normalize=False):
    """
    Examples:
    ---------
    >>> from gower.gower_dist import *
    >>> C = [x for i in range(3, 7) for x in all_possible_clusters(i)]
    >>> pairs = [(str(tuple(x)), cluster_neatness(x),
    ...           cluster_neatness(x, True)) for x in C]
    >>> for k, v1, v2 in sorted(pairs, key=lambda x: (x[2], x[1])):
    ...     print(f"{k:25}{v1:25}{v2:25}")
    (1, 1, 1)                                      0.0                      0.0
    (3,)                                           0.0                      0.0
    (1, 1, 1, 1)                                   0.0                      0.0
    (4,)                                           0.0                      0.0
    (1, 1, 1, 1, 1)                                0.0                      0.0
    (5,)                                           0.0                      0.0
    (1, 1, 1, 1, 1, 1)                             0.0                      0.0
    (6,)                                           0.0                      0.0
    (1, 1, 1, 1, 2)               0.011791383219954649      0.11607142857142858
    (1, 5)                        0.022222222222222223                  0.21875
    (1, 1, 1, 3)                  0.023809523809523808                 0.234375
    (1, 1, 1, 2)                  0.018333333333333333                     0.24
    (1, 1, 4)                                     0.03                0.2953125
    (1, 1, 2)                                     0.03                      0.3
    (1, 1, 2, 2)                   0.03428571428571429                   0.3375
    (1, 4)                                     0.03125       0.4090909090909091
    (1, 3)                        0.041666666666666664       0.4166666666666667
    (1, 1, 3)                      0.03333333333333333      0.43636363636363634
    (1, 2, 3)                     0.047619047619047616                  0.46875
    (2, 4)                         0.05952380952380952                0.5859375
    (1, 2, 2)                                    0.055                     0.72
    (2, 2, 2)                      0.09142857142857143                      0.9
    (1, 2)                        0.037037037037037035                      1.0
    (2, 3)                          0.0763888888888889                      1.0
    (2, 2)                                         0.1                      1.0
    (3, 3)                         0.10158730158730159                      1.0
    """
    if not isinstance(cluster_sizes, list):
        cluster_sizes = cluster_sizes.tolist()
    if cluster_sizes == [1, 2] and normalize:  # special case
        return 1.0
    total = sum(cluster_sizes)
    n_ones = int(math.sqrt(total))

    def f(x):
        # What if new elements are introduced?
        # How robust is the Gini coefficient of our clustering to new elements
        # comprising singletons and new, very large clusters?
        # In other words, not all values of 0 for the Gini coefficient are
        # equal. We want to find the one that is most robust to new elements.
        # In order to do this, we consider two scenarios:
        # 1. The new elements are all singletons
        # 2. The new elements are all in a new, single cluster
        # We then compute (1-gini(scenario_1))*(1-gini(scenario_2))
        a, b = gini_coefficient(n_ones * [1] + x, return_factors=True)
        c, d = gini_coefficient(x + [(total + 1) ** 2 + 1], return_factors=True)
        bd = b * d
        return bd - a * d - b * c + a * c, bd

    num, den = f(cluster_sizes)
    if not den:
        return 0.0
    if not normalize:
        return num / den

    maximal = (0.0, 1.0)
    for k in range(n_ones, 0, -1):
        mu = total // k
        add1 = total - mu * k
        num1, den1 = f([mu + 1] * add1 + [mu] * (k - add1))
        if num1 * maximal[1] > maximal[0] * den1:
            maximal = (num1, den1)

    if maximal[0]:
        return num * maximal[1] / (den * maximal[0])
    return num / den


def evaluate_clusters(sample, matrix, actual: pd.Series, method, precomputed):
    if method == cluster_optics_dbscan:
        assignments = cluster_optics_dbscan(
            reachability=precomputed.reachability_,
            core_distances=precomputed.core_distances_,
            ordering=precomputed.ordering_,
            eps=sample["eps"],
        )
    elif precomputed:
        assignments = method(metric="precomputed", **sample).fit_predict(matrix)
    else:
        assignments = method(**sample).fit_predict(matrix)
    assignments = fix_classes(assignments)
    unique, counts = np.unique(assignments, return_counts=True)
    # unique = [chr(i + 65) for i in unique]
    try:
        if precomputed:
            sil = silhouette_score(matrix, assignments, metric="precomputed")
        else:
            sil = silhouette_score(matrix, assignments)
    except ValueError:
        sil = np.nan
    nice = cluster_niceness(counts)
    neat = cluster_neatness(counts)
    gini = gini_coefficient(counts, False)
    ratio = len(counts) / sum(counts)
    out = {"sample": sample, "silhouette": sil, "niceness": nice,
           "neatness": neat, "gini": gini, "ratio": ratio,
           "assignments": assignments,
           "counts_dict": dict(zip(unique, counts))}
    if actual is not None:
        if actual.dtype == float:
            cr = correlation_ratio(assignments, actual)
            out["CR"] = cr
        else:
            out["AR"] = adjusted_rand_score(actual, assignments)
            out["AMI"] = adjusted_mutual_info_score(actual, assignments)
    return out


def sample_params(df, matrix, actual, method, samples, param, n_iter, precomputed,
                  use_mp):

    # do grid search to get best parameters
    if use_mp:
        results = process_map(partial(evaluate_clusters, matrix=matrix,
                                      actual=actual, method=method,
                                      precomputed=precomputed),
                              samples, chunksize=math.ceil(n_iter / 40))
    else:
        results = [evaluate_clusters(sample, matrix, actual, method,
                                     precomputed) for sample in tqdm(samples)]
    df_results = pd.DataFrame({key: [z[key] for z in results]
                               for key in results[0].keys() if key not in
                               ["sample", "assignments", "counts_dict"]})
    # for i in range(5):
    #     df_results.iloc[:, i] = gaussian_filter1d(df_results.iloc[:, i],
    #                                               sigma=df_results.iloc[:, i].std(),
    #                                               axis=0)
    best = np.argmax(df_results.neatness)
    best_params = results[best]

    # assign clusters
    df["cluster"] = best_params["assignments"]
    df.cluster = df.cluster.astype(str)

    neatest = df_results.neatness.max()
    df_results.neatness /= neatest

    # plt
    var = [z["sample"][param] for z in results]
    for col in df_results.columns:
        if col in ["counts_dict", "assignments"]:
            continue
        plt.plot(var, df_results[col])
    plt.axvline(best_params["sample"][param], c="black", ls="--")
    legend = ["silhouette", "niceness", "neatness (%.6f)" % neatest,
              "gini", "k/N"] + ((["CR"] if actual.dtype == float else ["AR", "AMI"])
                                if actual is not None else [])
    plt.legend(legend)

    # corr
    corr = associations(df, nom_nom_assoc="theil",
                        figsize=(df.shape[1], df.shape[1]))["corr"]

    # print results
    del best_params["assignments"]
    print(best_params)


def optimize_dbscan(df, actual=None, factor=10.0, offset=0.0, n_iter=1000,
                    use_mp=True, min_samples=1, precomputed=False, **kwargs):
    df = df.copy()

    # get distance matrix
    if precomputed:
        matrix = gower_matrix(df.to_numpy(), use_mp=use_mp, **kwargs)
    else:
        matrix = df.to_numpy()

    samples = [{"eps": offset + factor * z / n_iter, "min_samples": min_samples}
               for z in range(1, n_iter + 1)]
    sample_params(df, matrix, actual, DBSCAN, samples, "eps", n_iter, precomputed,
                  use_mp)

    return df


def optimize_gm(df, actual=None, n_iter=10, use_mp=True):
    df = df.copy()

    # get distance matrix
    matrix = df.to_numpy()

    samples = [{"n_components": z, "random_state": 42}
               for z in range(2, n_iter + 2)]
    sample_params(df, matrix, actual, GaussianMixture, samples, "n_components",
                  n_iter, False, use_mp)

    return df


def optimize_agglo(df, actual=None, n_iter=10, use_mp=True, precomputed=False,
                   **kwargs):
    df = df.copy()

    # get distance matrix
    if precomputed:
        matrix = gower_matrix(df.to_numpy(), use_mp=use_mp, **kwargs)
    else:
        matrix = df.to_numpy()

    samples = [{"n_clusters": z, "linkage": "average"}
               for z in range(2, n_iter + 2)]
    sample_params(df, matrix, actual, AgglomerativeClustering, samples, "n_clusters",
                  n_iter, precomputed, use_mp)

    return df


def optimize_cluster_optics_dbscan(df, actual=None, factor=10.0, offset=0.0,
                                   n_iter=1000, use_mp=True, min_samples=1,
                                   metric="minkowski", **kwargs):
    df = df.copy()

    # get distance matrix
    if metric == "precomputed":
        matrix = gower_matrix(df.to_numpy(), use_mp=use_mp, **kwargs)
    else:
        matrix = df.to_numpy()

    clust = OPTICS(min_samples=min_samples, metric=metric)
    clust.fit(matrix)

    samples = [{"eps": offset + factor * z / n_iter}
               for z in range(1, n_iter + 1)]
    sample_params(df, matrix, actual, cluster_optics_dbscan, samples, "eps",
                  n_iter, clust, use_mp)

    return df
