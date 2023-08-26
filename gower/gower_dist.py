import math
from functools import partial
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dython.nominal import associations, correlation_ratio
from scipy.sparse import issparse
from scipy.stats import norm
from sklearn.cluster import AgglomerativeClustering, DBSCAN, OPTICS, cluster_optics_dbscan
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score, davies_bouldin_score, \
    calinski_harabasz_score
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
    if hasattr(q, "__iter__"):
        x_n_cols = len(q)
        assert (X.shape[1] == x_n_cols), (X.shape, x_n_cols)
        out = np.array([np.nanpercentile(X.iloc[:, i], 100 - q[i]) -
                        np.nanpercentile(X.iloc[:, i], q[i])
                        for i in range(x_n_cols)])
    else:
        out = np.nanpercentile(X, 100 - q, axis=0) - \
              np.nanpercentile(X, q, axis=0)

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


def gower_matrix(data_x, data_y=None, cat_features=None, weight_cat=None,
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
                X_cat=X_cat, X_num=X_num, Y_cat=Y_cat, Y_num=Y_num,
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


def gower_get(xi_cat, xi_num, xj_cat, xj_num, feature_weight_cat,
              feature_weight_num, feature_weight_sum, g_t, h_t, knn,
              knn_models):
    # categorical columns
    sij_cat = np.where(xi_cat == xj_cat,
                       np.zeros_like(xi_cat),
                       np.ones_like(xi_cat))
    sum_cat = np.multiply(feature_weight_cat, sij_cat).sum(axis=1)

    # numerical columns
    abs_delta = np.absolute(xi_num - xj_num)
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

    return sum_sij


def call_gower_get(i, x_n_rows, y_n_rows, X_cat, X_num, Y_cat, Y_num,
                   weight_cat, weight_num, weight_sum, g_t, h_t, knn,
                   knn_models):
    j_start = i if x_n_rows == y_n_rows else 0
    # call the main function
    res = gower_get(X_cat.iloc[i, :],
                    X_num.iloc[i, :],
                    Y_cat.iloc[j_start:y_n_rows, :],
                    Y_num.iloc[j_start:y_n_rows, :],
                    weight_cat, weight_num, weight_sum, g_t, h_t, knn,
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


def transpose_counts(x):
    x = list(x)
    y = []
    while x:
        if len(x) == 1:
            return y + [1] * x[0]
        y.append(len(x))
        x = [i - 1 for i in x if i > 1]
    return y


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


def niceness(cluster_sizes: Union[np.ndarray[int], list[int]]) -> float:
    """
    Examples:
    ---------
    >>> from gower.gower_dist import *
    >>> C = [x for i in range(7) for x in all_possible_clusters(i)]
    >>> pairs = [(str(tuple(x)), niceness(x)) for x in C]
    >>> for k, v1 in sorted(pairs, key=lambda x: x[1]): print(f"{k:50}{v1:25}")
    (0,)                                                                    nan
    (1,)                                                                    nan
    (1, 1)                                                                  nan
    (2,)                                                                    nan
    (1, 1, 1)                                                               0.0
    (3,)                                                                    0.0
    (1, 1, 1, 1)                                                            0.0
    (4,)                                                                    0.0
    (1, 1, 1, 1, 1)                                                         0.0
    (5,)                                                                    0.0
    (1, 1, 1, 1, 1, 1)                                                      0.0
    (6,)                                                                    0.0
    (1, 1, 1, 1, 2)                                          0.3888888888888889
    (1, 1, 1, 2)                                                            0.5
    (1, 5)                                                   0.5555555555555556
    (1, 1, 2)                                                             0.625
    (1, 4)                                                   0.6666666666666666
    (1, 1, 1, 3)                                             0.6666666666666666
    (1, 1, 2, 2)                                             0.7222222222222222
    (1, 3)                                                                 0.75
    (1, 1, 4)                                                              0.75
    (1, 1, 3)                                                0.7777777777777778
    (1, 2, 2)                                                0.8888888888888888
    (2, 4)                                                   0.8888888888888888
    (1, 2, 3)                                                0.9166666666666666
    (1, 2)                                                                  1.0
    (2, 2)                                                                  1.0
    (2, 3)                                                                  1.0
    (2, 2, 2)                                                               1.0
    (3, 3)                                                                  1.0

    Parameters
    ----------
    cluster_sizes : Union[np.ndarray[int], list[int]]
        A 1D array of cluster sizes.

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
        x = np.array(x)

        n = x.sum()
        n_2 = n ** 2

        # compute factors
        gi0 = n_2 - np.square(x).sum()
        dof0 = n - len(x)
        # T = transpose_counts(x)
        # gi1 = n_2 - np.square(T).sum()
        # dof1 = n - len(T)
        #
        # return max(gi0 * dof0, gi1 * dof1)
        return gi0 * dof0

    total = sum(cluster_sizes)
    if total < 2:
        return np.nan

    out = f(cluster_sizes)
    out /= f(tidy_clusters(total))  # tidy_clusters maximizes f given total

    return out


def gini_coefficient(cluster_sizes: Union[np.ndarray[int], list[int]],
                     normalize=True, return_factors=False):
    """
    Examples:
    ---------
    >>> from gower.gower_dist import *
    >>> C = [x for i in range(7) for x in all_possible_clusters(i)]
    >>> pairs = [(str(tuple(x)), gini_coefficient(x, False),
    ...           gini_coefficient(x)) for x in C]
    >>> for k, v1, v2 in sorted(pairs, key=lambda x: (x[2], x[1])):
    ...     print(f"{k:25}{v1:25}{v2:25}")
    (0,)                                           0.0                      0.0
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
        n_singletons = int(-0.5 + math.sqrt(0.25 + total))  # solve n(n+1)=total
        num1, den1, _ = f([1] * n_singletons + [total - n_singletons])
        if num1:
            if return_factors:
                return num * den1, den * num1
            num *= den1
            num /= num1 * den
            den = 1
    if return_factors:
        return num, den
    if den:
        return num / den
    return 0.0


def neatness(cluster_sizes, normalize=True):
    """
    Examples:
    ---------
    >>> from gower.gower_dist import *
    >>> C = [x for i in range(7) for x in all_possible_clusters(i)]
    >>> pairs = [(str(tuple(x)), neatness(x, False),
    ...           neatness(x)) for x in C]
    >>> for k, v1, v2 in sorted(pairs, key=lambda x: (x[2], x[1])):
    ...     print(f"{k:25}{v1:25}{v2:25}")
    (0,)                                           nan                      nan
    (1,)                                           nan                      nan
    (1, 1)                                         nan                      nan
    (2,)                                           nan                      nan
    (1, 1, 1)                                      0.0                      0.0
    (3,)                                           0.0                      0.0
    (1, 1, 1, 1)                                   0.0                      0.0
    (4,)                                           0.0                      0.0
    (1, 1, 1, 1, 1)                                0.0                      0.0
    (5,)                                           0.0                      0.0
    (1, 1, 1, 1, 1, 1)                             0.0                      0.0
    (6,)                                           0.0                      0.0
    (1, 1, 1, 1, 2)               0.008253968253968255      0.07738095238095238
    (1, 1, 1, 2)                               0.01375      0.17142857142857143
    (1, 1, 1, 3)                                  0.02                   0.1875
    (1, 5)                        0.022222222222222223      0.20833333333333334
    (1, 1, 2)                                    0.025                    0.225
    (1, 1, 4)                                    0.028                   0.2625
    (1, 1, 2, 2)                                 0.032                      0.3
    (1, 1, 3)                                     0.03      0.37402597402597404
    (1, 3)                        0.041666666666666664                    0.375
    (1, 4)                                     0.03125      0.38961038961038963
    (1, 2, 3)                      0.04666666666666667                   0.4375
    (2, 4)                         0.06111111111111111       0.5729166666666666
    (1, 2, 2)                                    0.055       0.6857142857142857
    (2, 2, 2)                                    0.096                      0.9
    (1, 2)                        0.037037037037037035                      1.0
    (2, 3)                         0.08020833333333334                      1.0
    (3, 3)                         0.10666666666666667                      1.0
    (2, 2)                          0.1111111111111111                      1.0
    """
    if not isinstance(cluster_sizes, list):
        cluster_sizes = cluster_sizes.tolist()
    total = sum(cluster_sizes)
    if total < 3:
        return np.nan
    singletons = [1] * int(math.sqrt(total))
    large_cluster = [total ** 2]
    cluster_sizes = sorted(cluster_sizes)

    def g(x):
        a, b = gini_coefficient(x + singletons, return_factors=True)
        c, d = gini_coefficient(x + large_cluster, return_factors=True)
        return (b - a) * (d - c), d * b

    num, den = g(cluster_sizes)
    if not den:
        return 0.0
    if not normalize:
        return num / den

    maximal = (0, 1)
    for k in range(total - 1, 1, -1):
        mu = total // k
        add1 = total - mu * k
        num1, den1 = g([mu] * (k - add1) + [mu + 1] * add1)
        if num1 * maximal[1] > maximal[0] * den1:
            maximal = (num1, den1)

    if not maximal[0]:
        return 1.0
    return num * maximal[1] / (den * maximal[0])


def evaluate_clusters(sample, matrix, actual: pd.Series, method, precomputed):
    if method == cluster_optics_dbscan:
        clusters = cluster_optics_dbscan(
            reachability=precomputed.reachability_,
            core_distances=precomputed.core_distances_,
            ordering=precomputed.ordering_,
            eps=sample["eps"],
        )
    elif precomputed:
        clusters = method(metric="precomputed", **sample).fit_predict(matrix)
    else:
        clusters = method(**sample).fit_predict(matrix)
    clusters = fix_classes(clusters)
    _, counts = np.unique(clusters, return_counts=True)
    counts_dict = dict(zip(*np.unique(counts, return_counts=True)))
    try:
        if precomputed:
            sil = silhouette_score(matrix, clusters, metric="precomputed")
        else:
            sil = silhouette_score(matrix, clusters)
    except ValueError:
        sil = np.nan
    out = {"Silhouette": sil,
           "Niceness": niceness(counts),
           "Neatness": neatness(counts),
           "GiniCoeff": gini_coefficient(counts),
           "len(X)/sum(X)": len(counts) / sum(counts),
           "max(X)/sum(X)": max(counts) / sum(counts),
           "sample": sample,
           "clusters": clusters,
           "counts_dict": counts_dict}
    if not precomputed:
        out.update({"DaviesBouldin": davies_bouldin_score(matrix, clusters),
                    "CalinskiHarabasz": calinski_harabasz_score(matrix, clusters)})
    if actual is not None:
        if actual.dtype == float:
            cr = correlation_ratio(clusters, actual)
            out["CorrRatio"] = cr
        else:
            out["AdjRandIndex"] = adjusted_rand_score(actual, clusters)
            out["AdjMutualInfo"] = adjusted_mutual_info_score(actual, clusters)
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
                               ["sample", "clusters", "counts_dict"]})

    if actual.dtype != float:
        max_muti = np.max(df_results.AdjMutualInfo)
        max_rand = np.max(df_results.AdjRandIndex)
        amax_gini = np.argmax(df_results.GiniCoeff)
        amin_stu0 = np.argmin(np.maximum(df_results["max(X)/sum(X)"], df_results["len(X)/sum(X)"]))
        amin_stu1 = np.argmin(df_results["max(X)/sum(X)"] + df_results["len(X)/sum(X)"])
        amax_nice = np.argmax(df_results.Niceness)
        amax_neat = np.argmax(df_results.Neatness)
        amax_silh = np.argmax(df_results.Silhouette)
        print("Best possible MutualInfo SCORE: ", max_muti)
        print("Best possible RandIndex  SCORE: ", max_rand)
        print("New metrics:")
        print("Max(K, L)     MutualInfo loss:  ", df_results.AdjMutualInfo.iloc[amin_stu0] - max_muti)
        print("Max(K, L)     RandIndex  loss:  ", df_results.AdjRandIndex.iloc[amin_stu0] - max_rand)
        print("K + L         MutualInfo loss:  ", df_results.AdjMutualInfo.iloc[amin_stu1] - max_muti)
        print("K + L         RandIndex  loss:  ", df_results.AdjRandIndex.iloc[amin_stu1] - max_rand)
        print("Niceness      MutualInfo loss:  ", df_results.AdjMutualInfo.iloc[amax_nice] - max_muti)
        print("Niceness      RandIndex  loss:  ", df_results.AdjRandIndex.iloc[amax_nice] - max_rand)
        print("Neatness      MutualInfo loss:  ", df_results.AdjMutualInfo.iloc[amax_neat] - max_muti)
        print("Neatness      RandIndex  loss:  ", df_results.AdjRandIndex.iloc[amax_neat] - max_rand)
        print("Old metrics:")
        print("GiniCoeff     MutualInfo loss:  ", df_results.AdjMutualInfo.iloc[amax_gini] - max_muti)
        print("GiniCoeff     RandIndex  loss:  ", df_results.AdjRandIndex.iloc[amax_gini] - max_rand)
        print("Silhouette    MutualInfo loss:  ", df_results.AdjMutualInfo.iloc[amax_silh] - max_muti)
        print("Silhouette    RandIndex  loss:  ", df_results.AdjRandIndex.iloc[amax_silh] - max_rand)
        if not precomputed:
            amax_dabo = np.argmax(df_results.DaviesBouldin)
            amax_caha = np.argmax(df_results.CalinskiHarabasz)
            print("DaviesBouldin    MutualInfo loss:", df_results.AdjMutualInfo.iloc[amax_dabo] - max_muti)
            print("DaviesBouldin    RandIndex  loss:", df_results.AdjRandIndex.iloc[amax_dabo] - max_rand)
            print("CalinskiHarabasz MutualInfo loss:", df_results.AdjMutualInfo.iloc[amax_caha] - max_muti)
            print("CalinskiHarabasz RandIndex  loss:", df_results.AdjRandIndex.iloc[amax_caha] - max_rand)
        best = np.argmax(df_results.AdjRandIndex + df_results.AdjMutualInfo)
    else:
        best = np.argmin((df_results.CorrRatio - 0.5).abs())
    best_params = results[best]

    # assign clusters
    df["cluster"] = best_params["clusters"]
    df.cluster = df.cluster.astype(str)

    plt.figure(figsize=(10, 10))

    # plot param vs. metrics
    var = [z["sample"][param] for z in results]
    legend = ["Niceness", "Neatness", "GiniCoeff", "len(X)/sum(X)", "max(X)/sum(X)"] + (
        (["CorrRatio"] if actual.dtype == float else
         ["AdjRandIndex", "AdjMutualInfo", ] + (
             ["DaviesBouldin", "CalinskiHarabasz"] if not precomputed else [])
         ) + ["Silhouette"] if actual is not None else [])
    for col in legend:
        if col in ["CalinskiHarabasz", "DaviesBouldin"]:
            plt.plot(var, df_results[col] / df_results[col].max())
        else:
            plt.plot(var, df_results[col])

    plt.axvline(best_params["sample"][param], c="black", ls="--")
    plt.legend(legend)
    plt.show()

    # display corr
    n_cols = df.shape[1]
    if n_cols < 20:
        corr = associations(df, nom_nom_assoc="theil",
                            figsize=(n_cols, n_cols))["corr"]

    # print results
    del best_params["clusters"]
    print(best_params)

    return np.min((df_results.CorrRatio - 0.5).abs()) if actual.dtype == float \
        else np.max(df_results.AdjRandIndex + df_results.AdjMutualInfo)


def optimize_dbscan(df, actual=None, factor=0.25, offset=0.0, n_iter=1000,
                    use_mp=True, min_samples=1, precomputed=False, **kwargs):
    df = df.copy()

    if precomputed:
        matrix = gower_matrix(df.to_numpy(), use_mp=use_mp, **kwargs)
    else:
        matrix = df.to_numpy()

    samples = [{"eps": offset + factor * z / n_iter, "min_samples": min_samples}
               for z in range(1, n_iter + 1)]
    ni = sample_params(df, matrix, actual, DBSCAN, samples, "eps", n_iter, precomputed,
                       use_mp)

    return df, ni


def optimize_gm(df, actual=None, n_iter=10, use_mp=True):
    df = df.copy()

    matrix = df.to_numpy()

    samples = [{"n_components": z, "random_state": 42}
               for z in range(2, n_iter + 2)]
    sample_params(df, matrix, actual, GaussianMixture, samples, "n_components",
                  n_iter, False, use_mp)

    return df


def optimize_agglo(df, actual=None, n_iter=10, use_mp=True, precomputed=False,
                   **kwargs):
    df = df.copy()

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
