import math
from functools import partial
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dython.nominal import associations, correlation_ratio
from hdbscan import HDBSCAN
from kneed import KneeLocator
from scipy.sparse import issparse
from scipy.stats import norm
from sklearn.cluster import (
    DBSCAN,
    AgglomerativeClustering,
    OPTICS,
    cluster_optics_dbscan,
    KMeans,
    SpectralClustering,
    Birch,
    AffinityPropagation,
    MeanShift,
)
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
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
        assert X.shape[1] == x_n_cols, (X.shape, x_n_cols)
        out = np.array(
            [
                np.nanpercentile(X.iloc[:, i], 100 - q[i])
                - np.nanpercentile(X.iloc[:, i], q[i])
                for i in range(x_n_cols)
            ]
        )
    else:
        out = np.nanpercentile(X, 100 - q, axis=0) - np.nanpercentile(X, q, axis=0)

    out[out == 0] = 1
    return out


def get_num_weight(x):
    """
    This value is always between 1 and sqrt(len(x)).
    It represents the "resolution" of the column in terms of perplexity.
    Binary variables get the lowest weight of 1 due to no perplexity.
    """
    x = np.array([i for i in x if i is not None])
    x = x[~np.isnan(x)] * 1.0
    x = np.diff(np.sort(x))  # a pmf of ordered categories
    return math.sqrt(np.prod(x**-x))  # perplexity


def gower_matrix(
    data_x,
    data_y=None,
    cat_features=None,
    circular_features=None,
    weight_cat=None,
    weight_cir=None,
    weight_num=None,
    lower_q=0.0,
    c_t=0.0,
    knn=False,
    use_mp=True,
    **tqdm_kwargs
):
    # function checks
    X = data_x
    Y = data_x if data_y is None else data_y
    if isinstance(X, pd.DataFrame):
        if not np.array_equal(X.columns, Y.columns):
            raise TypeError("X and Y must have same columns!")
    elif X.shape[1] != Y.shape[1]:
        raise TypeError("X and Y must have same y-dim!")

    if issparse(X) or issparse(Y):
        raise TypeError("Sparse matrices are not supported!")

    x_n_rows, x_n_cols = X.shape
    y_n_rows, y_n_cols = Y.shape

    if cat_features is None:
        cat_features = get_cat_features(X)
    else:
        cat_features = np.array(cat_features)

    if circular_features is None:
        circular_features = np.zeros(x_n_cols, dtype=bool)
    else:
        circular_features = np.array(circular_features)

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(Y, pd.DataFrame):
        Y = pd.DataFrame(Y)

    Z = pd.concat((X, Y))

    x_index = range(x_n_rows)
    y_index = range(x_n_rows, x_n_rows + y_n_rows)

    Z_num = Z.loc[
        :, np.logical_not(cat_features) & np.logical_not(circular_features)
    ].astype(float)
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
            knn_models.append(
                (values, NearestNeighbors(n_neighbors=n_knn).fit(values.reshape(-1, 1)))
            )

    Z_cat = Z.loc[:, cat_features]
    cat_cols = Z_cat.shape[1]

    cir_mask = circular_features > 0
    Z_cir = Z.loc[:, cir_mask]
    periods = circular_features[cir_mask]

    # weights

    if weight_cir is None:
        weight_cir = np.ones(cir_mask.sum()) * np.sqrt(periods // 2)

    if (
        isinstance(weight_cat, str)
        and weight_cat == "uniform"
        or not isinstance(weight_cat, str)
        and weight_cat is None
    ):
        weight_cat = np.ones(cat_cols)
    elif isinstance(weight_cat, str):
        raise ValueError(f"Unknown weight_cat: {weight_cat}")
    weight_cat = np.array(weight_cat)

    if isinstance(weight_num, str):
        if weight_num == "uniform":
            weight_num = np.ones(num_cols)
        else:
            raise ValueError(f"Unknown weight_num: {weight_num}")
    elif weight_num is None:
        if use_mp:
            weight_num = process_map(get_num_weight, Z_num.T.to_numpy(), **tqdm_kwargs)
        else:
            weight_num = [
                get_num_weight(Z_num.iloc[:, col]) for col in tqdm(range(num_cols))
            ]
    weight_num = np.array(weight_num)

    print(weight_cat, weight_num, weight_cir)
    weight_sum = weight_cat.sum() + weight_num.sum() + weight_cir.sum()

    # distance matrix

    out = np.zeros((x_n_rows, y_n_rows))

    X_cat = Z_cat.iloc[x_index,]
    X_num = Z_num.iloc[x_index,]
    Y_cat = Z_cat.iloc[y_index,]
    Y_num = Z_num.iloc[y_index,]
    X_cir = Z_cir.iloc[x_index,]
    Y_cir = Z_cir.iloc[y_index,]

    h_t = np.zeros(num_cols)
    if np.any(c_t > 0):
        dist = norm(0, 1)
        h_t = (
            c_t
            * x_n_rows**-0.2
            * np.minimum(Z_num.std(), g_t / (dist.ppf(1 - lower_q) - dist.ppf(lower_q)))
        )
        print("h_t:", h_t)
    f = partial(
        call_gower_get,
        x_n_rows=x_n_rows,
        y_n_rows=y_n_rows,
        X_cat=X_cat,
        X_num=X_num,
        Y_cat=Y_cat,
        Y_num=Y_num,
        weight_cat=weight_cat,
        weight_cir=weight_cir,
        weight_num=weight_num,
        weight_sum=weight_sum,
        g_t=g_t,
        h_t=h_t,
        knn=knn,
        knn_models=knn_models.copy(),
        X_cir=X_cir,
        Y_cir=Y_cir,
        periods=periods,
    )
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


def gower_get(
    xi_cat,
    xi_num,
    xj_cat,
    xj_num,
    feature_weight_cat,
    feature_weight_cir,
    feature_weight_num,
    feature_weight_sum,
    g_t,
    h_t,
    knn,
    knn_models,
    xi_cir,
    xj_cir,
    periods,
):
    # categorical columns
    sij_cat = np.where(xi_cat == xj_cat, np.zeros_like(xi_cat), np.ones_like(xi_cat))
    sum_cat = np.multiply(feature_weight_cat, sij_cat).sum(axis=1)

    # circular columns
    sij_cir = np.minimum(
        np.absolute(xi_cir - xj_cir), periods - np.absolute(xi_cir - xj_cir)
    ) / (periods // 2)
    sum_cir = np.multiply(feature_weight_cir, sij_cir).sum(axis=1)

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
            neighbors = knn_model.kneighbors(xi.reshape(-1, 1), return_distance=False)
            neighbors = values[neighbors]
            for j, x in enumerate(xj_num.iloc[:, i]):
                if x in neighbors:
                    abs_delta.iloc[j, i] = 0.0
    sij_num = abs_delta.to_numpy() / g_t
    sij_num = np.minimum(sij_num, np.ones_like(sij_num))

    sum_num = np.multiply(feature_weight_num, sij_num).sum(axis=1)
    sums = np.add(np.add(sum_cat, sum_num), sum_cir)
    return np.divide(sums, feature_weight_sum)


def call_gower_get(
    i,
    x_n_rows,
    y_n_rows,
    X_cat,
    X_num,
    Y_cat,
    Y_num,
    weight_cat,
    weight_cir,
    weight_num,
    weight_sum,
    g_t,
    h_t,
    knn,
    knn_models,
    X_cir,
    Y_cir,
    periods,
):
    j_start = i if x_n_rows == y_n_rows else 0
    return gower_get(
        X_cat.iloc[i, :],
        X_num.iloc[i, :],
        Y_cat.iloc[j_start:y_n_rows, :],
        Y_num.iloc[j_start:y_n_rows, :],
        weight_cat,
        weight_cir,
        weight_num,
        weight_sum,
        g_t,
        h_t,
        knn,
        knn_models,
        X_cir.iloc[i, :],
        Y_cir.iloc[j_start:y_n_rows, :],
        periods,
    )


def smallest_indices(ary, n):
    """Returns the n smallest indices from a numpy array."""
    flat = np.nan_to_num(ary.flatten(), nan=999)
    indices = np.argpartition(-flat, -n)[-n:]
    indices = indices[np.argsort(flat[indices])]
    values = flat[indices]
    return {"index": indices, "values": values}


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
        assert all(i >= -1 for i in x), x
        x = [i for i in x if i != -1] + list(range(-1, -1 - list(x).count(-1), -1))
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
        out.extend(sorted([i] + j) for j in all_possible_clusters(n - i, memo))
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
    diff_floor = n - r_floor**2
    diff_ceil = r_ceil**2 - n
    if r > r_floor:
        if diff_ceil < diff_floor:
            return [r_floor] * diff_ceil + [r_ceil] * (r_ceil - diff_ceil)
        else:
            return [r_ceil] * diff_floor + [r_floor] * (r_floor - diff_floor)
    else:
        return [r_floor] * r_floor


def niceness(cluster_sizes: Union[np.ndarray[int], list[int]]) -> float:
    """
    Examples:
    ---------
    >>> from gower.gower_dist import *
    >>> C = [x for i in range(7) for x in all_possible_clusters(i)]
    >>> pairs = [(str(tuple(x)), niceness(x)) for x in C]
    >>> for k, v in sorted(pairs, key=lambda x: x[1]): print(f"{k:50}{v:25}")
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
        n_2 = n**2

        gi = n_2 - np.square(x).sum()
        dof = n - len(x)
        return gi * dof

    total = sum(cluster_sizes)
    if total < 2:
        return np.nan

    out = f(cluster_sizes)
    out /= f(tidy_clusters(total))  # tidy_clusters maximizes f given total

    return out


def gini_coefficient(
    cluster_sizes: Union[np.ndarray[int], list[int]],
    normalize=True,
    return_factors=False,
    repeat=False,
):
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
    (1,)                                           0.0                     -0.0
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
    cluster_sizes = sorted(cluster_sizes)
    raw = sum(cluster_sizes)
    raw4 = raw**4
    total = raw**2 + raw4 if repeat else raw
    n_singletons = int(-0.5 + math.sqrt(0.25 + total))  # solve n(n+1)=total

    def f(x, r=n_singletons, final=total - n_singletons):
        s = r * (r - 1) // 2
        n = len(x) * r + (final > 1)
        d = n * total
        G = sum(xi * (r * (n - i * r) - s) for i, xi in enumerate(x)) + final
        return d + total - 2 * G, d

    num, den = f(cluster_sizes, raw if repeat else 1, raw4 if repeat else 0)
    if normalize:
        num1, den1 = f([1])
        if num1:
            if return_factors:
                return num * den1, den * num1
            num *= den1
            num /= num1 * den
            den = 1
    if return_factors:
        return num, den
    return num / den if den else 0.0


def neatness(cluster_sizes, normalize=True):
    """
    Examples:
    ---------
    >>> from gower.gower_dist import *
    >>> C = [x for i in range(7) for x in all_possible_clusters(i)]
    >>> pairs = [(str(tuple(x)), neatness(x, False), neatness(x)) for x in C]
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
    (1, 1, 1, 1, 2)              0.0006702974444909929      0.08471814923427827
    (1, 1, 1, 2)                 0.0013227513227513227       0.1746031746031746
    (1, 1, 1, 3)                 0.0016457142857142857                    0.208
    (1, 1, 2)                    0.0029304029304029304      0.22252747252747251
    (1, 5)                        0.002197802197802198       0.2777777777777778
    (1, 1, 4)                     0.002406015037593985      0.30409356725146197
    (1, 1, 2, 2)                               0.00256      0.32355555555555554
    (1, 3)                       0.0049382716049382715                    0.375
    (1, 1, 3)                             0.0029296875               0.38671875
    (1, 4)                        0.003246753246753247      0.42857142857142855
    (1, 2, 3)                    0.0037426900584795323       0.4730344379467186
    (2, 4)                        0.004835164835164835       0.6111111111111112
    (1, 2, 2)                     0.005208333333333333                   0.6875
    (2, 2, 2)                     0.007218045112781955       0.9122807017543859
    (1, 2)                        0.007142857142857143                      1.0
    (2, 3)                        0.007575757575757576                      1.0
    (3, 3)                        0.007912087912087912                      1.0
    (2, 2)                         0.01316872427983539                      1.0
    """
    if not isinstance(cluster_sizes, list):
        cluster_sizes = cluster_sizes.tolist()
    total = sum(cluster_sizes)
    if total < 3:
        return np.nan
    singletons = total * [1]

    def g(x):
        a, b = gini_coefficient(x, return_factors=True, repeat=True)
        c, d = gini_coefficient(
            [total * i for i in x] + singletons, return_factors=True
        )
        return (b - a) * (d - c), d * b

    num, den = g(cluster_sizes)
    if not den:
        return 0.0
    if not normalize:
        return num / den

    maximal = (0, 1)
    for k in range(2, math.ceil(math.sqrt(total)) + 1):
        mu = total // k
        add1 = total - mu * k
        num1, den1 = g([mu] * (k - add1) + [mu + 1] * add1)
        if num1 * maximal[1] > maximal[0] * den1:
            maximal = (num1, den1)

    return num * maximal[1] / (den * maximal[0]) if maximal[0] else 1.0


def δ(ck, cl):
    values = np.ones([len(ck), len(cl)])
    for i in range(len(ck)):
        for j in range(len(cl)):
            values[i, j] = np.linalg.norm(ck[i] - cl[j])
    return np.min(values)


def Δ(ci):
    values = np.zeros([len(ci), len(ci)])
    for i in range(len(ci)):
        for j in range(len(ci)):
            values[i, j] = np.linalg.norm(ci[i] - ci[j])
    return np.max(values)


def dunn(k_list):
    δs = np.ones([len(k_list), len(k_list)])
    Δs = np.zeros([len(k_list), 1])
    l_range = list(range(len(k_list)))
    for k in l_range:
        for l in l_range[:k] + l_range[k + 1 :]:
            δs[k, l] = δ(k_list[k], k_list[l])
            Δs[k] = Δ(k_list[k])
    return np.min(δs) / np.max(Δs) if np.max(Δs) else 0


def inverse_sigmoid(x):
    return np.log(x / (1 - x))


def rescaled_silhouette(*args, **kwargs):
    return 0.5 * inverse_sigmoid(silhouette_score(*args, **kwargs) / 2 + 0.5)


def evaluate_clusters(sample, matrix, actual: pd.Series, method, precomputed):
    if method == cluster_optics_dbscan:
        clusters = cluster_optics_dbscan(
            reachability=precomputed.reachability_,
            core_distances=precomputed.core_distances_,
            ordering=precomputed.ordering_,
            eps=sample["eps"],
        )
    elif precomputed is not None:
        clusters = method(**{**sample, precomputed: "precomputed"}).fit_predict(matrix)
    else:
        clusters = method(**sample).fit_predict(matrix)
    clusters = fix_classes(clusters)
    _, counts = np.unique(clusters, return_counts=True)
    counts_dict = dict(zip(*np.unique(counts, return_counts=True)))
    try:
        if precomputed is not None:
            sil = rescaled_silhouette(matrix, clusters, metric="precomputed")
        else:
            sil = rescaled_silhouette(matrix, clusters)
    except ValueError:
        sil = np.nan
    out = {
        "Silhouette": sil,
        "Niceness": niceness(counts),
        "Neatness": neatness(counts),
        "GiniCoeff": gini_coefficient(counts),
        "len(X)/sum(X)": len(counts) / sum(counts),
        "max(X)/sum(X)": max(counts) / sum(counts),
        "sample": sample,
        "clusters": clusters,
        "counts_dict": counts_dict,
    }
    if precomputed is None:
        try:
            db = davies_bouldin_score(matrix, clusters)
        except ValueError:
            db = np.nan
        try:
            ch = calinski_harabasz_score(matrix, clusters)
        except ValueError:
            ch = np.nan
        try:
            di = dunn([matrix[clusters == i] for i in np.unique(clusters)])
        except ValueError:
            di = np.nan
        out |= {"DaviesBouldin": db, "CalinskiHarabasz": ch, "Dunn": di}
    if actual is not None:
        if actual.dtype == float:
            cr = correlation_ratio(clusters, actual)
            out["CorrRatio"] = cr
        else:
            out["AdjRandIndex"] = adjusted_rand_score(actual, clusters)
            out["AdjMutualInfo"] = adjusted_mutual_info_score(actual, clusters)
            out["Combined"] = (1 - out["GiniCoeff"]) * out["AdjRandIndex"] + out[
                "GiniCoeff"
            ] * out["AdjMutualInfo"]
    return out


def get_knee(X, k, **kwargs):
    knn = NearestNeighbors(n_neighbors=k, **kwargs).fit(X)

    # For each point, compute distances to its k-nearest neighbors
    distances, _ = knn.kneighbors(X)

    distances = np.sort(distances, axis=0)[::-1, k - 1]

    kn = KneeLocator(
        range(1, len(distances) + 1),
        distances,
        curve="convex",
        direction="decreasing",
        interp_method="polynomial",
    )

    return kn.knee_y


def simple_preprocess(df):
    # just numeric features
    matrix = df.select_dtypes(include=[np.number])
    matrix -= matrix.min()
    matrix /= matrix.max()
    matrix.fillna(1, inplace=True)
    weight = matrix.apply(get_num_weight)
    matrix *= weight / weight.sum()
    # return adjusted numeric features plus cat features
    return pd.concat([matrix, df.select_dtypes(exclude=[np.number])], axis=1).to_numpy()


def sample_params(
    df,
    matrix,
    actual,
    method,
    samples,
    param,
    n_iter,
    precomputed,
    use_mp,
    title,
    knee=None,
    plot_corr=False,
    **kwargs
):
    if isinstance(matrix, pd.DataFrame):
        matrix = simple_preprocess(matrix)
    # do grid search to get best parameters
    if use_mp:
        results = process_map(
            partial(
                evaluate_clusters,
                matrix=matrix,
                actual=actual,
                method=method,
                precomputed=precomputed,
            ),
            samples,
            chunksize=math.ceil(n_iter / 40),
        )
    else:
        results = [
            evaluate_clusters(sample, matrix, actual, method, precomputed)
            for sample in tqdm(samples)
        ]
    df_results = pd.DataFrame(
        {
            key: [z[key] for z in results]
            for key in results[0].keys()
            if key not in ["sample", "clusters", "counts_dict"]
        }
    )

    if actual is None:
        best = np.argmax(df_results.Neatness)
    elif actual.dtype != float:
        max_muti = np.max(df_results.AdjMutualInfo)
        max_rand = np.max(df_results.AdjRandIndex)
        max_combo = np.max(df_results.Combined)
        amax_gini = np.argmax(df_results.GiniCoeff)
        amin_stu0 = np.argmin(
            np.maximum(df_results["max(X)/sum(X)"], df_results["len(X)/sum(X)"])
        )
        amin_stu1 = np.argmin(df_results["max(X)/sum(X)"] + df_results["len(X)/sum(X)"])
        amax_nice = np.argmax(df_results.Niceness)
        amax_neat = np.argmax(df_results.Neatness)
        amax_silh = np.argmax(df_results.Silhouette)
        if knee is not None:
            n, d = df.shape
            k = min(2 * d, n) - 1
            if precomputed is not None:
                knee = get_knee(matrix, k, metric="precomputed", **kwargs)
            else:
                knee = get_knee(matrix, k, **kwargs)
        if knee is not None:
            knee_x = np.argmin(np.abs(np.array([x["eps"] for x in samples]) - knee))
        if precomputed is None:
            amax_dabo = np.argmax(df_results.DaviesBouldin)
            amax_caha = np.argmax(df_results.CalinskiHarabasz)
            amax_dunn = np.argmax(df_results.Dunn)
        results_table = pd.DataFrame(
            {
                "Metric": [
                    "Max(K, L)",
                    "K + L",
                    "Niceness",
                    "Neatness",
                    "GiniCoeff",
                    "Silhouette",
                    "Knee" if knee is not None else None,
                    "DaviesBouldin" if precomputed is None else None,
                    "CalinskiHarabasz" if precomputed is None else None,
                    "Dunn" if precomputed is None else None,
                ],
                "MutualInfo loss": [
                    df_results.AdjMutualInfo.iloc[amin_stu0],
                    df_results.AdjMutualInfo.iloc[amin_stu1],
                    df_results.AdjMutualInfo.iloc[amax_nice],
                    df_results.AdjMutualInfo.iloc[amax_neat],
                    df_results.AdjMutualInfo.iloc[amax_gini],
                    df_results.AdjMutualInfo.iloc[amax_silh],
                    df_results.AdjMutualInfo.iloc[knee_x] if knee is not None else None,
                    df_results.AdjMutualInfo.iloc[amax_dabo]
                    if precomputed is None
                    else None,
                    df_results.AdjMutualInfo.iloc[amax_caha]
                    if precomputed is None
                    else None,
                    df_results.AdjMutualInfo.iloc[amax_dunn]
                    if precomputed is None
                    else None,
                ],
                "RandIndex loss": [
                    df_results.AdjRandIndex.iloc[amin_stu0],
                    df_results.AdjRandIndex.iloc[amin_stu1],
                    df_results.AdjRandIndex.iloc[amax_nice],
                    df_results.AdjRandIndex.iloc[amax_neat],
                    df_results.AdjRandIndex.iloc[amax_gini],
                    df_results.AdjRandIndex.iloc[amax_silh],
                    df_results.AdjRandIndex.iloc[knee_x] if knee is not None else None,
                    df_results.AdjRandIndex.iloc[amax_dabo]
                    if precomputed is None
                    else None,
                    df_results.AdjRandIndex.iloc[amax_caha]
                    if precomputed is None
                    else None,
                    df_results.AdjRandIndex.iloc[amax_dunn]
                    if precomputed is None
                    else None,
                ],
                "Combined loss": [
                    df_results.Combined.iloc[amin_stu0],
                    df_results.Combined.iloc[amin_stu1],
                    df_results.Combined.iloc[amax_nice],
                    df_results.Combined.iloc[amax_neat],
                    df_results.Combined.iloc[amax_gini],
                    df_results.Combined.iloc[amax_silh],
                    df_results.Combined.iloc[knee_x] if knee is not None else None,
                    df_results.Combined.iloc[amax_dabo]
                    if precomputed is None
                    else None,
                    df_results.Combined.iloc[amax_caha]
                    if precomputed is None
                    else None,
                    df_results.Combined.iloc[amax_dunn]
                    if precomputed is None
                    else None,
                ],
            }
        )
        results_table = results_table.dropna()
        results_table.set_index("Metric", inplace=True)
        results_table.max_muti = max_muti
        results_table.max_rand = max_rand
        results_table.max_combo = max_combo
        best = np.argmax(df_results.Combined)
    else:
        best = np.argmin((df_results.CorrRatio - 0.5).abs())
    best_params = results[best]

    # assign clusters
    df["cluster"] = best_params["clusters"]
    df.cluster = df.cluster.astype(str)

    plt.style.use("dark_background")
    plt.figure(figsize=(10, 10))

    # plot param vs. metrics
    var = np.array([z["sample"][param] for z in results])
    legend = (
        (
            [
                "Niceness",
                "Neatness",
                "GiniCoeff",
                "len(X)/sum(X)",
                "max(X)/sum(X)",
            ]
            + (
                (
                    (
                        ["CorrRatio"]
                        if actual.dtype == float
                        else (
                            [
                                "AdjRandIndex",
                                "AdjMutualInfo",
                                "Combined",
                            ]
                            + (
                                [
                                    f"DaviesBouldin {df_results.DaviesBouldin.max()}",
                                    f"CalinskiHarabasz {df_results.CalinskiHarabasz.max()}",
                                    "Dunn",
                                ]
                                if precomputed is None
                                else []
                            )
                        )
                    )
                    + ["Silhouette"]
                )
                if actual is not None
                else []
            )
        )
        + ["Maximizing"]
    ) + (["Knee"] if knee is not None else [])
    colors = plt.get_cmap("Set3").colors
    for i, col in enumerate(legend[: -1 - (knee is not None)]):
        if " " in col:
            plt.plot(
                var,
                df_results[col.split()[0]] / df_results[col.split()[0]].max(),
                c=colors[i],
            )
        else:
            plt.plot(var, df_results[col], c=colors[i])

    plt.axvline(best_params["sample"][param], c="w", ls="--")
    if knee is not None:
        plt.axvline(var[np.argmin(np.abs(var - knee))], c="g", ls=":")
    plt.legend(legend)
    plt.title(title)
    plt.xlabel(param)
    plt.show()

    if plot_corr:
        # display corr
        n_cols = df.shape[1]
        associations(df, nom_nom_assoc="theil", figsize=(n_cols, n_cols))

    # print results
    del best_params["clusters"]
    print(best_params)
    if actual is not None:
        print(actual.value_counts())
        print(df_results.Combined.max())

    out = (
        (
            np.min((df_results.CorrRatio - 0.5).abs())
            if actual.dtype == float
            else np.max(df_results.Combined)
        )
        if actual is not None
        else np.max(df_results.Neatness)
    )
    return (out, results_table) if actual is not None and actual.dtype != float else out


def optimize_dbscan(
    df,
    title,
    actual=None,
    factor=0.5,
    offset=0.0,
    n_iter=100,
    min_samples=1,
    precomputed=None,
    use_mp=True,
    **kwargs
):
    df = df.copy()

    if precomputed:
        precomputed = "metric"

    if precomputed:
        matrix = gower_matrix(df.to_numpy(), use_mp=use_mp, **kwargs)
    else:
        matrix = df

    samples = [
        {"eps": offset + factor * z / n_iter, "min_samples": min_samples}
        for z in range(1, n_iter + 1)
    ]
    res = sample_params(
        df,
        matrix,
        actual,
        DBSCAN,
        samples,
        "eps",
        n_iter,
        precomputed,
        use_mp,
        title,
        knee=True,
    )

    return df, res


def optimize_gmm(df, title, actual=None, n_iter=10, use_mp=True):
    df = df.copy()

    if len(df) > 5000:
        covariance_type = "spherical"
    elif len(df) > 2000:
        covariance_type = "diag"
    else:
        covariance_type = "full"

    matrix = df

    samples = [
        {"n_components": z, "random_state": 42, "covariance_type": covariance_type}
        for z in range(2, n_iter + 2)
    ]
    res = sample_params(
        df,
        matrix,
        actual,
        GaussianMixture,
        samples,
        "n_components",
        n_iter,
        None,
        use_mp,
        title,
    )

    return df, res


def optimize_bgmm(df, title, actual=None, n_iter=10, use_mp=True):
    df = df.copy()

    if len(df) > 5000:
        covariance_type = "spherical"
    elif len(df) > 2000:
        covariance_type = "diag"
    else:
        covariance_type = "full"

    matrix = df

    samples = [
        {"n_components": z, "random_state": 42, "covariance_type": covariance_type}
        for z in range(2, n_iter + 2)
    ]
    res = sample_params(
        df,
        matrix,
        actual,
        BayesianGaussianMixture,
        samples,
        "n_components",
        n_iter,
        None,
        use_mp,
        title,
    )

    return df, res


def optimize_agglo(
    df,
    title,
    actual=None,
    factor=0.5,
    offset=0.0,
    n_iter=100,
    linkage="average",
    precomputed=None,
    use_mp=True,
    **kwargs
):
    df = df.copy()

    if precomputed:
        precomputed = "metric"

    if precomputed:
        matrix = gower_matrix(df.to_numpy(), use_mp=use_mp, **kwargs)
    else:
        matrix = df

    samples = [
        {
            "distance_threshold": offset + factor * z / n_iter,
            "linkage": linkage,
            "n_clusters": None,
        }
        for z in range(1, n_iter + 1)
    ]
    res = sample_params(
        df,
        matrix,
        actual,
        AgglomerativeClustering,
        samples,
        "distance_threshold",
        n_iter,
        precomputed,
        use_mp,
        title,
    )

    return df, res


def optimize_optics(
    df,
    title,
    actual=None,
    factor=0.5,
    offset=0.0,
    n_iter=100,
    use_mp=True,
    min_samples=1,
    metric="minkowski",
    **kwargs
):
    df = df.copy()

    if metric == "precomputed":
        matrix = gower_matrix(df.to_numpy(), use_mp=use_mp, **kwargs)
    else:
        matrix = df

    clust = OPTICS(min_samples=min_samples, metric=metric)
    clust.fit(matrix)

    samples = [{"eps": offset + factor * z / n_iter} for z in range(1, n_iter + 1)]
    res = sample_params(
        df,
        matrix,
        actual,
        cluster_optics_dbscan,
        samples,
        "eps",
        n_iter,
        clust,
        use_mp,
        title,
    )

    return df, res


def optimize_kmeans(
    df, title, actual=None, n_iter=10, use_mp=True, precomputed=None, **kwargs
):
    df = df.copy()

    if precomputed:
        matrix = gower_matrix(df.to_numpy(), use_mp=use_mp, **kwargs)
    else:
        matrix = df

    samples = [{"n_clusters": z, "random_state": 42} for z in range(2, n_iter + 2)]
    res = sample_params(
        df,
        matrix,
        actual,
        KMeans,
        samples,
        "n_clusters",
        n_iter,
        precomputed,
        use_mp,
        title,
    )

    return df, res


def optimize_hdbscan(
    df, title, actual=None, n_iter=100, use_mp=True, precomputed=None, **kwargs
):
    df = df.copy()

    if precomputed:
        precomputed = "metric"
        matrix = gower_matrix(df.to_numpy(), use_mp=use_mp, **kwargs)
    else:
        matrix = df

    samples = [{"min_cluster_size": z, "min_samples": 1} for z in range(2, n_iter + 1)]
    res = sample_params(
        df,
        matrix,
        actual,
        HDBSCAN,
        samples,
        "min_cluster_size",
        n_iter,
        precomputed,
        use_mp,
        title,
    )

    return df, res


def optimize_spectral(
    df, title, actual=None, n_iter=10, use_mp=True, precomputed=None, **kwargs
):
    df = df.copy()

    if precomputed:
        precomputed = "affinity"
        matrix = gower_matrix(df.to_numpy(), use_mp=use_mp, **kwargs)
    else:
        matrix = df

    samples = [{"n_clusters": z, "random_state": 42} for z in range(2, n_iter + 2)]
    res = sample_params(
        df,
        matrix,
        actual,
        SpectralClustering,
        samples,
        "n_clusters",
        n_iter,
        precomputed,
        use_mp,
        title,
    )

    return df, res


def optimize_birch(df, title, actual=None, n_iter=100, use_mp=True):
    df = df.copy()

    matrix = df

    samples = [
        {"threshold": z / n_iter, "n_clusters": None} for z in range(1, n_iter + 1)
    ]
    res = sample_params(
        df, matrix, actual, Birch, samples, "threshold", n_iter, None, use_mp, title
    )

    return df, res


def optimize_affinity(
    df, title, actual=None, n_iter=100, use_mp=True, precomputed=None, **kwargs
):
    df = df.copy()

    if precomputed:
        precomputed = "affinity"
        matrix = gower_matrix(df.to_numpy(), use_mp=use_mp, **kwargs)
    else:
        matrix = df

    samples = [
        {"damping": 0.5 + 0.5 * z / n_iter, "random_state": 42} for z in range(n_iter)
    ]
    res = sample_params(
        df,
        matrix,
        actual,
        AffinityPropagation,
        samples,
        "damping",
        n_iter,
        precomputed,
        use_mp,
        title,
    )

    return df, res


def optimize_meanshift(df, title, actual=None, n_iter=100, use_mp=True):
    df = df.copy()

    matrix = df

    samples = [
        {"bandwidth": 0.5 + 0.5 * z / n_iter, "random_state": 42} for z in range(n_iter)
    ]
    res = sample_params(
        df, matrix, actual, MeanShift, samples, "bandwidth", n_iter, None, use_mp, title
    )

    return df, res
