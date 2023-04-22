import math
from functools import partial
from typing import Union

import numpy as np
import pandas as pd
from dython.nominal import associations
from scipy.sparse import issparse
from scipy.stats import norm
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def get_percentile_range(X, q):
    return np.nanpercentile(X, 100 - q, axis=0) - np.nanpercentile(X, q, axis=0)


def get_num_weight(x):
    """
    This value is always between 1 and sqrt(len(x)).
    It represents the "resolution" of the column in terms of entropy.
    Binary variables get the lowest weight of 1 due to no entropy.
    """
    assert 0 <= np.nanmin(x) <= np.nanmax(x) <= 1, x
    x = np.array([i for i in x if i is not None])
    x = x[~np.isnan(x)] * 1.0
    x = np.diff(np.sort(x))  # a pmf of ordered categories
    return math.sqrt(np.prod(x ** -x))  # entropy


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
        if isinstance(X, pd.DataFrame):
            is_number = np.vectorize(lambda x: not np.issubdtype(x, np.number))
            cat_features = is_number(X.dtypes)
        else:
            cat_features = np.zeros(x_n_cols, dtype=bool)
            for col in range(x_n_cols):
                if not np.issubdtype(type(X[0, col]), np.number):
                    cat_features[col] = True
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
            weight_num = [get_num_weight(Z_num.loc[:, col]) for col in
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


def all_possible_clusters(n_elements, memo=None):
    """
    *** This function was (mostly) written by GitHub Copilot and ChatGPT. ***

    This function returns a list of all possible clusterings given a number of
    elements to cluster. The clusterings are returned as a list of lists of
    integers. The integers are the element counts per cluster.

    For example, if there are 3 elements, then the possible clusterings are:
    [[1, 1, 1], [1, 2], [3]]

    This function is not recommended for n_elements > 20.
    """
    if n_elements == 1:
        return [[1]]
    if memo is None:
        memo = {}
    if n_elements in memo:
        return memo[n_elements]
    out = []
    for i in range(1, n_elements):
        for j in all_possible_clusters(n_elements - i, memo):
            out.append(sorted([i] + j))
    out.append([n_elements])
    out = [c for i, c in enumerate(out) if c not in out[:i]]
    memo[n_elements] = out
    return out


def cluster_niceness(cluster_sizes: Union[np.ndarray[int], list[int]]) -> float:
    """
    This value tells you to what extent clusters are "nice". It is not a measure
    of the separation between clusters such as the Davies-Bouldin index, but
    rather a measure of how well clusters are distributed.

    If the elements comprise one cluster, or the clusters are all singletons,
    the value is 0 as useless clusters are not "nice".
    If the elements are evenly distributed, and the number of clusters equals
    the number of elements per cluster, the value is 1.
    Otherwise, the value is on the open interval (0, 1).

    One additional property of this function is that it is symmetric with
    respect to the number of clusters and number of elements per cluster when
    the elements are evenly distributed:
        cluster_niceness([a]*b) == cluster_niceness([b]*a)

    This function is designed to be used in conjunction with grid search and
    DBSCAN to find the best value for the "eps" parameter.

    Inputs:
        X: A 1D array of cluster sizes.

    Outputs:
        A float on the closed interval [0, 1].

    Examples of clusterings of six elements:
        >>> from gower.gower_dist import all_possible_clusters, cluster_niceness
        >>> C = all_possible_clusters(n_elements=6)
        >>> pairs = [(str(x), cluster_niceness(x)) for x in C]
        >>> for k, v in sorted(pairs, key=lambda x: x[1]):
        >>>     print(f"{k:20}{v}")
        [1, 1, 1, 1, 1, 1]  0.0
        [6]                 0.0
        [1, 1, 1, 1, 2]     0.3701904728842866
        [1, 5]              0.5288435326918379
        [1, 1, 1, 3]        0.6346122392302055
        [1, 1, 2, 2]        0.6874965924993893
        [1, 1, 4]           0.7139387691339812
        [2, 4]              0.8461496523069406
        [1, 2, 3]           0.8725918289415325
        [2, 2, 2]           0.9519183588453083
        [3, 3]              0.9519183588453083

    Note that only a square number of elements permits a return value of 1.0,
    but any number of elements could return a value of 0.0 (besides 1, which
    returns NaN).
    """
    # check inputs
    if any(x < 1 or x != int(x) for x in cluster_sizes):
        raise ValueError("The number of elements in a given cluster must be a "
                         "natural number.")

    # convert to numpy array
    if not isinstance(cluster_sizes, np.ndarray):
        cluster_sizes = np.array(cluster_sizes)

    # compute number of clusters and elements
    n_clusters = len(cluster_sizes)
    n_elements = cluster_sizes.sum()
    n_elements2 = n_elements ** 2

    # compute factors
    f1 = n_elements2 - np.square(cluster_sizes).sum()
    f2 = n_elements - n_clusters
    numer = f1 * f2 / n_elements2

    # compute (sqrt(n_elements)-1)^2
    denom = n_elements - 2 * math.sqrt(n_elements) + 1

    return numer / denom


def evaluate_clusters(sample, matrix):
    assignments = fix_classes(DBSCAN(metric="precomputed",
                                     **sample).fit_predict(matrix))
    _, counts = np.unique(assignments, return_counts=True)
    return sample, cluster_niceness(counts)


def optimize_clusters(df, factor=0.5, n_iter=100, use_mp=True, **tqdm_kwargs):
    df = df.copy()
    samples = [{"eps": 2 * factor * z / n_iter, "min_samples": 1} for z in
               range(1, n_iter + 1)]
    matrix = gower_matrix(df.to_numpy(), use_mp=use_mp, **tqdm_kwargs)
    if use_mp:
        results = process_map(partial(evaluate_clusters, matrix=matrix),
                              samples, chunksize=math.ceil(n_iter / 40))
    else:
        results = [evaluate_clusters(sample, matrix) for sample in
                   tqdm(samples)]

    best_params = max(results, key=lambda z: z[1])
    df["cluster"] = DBSCAN(metric="precomputed", **best_params[0]
                           ).fit_predict(matrix)
    df.cluster = df.cluster.astype(str)
    _, counts = np.unique(df.cluster, return_counts=True)
    class_sizes, class_counts = np.unique(counts, return_counts=True)

    corr = associations(df, nom_nom_assoc="theil",
                        figsize=(df.shape[1], df.shape[1]))["corr"]
    print(*best_params, (corr.cluster.mean() + corr.T.cluster.mean()) / 2)
    print(dict(zip(class_sizes, class_counts)))

    df["per_cluster"] = df.groupby("cluster").transform("count").iloc[:, 0]
    df.sort_values(["per_cluster", "cluster"], ascending=[False, True],
                   inplace=True)
    df.drop("per_cluster", axis=1, inplace=True)

    return df
