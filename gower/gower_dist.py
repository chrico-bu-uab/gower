import math
from functools import partial

import numpy as np
import pandas as pd
from dython.nominal import associations
from scipy.sparse import issparse
from scipy.stats import norm
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def get_num_weight(x: pd.Series):
    """
    This value is always between 1 and 1+log2(len(x)).
    It represents the "resolution" of the column in terms of entropy.
    Binary variables get the lowest weight of 1 due to no entropy.
    """
    assert 0 <= np.nanmin(x) <= np.nanmax(x) <= 1, x
    x = np.array([i for i in x if i is not None])
    x = x[~np.isnan(x)] * 1.0
    x = np.diff(np.sort(x))  # a pmf of ordered categories
    return 1 + math.log2(np.prod(x ** -x))  # entropy


def fix_classes(x):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    x = [i for i in x if i is not None]
    if "-1" in x:
        x = [int(i) for i in x]
    if -1 in x:
        assert not any(i < -1 for i in x), x
        x = [i for i in x if i != -1] + list(range(-1, -1 - list(x).count(-1), -1))
    return x


def cluster_niceness(X: np.ndarray[int]):
    """
    This value tells you to what extent clusters are "nice". It is not a measure
    of the separation between clusters.

    If there is only one cluster, or the clusters are all singletons, the value
    is 0. Useless clusters are not "nice".
    If the elements are evenly distributed, and the number of clusters equals
    the square root of the number of elements, the value is 1.
    Otherwise, the value is on the open interval (0, 1).

    This function is designed to be used in conjunction with grid search and
    DBSCAN to find the best value for the "eps" parameter.

    Inputs:
        X: A 1D array of cluster sizes.

    Outputs:
        A float on the closed interval [0, 1].

    Examples:
        >>> cluster_niceness(np.zeros(100) + 1)
        0.0
        >>> cluster_niceness(np.zeros(25) + 4)
        0.8888888888888888
        >>> cluster_niceness(np.zeros(10) + 10)
        1.0
        >>> cluster_niceness(np.zeros(4) + 25)
        0.8888888888888888
        >>> cluster_niceness(np.zeros(1) + 100)
        0.0
    """
    ttl = np.sum(X)
    return (ttl - np.sum(np.square(X)) / ttl) * (1 - len(X) / ttl) / \
        (ttl - 2 * math.sqrt(ttl) + 1)


def evaluate_clusters(sample, matrix):
    assignments = fix_classes(DBSCAN(metric="precomputed", **sample).fit_predict(matrix))
    _, counts = np.unique(assignments, return_counts=True)
    return sample, cluster_niceness(counts)


def optimize_clusters(df, factor=0.5, n_iter=100, use_mp=True, **kwargs):
    df = df.copy()
    samples = [{"eps": 2 * factor * z / n_iter, "min_samples": 1} for z in range(1, n_iter + 1)]
    matrix = gower_matrix(df.to_numpy(), use_mp=use_mp, **kwargs)
    if use_mp:
        results = process_map(partial(evaluate_clusters, matrix=matrix), samples, chunksize=max(n_iter // 32, 1))
    else:
        results = [evaluate_clusters(sample, matrix) for sample in tqdm(samples)]

    best_params = max(results, key=lambda z: z[1])
    df["cluster"] = DBSCAN(metric="precomputed", **best_params[0]).fit_predict(matrix)
    df.cluster = df.cluster.astype(str)
    _, counts = np.unique(df.cluster, return_counts=True)
    class_sizes, class_counts = np.unique(counts, return_counts=True)

    corr = associations(df, nom_nom_assoc="theil", figsize=(df.shape[1], df.shape[1]))["corr"]
    print(*best_params, (corr.cluster.mean() + corr.T.cluster.mean()) / 2)
    print(dict(zip(class_sizes, class_counts)))

    df["count_per_cluster"] = df.groupby("cluster").transform("count").iloc[:, 0]
    df.sort_values(["count_per_cluster", "cluster"], ascending=[False, True], inplace=True)
    df.drop("count_per_cluster", axis=1, inplace=True)

    return df


def get_cat_features(X):
    if not isinstance(X, np.ndarray):
        is_number = np.vectorize(lambda x: not np.issubdtype(x, np.number))
        cat_features = is_number(X.dtypes)
    else:
        x_n_cols = X.shape[1]
        cat_features = np.zeros(x_n_cols, dtype=bool)
        for col in range(x_n_cols):
            if not np.issubdtype(type(X[0, col]), np.number):
                cat_features[col] = True
    return cat_features


def get_percentiles(X, R):
    return [np.nanpercentile(X, p, axis=0) for p in R]


def gower_matrix(data_x, data_y=None, weight_cat=None, weight_num=None,
                 cat_features=None, R=(0, 100), c=0.0, knn=False,
                 use_mp=True, return_weight_num=False, **tqdm_kwargs):
    # function checks
    X = data_x
    if data_y is None:
        Y = data_x
    else:
        Y = data_y
    if not isinstance(X, np.ndarray):
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

    num_cols = Z_num.shape[1]

    P0, P1 = get_percentiles(Z_num, R)
    knn_models = []
    if knn:
        n_knn = int(math.sqrt(x_n_rows))
        for col in range(num_cols):
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
        #     weight_cat = process_map(get_cat_weight, Z_cat.T, **tqdm_kwargs)
        # else:
        #     weight_cat = [get_cat_weight(Z_cat[:, col]) for col in tqdm(range(cat_cols))]
        weight_cat = np.ones(cat_cols)
    weight_cat = np.array(weight_cat)

    if isinstance(weight_num, str):
        if weight_num == "uniform":
            weight_num = np.ones(num_cols)
        else:
            raise ValueError("Unknown weight_num: {}".format(weight_num))
    elif weight_num is None:
        if use_mp:
            weight_num = process_map(get_num_weight, Z_num.T.to_numpy(), **tqdm_kwargs)
        else:
            weight_num = [get_num_weight(Z_num.loc[:, col]) for col in tqdm(range(num_cols))]
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
    if c > 0:
        dist = norm(0, 1)
        h_t = c * x_n_rows ** -0.2 * np.minimum(
            Z_num.std(),
            (P1 - P0) / (dist.ppf(R[1] / 100) - dist.ppf(R[0] / 100)))
        print("h_t:", h_t)
    g = partial(call_gower_get, x_n_rows=x_n_rows, y_n_rows=y_n_rows,
                X_cat=X_cat, X_num=X_num, Y_cat=Y_cat, Y_num=Y_num,
                weight_cat=weight_cat, weight_num=weight_num,
                weight_sum=weight_sum, IQR=P1 - P0, h_t=h_t,
                knn_models=knn_models)
    if use_mp:
        processed = process_map(g, range(x_n_rows), **tqdm_kwargs)
    else:
        processed = list(map(g, tqdm(range(x_n_rows))))
    for i, res in enumerate(processed):
        j_start = i if x_n_rows == y_n_rows else 0
        out[i, j_start:] = res
        if x_n_rows == y_n_rows:
            out[i:, j_start] = res

    max_distance = np.nanmax(out)
    assert math.isclose(max_distance, 1) or (max_distance < 1), max_distance

    return out if not return_weight_num else (out, weight_num)


def gower_get(xi_cat, xi_num, xj_cat, xj_num, feature_weight_cat,
              feature_weight_num, feature_weight_sum, IQR, h_t, knn_models):
    # categorical columns
    sij_cat = np.where(xi_cat == xj_cat,
                       np.zeros_like(xi_cat),
                       np.ones_like(xi_cat))
    sum_cat = np.multiply(feature_weight_cat, sij_cat).sum(axis=1)

    # numerical columns
    abs_delta = np.abs(xi_num - xj_num)
    abs_delta = np.maximum(abs_delta - h_t, np.zeros_like(abs_delta))
    xi_num = xi_num.to_numpy()
    if knn_models:
        for i, (values, knn_model) in enumerate(knn_models):
            xi = xi_num[i]
            if np.isnan(xi).any():
                continue
            neighbors = knn_model.kneighbors(xi.reshape(-1, 1),
                                             return_distance=False)
            neighbors = values[neighbors]
            for j, x in enumerate(xj_num.iloc[:, i]):
                if x in neighbors:
                    abs_delta.iloc[j, i] = 0.0
    sij_num = abs_delta.to_numpy() / IQR
    sij_num = np.minimum(sij_num, np.ones_like(sij_num))

    sum_num = np.multiply(feature_weight_num, sij_num).sum(axis=1)
    sums = np.add(sum_cat, sum_num)
    sum_sij = np.divide(sums, feature_weight_sum)

    return sum_sij


def call_gower_get(i, x_n_rows, y_n_rows, X_cat, X_num, Y_cat, Y_num,
                   weight_cat, weight_num, weight_sum, IQR, h_t, knn_models):
    j_start = i if x_n_rows == y_n_rows else 0
    # call the main function
    res = gower_get(X_cat.iloc[i, :],
                    X_num.iloc[i, :],
                    Y_cat.iloc[j_start:y_n_rows, :],
                    Y_num.iloc[j_start:y_n_rows, :],
                    weight_cat, weight_num, weight_sum, IQR, h_t, knn_models)
    return res


def smallest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = np.nan_to_num(ary.flatten(), nan=999)
    indices = np.argpartition(-flat, -n)[-n:]
    indices = indices[np.argsort(flat[indices])]
    values = flat[indices]
    return {'index': indices, 'values': values}


def gower_topn(data_x, data_y=None, weight=None, cat_features=None, n=5):
    if data_x.shape[0] >= 2:
        TypeError("Only support `data_x` of 1 row. ")
    dm = gower_matrix(data_x, data_y, weight, cat_features)

    return smallest_indices(np.nan_to_num(dm[0], nan=1), n)
