import math
from functools import partial
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dython.nominal import associations  # , correlation_ratio
from kneed import KneeLocator
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.sparse import issparse
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.cluster import *
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
    # davies_bouldin_score,
    # calinski_harabasz_score,
    pairwise_distances,
)
from sklearn.metrics.cluster._unsupervised import check_number_of_labels
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y, _safe_indexing, validate_params
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


# Forked from https://pypi.org/project/gower/

# Everything in this package is geared to use Manhattan distance! :)


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
    return math.sqrt(np.prod(x ** -x))  # perplexity


def check_data(data_x, data_y):
    X = data_x
    Y = data_x if data_y is None else data_y

    if isinstance(X, pd.DataFrame):
        if not np.array_equal(X.columns, Y.columns):
            raise TypeError("X and Y must have same columns!")
    elif X.shape[1] != Y.shape[1]:
        raise TypeError("X and Y must have same y-dim!")

    if issparse(X) or issparse(Y):
        raise TypeError("Sparse matrices are not supported!")

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(Y, pd.DataFrame):
        Y = pd.DataFrame(Y)

    return X, Y


def gower_matrix(
        data_x,
        data_y=None,
        cat_features=None,
        circular_features=None,
        weight_cat=None,
        weight_cir=None,
        weight_num=None,
        q=0.0,
        c_t=0.0,
        knn=False,
        use_mp=True,
        **tqdm_kwargs
):
    """
    Please refer to "Distances with Mixed-Type Variables, some
    Modified Gower’s Coefficients" by Marcello D'Orazio for information on
    parameters `q`, `c_t`, and `knn`.

    Parameters
    ----------
    data_x : array-like
        A 2D array of data.
    data_y : array-like, optional
        A 2D array of data. If None, then `data_y` is set to `data_x`.
    cat_features : array-like, optional
        A 1D array of boolean values indicating whether a column is categorical.
        If None, then `cat_features` is set to the result of `get_cat_features`.
    circular_features : array-like, optional
        A 1D array of integer values indicating the periodicities of columns.
        If None, then `circular_features` is set to zeros.
    weight_cat : array-like, optional
        A 1D array of weights for categorical columns.
        If None, then `weight_cat` is set to ones.
    weight_cir : array-like, optional
        A 1D array of weights for circular columns.
        If None, then `weight_cir` is a function of the periodicities of the
        circular columns.
    weight_num : array-like, optional
        A 1D array of weights for numerical columns.
        If None, then `weight_num` is set to the result of `get_num_weight`.
    q : float or array-like, optional
        A float or 1D array of floats between 0 and 0.5.
        See D'Orazio for more information.
    c_t : float or array-like, optional
        A float or 1D array of floats between 0 and 1.
        See D'Orazio for more information.
    knn : bool or array-like, optional
        A bool or 1D array of bools indicating whether to use k-nearest neighbors
        for numerical columns.
        If True, then `knn` is set to ones.
        See D'Orazio for more information.
    use_mp : bool, optional
        A bool indicating whether to use multiprocessing.
    tqdm_kwargs : dict, optional
        A dict of keyword arguments to pass to `tqdm`.

    Returns
    -------
    array-like
        A 2D array of distances.
    """
    X, Y = check_data(data_x, data_y)

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

    Z = pd.concat((X, Y))

    Z_num = Z.loc[
            :, np.logical_not(cat_features) & np.logical_not(circular_features)
            ].astype(float)
    Z_num -= Z_num.min()
    Z_num /= Z_num.max()
    Z_num.fillna(1, inplace=True)

    num_cols = Z_num.shape[1]

    g_t = get_percentile_range(Z_num, 100 * q)
    knn_models = []
    if np.any(knn):
        if isinstance(knn, bool):
            knn = np.ones(num_cols, dtype=bool)
        n_knn = round(math.sqrt(x_n_rows))
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

    # print(weight_cat, weight_num, weight_cir)
    weight_sum = weight_cat.sum() + weight_num.sum() + weight_cir.sum()

    # distance matrix

    out = np.zeros((x_n_rows, y_n_rows))

    x_index = range(x_n_rows)
    y_index = range(x_n_rows, x_n_rows + y_n_rows)

    X_cat = Z_cat.iloc[x_index, ]
    X_num = Z_num.iloc[x_index, ]
    X_cir = Z_cir.iloc[x_index, ]
    Y_cat = Z_cat.iloc[y_index, ]
    Y_num = Z_num.iloc[y_index, ]
    Y_cir = Z_cir.iloc[y_index, ]

    h_t = np.zeros(num_cols)
    if np.any(c_t > 0):
        dist = norm(0, 1)
        h_t = (
                c_t
                * x_n_rows ** -0.2
                * np.minimum(Z_num.std(), g_t / (dist.ppf(1 - q) - dist.ppf(q)))
        )
        print("h_t:", h_t)
    f = partial(
        compute_with_gower_get,
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
        knn_models=knn_models,
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


def compute_with_gower_get(
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

    # categorical columns
    xi_cat = X_cat.iloc[i, :]
    xj_cat = Y_cat.iloc[j_start:y_n_rows, :]
    sij_cat = np.where(xi_cat == xj_cat, np.zeros_like(xi_cat), np.ones_like(xi_cat))
    sum_cat = np.multiply(weight_cat, sij_cat).sum(axis=1)

    # circular columns
    xi_cir = X_cir.iloc[i, :]
    xj_cir = Y_cir.iloc[j_start:y_n_rows, :]
    sij_cir = np.absolute(xi_cir - xj_cir)
    sij_cir = np.minimum(sij_cir, periods - sij_cir) / (periods // 2)
    sum_cir = np.multiply(weight_cir, sij_cir).sum(axis=1)

    # numerical columns
    xi_num = X_num.iloc[i, :]
    xj_num = Y_num.iloc[j_start:y_n_rows, :]
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
    sum_num = np.multiply(weight_num, sij_num).sum(axis=1)

    sums = np.add(np.add(sum_cat, sum_num), sum_cir)
    return np.divide(sums, weight_sum)


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


# Clustering Metrics


@validate_params(
    {
        "X": ["array-like"],
        "labels": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def calinski_harabasz_score(X, labels, p, **kwargs):
    """
    See scikit-learn's documentation for more information.
    This version generalizes to any p-norm.
    """
    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    n_samples, _ = X.shape
    n_labels = len(le.classes_)

    check_number_of_labels(n_labels, n_samples)

    extra_disp, intra_disp = 0.0, 0.0
    mean = np.mean(X, axis=0)
    for k in range(n_labels):
        cluster_k = X[labels == k]
        mean_k = np.mean(cluster_k, axis=0)
        extra_disp += len(cluster_k) * np.linalg.norm(mean_k - mean, p) ** p
        intra_disp += np.linalg.norm(cluster_k - mean_k, p) ** p

    return (
        1.0
        if intra_disp == 0.0
        else extra_disp * (n_samples - n_labels) / (intra_disp * (n_labels - 1.0))
    )


@validate_params(
    {
        "X": ["array-like"],
        "labels": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def davies_bouldin_score(X, labels, **kwargs):
    """
    See scikit-learn's documentation for more information.
    This version allows keyword arguments to be passed to `pairwise_distances`.
    """
    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples, _ = X.shape
    n_labels = len(le.classes_)
    check_number_of_labels(n_labels, n_samples)

    intra_dists = np.zeros(n_labels)
    centroids = np.zeros((n_labels, len(X[0])), dtype=float)
    for k in range(n_labels):
        cluster_k = _safe_indexing(X, labels == k)
        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid
        intra_dists[k] = np.average(pairwise_distances(cluster_k, [centroid], **kwargs))

    centroid_distances = pairwise_distances(centroids, **kwargs)

    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0

    centroid_distances[centroid_distances == 0] = np.inf
    combined_intra_dists = intra_dists[:, None] + intra_dists
    scores = np.max(combined_intra_dists / centroid_distances, axis=1)
    return np.mean(scores)


def dunn_score(X, func=None, **kwargs):
    """
    Dunn Index for Cluster Validation
    @param X: list of clusters
    @param func: function or functions to compute distances
    @param kwargs: arguments to pass to cdist
    @return: Dunn Index
    """
    if func is None:
        f_diameter, f_separation = np.max, np.min
    elif callable(func):
        f_diameter = f_separation = func
    else:  # assume dictionary
        f_diameter = func["diameter"]
        f_separation = func["separation"]

    def get_dist(x, y, f):
        """
        @param x: cluster x's data
        @param y: cluster y's data
        @param f: function to take of distances between x and y
        @return: single distance between x and y
        """
        distances = cdist(x, y, **kwargs)
        return f(distances)

    # compute diameters
    diameters = np.array([get_dist(x, x, f_diameter) for x in X])
    largest_diameter = np.max(diameters)

    if not largest_diameter:
        return np.nan

    # compute separations
    min_separation = np.inf
    n = len(X)
    for i in range(n - 1):
        for j in range(i + 1, n):
            separation = get_dist(X[i], X[j], f_separation)
            if separation < min_separation:
                min_separation = separation

    if min_separation == np.inf:
        return np.nan

    # compute Dunn Index
    return min_separation / largest_diameter


def reconstruct_observations(matrix, plot=False):
    """
    methodology loosely based on https://stats.stackexchange.com/a/12503/369868
    """
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0)
    mat = matrix - matrix.mean(axis=0)
    pca = PCA(random_state=42)
    mat = pca.fit_transform(mat.T)
    x_axis = range(1, len(mat) + 1)
    y_axis = pca.explained_variance_
    elbow = KneeLocator(
        x_axis,
        y_axis,
        curve="convex",
        direction="decreasing",
        interp_method="polynomial"
    )
    elbow_x = elbow.elbow
    if plot:
        plt.xlabel("d")
        plt.ylabel("Explained Variance")
        plt.plot(x_axis, y_axis, "bx-")
        plt.plot(x_axis, elbow.Ds_y)
        ylim = plt.ylim()
        plt.vlines(elbow_x, ylim[0], ylim[1], linestyles="--")
        plt.show()
    return mat[:, :elbow_x]


def get_elbow(X, k, plot=False, **kwargs):
    """
    See https://stats.stackexchange.com/q/541340
    """
    # compute distances
    knn = NearestNeighbors(n_neighbors=k, **kwargs).fit(X)
    distances, _ = knn.kneighbors(X)
    distances = np.sort(distances, axis=0)[:, k - 1]
    x_axis = range(1, len(distances) + 1)

    # find elbow
    elbow = KneeLocator(
        x_axis,
        distances,
        curve="convex",
        interp_method="polynomial"
    )

    if plot:
        plt.xlabel("k")
        plt.ylabel("Distance")
        plt.plot(x_axis, distances, "bx-")
        plt.plot(x_axis, elbow.Ds_y)
        ylim = plt.ylim()
        plt.vlines(elbow.elbow, ylim[0], ylim[1], linestyles="--")
        plt.show()

    return elbow.elbow_y


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


def nice_helper(n):
    r = math.sqrt(n)
    r_floor = int(r)
    r_ceil = r_floor + 1
    diff_floor = n - r_floor ** 2
    diff_ceil = r_ceil ** 2 - n
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
    """

    def f(x):
        x = np.array(x)
        n = x.sum()
        gi = n ** 2 - np.square(x).sum()
        dof = n - len(x)
        return gi * dof

    total = sum(cluster_sizes)
    if total < 3:
        return np.nan

    out = f(cluster_sizes)
    out /= f(nice_helper(total))  # nice_helper maximizes f given total

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
    raw4 = raw ** 4
    total = raw ** 2 + raw4 if repeat else raw
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


def neat_helper(f, num, den, total, normalize):
    if not den:
        return 0.0
    if not normalize:
        return num / den

    num_max, den_max = (0, 1)
    for k in range(2, math.ceil(math.sqrt(total)) + 1):
        mu = total // k
        add1 = total - mu * k
        num1, den1 = f([mu] * (k - add1) + [mu + 1] * add1)
        if num1 * den_max > num_max * den1:
            num_max, den_max = (num1, den1)

    return num * den_max / (den * num_max) if num_max else 1.0


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

    def f(x):
        a, b = gini_coefficient(x, return_factors=True, repeat=True)
        c, d = gini_coefficient(
            [total * i for i in x] + singletons, return_factors=True
        )
        return (b - a) * (d - c), d * b

    num, den = f(cluster_sizes)
    return neat_helper(f, num, den, total, normalize)


def tidiness(cluster_sizes):
    """
    Examples:
    ---------
    >>> from gower.gower_dist import *
    >>> C = [x for i in range(7) for x in all_possible_clusters(i)]
    >>> pairs = [(str(tuple(x)), tidiness(x)) for x in C]
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
    (1, 1, 1, 1, 2)                                          0.3333333333333333
    (1, 5)                                                   0.3333333333333333
    (1, 1, 2)                                                               0.5
    (1, 3)                                                                  0.5
    (1, 1, 1, 2)                                                            0.5
    (1, 4)                                                                  0.5
    (1, 1, 1, 3)                                             0.6666666666666666
    (1, 1, 2, 2)                                             0.6666666666666666
    (1, 1, 4)                                                0.6666666666666666
    (2, 4)                                                   0.6666666666666666
    (1, 2)                                                                  1.0
    (2, 2)                                                                  1.0
    (1, 1, 3)                                                               1.0
    (1, 2, 2)                                                               1.0
    (2, 3)                                                                  1.0
    (1, 2, 3)                                                               1.0
    (2, 2, 2)                                                               1.0
    (3, 3)                                                                  1.0
    """
    total = sum(cluster_sizes)
    if total < 3:
        return np.nan
    f = lambda x: max(max(x), len(x))
    return (total - f(cluster_sizes)) / (total - f(nice_helper(total)))


def get_closest_points(x, y):
    d = np.abs(x[:, np.newaxis] - y)
    c = np.argwhere(d == np.min(d))
    return round((np.mean(x[c[:, 0]]) + np.mean(y[c[:, 1]])) / 2)


def weighted_quantiles(values, weights, quantiles=0.5, interpolate=True):
    """
    https://stackoverflow.com/a/75321415/5295786
    """
    i = values.argsort()
    sorted_weights = weights[i]
    sorted_values = values[i]
    Sn = sorted_weights.cumsum()

    if interpolate:
        Pn = (Sn - sorted_weights / 2) / Sn[-1]
        return np.interp(quantiles, Pn, sorted_values)
    else:
        return sorted_values[np.searchsorted(Sn, quantiles * Sn[-1])]


def kernel_weighted_median(indices):
    """
    Used for the Ensemble method below.
    """
    indices = np.array(indices)
    if len(indices) == 1:
        return indices.item()
    weights = np.exp(-np.square(indices - indices.mean()) / (2 * indices.var()))
    return round(weighted_quantiles(indices, weights))


# Cluster evaluation


def fix_classes(x):
    if isinstance(x, (np.ndarray, pd.Series)):
        x = x.tolist()
    x = [i for i in x if i is not None]
    if "-1" in x:
        x = [int(i) for i in x]
    if -1 in x:
        assert all(i >= -1 for i in x), x
        x = [i for i in x if i != -1] + list(range(-1, -1 - x.count(-1), -1))
    return x


def evaluate_clusters(sample, matrix, obs, actual: pd.Series, method, precomputed):
    matrix = matrix.to_numpy() if isinstance(matrix, pd.DataFrame) else matrix
    if method == cluster_optics_dbscan:
        estimator = None
        clusters = cluster_optics_dbscan(
            reachability=precomputed.reachability_,
            core_distances=precomputed.core_distances_,
            ordering=precomputed.ordering_,
            eps=sample["eps"],
        )
    elif precomputed is not None:
        estimator = method(**{**sample, precomputed: "precomputed"})
        clusters = estimator.fit_predict(matrix)
    else:
        estimator = method(**sample)
        clusters = estimator.fit_predict(matrix)
    clusters = fix_classes(clusters)
    _, counts = np.unique(clusters, return_counts=True)
    counts_dict = dict(zip(*np.unique(counts, return_counts=True)))
    try:
        if precomputed is not None:
            sil = silhouette_score(matrix, clusters, metric="precomputed")
        else:
            sil = silhouette_score(matrix, clusters)
    except ValueError:
        sil = np.nan
    out = {
        "Silhouette": sil,
        "Niceness": niceness(counts),
        "GiniCoeff": gini_coefficient(counts),
        "Neatness": neatness(counts),
        "sample": sample,
        "clusters": clusters,
        "counts_dict": counts_dict,
    }
    kwargs = dict(metric="minkowski", p=1)
    if hasattr(estimator, "bic"):
        out["BIC"] = estimator.bic(matrix)
    try:
        db = davies_bouldin_score(obs, clusters, **kwargs)
    except ValueError:
        db = np.nan
    try:
        ch = calinski_harabasz_score(obs, clusters, **kwargs)
    except ValueError:
        ch = np.nan
    di = dunn_score([obs[clusters == i] for i in np.unique(clusters)], **kwargs)
    out |= {"DaviesBouldin": db, "CalinskiHarabasz": ch, "Dunn": di}
    if actual is not None:
        out["AdjRandIndex"] = adjusted_rand_score(actual, clusters)
        out["AdjMutualInfo"] = adjusted_mutual_info_score(actual, clusters)
        g = gini_coefficient(np.unique(actual, return_counts=True)[1])
        out["Combined"] = (1 - g) * out["AdjRandIndex"] + g * out["AdjMutualInfo"]
    return out


# Cluster optimization


def simple_preprocess(df):
    df = df.copy()
    df -= df.min()
    df /= df.max()
    df.fillna(0.5, inplace=True)
    weight = df.apply(get_num_weight)
    return df * weight / weight.sum()


def get_data_points(df_results, column, indices):
    return [
        df_results[column].iloc[index] if index is not None else None
        for index in indices
    ]


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
        elbow=None,
        plot_corr=False,
        **kwargs
):
    if precomputed is None:
        matrix = simple_preprocess(matrix)
        obs = df
    else:
        obs = reconstruct_observations(matrix)
    # do grid search to get best parameters
    if use_mp:
        results = process_map(
            partial(
                evaluate_clusters,
                matrix=matrix,
                obs=obs,
                actual=actual,
                method=method,
                precomputed=precomputed,
            ),
            samples,
            chunksize=math.ceil(n_iter / 40),
        )
    else:
        results = [
            evaluate_clusters(sample, matrix, obs, actual, method, precomputed)
            for sample in tqdm(samples)
        ]
    df_results = pd.DataFrame(
        {
            key: [z[key] for z in results]
            for key in results[0].keys()
            if key not in ["sample", "clusters", "counts_dict"]
        }
    )

    if elbow is not None:
        if precomputed is not None:
            kwargs["metric"] = "precomputed"
        elbow = get_elbow(matrix, obs.shape[1] + 1, **kwargs)
        if elbow is None:
            elbow_x = None
        else:
            elbow_x = np.argmin(np.abs(np.array([x[param] for x in samples]) - elbow))
    else:
        elbow_x = None

    # smooth results for peak finding and plotting
    # do NOT smooth the extrinsic metrics!!!
    for col in df_results.columns:
        if col not in ["AdjMutualInfo", "AdjRandIndex", "Combined"]:
            df_results[col] = gaussian_filter1d(df_results[col], 1, mode="nearest")

    def get_peaks(x):
        peaks, _ = find_peaks(x)
        return peaks[np.argmax(x[peaks])] if peaks.size else -1

    if "BIC" in df_results.columns:
        bic = get_peaks(-df_results.BIC)
    else:
        bic = None
    kwds = pd.Series(dict(DaviesBouldin=get_peaks(-df_results.DaviesBouldin),
                          CalinskiHarabasz=get_peaks(df_results.CalinskiHarabasz),
                          Dunn=get_peaks(df_results.Dunn),
                          Silhouette=get_peaks(df_results.Silhouette),
                          GiniCoeff=get_peaks(df_results.GiniCoeff),
                          Niceness=get_peaks(df_results.Niceness),
                          Neatness=get_peaks(df_results.Neatness)))
    if elbow_x is not None:
        kwds["Elbow"] = elbow_x
    if bic is not None:
        kwds["BIC"] = bic
    ensemble = kernel_weighted_median(kwds)
    if actual is None:
        best = ensemble
        results_table = None
    else:
        max_muti = np.max(df_results.AdjMutualInfo)
        max_rand = np.max(df_results.AdjRandIndex)
        max_combo = np.max(df_results.Combined)
        indices = [
            bic if "BIC" in df_results.columns else None,
            kwds["DaviesBouldin"],
            kwds["CalinskiHarabasz"],
            kwds["Dunn"],
            kwds["Silhouette"],
            kwds["GiniCoeff"],
            kwds["Niceness"],
            kwds["Neatness"],
            kwds["Elbow"] if elbow_x is not None else None,
            ensemble,
        ]
        results_table = pd.DataFrame(
            {
                "Metric": [
                    "BIC",
                    "DaviesBouldin",
                    "CalinskiHarabasz",
                    "Dunn",
                    "Silhouette",
                    "GiniCoeff",
                    "Niceness",
                    "Neatness",
                    "Elbow",
                    "Ensemble",
                ],
                "AdjMutualInfo": get_data_points(df_results, "AdjMutualInfo", indices),
                "AdjRandIndex": get_data_points(df_results, "AdjRandIndex", indices),
                "Combined": get_data_points(df_results, "Combined", indices),
            }
        )
        results_table = results_table.fillna(0)
        results_table.set_index("Metric", inplace=True)
        results_table.max_muti = max_muti
        results_table.max_rand = max_rand
        results_table.max_combo = max_combo
        best = get_peaks(df_results.Combined)
    best_params = results[best]

    # assign clusters
    df["cluster"] = best_params["clusters"]

    if actual is not None:
        plt.style.use("dark_background")
        fig, ax = plt.subplots(layout='constrained', figsize=(16, 9))
        ax.grid(False)

        # plot param vs. metrics
        var = np.array([z["sample"][param] for z in results])
        legend = (
                (["BIC %0.2f" % df_results.BIC.max()] if "BIC" in df_results.columns else []) +
                [
                    "DaviesBouldin %0.2f" % df_results.DaviesBouldin.max(),
                    "CalinskiHarabasz %0.2f" % df_results.CalinskiHarabasz.max(),
                    "Dunn %0.2f" % df_results.Dunn.max(),
                    "Silhouette",
                    "GiniCoeff",
                    "Niceness",
                    "Neatness",
                ]
                + (["Elbow"] if elbow is not None else [])
                + ["Ensemble", "Maximizing", "AdjRandIndex", "AdjMutualInfo", "Combined"]
        )
        # https://stats.stackexchange.com/a/336149/369868 :)
        colors = [
            "#e6194b",
            "#3cb44b",
            "#ffe119",
            "#0082c8",
            "#f58231",
            "#911eb4",
            "#46f0f0",
            "#f032e6",
            "#d2f53c",
            "#fabebe",
            "#008080",
            "#e6beff",
            "#aa6e28",
            "#fffac8",
            "#800000",
            "#aaffc3",
            "#808000",
            "#ffd8b1",
            "#000080",
            "#808080",
            "#ffffff",
            "#000000"
        ]
        for i, col in enumerate(legend):
            if " " in col:
                ax.plot(
                    var,
                    df_results[col.split()[0]] / df_results[col.split()[0]].max(),
                    '-D',
                    c=colors[i],
                    alpha=0.4,
                    markevery=[kwds[col.split()[0]]]
                )
            elif col == "Elbow":
                ax.axvline(var[elbow_x], c=colors[i], ls=":", alpha=0.4)
            elif col == "Ensemble":
                ax.axvline(
                    var[ensemble],
                    c=colors[i],
                    ls=":",
                    alpha=0.4,
                )
            elif col == "Maximizing":
                ax.axvline(best_params["sample"][param], c=colors[i], ls="--", alpha=0.4)
            elif col in ["AdjRandIndex", "AdjMutualInfo", "Combined"]:
                ax.plot(var, df_results[col], c=colors[i], alpha=0.4)
            else:
                ax.plot(var, df_results[col], '-D', c=colors[i], alpha=0.4, markevery=[kwds[col]])

        fig.legend(legend, loc="outside center right")
        plt.title(title)
        plt.xlabel(param)
        plt.show()

        # print results
        del best_params["clusters"]
        print(best_params)
        print(actual.value_counts())

    if plot_corr:
        # display corr
        n_cols = df.shape[1]
        associations(df, nom_nom_assoc="theil", figsize=(n_cols, n_cols))

    out = np.max(df_results.Combined) if actual is not None else np.max(results[ensemble])
    return (out, results_table) if actual is not None else out


def optimize_dbscan(
        df,
        title,
        actual=None,
        factor=1.0,
        offset=0.0,
        n_iter=100,
        min_samples=1,
        precomputed=None,
        use_mp=True,
        **kwargs
):
    df = df.copy()

    samples = [
        {"eps": offset + factor * z / n_iter, "min_samples": min_samples}
        for z in range(1, n_iter + 1)
    ]

    if precomputed:
        precomputed = "metric"
        matrix = gower_matrix(df, use_mp=use_mp, **kwargs)
    else:
        matrix = df
        for sample in samples:
            sample["metric"] = "minkowski"
            sample["p"] = 1

    if precomputed:
        res = sample_params(df,
                            matrix,
                            actual,
                            DBSCAN,
                            samples,
                            "eps",
                            n_iter,
                            precomputed,
                            use_mp,
                            title,
                            elbow=True)
    else:
        res = sample_params(df,
                            matrix,
                            actual,
                            DBSCAN,
                            samples,
                            "eps",
                            n_iter,
                            precomputed,
                            use_mp,
                            title,
                            elbow=True,
                            metric="minkowski")

    return df.cluster.to_numpy(), res


def optimize_gmm(df, title, actual=None, bayes=False, n_iter=20, use_mp=True):
    df = df.copy()

    if len(df) > 10000:
        covariance_type = "spherical"
    elif len(df) > 1000:
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
        BayesianGaussianMixture if bayes else GaussianMixture,
        samples,
        "n_components",
        n_iter,
        None,
        use_mp,
        title,
    )

    return df.cluster.to_numpy(), res


def optimize_agglo(
        df,
        title,
        actual=None,
        factor=1.0,
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
        matrix = gower_matrix(df, use_mp=use_mp, **kwargs)
    else:
        matrix = df

    samples = [
        {
            "distance_threshold": offset + factor * z / n_iter,
            "linkage": linkage,
            "n_clusters": None,
        }
        if precomputed
        else {
            "distance_threshold": offset + factor * z / n_iter,
            "linkage": linkage,
            "n_clusters": None,
            "metric": "l1",
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
        elbow=True,
    )

    return df.cluster.to_numpy(), res


def optimize_optics(
        df,
        title,
        actual=None,
        factor=1.0,
        offset=0.0,
        n_iter=100,
        use_mp=True,
        min_samples=1,
        metric="minkowski",
        **kwargs
):
    df = df.copy()

    if metric == "precomputed":
        matrix = gower_matrix(df, use_mp=use_mp, **kwargs)
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

    return df.cluster.to_numpy(), res


def optimize_kmeans(
        df, title, actual=None, n_iter=20, use_mp=True, precomputed=None, **kwargs
):
    df = df.copy()

    matrix = gower_matrix(df, use_mp=use_mp, **kwargs) if precomputed else df
    samples = [{"n_clusters": z, "random_state": 42} for z in range(2, n_iter + 2)]
    res = sample_params(
        df,
        matrix,
        actual,
        KMeans,
        samples,
        "n_clusters",
        n_iter,
        None,
        use_mp,
        title,
    )

    return df.cluster.to_numpy(), res


def optimize_hdbscan(
        df, title, actual=None, n_iter=100, use_mp=True, precomputed=None, **kwargs
):
    df = df.copy()

    if precomputed:
        precomputed = "metric"
        matrix = gower_matrix(df, use_mp=use_mp, **kwargs)
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

    return df.cluster.to_numpy(), res


def optimize_spectral(
        df, title, actual=None, n_iter=20, use_mp=True, precomputed=None, **kwargs
):
    df = df.copy()

    if precomputed:
        precomputed = "affinity"
        matrix = gower_matrix(df, use_mp=use_mp, **kwargs)
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

    return df.cluster.to_numpy(), res


def optimize_birch(df, title, actual=None, n_iter=100, factor=1.0, use_mp=True):
    df = df.copy()

    matrix = df

    samples = [
        {"threshold": factor * z / n_iter, "n_clusters": None}
        for z in range(1, n_iter + 1)
    ]
    res = sample_params(
        df, matrix, actual, Birch, samples, "threshold", n_iter, None, use_mp, title
    )

    return df.cluster.to_numpy(), res


def optimize_affinity(
        df, title, actual=None, n_iter=100, use_mp=True, precomputed=None, **kwargs
):
    df = df.copy()

    if precomputed:
        precomputed = "affinity"
        matrix = gower_matrix(df, use_mp=use_mp, **kwargs)
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
        elbow=True,
    )

    return df.cluster.to_numpy(), res


def optimize_meanshift(df, title, actual=None, n_iter=100, use_mp=True):
    df = df.copy()

    matrix = df

    samples = [
        {"bandwidth": estimate_bandwidth(df, quantile=z / n_iter)}
        for z in range(1, n_iter + 1)
    ]
    res = sample_params(
        df, matrix, actual, MeanShift, samples, "bandwidth", n_iter, None, use_mp, title
    )

    return df.cluster.to_numpy(), res
