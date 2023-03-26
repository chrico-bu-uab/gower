from functools import partial
from math import isclose, sqrt

import numpy as np
from scipy.sparse import issparse
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def call_gower_get(i, x_n_rows, y_n_rows, X_cat, X_num, Y_cat, Y_num,
      weight_cat, weight_num, weight_sum, num_ranges, h_t, knn_models):
    j_start = i
    if x_n_rows != y_n_rows:
        j_start = 0
    # call the main function
    res = gower_get(X_cat[i, :],
                    X_num[i, :],
                    Y_cat[j_start:y_n_rows, :],
                    Y_num[j_start:y_n_rows, :],
                    weight_cat,
                    weight_num,
                    weight_sum,
                    num_ranges,
                    h_t,
                    knn_models)
    return res


def get_cat_weight(x):
    one_hot = OneHotEncoder().fit_transform(np.array(x).reshape(-1, 1)).toarray()
    unbiased_var_sum = np.square(one_hot - one_hot.mean(axis=0)).sum() / (len(x) - 1)
    if isclose(unbiased_var_sum, 0) or isclose(unbiased_var_sum, 1):
        return 0
    n, k = one_hot.shape
    N = n * k
    biased_kurtosis_flat = k + 1 / (k - 1) - 2
    unbiased_kurtosis_flat = 1 / (N - 2) / (N - 3) * ((N ** 2 - 1) * biased_kurtosis_flat - 3 * (N - 1) ** 2) + 3
    return (1 - unbiased_var_sum) * unbiased_var_sum * unbiased_kurtosis_flat


def get_percentiles(X, R):
    return [np.nanpercentile(X, p, axis=0) for p in R]


def gower_matrix(data_x, data_y=None, weight=None, cat_features=None, R=(0, 100), c=0.0,
                 knn=False, use_mp=True, **tqdm_kwargs):
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
        if not isinstance(X, np.ndarray):
            is_number = np.vectorize(lambda x: not np.issubdtype(x, np.number))
            cat_features = is_number(X.dtypes)
        else:
            cat_features = np.zeros(x_n_cols, dtype=bool)
            for col in range(x_n_cols):
                if not np.issubdtype(type(X[0, col]), np.number):
                    cat_features[col] = True
    else:
        cat_features = np.array(cat_features)

    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)

    Z = np.concatenate((X, Y))

    x_index = range(0, x_n_rows)
    y_index = range(x_n_rows, x_n_rows + y_n_rows)

    Z_num = Z[:, np.logical_not(cat_features)].astype(np.float64)

    num_cols = Z_num.shape[1]
    num_ranges = np.zeros(num_cols)
    num_max = np.zeros(num_cols)

    knn_models = []
    n_knn = int(sqrt(x_n_rows))
    for col in range(num_cols):
        col_array = Z_num[:, col]
        p0, p1 = get_percentiles(col_array, R)

        if np.isnan(p1):
            p1 = 0.0
        if np.isnan(p0):
            p0 = 0.0
        num_max[col] = p1
        num_ranges[col] = abs(1 - p0 / p1) if (p1 != 0) else 0.0

        if knn:
            col_array[np.isnan(col_array)] = 1.0
            knn_models.append(NearestNeighbors(n_neighbors=n_knn).fit(col_array.reshape(-1, 1)))

    # This is to normalize the numeric values between 0 and 1.
    Z_num = np.divide(Z_num, num_max, out=np.zeros_like(Z_num), where=num_max != 0)
    Z_cat = Z[:, cat_features]

    if isinstance(weight, np.array):
        weight_cat = weight[cat_features]
        weight_num = weight[np.logical_not(cat_features)]
        weight_sum = weight.sum()
    else:
        if weight == "uniform":
            weight_cat = np.ones(Z_cat.shape[1])
        else:
            if use_mp:
                weight_cat = process_map(get_cat_weight, Z_cat.T, **tqdm_kwargs)
            else:
                weight_cat = [get_cat_weight(Z_cat[:, col]) for col in tqdm(range(Z_cat.shape[1]))]
        weight_cat = np.array(weight_cat)
        weight_cat /= 4 * weight_cat.max(initial=0)
        weight_num = np.ones(num_cols)
        weight_sum = Z.shape[1]

    out = np.zeros((x_n_rows, y_n_rows), dtype=np.float64)

    X_cat = Z_cat[x_index, ]
    X_num = Z_num[x_index, ]
    Y_cat = Z_cat[y_index, ]
    Y_num = Z_num[y_index, ]

    h_t = np.zeros(num_cols, dtype=np.float64)
    if c > 0:
        p0, p1 = get_percentiles(X_num, R)
        dist = norm(0, 1)
        h_t = c * x_n_rows ** -0.2 * np.minimum(np.nanstd(X_num, axis=0), (p1 - p0) / (dist.ppf(p1) - dist.ppf(p0)))
    g = partial(call_gower_get, x_n_rows=x_n_rows, y_n_rows=y_n_rows, X_cat=X_cat, X_num=X_num, Y_cat=Y_cat, Y_num=Y_num,
                weight_cat=weight_cat, weight_num=weight_num, weight_sum=weight_sum, num_ranges=num_ranges, h_t=h_t,
                knn_models=knn_models)
    if use_mp:
        processed = process_map(g, range(x_n_rows), **tqdm_kwargs)
    else:
        processed = list(map(g, tqdm(range(x_n_rows))))
    for i, res in enumerate(processed):
        j_start = i
        if x_n_rows != y_n_rows:
            j_start = 0
        out[i, j_start:] = res
        if x_n_rows == y_n_rows:
            out[i:, j_start] = res

    max_distance = np.nanmax(out)
    assert max_distance <= 1, max_distance

    return out


def gower_get(xi_cat, xi_num, xj_cat, xj_num, feature_weight_cat,
              feature_weight_num, feature_weight_sum, ranges_of_numeric, h_t, knn_models):
    # categorical columns
    sij_cat = np.where(xi_cat == xj_cat, np.zeros_like(xi_cat), np.ones_like(xi_cat))
    sum_cat = np.multiply(feature_weight_cat, sij_cat).sum(axis=1)

    # numerical columns
    abs_delta = np.maximum(np.absolute(xi_num - xj_num) - h_t, 0)
    xj_num = np.where(np.isnan(xj_num), 1.0, xj_num)
    if knn_models:
        for i, knn_model in enumerate(knn_models):
            neighbors = knn_model.kneighbors(xi_num[i].reshape(-1, 1), return_distance=False)
            for j, x in enumerate(xj_num[:, i]):
                if x in neighbors:
                    abs_delta[j, i] = 0.0
    sij_num = np.divide(abs_delta, ranges_of_numeric, out=np.zeros_like(abs_delta), where=ranges_of_numeric != 0)
    sij_num = np.minimum(sij_num, np.ones_like(sij_num))

    sum_num = np.multiply(feature_weight_num, sij_num).sum(axis=1)
    sums = np.add(sum_cat, sum_num)
    sum_sij = np.divide(sums, feature_weight_sum)

    return sum_sij


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
