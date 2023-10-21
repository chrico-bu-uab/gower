import pytest

from gower.gower_dist import *

np.set_printoptions(precision=4, suppress=True)
pd.set_option("display.max_columns", 999)


def test_answer():
    print()

    Xd = pd.DataFrame(
        {
            "age": [19, 30, 21, 30, 19, 30, 21, 30, 19, 30],
            "gender": ["M", "M", "N", "M", "F", "F", "F", "F", None, None],
            "civil_status": ["MARRIED", "SINGLE", "SINGLE", "SINGLE", "MARRIED",
                             "SINGLE", "WIDOW", "DIVORCED", "DIVORCED", "MARRIED"],
            "salary": [3000.0, 1200.0, 32000.0, 1800.0, 2900.0, 1100.0, 10000.0,
                       1500.0, 1200.0, None],
            "has_children": [1, 0, 1, 1, 1, 0, 0, 1, 1, None],
            "available_credit": [22000, 100, 2200, None, 2000, 100, 6000, 2200, 0, None],
            "default_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "balance": [10, None, 9, 15, 5, 0, 8, 2, None, None],
            "day_of_week": [1, 5, 1, 1, 6, 1, 3, 1, 2, None],
        }
    )
    circular_features = (Xd.columns == "day_of_week") * 7

    Xd["uniform"] = Xd["auto"] = Xd["q"] = Xd["h_t"] = Xd["knn"] = None
    estimator = DBSCAN(eps=0.3, min_samples=1, metric="precomputed")

    aaa = gower_matrix(Xd.iloc[:, :-5], circular_features=circular_features, weight_num="uniform")
    assert aaa[0][1] == pytest.approx(0.5923871944907544), aaa[0][1]
    Xd.iloc[:, -5] = estimator.fit_predict(aaa)

    aaa = gower_matrix(Xd.iloc[:, :-5], circular_features=circular_features)
    assert aaa[0][1] == pytest.approx(0.4991186854937608), aaa[0][1]
    Xd.iloc[:, -4] = estimator.fit_predict(aaa)

    aaa = gower_matrix(Xd.iloc[:, :-5], circular_features=circular_features, q=0.25)
    assert aaa[0][1] == pytest.approx(0.609687061116527), aaa[0][1]
    Xd.iloc[:, -3] = estimator.fit_predict(aaa)

    aaa = gower_matrix(Xd.iloc[:, :-5], circular_features=circular_features, q=0.25, c_t=1.06)
    assert aaa[0][1] == pytest.approx(0.40686992125796867), aaa[0][1]
    Xd.iloc[:, -2] = estimator.fit_predict(aaa)

    aaa = gower_matrix(Xd.iloc[:, :-5], circular_features=circular_features, knn=True)
    assert aaa[0][1] == pytest.approx(0.47661833992938185), aaa[0][1]
    Xd.iloc[:, -1] = estimator.fit_predict(aaa)

    symbols = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
    i = 0
    for col in ["uniform", "auto", "q", "h_t", "knn"]:
        Xd[col] = Xd[col].apply(lambda j: symbols[i + j])
        i += len(Xd[col].dropna().unique())

    for col in ["uniform", "auto", "q", "h_t", "knn"]:
        for col2 in ["uniform", "auto", "q", "h_t", "knn"]:
            if col == col2:
                continue
            m = adjusted_mutual_info_score(Xd[col], Xd[col2])
            r = adjusted_rand_score(Xd[col], Xd[col2])
            assert m < 1, (col, col2)
            assert r < 1, (col, col2)

    print(Xd.sort_values(["uniform", "auto", "q", "h_t", "knn"]))

    print(aaa)
