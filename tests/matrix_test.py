import pytest

from gower.gower_dist import *

np.set_printoptions(precision=4, suppress=True)
pd.set_option("display.max_columns", 999)


def test_answer():
    print()

    Xd = pd.DataFrame(
        {
            "age": [21, 21, 19, 30, 21, 21, 19, 30, None],
            "gender": ["M", "M", "N", "M", "F", "F", "F", "F", None],
            "civil_status": [
                "MARRIED",
                "SINGLE",
                "SINGLE",
                "SINGLE",
                "MARRIED",
                "SINGLE",
                "WIDOW",
                "DIVORCED",
                None,
            ],
            "salary": [
                3000.0,
                1200.0,
                32000.0,
                1800.0,
                2900.0,
                1100.0,
                10000.0,
                1500.0,
                None,
            ],
            "has_children": [1, 0, 1, 1, 1, 0, 0, 1, None],
            "available_credit": [2200, 100, 22000, 1100, 2000, 100, 6000, 2200, None],
            "has_default": [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, None],
            "balance": [970678, 610446, 877380, 1086427, 500000, 0, 800000, 400000, None],
        }
    )
    Xd["uniform"] = Xd["auto"] = Xd["R"] = Xd["h_t"] = Xd["knn"] = None
    gm = GaussianMixture(n_components=4, random_state=0)

    aaa = gower_matrix(Xd.iloc[:, :-5], weight_num="uniform")
    assert aaa[0][1] == pytest.approx(0.3419647260257232), aaa[0][1]
    Xd.iloc[:-1, -5] = gm.fit_predict(aaa[:-1, :-1])

    aaa = gower_matrix(Xd.iloc[:, :-5])
    assert aaa[0][1] == pytest.approx(0.30162133530643037), aaa[0][1]
    Xd.iloc[:-1, -4] = gm.fit_predict(aaa[:-1, :-1])

    aaa = gower_matrix(Xd.iloc[:, :-5], q=0.25)
    assert aaa[0][1] == pytest.approx(0.4967784664107954), aaa[0][1]
    Xd.iloc[:-1, -3] = gm.fit_predict(aaa[:-1, :-1])

    aaa = gower_matrix(Xd.iloc[:, :-5], q=0.25, c_t=1.06)
    assert aaa[0][1] == pytest.approx(0.19572596545315152), aaa[0][1]
    Xd.iloc[:-1, -2] = gm.fit_predict(aaa[:-1, :-1])

    aaa = gower_matrix(Xd.iloc[:, :-5], knn=True)
    assert aaa[0][1] == pytest.approx(0.30162133530643037), aaa[0][1]
    Xd.iloc[:-1, -1] = gm.fit_predict(aaa[:-1, :-1])

    i = 65
    dfs = []
    for col in ["uniform", "auto", "R", "h_t", "knn"]:
        Xd[col] = Xd[col].apply(lambda x: chr(i + x) if x is not None else None)
        i += len(Xd[col].dropna().unique())
        dfs.append(hamming_similarity(Xd[col].dropna()))

    for col in ["uniform", "auto", "R", "h_t", "knn"]:
        for col2 in ["uniform", "auto", "R", "h_t", "knn"]:
            if col == col2:
                continue
            print(adjusted_mutual_info_score(Xd.iloc[:-1][col], Xd.iloc[:-1][col2]))
            print(adjusted_rand_score(Xd.iloc[:-1][col], Xd.iloc[:-1][col2]))

    print(Xd.sort_values(["uniform", "auto", "R", "h_t", "knn"]))
