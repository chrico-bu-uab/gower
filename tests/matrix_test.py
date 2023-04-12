import numpy as np
import pandas as pd
import pytest
from sklearn.mixture import GaussianMixture

import gower

np.set_printoptions(precision=0, suppress=True)
pd.set_option('display.max_columns', 999)


def test_answer():
    print()

    Xd = pd.DataFrame({'age': [21, 21, 19, 30, 21, 21, 19, 30, None],
                       'gender': ['M', 'M', 'N', 'M', 'F', 'F', 'F', 'F', None],
                       'civil_status': ['MARRIED', 'SINGLE', 'SINGLE', 'SINGLE', 'MARRIED', 'SINGLE', 'WIDOW',
                                        'DIVORCED', None],
                       'salary': [3000.0, 1200.0, 32000.0, 1800.0, 2900.0, 1100.0, 10000.0, 1500.0, None],
                       'has_children': [1, 0, 1, 1, 1, 0, 0, 1, None],
                       'available_credit': [2200, 100, 22000, 1100, 2000, 100, 6000, 2200, None]})
    X = np.asarray(Xd)
    Xd['uniform'] = None
    Xd['auto'] = None
    Xd['R'] = None
    Xd['h_t'] = None
    Xd['knn'] = None
    gm = GaussianMixture(n_components=5, random_state=0)

    aaa = gower.gower_matrix(X, weight_cat="uniform", weight_num="uniform")
    assert aaa[0][1] == pytest.approx(0.3590238), aaa[0][1]
    Xd.iloc[:-1, -5] = gm.fit_predict(aaa[:-1, :-1])

    aaa = gower.gower_matrix(X)
    assert aaa[0][1] == pytest.approx(0.32100430130958557), aaa[0][1]
    Xd.iloc[:-1, -4] = gm.fit_predict(aaa[:-1, :-1])

    aaa = gower.gower_matrix(X, R=(30, 70))
    assert aaa[0][1] == pytest.approx(0.6944147348403931), aaa[0][1]
    Xd.iloc[:-1, -3] = gm.fit_predict(aaa[:-1, :-1])

    aaa = gower.gower_matrix(X, R=(30, 70), c=3.5)
    assert aaa[0][1] == pytest.approx(0.15279258787631989), aaa[0][1]
    Xd.iloc[:-1, -2] = gm.fit_predict(aaa[:-1, :-1])

    aaa = gower.gower_matrix(X, knn=True)
    assert aaa[0][1] == pytest.approx(0.17561160027980804), aaa[0][1]
    Xd.iloc[:-1, -1] = gm.fit_predict(aaa[:-1, :-1])

    i = 65
    for col in Xd.iloc[:, 6:].columns:
        Xd[col] = Xd[col].apply(lambda x: chr(i + x) if x is not None else None)
        i += len(Xd[col].dropna().unique())

    print(Xd.sort_values(['uniform', 'auto', 'R', 'h_t', 'knn']))
