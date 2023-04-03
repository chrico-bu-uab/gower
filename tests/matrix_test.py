import numpy as np
import pandas as pd
import pytest
from sklearn.mixture import GaussianMixture

import gower

np.set_printoptions(precision=0, suppress=True)
pd.set_option('display.max_columns', None)


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
    Xd['h_t'] = None
    Xd['knn'] = None

    aaa = gower.gower_matrix(X, weight_cat="uniform", weight_num="uniform")
    assert aaa[0][1] == pytest.approx(0.3590238), aaa[0][1]
    Xd.iloc[:-1, -4] = GaussianMixture(n_components=4, random_state=0).fit_predict(aaa[:-1, :-1])
    print(pd.DataFrame(aaa).describe())

    aaa = gower.gower_matrix(X)
    assert aaa[0][1] == pytest.approx(0.27430869751224973), aaa[0][1]
    Xd.iloc[:-1, -3] = GaussianMixture(n_components=4, random_state=0).fit_predict(aaa[:-1, :-1])
    print(pd.DataFrame(aaa).describe())

    aaa = gower.gower_matrix(X, R=(30, 70), c=2)
    assert aaa[0][1] == pytest.approx(0.8351356636176901), aaa[0][1]
    Xd.iloc[:-1, -2] = GaussianMixture(n_components=4, random_state=0).fit_predict(aaa[:-1, :-1])
    print(pd.DataFrame(aaa).describe())

    aaa = gower.gower_matrix(X, knn=True)
    assert aaa[0][1] == pytest.approx(0.16781469923188938), aaa[0][1]
    Xd.iloc[:-1, -1] = GaussianMixture(n_components=4, random_state=0).fit_predict(aaa[:-1, :-1])
    print(pd.DataFrame(aaa).describe())

    print(Xd.iloc[:, :6])
    print(Xd.iloc[:, 6:])
