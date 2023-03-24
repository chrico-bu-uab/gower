import numpy as np
import pandas as pd
import pytest

import gower

from sklearn.cluster import DBSCAN

np.set_printoptions(precision=0, suppress=True)
pd.set_option('display.max_columns', None)


def test_answer():
    Xd = pd.DataFrame({'age': [21, 21, 19, 30, 21, 21, 19, 30, None],
                       'gender': ['M', 'M', 'N', 'M', 'F', 'F', 'F', 'F', None],
                       'civil_status': ['MARRIED', 'SINGLE', 'SINGLE', 'SINGLE', 'MARRIED', 'SINGLE', 'WIDOW',
                                        'DIVORCED', None],
                       'salary': [3000.0, 1200.0, 32000.0, 1800.0, 2900.0, 1100.0, 10000.0, 1500.0, None],
                       'has_children': [1, 0, 1, 1, 1, 0, 0, 1, None],
                       'available_credit': [2200, 100, 22000, 1100, 2000, 100, 6000, 2200, None]})
    X = np.asarray(Xd)
    Xd['a'] = None
    Xd['b'] = None
    Xd['c'] = None
    Xd['d'] = None

    dbscan = DBSCAN(eps=0.13, min_samples=1, metric="precomputed")

    aaa = gower.gower_matrix(X, weight=np.ones(6))
    assert aaa[0][1] == pytest.approx(0.3590238094329834), aaa[0][1]
    Xd.iloc[:-1, -4] = dbscan.fit_predict(aaa[:-1, :-1])

    aaa = gower.gower_matrix(X)
    assert aaa[0][1] == pytest.approx(0.2916707939830703), aaa[0][1]
    Xd.iloc[:-1, -3] = dbscan.fit_predict(aaa[:-1, :-1])

    aaa = gower.gower_matrix(X, R=(30, 70), c=2)
    assert aaa[0][1] == pytest.approx(0.445336398682882), aaa[0][1]
    Xd.iloc[:-1, -2] = dbscan.fit_predict(aaa[:-1, :-1])

    aaa = gower.gower_matrix(X, knn=True)
    assert aaa[0][1] == pytest.approx(0.12500412731640362), aaa[0][1]
    Xd.iloc[:-1, -1] = dbscan.fit_predict(aaa[:-1, :-1])

    print()
    print(Xd)
