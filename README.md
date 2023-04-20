<!-- badges: start -->
[![Build Status](https://travis-ci.com/wwwjk366/gower.svg?branch=master)](https://travis-ci.com/wwwjk366/gower)
[![PyPI version](https://badge.fury.io/py/gower.svg)](https://pypi.org/project/gower/)
[![Downloads](https://pepy.tech/badge/gower/month)](https://pepy.tech/project/gower/month)
<!-- badges: end -->

# Introduction

Gower's distance calculation in Python. Gower Distance is a distance measure that can be used to calculate distance between two entity whose attribute has a mixed of categorical and numerical values. [Gower (1971) A general coefficient of similarity and some of its properties. Biometrics 27 857–874.](https://www.jstor.org/stable/2528823?seq=1) 

More details and examples can be found on my personal website here:(https://www.thinkdatascience.com/post/2019-12-16-introducing-python-package-gower/)

Core functions are wrote by [Marcelo Beckmann](https://sourceforge.net/projects/gower-distance-4python/files/).

# Examples

## Installation

```
pip install gower
```

## Generate some data

```python
import numpy as np
import pandas as pd
import gower

Xd=pd.DataFrame({'age':[21,21,19, 30,21,21,19,30,None],
'gender':['M','M','N','M','F','F','F','F',None],
'civil_status':['MARRIED','SINGLE','SINGLE','SINGLE','MARRIED','SINGLE','WIDOW','DIVORCED',None],
'salary':[3000.0,1200.0 ,32000.0,1800.0 ,2900.0 ,1100.0 ,10000.0,1500.0,None],
'has_children':[1,0,1,1,1,0,0,1,None],
'available_credit':[2200,100,22000,1100,2000,100,6000,2200,None]})
Yd = Xd.iloc[1:3,:]
X = np.asarray(Xd)
Y = np.asarray(Yd)

```

## Find the distance matrix

```python
gower.gower_matrix(X)
```




    array([[0.        , 0.3058583 , 0.68839877, 0.29736742, 0.13945927,
            0.4434072 , 0.52530207, 0.42574698,        nan],
           [0.3058583 , 0.        , 0.72049697, 0.29239914, 0.44015913,
            0.1375489 , 0.42052023, 0.57465973,        nan],
           [0.68839877, 0.72049697, 0.        , 0.71200609, 0.69097799,
            0.72116582, 0.71061689, 0.84038565,        nan],
           [0.29736742, 0.29239914, 0.71200609, 0.        , 0.43166825,
            0.42994804, 0.68578945, 0.28627369,        nan],
           [0.13945927, 0.44015913, 0.69097799, 0.43166825, 0.        ,
            0.30394793, 0.39100125, 0.29010845,        nan],
           [0.4434072 , 0.1375489 , 0.72116582, 0.42994804, 0.30394793,
            0.        , 0.28430903, 0.43844853,        nan],
           [0.52530207, 0.42052023, 0.71061689, 0.68578945, 0.39100125,
            0.28430903, 0.        , 0.5404089 ,        nan],
           [0.42574698, 0.57465973, 0.84038565, 0.28627369, 0.29010845,
            0.43844853, 0.5404089 , 0.        ,        nan],
           [       nan,        nan,        nan,        nan,        nan,
                   nan,        nan,        nan,        nan]])


## Find Top n results

```python
gower.gower_topn(Xd.iloc[0:2,:], data_y=Xd.iloc[:,])
```




    {'index': array([0, 4, 3, 1, 7]),
     'values': array([0.        , 0.13945927, 0.29736742, 0.3058583 , 0.42574698])}
