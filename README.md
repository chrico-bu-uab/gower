# Examples

## Installation

```
git clone https://github.com/chrico-bu-uab/gower.git
pip install -e gower
```

## Generate some data and compute the Gower distance matrix

```python
import numpy as np
import pandas as pd
import gower

np.set_printoptions(precision=4, suppress=True)

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

gower.gower_matrix(Xd)
```
    [[0.     0.4766 0.4287 0.2789 0.3715 0.6697 0.5685 0.5841 0.4987 0.4855]
     [0.4766 0.     0.4548 0.3535 0.4507 0.3056 0.438  0.5051 0.5416 0.4886]
     [0.4287 0.4548 0.     0.3987 0.4037 0.5354 0.4882 0.5173 0.463  0.4771]
     [0.2789 0.3535 0.3987 0.     0.4289 0.4341 0.5503 0.4148 0.5017 0.4583]
     [0.3715 0.4507 0.4037 0.4289 0.     0.3878 0.3753 0.3034 0.311  0.4212]
     [0.6697 0.3056 0.5354 0.4341 0.3878 0.     0.3385 0.2217 0.4868 0.4791]
     [0.5685 0.438  0.4882 0.5503 0.3753 0.3385 0.     0.3994 0.3842 0.3491]
     [0.5841 0.5051 0.5173 0.4148 0.3034 0.2217 0.3994 0.     0.2695 0.399 ]
     [0.4987 0.5416 0.463  0.5017 0.311  0.4868 0.3842 0.2695 0.     0.3441]
     [0.4855 0.4886 0.4771 0.4583 0.4212 0.4791 0.3491 0.399  0.3441 0.    ]]
