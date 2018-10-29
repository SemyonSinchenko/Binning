# Binning
Monotonic binning (WOE - Weight Of Evidence) in Python.
This project is licensed under the terms of the MIT license.

The main method of class:

`
Monotonic binning.fit_transform(self, df: pd.DataFrame, target: str,
                      column: str, minbins: int=20, maxbins: int=200,
                      tol: float=0.0001, stopping: int=2)`

* df: pandas.DataFrame, DataFrame
* target: String, target column
* column: String, column for binning
* minbins: Int, the minimal number of initial bins (default=20)
* maxbins: Int, the maximal number of initial bins (default=200)
* tol: Float, tolerance for stopping (default=0.0001)
* stopping: Int, the minimal number of final bins
* return: numpy.array(n-obs) - binned column


Example of usage:

`import pandas as pd`
`import numpy as np`
`import MonotonicBinning`


`X = np.random.normal(0, 3, size=(10000, 3))`
`W = np.random.rand(3)`
`b = np.random.rand() * np.ones(10000)`
`y = X.dot(W) + b`
`labels = 1 / (1 + np.exp(y))`
`labels[labels >= .5] = 1`
`labels[labels < .5] = 0`

`df = pd.DataFrame(X, columns=['var_' + str(i) for i in range(X.shape[1])])`
`df['target'] = labels`
`binner = MonotonicBinning()`

`df['var_0'] = binner.fit_transform(df, target='target', column='var_0')`
`print(df.groupby('var_0').agg({'target': ['mean', 'count']}))`
