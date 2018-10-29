#%%
import numpy as np
import pandas as pd
import scipy.stats as sp
from copy import deepcopy


class MonotonicBinning(object):
    """
    Create the monotonic linear binning of given column.
    Monotonic here mean that ratio of bad and good observations (y=1 and y=0) in different bins are close by.

    Example of usage:
        import pandas as pd
        import numpy as np
        import MonotonicBinning

        X = np.random.normal(0, 3, size=(10000, 3))
        W = np.random.rand(3)
        b = np.random.rand() * np.ones(10000)
        y = X.dot(W) + b

        labels = 1 / (1 + np.exp(y))
        labels[labels >= .5] = 1
        labels[labels < .5] = 0

        df = pd.DataFrame(X, columns=['var_' + str(i) for i in range(X.shape[1])])
        df['target'] = labels

        binner = MonotonicBinning()
        df['var_0'] = binner.fit_transform(df, target='target', column='var_0')

        print(df.groupby('var_0').agg({'target': ['mean', 'count']}))
    """

    def __init__(self):
        self.bins = None

    @staticmethod
    def __create_bins(x, y, n):
        """
        Inner (private) method. Binned variable and calculate Spearman correlation.
        Return bins, correlation.
        :param x: Variable
        :param y: Target
        :param n: The number of bins
        :return: Spearman correlation coefficient (absolute value)
        """

        # Bin x
        binned_x, bins = pd.cut(x, bins=n, retbins=True, labels=False)

        # Create temporary DataFrame with binned x and y
        tmp_df = pd.DataFrame({'x': binned_x, 'y': y})

        # Group tmp_df by bins
        tmp_df = tmp_df.groupby('x', as_index=False).agg({'y': 'mean'})

        # Calculate correlations (Spearman)
        corr = sp.spearmanr(tmp_df['x'], tmp_df['y'])[0]

        # Return absolute value of corr
        return np.abs(corr), bins

    def fit_transform(self, df: pd.DataFrame, target: str,
                      column: str, minbins: int=20, maxbins: int=200,
                      tol: float=0.0001, stopping: int=2):
        """
        Monotonic binning.
        :param df: pandas.DataFrame, DataFrame
        :param target: String, target column
        :param column: String, column for binning
        :param minbins: Int, the minimal number of initial bins (default=20)
        :param maxbins: Int, the maximal number of initial bins (default=200)
        :param tol: Float, tolerance for stopping (default=0.0001)
        :param stopping: Int, the minimal number of final bins
        :return: numpy.array(n-obs) - binned column
        """

        assert (target in df.columns), '{} not in the data frame!'.format(target)
        assert (column in df.columns), '{} not in the data frame!'.format(column)

        # Make deepcopy to avoid side-effects
        x = deepcopy(df[column])
        y = deepcopy(df[target])

        # N - the number of initial bins (min of nbins and n-unique values)
        n = min(maxbins, max(len(x.unique()), minbins))

        while n > stopping:
            # Create next bins
            corr, bins = MonotonicBinning.__create_bins(x, y, n)

            # Check that corr is jet bad
            if np.abs(1 - corr) < tol:
                break
            else:
                # Else decrease n
                n -= 1

        self.bins = bins

        return pd.cut(x, bins=self.bins, retbins=False, labels=False)

    def transform(self, df: pd.DataFrame, column: str):
        """
        Apply binning to another DataFrame.
        :param df: pandas.DataFrame, data frame
        :param column: String, column for binning
        :return: numpy.array(n-obs) - binned column
        """

        assert (column in df.columns), '{} not in the data frame!'.format(column)

        return pd.cut(df[column], bins=self.bins, labels=False).values

    def print_bins(self):
        """
        Print bins.
        :return: None
        """

        print(self.bins)

