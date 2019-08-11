import numpy as np
from functools import partial

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals.joblib import Parallel, delayed


class TargetCategoryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, builder, columns=None, n_jobs=1, true_label=None):
        self.vc = dict()
        self.columns = columns
        self.n_jobs = n_jobs
        self.true_label = true_label
        self.builder = builder

    def fit(self, df, y=None):
        if self.columns is None:
            columns = df.select_dtypes(include=['object'])
        else:
            columns = self.columns

        if self.true_label is not None:
            target = (y == self.true_label)
        else:
            target = y

        encoders = Parallel(n_jobs=self.n_jobs)(
            delayed(self.builder)(df[col], target)
            for col in columns
        )

        self.vc = dict(zip(columns, encoders))

        return self

    def transform(self, df):
        res = df.copy()
        for col, encoder in self.vc.items():
            res[col] = encoder(res[col])
        return res


def map_encoder(vals, mapping):
    return vals.map(lambda x: mapping.get(x, mapping.get('nan', 0)))


def build_count_encoder(column, __):
    entries = column.replace(np.nan, 'nan').value_counts()
    entries = entries.sort_values(ascending=False).index
    mapping = dict(zip(entries, range(len(entries))))
    return partial(map_encoder, mapping=mapping)


def count_encoder(columns=None, n_jobs=1):
    return TargetCategoryEncoder(build_count_encoder, columns, n_jobs)
