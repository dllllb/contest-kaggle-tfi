import math
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.gaussian_process import GaussianProcess, GaussianProcessRegressor
from sklearn.linear_model import HuberRegressor, ARDRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import clone as sk_clone
from sklearn.metrics import make_scorer
from sklearn.preprocessing import Imputer

import dstools.dstools.ml.transformers as tr
    

def update_model_stats(stats_file, params, results):
    import json
    import os.path
    
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    else:
        stats = []
        
    stats.append({**results, **params})
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)

        
def run_experiment(evaluator, params, stats_file):    
    import time
    
    params = init_params(params)
    start = time.time()
    scores = evaluator(params)
    exec_time = time.time() - start
    update_model_stats(stats_file, params, {**scores, 'exec-time-sec': exec_time})

    
def init_params(overrides):
    defaults = {
        'validation-type': 'split',
        'n_folds': 100,
    }
    return {**defaults, **overrides}
    

def rmse(true, pred):
    return math.sqrt(mean_squared_error(true, pred))


def split_test(est, n_tests):
    df = pd.read_csv('train.csv.gz', index_col='Id')
    features = df.drop(['revenue'], axis=1)
    target = df.revenue
    
    scores = []
    for i in range(n_tests):
        m = sk_clone(est)
        xtr, xtst, ytr, ytst = train_test_split(features, target, test_size=.2)
        m.fit(xtr, ytr)
        scorer = make_scorer(rmse)
        scores.append(scorer(m, xtst, ytst))

    scores = np.array(scores)
    return {'RMSE-mean': scores.mean(), 'RMSE-STD': scores.std()}


def submit(est):
    df = pd.read_csv('train.csv.gz', index_col='Id')
    features = df.drop(['revenue'], axis=1)
    labels = df.revenue

    model = est.fit(features, labels)

    df_test = pd.read_csv('test.csv.gz', index_col='Id')

    y_pred = model.predict(df_test)

    res_df = pd.DataFrame({'Prediction': y_pred}, index=df_test.index)
    res_df.to_csv('results.csv', index_label='Id')
    
    
def outliers_filter(features, target):
    threshold = target.mean()+target.std()*3
    return features[target < threshold], target[target < threshold]


class SamplesFilteringPipeline(BaseEstimator):
    def __init__(self, pipeline, samples_filter):
        self.pipeline = pipeline
        self.samples_filter = samples_filter

    def fit(self, X, y):
        X_filtered, y_filtered = self.samples_filter(X, y)
        return self.pipeline.fit(X_filtered, y_filtered)

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    
def no_outliers_pipeline(est):
    return SamplesFilteringPipeline(est, outliers_filter)


def days_to_delta(df):
    delta = np.timedelta64(1, 'D')
    days_open = (pd.to_datetime('2015-02-01') - pd.to_datetime(df['Open Date'])) / delta
    dfc = df.drop('Open Date', axis=1).copy()
    dfc['days_open'] = days_open
    return dfc


def validate(params):
    category_encoding = params['category_encoding']
    
    if category_encoding == 'onehot':
        df2dict = FunctionTransformer(
            lambda x: x.to_dict(orient='records'), validate=False)
            
        transf = make_pipeline(
            FunctionTransformer(days_to_delta, validate=False),
            df2dict,
            DictVectorizer(sparse=False),
        )
    elif category_encoding == 'empyrical_bayes':
        transf = make_pipeline(
        FunctionTransformer(days_to_delta, validate=False),
            tr.empirical_bayes_encoder_normal_distr(),
            Imputer()
        )
    elif category_encoding == 'count':
        transf = make_pipeline(
            FunctionTransformer(days_to_delta, validate=False),
            tr.count_encoder(),
            Imputer()
        )
    
    reg_type = params['regressor_type']
    
    if reg_type == 'rfr':
        reg = make_pipeline(
            SelectKBest(f_regression, params['k_best']),
            RandomForestRegressor(
                n_jobs=params['n_jobs'],
                n_estimators=params['n_estimators'],
                max_features=params['max_features'],
                max_depth=params['max_depth'],
                random_state=1))
    elif reg_type == 'huber':
        reg = HuberRegressor(epsilon=params['epsilon'])
    elif reg_type == 'ard':
        reg = ARDRegression()
        
    est = make_pipeline(transf, reg)
    
    if params['drop_outliers']:
        est = no_outliers_pipeline(est)
            
    valid_mode = params['valid_mode']
    n_folds = params['n_folds']
    if valid_mode == 'split':
        return split_test(est, n_folds)


def test_validate():
    params = {
        "category_encoding": "empyrical_bayes",
        "valid_mode": "split",
        "k_best": 20,
        "n_jobs": 4,
        "n_estimators": 100,
        "max_features": 0.2,
        "max_depth": 2,
        "regressor_type": "rfr",
        "drop_outliers": True,
        'n_folds': 3,
    }
    print(validate(params))
