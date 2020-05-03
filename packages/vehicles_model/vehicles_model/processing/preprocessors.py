# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 07:37:18 2020

@author: ali95
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from math import sqrt

import joblib
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


class Pipeline:

    def __init__(self, target, categorical_to_impute, numerical_to_impute,
                 categoricals, ohe_catgs, ord_catgs, unneeded_vars,
                 numerical_ranges, alpha=0.0003, val_size=0.3, test_size=0.33,
                 random_state=300, rare_perc=0.01):
        # data sets
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        # engineering parameters (to be learnt from data)
        self.imputing_dict = {}
        self.frequent_category_dict = {}
        self.encoding_dict = {}
        self.medians_dict = {}

        self.scaler = MinMaxScaler()
        self.ohe = OneHotEncoder(sparse=False)
        self.selector = SelectFromModel(Lasso(alpha=alpha,
                                              random_state=random_state))
        self.model = RandomForestRegressor(n_estimators=300,
                                           min_samples_split=2,
                                           min_samples_leaf=10,
                                           max_features='auto',
                                           max_depth=15,
                                           n_jobs=-1,
                                           random_state=random_state)

        # groups of variables to engineer
        self.target = target
        self.categorical_to_impute = categorical_to_impute
        self.numerical_to_impute = numerical_to_impute
        self.categoricals = categoricals
        self.ohe_catgs = ohe_catgs
        self.ord_catgs = ord_catgs
        self.unneeded_vars = unneeded_vars
        self.numerical_ranges = numerical_ranges

        # more parameters
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.rare_perc = rare_perc

    # functions that learn from training set
    def get_frequent_labels(self):
        for var in self.categoricals:
            tmp = self.X_train.groupby(
                var)[self.target].count()/len(self.X_train)
            self.frequent_category_dict[var] = tmp[tmp > self.rare_perc].index
        return self

    def get_oh_encodings(self):
        self.ohe.fit(self.X_train[self.ohe_catgs])
        return self

    def get_ord_encodings(self):
        for var in self.ord_catgs:
            order_labels = self.X_train.groupby(
                var)[self.target].mean().sort_values().index
            self.encoding_dict[var] = {
                k: i for i, k in enumerate(order_labels, 0)}
        return self

    def get_medians(self):
        for var in self.numerical_to_impute:
            self.medians_dict[var] = self.X_train[var].median()
        return self

    def get_trained_scaler(self):
        self.scaler.fit(self.X_train)
        return self

    def get_selected_features(self):
        print(self.X_train.columns)
        self.selector.fit(self.X_train, self.y_train)
        self.features = list(
            set(self.X_train.columns[self.selector.get_support()]))
        print(self.features)
        return self

    ################### functions that transform sets #########################
    def split_data(self, data):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, data[self.target],
                                                                                test_size=self.val_size,
                                                                                random_state=self.random_state)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_test, self.X_test[self.target],
                                                                            test_size=self.test_size,
                                                                            random_state=self.random_state)
        return self

    def drop_unneeded(self, df):
        df = df.copy()
        df.drop(self.unneeded_vars, axis=1, inplace=True)
        return df

    def replace_missing_catgs(self, df):
        df = df.copy()
        df[self.categorical_to_impute].fillna('Missing')

        return df

    def replace_rare_labels(self, df):
        df = df.copy()
        for var in self.categoricals:
            df[var] = np.where(df[var].isin(self.frequent_category_dict[var]),
                               df[var], 'Rare')
        return df

    def apply_oh_encodings(self, df):
        df = df.copy()
        df = pd.concat([df.reset_index(drop=True), pd.DataFrame(self.ohe.transform(df[self.ohe_catgs]),
                                                                columns=np.concatenate(self.ohe.categories_).ravel())], axis=1)
        df.drop(self.ohe_catgs, axis=1, inplace=True)
        return df

    def apply_ord_encodings(self, df):
        df = df.copy()
        for var in self.ord_catgs:
            df[var] = df[var].map(self.encoding_dict[var])
        return df

    def transform_description(self, df, var='description'):
        df = df.copy()
        splits = df[var].str.split()
        df['desc_n_words'] = splits.apply(len)
        df['mean_length'] = splits.apply(
            lambda x: np.mean([len(word) for word in x]))
        df['n_figures'] = splits.apply(lambda x: np.sum(
            [np.char.isnumeric(word) for word in x]))
        df.drop(var, axis=1, inplace=True)
        return df

    def impute_outliers(self, df):
        df = df.copy()
        for var in self.numerical_to_impute:
            df[var] = np.where(df[var].between(self.numerical_ranges[var][0], self.numerical_ranges[var][1]),
                               df[var], self.medians_dict[var])
        return df

    def apply_scaler(self, df):
        df = df.copy()
        df = pd.DataFrame(self.scaler.transform(
            df), index=df.index, columns=df.columns)
        return df

    # fit function that will run all preceding functions

    def fit(self, data):
        print('Started fitting pipeline')
        # split the data
        self.split_data(data)
        print('finished splitting data')

        # drop unnecessary variables
        self.X_train = self.drop_unneeded(self.X_train)
        self.X_val = self.drop_unneeded(self.X_val)
        self.X_test = self.drop_unneeded(self.X_test)
        print('dropped unneeded vars')

        self.X_train = self.replace_missing_catgs(self.X_train)
        self.X_val = self.replace_missing_catgs(self.X_val)
        self.X_test = self.replace_missing_catgs(self.X_test)
        print('replaced missings catgs')

        # acquire frequent labels
        self.get_frequent_labels()

        # replace rare labels
        self.X_train = self.replace_rare_labels(self.X_train)
        self.X_val = self.replace_rare_labels(self.X_val)
        self.X_test = self.replace_rare_labels(self.X_test)
        print('replaced rare labels')

        # acquire one-hot encodings
        self.get_oh_encodings()

        # apply oh-encoding on sets
        self.X_train = self.apply_oh_encodings(self.X_train)
        self.X_val = self.apply_oh_encodings(self.X_val)
        self.X_test = self.apply_oh_encodings(self.X_test)
        print('applied ohe')

        # acquire ordinal encodings
        self.get_ord_encodings()

        # apply ordinal encodings on sets and drop nulls
        self.X_train = self.apply_ord_encodings(self.X_train)
        self.X_val = self.apply_ord_encodings(self.X_val)
        self.X_test = self.apply_ord_encodings(self.X_test)
        print('applied ordinal encodings')

        # drop rows with unidentified ordinal labels
        self.X_train = self.X_train.dropna(
            subset=self.ord_catgs+['description']).reset_index(drop=True)
        self.X_val = self.X_val.dropna(
            subset=self.ord_catgs+['description']).reset_index(drop=True)
        self.X_test = self.X_test.dropna(
            subset=self.ord_catgs+['description']).reset_index(drop=True)
        print('dropped unidentified and reset index')

       # apply description transformation
        self.X_train = self.transform_description(self.X_train)
        self.X_val = self.transform_description(self.X_val)
        self.X_test = self.transform_description(self.X_test)
        print('transformed descriptions')

        # learn numerical medians
        self.get_medians()

        # filter out outliers
        self.X_train = self.impute_outliers(self.X_train)
        self.X_val = self.impute_outliers(self.X_val)
        self.X_test = self.impute_outliers(self.X_test)
        print('imputed with medians')

        #  log-transform target var and pop from main sets
        self.y_train = np.log(self.X_train.pop(self.target))
        self.y_val = np.log(self.X_val.pop(self.target))
        self.y_test = np.log(self.X_test.pop(self.target))
        print('log-transformed prices')
        # train the data scaler
        self.get_trained_scaler()

        # apply the scaler
        self.X_train = self.apply_scaler(self.X_train)
        self.X_val = self.apply_scaler(self.X_val)
        self.X_test = self.apply_scaler(self.X_test)
        print('scaled data')

        # train the selector
        self.get_selected_features()

        # select the features
        self.X_train = self.X_train[self.features]
        self.X_val = self.X_val[self.features]
        self.X_test = self.X_test[self.features]
        print('selected features')
        # train the ML model
        self.model.fit(self.X_train, self.y_train)

        print('Model training finished')

        return self

    # transformations to new data
    def transform(self, data):
        data = data.copy()
        data = self.drop_unneeded(data)
        data = self.replace_missing_catgs(data)
        data = self.replace_rare_labels(data)
        data = self.apply_oh_encodings(data)
        data = self.apply_ord_encodings(data)
        data = data.dropna(subset=self.ord_catgs +
                           ['description']).reset_index(drop=True)
        data = self.transform_description(data)
        data = self.impute_outliers(data)
        if self.target in data:
            data.drop(self.target, axis=1, inplace=True)
        data = self.apply_scaler(data)
        data = data[self.features]

        print('Data transformation finished')

        return data

    # predicts new data
    def predict(self, data):
        data = self.transform(data)
        preds = np.exp(self.model.predict(data))
        return preds

    def evaluate_model(self):
        print('train set rmse = ', sqrt(
            mse(np.exp(self.y_train), np.exp(self.model.predict(self.X_train)))))
        print('val set rmse = ', sqrt(
            mse(np.exp(self.y_val), np.exp(self.model.predict(self.X_val)))))
        print('test set rmse = ', sqrt(
            mse(np.exp(self.y_test), np.exp(self.model.predict(self.X_test)))))
        print()
        print('train r2 score = ', r2_score(np.exp(self.y_train),
                                            np.exp(self.model.predict(self.X_train))))
        print('val r2 score = ', r2_score(np.exp(self.y_val),
                                          np.exp(self.model.predict(self.X_val))))
        print('test r2 score = ', r2_score(np.exp(self.y_test),
                                           np.exp(self.model.predict(self.X_test))))
