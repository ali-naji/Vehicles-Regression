# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 07:37:18 2020

@author: ali95
"""
import vehicles_model
import pathlib


PACKAGE_ROOT = pathlib.Path(vehicles_model.__file__).resolve().parent
DATASET_DIR = PACKAGE_ROOT / 'datasets'
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
PIPELINE_FILENAME = "neuralnet_pipeline_v"

TARGET = 'price'

VARS_TO_DROP = ['id', 'url', 'region_url', 'image_url', 'vin', 'county']

CATEGORICAL_TO_IMPUTE = ['manufacturer', 'model', 'condition', 'cylinders',
                         'fuel', 'title_status', 'transmission', 'drive',
                         'size', 'type', 'paint_color', 'description']

CATEGORICALS = ['manufacturer', 'model', 'condition', 'cylinders',
                'fuel', 'title_status', 'transmission', 'drive',
                'size', 'type', 'paint_color']

NUMERICAL_TO_IMPUTE = ['year', 'price', 'odometer', 'lat', 'long']

OHE_FEATS = ['condition', 'cylinders', 'fuel', 'title_status', 'transmission',
             'drive', 'size']

ORD_FEATS = ['region', 'manufacturer', 'model', 'type', 'paint_color', 'state']

VARS_TO_DROP = ['id', 'url', 'region_url', 'image_url', 'vin', 'county']

NUM_RANGES = {'price': (500.0, 1000000.0), 'year': (1890.0, 2021.0), 'odometer': (0.0, 5000000.0),
              'long': (-125.0, -66.0), 'lat': (23.0, 50.0)}
