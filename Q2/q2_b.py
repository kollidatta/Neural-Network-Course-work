# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 17:07:34 2019

@author: prabhakar reddy
"""


import pandas as pd
import scipy.io as sio

df = pd.read_csv('financedata1.csv')
df.columns = ['Attr_a', 'Attr_b', 'Attr_c', 'Attr_d', 'Attr_e',  'Attr_f', 'Attr_g', 'Attr_h', 
              'Attr_i', 'Attr_j', 'Attr_k', 'Attr_l', 'Attr_m', 'Attr_n', 'Attr_o', 'Attr_p', 'Output']
df = df.drop('Attr_a', axis = 1)
#df = df.drop('Output',axis =1)
# Categorical boolean mask
categorical_feature_mask = df.dtypes==object
# filter categorical columns using mask and turn it into a list
categorical_cols = df.columns[categorical_feature_mask].tolist()

# import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
# instantiate OneHotEncoder
ohe = OneHotEncoder(categorical_features = categorical_feature_mask, sparse=False ) 
# categorical_features = boolean mask for categorical columns
# sparse = False output an array not sparse matrix

# import labelencoder
from sklearn.preprocessing import LabelEncoder
# instantiate labelencoder object
le = LabelEncoder()

# apply le on categorical feature columns
df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))
df[categorical_cols].head(10)

# apply OneHotEncoder on categorical feature columns
X_ohe = ohe.fit_transform(df) # It returns an numpy array

sio.savemat('test2.mat', {'struct':df.to_dict("list")}, oned_as = 'column')
