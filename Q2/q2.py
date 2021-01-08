# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 18:07:47 2019

@author: prabhakar reddy
"""

import pandas as pd
import scipy.io as sio

df = pd.read_csv('financedata1.csv')
df.columns = ['Attr_a', 'Attr_b', 'Attr_c', 'Attr_d', 'Attr_e',  'Attr_f', 'Attr_g', 'Attr_h', 
              'Attr_i', 'Attr_j', 'Attr_k', 'Attr_l', 'Attr_m', 'Attr_n', 'Attr_o', 'Attr_p', 'Output']
df = df.drop('Attr_a', axis = 1)

#df1 = df['Attr_b']
# Categorical boolean mask
categorical_feature_mask = df.dtypes==object
# filter categorical columns using mask and turn it into a list
categorical_cols = df.columns[categorical_feature_mask].tolist()

# import labelencoder
from sklearn.preprocessing import LabelEncoder
# instantiate labelencoder object
le = LabelEncoder()

# apply le on categorical feature columns
df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))
df[categorical_cols].head(10)


from sklearn import preprocessing

x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
df.columns = ['Attr_a', 'Attr_b', 'Attr_c', 'Attr_d', 'Attr_e',  'Attr_f', 'Attr_g', 'Attr_h', 
              'Attr_i', 'Attr_j', 'Attr_k', 'Attr_l', 'Attr_m', 'Attr_n', 'Attr_o', 'Output']

sio.savemat('test1.mat', {'struct':df.to_dict("list")}, oned_as = 'column')
#
#df_mat = df.to_dict()
#
#sio.savemat('finance', df_mat, appendmat=True, format='5', 
#                 long_field_names=False, do_compression=False, oned_as='row')


