import numpy as np
import pandas as pd
import os

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

from imblearn.over_sampling import SMOTE
from collections import Counter

import pickle

import Code.Settings as s
from Code.FeatureSelection import feature_selection

def make_sure_folder_exists(path):
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        os.makedirs(folder)

def history_preprocess_data(X, y, window_len):
    a = X[np.arange(X.shape[0] - window_len + 1)[:, None] + np.arange(window_len)]
    b = y[window_len-1:]
    b = b.T

    return a, b

def check_files(subset):

    features_path = './Review/preprocess_data/%s/features' %subset

    if os.path.isfile(features_path):
        check = True
    else:
        check = False

    return check

def data_preprocess_ML(df, subset):

    target_file = './Review/preprocess_data/ML/%s/' %subset
    make_sure_folder_exists(target_file)

    target_file_features = './Review/preprocess_data/%s/' %subset
    make_sure_folder_exists(target_file_features)

    check = check_files(subset)

    if check == True:

        print('Loading data...')

        
        with open(target_file_features + 'features', 'rb') as f:
            colX = pickle.load(f)

    else:
        print('Calculating VIF')

        if subset == 'squid':
            colX = [c for c in feature_selection(df, [c for c in df.columns if "squid" in c], subset)]
        if subset == 'M':
            colX = [c for c in feature_selection(df, [c for c in df.columns if "M_" in c], subset)]
        if subset == 'PRODUCT':
            colX = [c for c in feature_selection(df, [c for c in df.columns if "Product_" in c], subset)]
        if subset == 'PROCESS':
            colX = [c for c in feature_selection(df, [c for c in df.columns if "Process_" in c], subset)]
        if subset == 'PRODUCT-PROCESS':
            colX = [c for c in feature_selection(df, [c for c in df.columns if "Product_" in c or "Process_" in c], subset)]
        if subset == 'M-PRODUCT':
            colX = [c for c in feature_selection(df, [c for c in df.columns if "M_" in c or "Product_" in c], subset)]
        if subset == 'M-PROCESS':
            colX = [c for c in feature_selection(df, [c for c in df.columns if "M_" in c or "Process_" in c], subset)]
        if subset == 'M-PRODUCT-PROCESS':
            colX = [c for c in feature_selection(df, [c for c in df.columns if "M_" in c or "Product_" in c or "Process_" in c], subset)]

        if subset == 'squid-M':
            colX = [c for c in feature_selection(df, [c for c in df.columns if "squid" in c or "M_" in c], subset)]
        if subset == 'squid-PRODUCT':
            colX = [c for c in feature_selection(df, [c for c in df.columns if "squid" in c or "Product_" in c], subset)]
        if subset == 'squid-PROCESS':
            colX = [c for c in feature_selection(df, [c for c in df.columns if "squid" in c or "Process_" in c], subset)]
        if subset == 'squid-PRODUCT-PROCESS':
            colX = [c for c in
                    feature_selection(df, [c for c in df.columns if "squid" in c or "Product_" in c or "Process_" in c], subset)]
        if subset == 'squid-M-PRODUCT':
            colX = [c for c in
                    feature_selection(df, [c for c in df.columns if "squid" in c or "M_" in c or "Product_" in c], subset)]
        if subset == 'squid-M-PROCESS':
            colX = [c for c in
                    feature_selection(df, [c for c in df.columns if "squid" in c or "M_" in c or "Process_" in c], subset)]
        if subset == 'squid-M-PRODUCT-PROCESS':
            colX = [c for c in feature_selection(df, [c for c in df.columns if "squid" in c or "M_" in c or "Product_" in c or "Process_" in c], subset)]

    groups = np.array(df['projectID'])

    projects = np.unique(groups)

    scaler = MinMaxScaler()

    projects.sort()
    j = 1
    input_list = []
    target_list = []
    group_list = []
    for i in projects:
        a = df[df.projectID.isin([i])]
        a = a.sort_index(axis=0, inplace=False)
        a_scaled = scaler.fit_transform(a[colX])
#        a_scaled = a[colX]
        y = a[s.TARGET]
        lb = preprocessing.LabelBinarizer()
        y = lb.fit_transform(y).ravel()

        # print(i)
        # print(Counter(y)[1])

        if s.OVERSAMPLING == 'True':
            if Counter(y)[1] > 1:
                sm = SMOTE(random_state=0, n_jobs=-1, k_neighbors=1)
                a_scaled, y = sm.fit_resample(a_scaled, y)


        group_list.append([i]*len(y))
        target_list.append(y)
        input_list.append(a_scaled)
        j += 1

    X = np.concatenate(input_list)
    y = np.concatenate(target_list)
    groups = np.concatenate(group_list)

    print(X.shape)
    print(y.shape)
    print(groups.shape)
    print('Using: %s' %colX)

       
    with open(target_file_features + 'features', 'wb') as f:
        pickle.dump(colX, f)

    return X, y, groups, colX

def data_preprocess_DL(df, subset):

    target_file = './Review/preprocess_data/DL/%s/' %subset
    make_sure_folder_exists(target_file)

    target_file_features = './Review/preprocess_data/%s/' %subset
    make_sure_folder_exists(target_file_features)

    check = check_files(subset)

    if check == True:

        print('Loading data...')

        with open(target_file_features + 'features', 'rb') as f:
            colX = pickle.load(f)

    else:
        print('Calculating VIF')

        if subset == 'squid':
            colX = [c for c in feature_selection(df, [c for c in df.columns if "squid" in c], subset)]
        if subset == 'M':
            colX = [c for c in feature_selection(df, [c for c in df.columns if "M_" in c], subset)]
        if subset == 'PRODUCT':
            colX = [c for c in feature_selection(df, [c for c in df.columns if "Product_" in c], subset)]
        if subset == 'PROCESS':
            colX = [c for c in feature_selection(df, [c for c in df.columns if "Process_" in c], subset)]
        if subset == 'PRODUCT-PROCESS':
            colX = [c for c in feature_selection(df, [c for c in df.columns if "Product_" in c or "Process_" in c], subset)]
        if subset == 'M-PRODUCT':
            colX = [c for c in feature_selection(df, [c for c in df.columns if "M_" in c or "Product_" in c], subset)]
        if subset == 'M-PROCESS':
            colX = [c for c in feature_selection(df, [c for c in df.columns if "M_" in c or "Process_" in c], subset)]
        if subset == 'M-PRODUCT-PROCESS':
            colX = [c for c in feature_selection(df, [c for c in df.columns if "M_" in c or "Product_" in c or "Process_" in c], subset)]

        if subset == 'squid-M':
            colX = [c for c in feature_selection(df, [c for c in df.columns if "squid" in c or "M_" in c], subset)]
        if subset == 'squid-PRODUCT':
            colX = [c for c in feature_selection(df, [c for c in df.columns if "squid" in c or "Product_" in c], subset)]
        if subset == 'squid-PROCESS':
            colX = [c for c in feature_selection(df, [c for c in df.columns if "squid" in c or "Process_" in c], subset)]
        if subset == 'squid-PRODUCT-PROCESS':
            colX = [c for c in
                    feature_selection(df, [c for c in df.columns if "squid" in c or "Product_" in c or "Process_" in c], subset)]
        if subset == 'squid-M-PRODUCT':
            colX = [c for c in
                    feature_selection(df, [c for c in df.columns if "squid" in c or "M_" in c or "Product_" in c], subset)]
        if subset == 'squid-M-PROCESS':
            colX = [c for c in
                    feature_selection(df, [c for c in df.columns if "squid" in c or "M_" in c or "Process_" in c], subset)]
        if subset == 'squid-M-PRODUCT-PROCESS':
            colX = [c for c in feature_selection(df, [c for c in df.columns if "squid" in c or "M_" in c or "Product_" in c or "Process_" in c], subset)]

    groups = np.array(df['projectID'])

    projects = np.unique(groups)

    scaler = MinMaxScaler()

    projects.sort()
    j = 1
    input_list = []
    target_list = []
    group_list = []
    for i in projects:
        a = df[df.projectID.isin([i])]
        a = a.sort_index(axis=0, inplace=False)
        a_scaled = scaler.fit_transform(a[colX])
#        a_features = a[colX]
        y = a[s.TARGET]
        lb = preprocessing.LabelBinarizer()
        y = lb.fit_transform(y).ravel()

        if s.OVERSAMPLING == 'True':
            if Counter(y)[1] > 1:
                sm = SMOTE(random_state=0, n_jobs=-1, k_neighbors=1)
                a_features, y = sm.fit_resample(a_features, y)

        x, y = history_preprocess_data(np.array(a_features), np.array(y), s.WINDOW_LEN)

        group_list.append([i]*len(y))
        target_list.append(y)
        input_list.append(x)
        j += 1

    X = np.concatenate(input_list)
    y = np.concatenate(target_list)
    groups = np.concatenate(group_list)

    print(X.shape)
    print(y.shape)
    print(groups.shape)
    print('Using: %s' %colX)


    with open(target_file_features + 'features', 'wb') as f:
        pickle.dump(colX, f)

    return X, y, groups, colX