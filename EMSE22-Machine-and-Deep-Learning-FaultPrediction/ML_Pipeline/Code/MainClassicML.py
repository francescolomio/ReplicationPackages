########################################################################################################################
import pandas as pd
import numpy as np
from sklearn import preprocessing

from imblearn.over_sampling import SMOTE
from collections import Counter


import Code.Settings as s
from Code.Classification_crossValidation import cross_validate, classifiers
from Code.FeatureSelection import feature_selection
from Code.PreProc import data_preprocess_ML

########################################################################################################################
def ClassicMLModel(subset, oversampling):

    # Read in data_raw and create the variable df to manipulate it
    if subset == 'squid':
        print('Using squid dataset...')
        PATH = './Review/Dataset/FullTable.csv'
    else:
        PATH = s.PATH

    df = pd.read_csv(PATH, low_memory=False)

    if subset == 'squid':
        print('Removing beam project')
        df = df[df.projectID != 'beam']


    # remove infinite values and NaN values
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    # variables assignement


    X, y, groups, colX = data_preprocess_ML(df, subset)

    model = classifiers()

    for clf, name in model:

        print("Evaluating %s classifier" % name)
        mean_fpr, mean_tpr, mean_auc, std_auc, tprs, importances_random, P, scores, column_names = cross_validate(clf,
                                                                        X, y, colX, groups, name, oversampling, subset)
    return mean_fpr, mean_tpr, mean_auc, std_auc, tprs, importances_random, P, scores, column_names
