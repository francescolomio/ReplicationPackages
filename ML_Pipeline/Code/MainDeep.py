#######################################################################################################################
# common libraries imports
import os
import sys

import numpy as np
from tensorflow import keras
import pandas as pd

# ad-hoc libraries imports
import Code.Settings as s
# from Code.PreProc import PreProc, make_sure_folder_exists
from Code.models_DL import resnet, fcn

from Code.PreProc import data_preprocess_DL, make_sure_folder_exists

#######################################################################################################################
def DeepLearningModel():

    #create folders for storing trained models
    make_sure_folder_exists(s.trainedmodels_folder)
    make_sure_folder_exists(s.history_folder)

    #######################################################################################################################
    """Preparing the data for the computation"""
    
    if s.FEATURE_DL == 'squid':
        PATH = './Review/Dataset/FullTable.csv'
    else:
        PATH = s.PATH

    df = pd.read_csv(PATH, low_memory=False)

    if s.FEATURE_DL == 'squid':
        print('Removing beam project')
        df = df[df.projectID != 'beam']

    # remove infinite values and NaN values
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    X, Y, groups, colX = data_preprocess_DL(df, sys.argv[2])

    #Calculate positive class occurrency
    targets = np.expand_dims(Y, axis=1).astype(int)
    counts = np.bincount(targets[:, 0])
    print(
        "Number of positive samples in training data: {} ({:.2f}% of total)".format(
            counts[1], 100 * float(counts[1]) / len(targets)
        )
    )

    # calculate class weights
    weight_for_0 = 1.0 - (counts[0] / (counts[0]+counts[1]))
    weight_for_1 = 1.0 - (counts[1] / (counts[0]+counts[1]))

    class_weight = {0: weight_for_0,
                    1: weight_for_1}

    # define metrics for training
    metrics = [
        keras.metrics.FalseNegatives(name="fn"),
        keras.metrics.FalsePositives(name="fp"),
        keras.metrics.TrueNegatives(name="tn"),
        keras.metrics.TruePositives(name="tp"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.BinaryAccuracy(name="accuracy")
    ]
    if s.MODEL=='resnet':
        print('Using Resnet')
        model = resnet(X, s.NB_CLASSES)
    elif s.MODEL=='fcn':
        print('Using FCN')
        model = fcn(X, s.NB_CLASSES)
    model.compile(optimizer=keras.optimizers.Adadelta(),
                  loss="binary_crossentropy",
                  metrics=metrics)

    model.summary()
    return X, targets, model, class_weight, groups, colX
    #######################################################################################################################
