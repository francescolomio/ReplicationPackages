from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneGroupOut
import pickle
import datetime
import numpy as np
import pandas as pd
from tensorflow import keras

import matplotlib.pyplot as plt

import os

from Code.PreProc import make_sure_folder_exists
import Code.Settings as s

def trainCycle(X, targets, model, class_weight, groups, oversampling, subset, column_names):

    name = s.MODEL
    #####Import Dataframe for column names
    df = pd.read_csv(s.PATH, low_memory=False)
    # column_names = [c for c in df.columns if s.FEATURE in c]
    print(column_names)

    # Define list for storing results
    if s.VALIDATION == 'StratifiedKFold':
        cv = StratifiedKFold(n_splits=s.n_splits)

    if s.VALIDATION == 'LOGO':
        cv = LeaveOneGroupOut()
        splits_indices = cv.split(X, targets, groups=groups)

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    # Aggregate the importances over folds here:
    N, _, P = X.shape
    importances_random = np.zeros(P)

    TP = []
    FP = []
    TN = []
    FN = []
    tnList = []
    fpList = []
    fnList = []
    tpList = []
    precisionList = []
    f1List = []
    mccList = []

    i = 1
    count = 0
    # for train, test in cv.split(X, y):
    train_splits = []
    test_splits = []
    train_groups = []
    test_groups = []
    train_anomaly_percentage = []
    test_anomaly_percentage = []
    train_anomaly_absolute = []
    test_anomaly_absolute = []
    counterfold = 1

    # Loop over crossvalidation folds:
    scores = []  # Collect accuracies here

    # fold = 0
    #######################################################################################################################
    """Main Computation"""

    for train, test in splits_indices:
        print('Fold %s of %s' %(counterfold, len(np.unique(groups))))
        counterfold += 1

        ####save logs for tensorboard visualizatino
        log_dir = "logs/%s_%s/" % (s.MODEL, s.FEATURE) + datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S") + 'fold_%s' % (counterfold)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=s.trainedmodels_folder + 'fold_%s' % (counterfold),
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            save_freq='epoch')

        # print('Starting fold %s' % (counterfold))

        train_splits.append(train)
        test_splits.append(test)
        train_groups.append(groups[train])
        test_groups.append(groups[test])
        count += 1

        X_train = X[train, :]
        y_train = targets[train]
        X_test = X[test, :]
        y_test = targets[test]

        a, b = np.unique(y_train, return_counts=True)[1]
        train_anomaly_percentage.append(b / (a + b))
        train_anomaly_absolute.append(b)
        c, d = np.unique(y_test, return_counts=True)[1]
        test_anomaly_percentage.append(d / (c + d))
        test_anomaly_absolute.append(d)



        hist = model.fit(X_train, y_train,
                         batch_size=s.MINI_BATCH_SIZE,
                         epochs=s.nb_epochs, verbose=2,
                         validation_data=(X_test, y_test),
                         class_weight=class_weight,
                         callbacks=[tensorboard_callback, model_checkpoint_callback])


        y_prob = model.predict(X_test)
        # y_prob = model.predict_proba(X_test)

        # convert the predicted from binary to integer
        y_pred = y_prob.round()

        # Compute ROC curve and area under the curve
        fpr, tpr, thresholds = roc_curve(y_test, y_prob[:])

        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        accuracy = (accuracy_score(y_test, y_pred))
        precision = (precision_score(y_test, y_pred))
        recall = (recall_score(y_test, y_pred))
        f1 = (f1_score(y_test, y_pred))
        mcc = (matthews_corrcoef(y_test, y_pred))
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        TN.append(tn)
        TP.append(tp)
        FN.append(fn)
        FP.append(fp)

        tnList.append(tn / (tn + fp))
        tpList.append(tp / (fn + tp))
        fpList.append(fp / (tn + fp))
        fnList.append(fn / (fn + tp))

        precisionList.append(precision)
        # recallList.append(recall)
        f1List.append(f1)
        mccList.append(mcc)

        print('Fold %s finished with following resutls:' % (counterfold))
        print(" accuracy: %s\n precision: %s\n recall: %s" % (accuracy, precision, recall))
        print(" auc: %s" % (roc_auc))

        # save model
        print("Saving Model...")
        model.save(s.trainedmodels_folder + '%s_%s.h5' % (s.MODEL, counterfold))

        print('Saving trainin history')
        with open(s.history_folder + 'trainHistoryDict_%s' % (counterfold), 'wb') as f:
            pickle.dump(hist.history, f)

            target_file = './Review/Folds_Information/%s/%s/Oversampling_%s/%s/' % (
            name, s.VALIDATION, oversampling, subset)

            make_sure_folder_exists(target_file)
            with open(target_file + 'Train_split.data_raw', 'wb') as fp:  # Pickling
                pickle.dump(train_splits, fp)
            with open(target_file + 'Test_split.data_raw', 'wb') as fp:  # Pickling
                pickle.dump(test_splits, fp)
            with open(target_file + 'Train_Groups.data_raw', 'wb') as fp:  # Pickling
                pickle.dump(train_groups, fp)
            with open(target_file + 'Test_Groups.data_raw', 'wb') as fp:  # Pickling
                pickle.dump(test_groups, fp)
            with open(target_file + 'Train_anomalies.data_raw', 'wb') as fp:  # Pickling
                pickle.dump(train_anomaly_percentage, fp)
            with open(target_file + 'Test_anomalies.data_raw', 'wb') as fp:  # Pickling
                pickle.dump(test_anomaly_percentage, fp)
            with open(target_file + 'Train_anomalies_absolute.data_raw', 'wb') as fp:  # Pickling
                pickle.dump(train_anomaly_absolute, fp)
            with open(target_file + 'Test_anomalies_absolute.data_raw', 'wb') as fp:  # Pickling
                pickle.dump(test_anomaly_absolute, fp)
            

    # Average the metrics over folds

    print("confusion matrix " + str(name))
    tnList = 100 * np.array(tnList)
    tpList = 100 * np.array(tpList)
    fnList = 100 * np.array(fnList)
    fpList = 100 * np.array(fpList)
    precisionList = 100 * np.array(precisionList)
    f1List = 100 * np.array(f1List)
    mccList = 100 * np.array(mccList)

    target_file = './Review/Metrics_folds/%s/%s/Oversampling_%s/%s/' % (name, s.VALIDATION, oversampling, subset)

    make_sure_folder_exists(target_file)
    with open(target_file + 'TN.data_raw', 'wb') as fp:  # Pickling
        pickle.dump(TN, fp)
    with open(target_file + 'TP.data_raw', 'wb') as fp:  # Pickling
        pickle.dump(TP, fp)
    with open(target_file + 'FN.data_raw', 'wb') as fp:  # Pickling
        pickle.dump(FN, fp)
    with open(target_file + 'FP.data_raw', 'wb') as fp:  # Pickling
        pickle.dump(FP, fp)
    with open(target_file + 'TNR.data_raw', 'wb') as fp:  # Pickling
        pickle.dump(tnList, fp)
    with open(target_file + 'TPR.data_raw', 'wb') as fp:  # Pickling
        pickle.dump(tpList, fp)
    with open(target_file + 'FNR.data_raw', 'wb') as fp:  # Pickling
        pickle.dump(fnList, fp)
    with open(target_file + 'FPR.data_raw', 'wb') as fp:  # Pickling
        pickle.dump(fpList, fp)
    with open(target_file + 'precision.data_raw', 'wb') as fp:  # Pickling
        pickle.dump(precisionList, fp)
    with open(target_file + 'F1.data_raw', 'wb') as fp:  # Pickling
        pickle.dump(f1List, fp)
    with open(target_file + 'MCC.data_raw', 'wb') as fp:  # Pickling
        pickle.dump(mccList, fp)

    # Average the TPR over folds

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    # mean_auc = auc(mean_fpr, mean_tpr)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    auc_meanpercent = 100 * mean_auc
    auc_stdpercent = 100 * std_auc
    with open(target_file + 'AUC.data_raw', 'wb') as fp:  # Pickling
        pickle.dump(aucs, fp)

    #######################################################################################################################
    """Show metrics"""

    print("TN: %.02f %% ± %.02f %% - FN: %.02f %% ± %.02f %%" % (np.mean(tnList),
                                                                 np.std(tnList),
                                                                 np.mean(fnList),
                                                                 np.std(fnList)))
    print("FP: %.02f %% ± %.02f %% - TP: %.02f %% ± %.02f %%" % (np.mean(fpList),
                                                                 np.std(fpList),
                                                                 np.mean(tpList),
                                                                 np.std(tpList)))

    print(
        "Precision: %.02f %% ± %.02f %% - F1: %.02f %% ± %.02f %% - MCC: %.02f %% ± %.02f %%" % (np.mean(precisionList),
                                                                                                 np.std(precisionList),
                                                                                                 np.mean(f1List),
                                                                                                 np.std(f1List),
                                                                                                 np.mean(mccList),
                                                                                                 np.std(mccList)))

    print("AUC: %.02f %% ± %.02f %%" % (auc_meanpercent, auc_stdpercent))

    #######################################################################################################################
    """Save metrics"""

    metrics_tosave = pd.DataFrame(
        {'AUC_Mean': auc_meanpercent, 'AUC_std': auc_stdpercent,
         'Confusion_Matrix_TNR_mean': np.mean(tnList), 'Confusion_Matrix_TNR_std': np.std(tnList),
         'Confusion_Matrix_FNR_mean': np.mean(fnList), 'Confusion_Matrix_FNR_std': np.std(fnList),
         'Confusion_Matrix_FPR_mean': np.mean(fpList), 'Confusion_Matrix_FPR_std': np.std(fpList),
         'Confusion_Matrix_TPR_mean': np.mean(tpList), 'Confusion_Matrix_TPR_std': np.std(tpList),
         'Confusion_Matrix_Precision_mean': np.mean(precisionList),
         'Confusion_Matrix_Precision_std': np.std(precisionList),
         'Confusion_Matrix_F1_mean': np.mean(f1List), 'Confusion_Matrix_F1_std': np.std(f1List),
         'Confusion_Matrix_MCC_mean': np.mean(mccList), 'Confusion_Matrix_MCC_std': np.std(mccList)}, index=[0])

    #target_file = "Metrics/%s" % name
    target_file = './Review/Metrics/%s/%s/Oversampling_%s/%s/' %(name, s.VALIDATION, s.OVERSAMPLING, s.FEATURE)
    if s.VALIDATION == 'TimeValidation':
        target_file = './Review/Metrics/%s/%s/%s/Oversampling_%s/%s/' % (name, s.VALIDATION, project, s.OVERSAMPLING, s.FEATURE)
    make_sure_folder_exists(target_file)
    metrics_tosave.to_csv(target_file + ".csv",
                          columns=['AUC_Mean', 'AUC_std','Confusion_Matrix_TNR_mean', 'Confusion_Matrix_TNR_std',
                                                         'Confusion_Matrix_FNR_mean', 'Confusion_Matrix_FNR_std',
                                                         'Confusion_Matrix_FPR_mean', 'Confusion_Matrix_FPR_std',
                                                         'Confusion_Matrix_TPR_mean', 'Confusion_Matrix_TPR_std',
                                                         'Confusion_Matrix_Precision_mean',
                                                         'Confusion_Matrix_Precision_std',
                                                         'Confusion_Matrix_F1_mean', 'Confusion_Matrix_F1_std',
                                                         'Confusion_Matrix_MCC_mean', 'Confusion_Matrix_MCC_std'],
                          sep=';', index=False)


    plt.figure(1)
    plt.clf()

    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim(s.xlim)
    plt.ylim(s.ylim)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    target_file = './Review/AUCs/%s/%s/Oversampling_%s/%s/' %(name, s.VALIDATION, oversampling, subset)

    make_sure_folder_exists(target_file)
    plt.savefig(target_file + ".pdf", bbox_inches="tight")

    return mean_fpr, mean_tpr, mean_auc, std_auc, tprs, importances_random, P, scores, column_names
