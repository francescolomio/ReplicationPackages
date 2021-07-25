import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def make_sure_folder_exists(path):
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        os.makedirs(folder)

def create_dataframes_squid(experiment, oversampling):
    #Creates dataframes with results per fold. This is used for the plots
    models_results_dataframe_full = []
    if experiment == 'ALL':
        subset = 'COMBINED'
    else:
        subset = experiment
    for i in models_plot:
        models_results_dataframe_full.append([i] * 32)
    project_dataframe = pd.DataFrame({'Fold': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                                               '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32'] * len(models),
                               'Label': [subset] * len(models) * 32,
                               'Oversampling': [oversampling] * len(models) * 32,
                               'Model': [j for i in models_results_dataframe_full for j in i],
                               'AUC': [j for i in auc for j in i],
                               'F1': [x / 100 for x in [j for i in F1 for j in i]],
                               'Precision': [x / 100 for x in [j for i in precision for j in i]],
                               'Recall': [x / 100 for x in [j for i in recall for j in i]],
                               'MCC': [x / 100 for x in [j for i in MCC for j in i]],
                               'FNR': [x / 100 for x in [j for i in fnr for j in i]],
                               'TNR': [x / 100 for x in [j for i in tnr for j in i]],
                               'FPR': [x / 100 for x in [j for i in fpr for j in i]],
                               'Anomalies': [j for i in Test_anomalies_absolute for j in i],
                               'TP': [j for i in tp for j in i]})

    return project_dataframe

def create_dataframes(experiment, oversampling):
    #Creates dataframes with results per fold. This is used for the plots
    models_results_dataframe_full = []
    if experiment == 'ALL':
        subset = 'COMBINED'
    else:
        subset = experiment
    for i in models_plot:
        models_results_dataframe_full.append([i] * 29)
    project_dataframe = pd.DataFrame({'Fold': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                                               '20', '21', '22', '23', '24', '25', '26', '27', '28', '29'] * len(models),
                               'Label': [subset] * len(models) * 29,
                               'Oversampling': [oversampling] * len(models) * 29,
                               'Model': [j for i in models_results_dataframe_full for j in i],
                               'AUC': [j for i in auc for j in i],
                               'F1': [x / 100 for x in [j for i in F1 for j in i]],
                               'Precision': [x / 100 for x in [j for i in precision for j in i]],
                               'Recall': [x / 100 for x in [j for i in recall for j in i]],
                               'MCC': [x / 100 for x in [j for i in MCC for j in i]],
                               'FNR': [x / 100 for x in [j for i in fnr for j in i]],
                               'TNR': [x / 100 for x in [j for i in tnr for j in i]],
                               'FPR': [x / 100 for x in [j for i in fpr for j in i]],
                               'Anomalies': [j for i in Test_anomalies_absolute for j in i],
                               'TP': [j for i in tp for j in i]})

    return project_dataframe

#### Load all folds information

path_folds = './Risultati_review/Review/Folds_Information/'
path_results = './Risultati_review/Review/Metrics_folds/'
models = ['RandomForestClassifier',
          'GradientBoostingClassifier',
          'XGBoost',
          'fcn',
          'resnet']

EXPERIMENTS = ['squid', 'M', 'PRODUCT', 'PROCESS', 'PRODUCT-PROCESS', 'M-PRODUCT', 'M-PROCESS', 'M-PRODUCT-PROCESS']
oversampling = 'Oversampling_False'
accuracies = ['AUC', 'F1', 'precision', 'TP', 'MCC', 'FN', 'FP', 'TN']
# for oversampling in OVERSAMPLING:
print(oversampling)
for experiment in EXPERIMENTS:
    print(experiment)
    folder = '/LOGO/'+oversampling+'/'+experiment+'/'

    Test_anomalies_absolute = []
    Test_anomalies = []
    Test_Groups = []

    auc = []
    F1 = []
    precision = []
    recall = []
    MCC = []
    fnr = []
    fpr = []
    tnr = []

    tp = []
    fn = []
    fp = []
    tn = []


    for i in models:
        with open(path_folds + i + folder + 'Test_anomalies_absolute.data_raw', 'rb') as f:
            Test_anomalies_absolute.append(pickle.load(f))
        with open(path_folds + i + folder + 'Test_anomalies.data_raw', 'rb') as f:
            Test_anomalies.append(pickle.load(f))
        with open(path_folds + i + folder + 'Test_Groups.data_raw', 'rb') as f:
            Test_Groups.append(pickle.load(f))

        with open(path_results+i+folder + 'AUC.data_raw', 'rb') as f:
            auc.append(pickle.load(f))
        with open(path_results+i+folder + 'F1.data_raw', 'rb') as f:
            F1.append(pickle.load(f))
        with open(path_results+i+folder + 'precision.data_raw', 'rb') as f:
            precision.append(pickle.load(f))
        with open(path_results+i+folder + 'TPR.data_raw', 'rb') as f:
            recall.append(pickle.load(f))
        with open(path_results+i+folder + 'MCC.data_raw', 'rb') as f:
            MCC.append(pickle.load(f))
        with open(path_results+i+folder + 'FNR.data_raw', 'rb') as f:
            fnr.append(pickle.load(f))
        with open(path_results+i+folder + 'FPR.data_raw', 'rb') as f:
            fpr.append(pickle.load(f))
        with open(path_results+i+folder + 'TNR.data_raw', 'rb') as f:
            tnr.append(pickle.load(f))

        with open(path_results + i + folder + 'TP.data_raw', 'rb') as f:
            tp.append(pickle.load(f))
        with open(path_results + i + folder + 'FN.data_raw', 'rb') as f:
            fn.append(pickle.load(f))
        with open(path_results + i + folder + 'FP.data_raw', 'rb') as f:
            fp.append(pickle.load(f))
        with open(path_results + i + folder + 'TN.data_raw', 'rb') as f:
            tn.append(pickle.load(f))


    all_metrics = [(auc, 'AUC'),
                   (F1, 'F1'),
                   (precision, 'Precision'),
                   (recall, 'Recall'),
                   (MCC, 'MCC'),
                   (fnr, 'FNR'),
                   (fpr, 'FPR'),
                   (tnr, 'TNR')]


    models_plot = ['RandomForest',
                   'GradientBoost',
                   'XGBoost',
                   'FCN',
                   'ResNet']

    if experiment == 'squid':
        project_squid_oversampling = create_dataframes_squid(experiment, oversampling)
    if experiment == 'PROCESS':
        project_process_oversampling = create_dataframes(experiment, oversampling)
    if experiment == 'PRODUCT':
        project_product_oversampling = create_dataframes(experiment, oversampling)
    if experiment == 'M':
        project_M_oversampling = create_dataframes(experiment, oversampling)
    if experiment == 'PRODUCT-PROCESS':
        project_product_process_oversampling = create_dataframes(experiment, oversampling)
    if experiment == 'M-PRODUCT':
        project_M_product_oversampling = create_dataframes(experiment, oversampling)
    if experiment == 'M-PROCESS':
        project_M_process_oversampling = create_dataframes(experiment, oversampling)
    if experiment == 'M-PRODUCT-PROCESS':
        project_all_result_dataframe_oversampling = create_dataframes(experiment, oversampling)


# result_dataframes_list_oversampling = [project_squid_oversampling, project_process_oversampling, project_product_oversampling, project_M_oversampling,
#                                        project_product_process_oversampling, project_M_product_oversampling, project_M_process_oversampling, project_all_result_dataframe_oversampling]
result_dataframes_list_oversampling = [project_process_oversampling, project_product_oversampling, project_M_oversampling,
                                       project_product_process_oversampling, project_M_product_oversampling, project_M_process_oversampling, project_all_result_dataframe_oversampling]



result_dataframe_oversampling_true = pd.concat(result_dataframes_list_oversampling)
target_file = './Risultati_review/Review/Table/' + oversampling + '/'
make_sure_folder_exists(target_file)
result_dataframe_oversampling_true.to_csv(target_file + 'result_dataframe.csv', index=False)
project_squid_oversampling.to_csv(target_file + 'result_dataframe_squid.csv', index=False)


#####print table for latex
result_dataframe_oversampling_true.drop(['Oversampling', 'Anomalies', 'TP'], axis=1, inplace=True)
table = result_dataframe_oversampling_true.groupby(['Label', 'Model']).mean()
table = table.T
target_file = './Risultati_review/Review/Table/' + oversampling + '/'
make_sure_folder_exists(target_file)
table.to_csv(target_file + 'result_metrics_table.csv')

project_squid_oversampling.drop(['Oversampling', 'Anomalies', 'TP'], axis=1, inplace=True)
table = project_squid_oversampling.groupby(['Label', 'Model']).mean()
table = table.T
target_file = './Risultati_review/Review/Table/' + oversampling + '/'
make_sure_folder_exists(target_file)
table.to_csv(target_file + 'result_squid_table.csv')

pd.set_option('display.max_columns', None)

import seaborn as sns

#print squid plot

for metric, name in all_metrics:
    fig_dims = (8, 6)
    fig, ax = plt.subplots(figsize=fig_dims, dpi=600)
    sns.set_style('white')
    y_range = np.arange(0,1.1,.1)
    if name == 'MCC':
        y_range = np.arange(round(min(project_squid_oversampling.MCC), 1),1.1,.1)
    boxprops = dict(linestyle='-', linewidth=.7)
    medianprops = dict(linestyle='-', linewidth=.7)
    meanlineprops = dict(linestyle=':', linewidth=.7, color='black')
    flierprops = dict(marker='.', markersize=1)
    whiskerprops = dict(linestyle='-', linewidth=.7)
    capprops = dict(linestyle='-', linewidth=.7)
    sns.boxplot(y=name, x='Model',
                     data=project_squid_oversampling,
                     hue='Model',
                meanline=True,
                showmeans=True,
                dodge=False,
                width=0.5, palette='bright', boxprops=boxprops, medianprops=medianprops, flierprops=flierprops, whiskerprops=whiskerprops, capprops=capprops, meanprops=meanlineprops)
    plt.plot([], [], '-', linewidth=1, color='Black', label='median')
    plt.plot([], [], ':', linewidth=1, color='black', label='mean')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
    # y_range = np.arange(0, 1.1, .1)
    ax.set_yticks(y_range)
    plt.xticks(fontsize="large")
    ax.legend(bbox_to_anchor=(0, 1.1, 1, 0.005), loc=1, ncol=5, mode="expand", borderaxespad=0., fontsize="medium")

    # plt.show()
    target_file = './Risultati_review/Review/Plots/Squid/' + oversampling + '/'
    make_sure_folder_exists(target_file)
    fig.savefig(target_file+name+'_boxpot_squid.pdf', bbox_inches = 'tight')
    plt.close()


#print metrics comparison plots

for metric, name in all_metrics:
    fig_dims = (12, 6)
    fig, ax = plt.subplots(figsize=fig_dims, dpi=600)
    sns.set_style('white')
    y_range = np.arange(0, 1.1, .1)
    if name == 'MCC':
        y_range = np.arange(round(min(result_dataframe_oversampling_true.MCC), 1),1.1,.1)
        # print('True')
    boxprops = dict(linestyle='-', linewidth=.7)
    medianprops = dict(linestyle='-', linewidth=.7)
    meanlineprops = dict(linestyle=':', linewidth=.7, color='black')
    flierprops = dict(marker='.', markersize=1)
    whiskerprops = dict(linestyle='-', linewidth=.7)
    capprops = dict(linestyle='-', linewidth=.7)
    sns.boxplot(y=name, x='Label',
                     data=result_dataframe_oversampling_true,
                     hue='Model',
                meanline=True,
                showmeans=True,
                width=0.5, palette='bright', boxprops=boxprops, medianprops=medianprops, flierprops=flierprops, whiskerprops=whiskerprops, capprops=capprops, meanprops=meanlineprops)
    plt.plot([], [], '-', linewidth=1, color='Black', label='median')
    plt.plot([], [], ':', linewidth=1, color='black', label='mean')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels(['Kamei [49]','Rahman [99]','SQ','Kamei [49] + Rahman [99]','SQ + Rahman [99]','SQ + Kamei [49]','SQ + Kamei [49] + Rahman [99]'], rotation=30)
    ax.set_yticks(y_range)
    ax.legend(bbox_to_anchor=(0, 1.005, 1, 0.005), loc=3, ncol=5, mode="expand", borderaxespad=0., fontsize="medium")

    target_file = './Risultati_review/Review/Plots/Comparison/' + oversampling + '/'
    make_sure_folder_exists(target_file)
    fig.savefig(target_file+name+'_boxpot.pdf', bbox_inches = 'tight')
    plt.close()
