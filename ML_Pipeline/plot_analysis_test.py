import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def make_sure_folder_exists(path):
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        os.makedirs(folder)

def create_dataframes(a, experiment, oversampling):
    #Creates dataframes with results per fold. This is used for the plots
    models_results_dataframe_full = []
    subset = experiment
    for i in models_plot:
        models_results_dataframe_full.append([i] * 29)
    project_dataframe = pd.DataFrame({'Fold': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                                               '20', '21', '22', '23', '24', '25', '26', '27', '28', '29'] * len(models),
                                      'Features': [a] * len(models) * 29,
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
analysis = ['metrics', 'squidrules', 'squidtype']

for a in analysis:
    path_folds = './ML_Pipeline/%s/Folds_Information/' %a
    path_results = './ML_Pipeline/%s/Metrics_folds/' %a
    models = ['RandomForestClassifier',
              'GradientBoostingClassifier',
              'XGBoost',
              'fcn',
              'resnet']

    if a=='metrics':
        EXPERIMENTS = ['M', 'M-PROCESS', 'M-PRODUCT', 'M-PRODUCT-PROCESS', 'PROCESS', 'PRODUCT', 'PRODUCT-PROCESS']
    else:
        EXPERIMENTS = ['squid', 'squid-M', 'squid-M-PROCESS', 'squid-M-PRODUCT', 'squid-M-PRODUCT-PROCESS', 'squid-PROCESS', 'squid-PRODUCT', 'squid-PRODUCT-PROCESS']
    oversampling = 'Oversampling_True'
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

        if a=='squidtype':
            if experiment == 'squid-PRODUCT':
                project_SPD_oversampling_TYPE = create_dataframes(a, experiment, oversampling)
            if experiment == 'squid-PROCESS':
                project_SPR_oversampling_TYPE = create_dataframes(a, experiment, oversampling)
            if experiment == 'squid-M':
                project_SM_oversampling_TYPE = create_dataframes(a, experiment, oversampling)
            if experiment == 'squid-PRODUCT-PROCESS':
                project_SPP_process_oversampling_TYPE = create_dataframes(a, experiment, oversampling)
            if experiment == 'squid-M-PROCESS':
                project_SMPR_oversampling_TYPE = create_dataframes(a, experiment, oversampling)
            if experiment == 'squid-M-PRODUCT':
                project_SMPD_oversampling_TYPE = create_dataframes(a, experiment, oversampling)
            if experiment == 'squid-M-PRODUCT-PROCESS':
                project_SMPP_dataframe_oversampling_TYPE = create_dataframes(a, experiment, oversampling)
            if experiment == 'squid':
                project_S_dataframe_oversampling_TYPE = create_dataframes(a, experiment, oversampling)
        if a=='metrics':
            if experiment == 'PRODUCT':
                project_PD_oversampling = create_dataframes(a, experiment, oversampling)
            if experiment == 'PROCESS':
                project_PR_oversampling = create_dataframes(a, experiment, oversampling)
            if experiment == 'M':
                project_M_oversampling = create_dataframes(a, experiment, oversampling)
            if experiment == 'PRODUCT-PROCESS':
                project_PP_process_oversampling = create_dataframes(a, experiment, oversampling)
            if experiment == 'M-PROCESS':
                project_MPR_oversampling = create_dataframes(a, experiment, oversampling)
            if experiment == 'M-PRODUCT':
                project_MPD_oversampling = create_dataframes(a, experiment, oversampling)
            if experiment == 'M-PRODUCT-PROCESS':
                project_MPP_dataframe_oversampling = create_dataframes(a, experiment, oversampling)
        if a == 'squidrules':
            if experiment == 'squid-PRODUCT':
                project_SPD_oversampling = create_dataframes(a, experiment, oversampling)
            if experiment == 'squid-PROCESS':
                project_SPR_oversampling = create_dataframes(a, experiment, oversampling)
            if experiment == 'squid-M':
                project_SM_oversampling = create_dataframes(a, experiment, oversampling)
            if experiment == 'squid-PRODUCT-PROCESS':
                project_SPP_process_oversampling = create_dataframes(a, experiment, oversampling)
            if experiment == 'squid-M-PROCESS':
                project_SMPR_oversampling = create_dataframes(a, experiment, oversampling)
            if experiment == 'squid-M-PRODUCT':
                project_SMPD_oversampling = create_dataframes(a, experiment, oversampling)
            if experiment == 'squid-M-PRODUCT-PROCESS':
                project_SMPP_dataframe_oversampling = create_dataframes(a, experiment, oversampling)
            if experiment == 'squid':
                project_S_dataframe_oversampling = create_dataframes(a, experiment, oversampling)

result_dataframes_list = [project_PD_oversampling, project_PR_oversampling, project_M_oversampling, project_PP_process_oversampling, project_MPR_oversampling,
                          project_MPD_oversampling, project_MPP_dataframe_oversampling, project_SPD_oversampling, project_SPR_oversampling, project_SM_oversampling,
                          project_SPP_process_oversampling,project_SMPR_oversampling,project_SMPD_oversampling,project_SMPP_dataframe_oversampling,project_S_dataframe_oversampling,
                          project_SPD_oversampling_TYPE,project_SPR_oversampling_TYPE,project_SM_oversampling_TYPE,project_SPP_process_oversampling_TYPE,project_SMPR_oversampling_TYPE,
                          project_SMPD_oversampling_TYPE,project_SMPP_dataframe_oversampling_TYPE,project_S_dataframe_oversampling_TYPE]

result_dataframe = pd.concat(result_dataframes_list)
result_dataframe.to_csv('result_dataframe.csv', index=False)


pd.set_option('display.max_columns', None)

import seaborn as sns

#print full_dataset plot
# for metric, name in all_metrics:
#     fig_dims = (8, 6)
#     fig, ax = plt.subplots(figsize=fig_dims, dpi=600)
#     sns.set_style('white')
#     y_range = np.arange(0,1.1,.1)
#     if name == 'MCC':
#         y_range = np.arange(round(min(project_all_result_dataframe.MCC), 1),1.1,.1)
#     boxprops = dict(linestyle='-', linewidth=.7)
#     medianprops = dict(linestyle='-', linewidth=.7)
#     meanlineprops = dict(linestyle=':', linewidth=.7, color='black')
#     flierprops = dict(marker='.', markersize=1)
#     whiskerprops = dict(linestyle='-', linewidth=.7)
#     capprops = dict(linestyle='-', linewidth=.7)
#     sns.boxplot(y=name, x='Model',
#                      data=project_all_result_dataframe_oversampling,
#                      hue='Model',
#                 meanline=True,
#                 showmeans=True,
#                 dodge=False,
#                 width=0.5, palette='bright', boxprops=boxprops, medianprops=medianprops, flierprops=flierprops, whiskerprops=whiskerprops, capprops=capprops, meanprops=meanlineprops)
#     plt.plot([], [], '-', linewidth=1, color='Black', label='median')
#     plt.plot([], [], ':', linewidth=1, color='black', label='mean')
#     ax.set_xlabel('')
#     ax.set_ylabel('')
#     ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
#     # y_range = np.arange(0, 1.1, .1)
#     ax.set_yticks(y_range)
#     plt.xticks(fontsize="large")
#     ax.legend(bbox_to_anchor=(0, 1.005, 1, 0.005), loc=1, ncol=5, mode="expand", borderaxespad=0., fontsize="medium")
#
#     # plt.show()
#     target_file = 'Plots/Combined/'
#     make_sure_folder_exists(target_file)
#     fig.savefig(target_file+name+'_boxpot_fulldataset.pdf', bbox_inches = 'tight')
#     # plt.close()

metric_results = result_dataframe.loc[result_dataframe['Features']=='metrics']
rules_metric_results = result_dataframe.loc[result_dataframe['Features']=='squidrules']
ruletype_metric_results = result_dataframe.loc[result_dataframe['Features']=='squidtype']
rule_vs_type = result_dataframe.loc[((result_dataframe['Features'] == 'squidtype') | (result_dataframe['Features'] == 'squidrules')) & (result_dataframe['Label'] == 'squid')]

#Plot for metrics comparison
for metric, name in all_metrics:
    fig_dims = (12, 6)
    fig, ax = plt.subplots(figsize=fig_dims, dpi=600)
    sns.set_style('white')
    y_range = np.arange(0, 1.1, .1)
    if name == 'MCC':
        y_range = np.arange(round(min(metric_results.MCC), 1),1.1,.1)
        # print('True')
    boxprops = dict(linestyle='-', linewidth=.7)
    medianprops = dict(linestyle='-', linewidth=.7)
    meanlineprops = dict(linestyle=':', linewidth=.7, color='black')
    flierprops = dict(marker='.', markersize=1)
    whiskerprops = dict(linestyle='-', linewidth=.7)
    capprops = dict(linestyle='-', linewidth=.7)
    sns.boxplot(y=name, x='Label',
                     data=metric_results,
                     hue='Model',
                meanline=True,
                showmeans=True,
                width=0.5, palette='bright', boxprops=boxprops, medianprops=medianprops, flierprops=flierprops, whiskerprops=whiskerprops, capprops=capprops, meanprops=meanlineprops)
    plt.plot([], [], '-', linewidth=1, color='Black', label='median')
    plt.plot([], [], ':', linewidth=1, color='black', label='mean')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
    ax.set_yticks(y_range)
    ax.legend(bbox_to_anchor=(0, 1.005, 1, 0.005), loc=3, ncol=5, mode="expand", borderaxespad=0., fontsize="medium")

    target_file = 'Plots/Comparison/Metrics/'
    make_sure_folder_exists(target_file)
    fig.savefig(target_file+name+'_boxpot.pdf', bbox_inches = 'tight')

#Plot for SQrules comparison
for metric, name in all_metrics:
    fig_dims = (12, 6)
    fig, ax = plt.subplots(figsize=fig_dims, dpi=600)
    sns.set_style('white')
    y_range = np.arange(0, 1.1, .1)
    if name == 'MCC':
        y_range = np.arange(round(min(rules_metric_results.MCC), 1),1.1,.1)
        # print('True')
    boxprops = dict(linestyle='-', linewidth=.7)
    medianprops = dict(linestyle='-', linewidth=.7)
    meanlineprops = dict(linestyle=':', linewidth=.7, color='black')
    flierprops = dict(marker='.', markersize=1)
    whiskerprops = dict(linestyle='-', linewidth=.7)
    capprops = dict(linestyle='-', linewidth=.7)
    sns.boxplot(y=name, x='Label',
                     data=rules_metric_results,
                     hue='Model',
                meanline=True,
                showmeans=True,
                width=0.5, palette='bright', boxprops=boxprops, medianprops=medianprops, flierprops=flierprops, whiskerprops=whiskerprops, capprops=capprops, meanprops=meanlineprops)
    plt.plot([], [], '-', linewidth=1, color='Black', label='median')
    plt.plot([], [], ':', linewidth=1, color='black', label='mean')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
    ax.set_yticks(y_range)
    ax.legend(bbox_to_anchor=(0, 1.005, 1, 0.005), loc=3, ncol=5, mode="expand", borderaxespad=0., fontsize="medium")

    target_file = 'Plots/Comparison/SquidRules/'
    make_sure_folder_exists(target_file)
    fig.savefig(target_file+name+'_boxpot.pdf', bbox_inches = 'tight')

#Plot for SQrules types comparison
for metric, name in all_metrics:
    fig_dims = (12, 6)
    fig, ax = plt.subplots(figsize=fig_dims, dpi=600)
    sns.set_style('white')
    y_range = np.arange(0, 1.1, .1)
    if name == 'MCC':
        y_range = np.arange(round(min(ruletype_metric_results.MCC), 1), 1.1, .1)
        # print('True')
    boxprops = dict(linestyle='-', linewidth=.7)
    medianprops = dict(linestyle='-', linewidth=.7)
    meanlineprops = dict(linestyle=':', linewidth=.7, color='black')
    flierprops = dict(marker='.', markersize=1)
    whiskerprops = dict(linestyle='-', linewidth=.7)
    capprops = dict(linestyle='-', linewidth=.7)
    sns.boxplot(y=name, x='Label',
                data=ruletype_metric_results,
                hue='Model',
                meanline=True,
                showmeans=True,
                width=0.5, palette='bright', boxprops=boxprops, medianprops=medianprops, flierprops=flierprops, whiskerprops=whiskerprops, capprops=capprops, meanprops=meanlineprops)
    plt.plot([], [], '-', linewidth=1, color='Black', label='median')
    plt.plot([], [], ':', linewidth=1, color='black', label='mean')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.set_yticks(y_range)
    ax.legend(bbox_to_anchor=(0, 1.005, 1, 0.005), loc=3, ncol=5, mode="expand", borderaxespad=0., fontsize="medium")

    target_file = 'Plots/Comparison/SquidType/'
    make_sure_folder_exists(target_file)
    fig.savefig(target_file + name + '_boxpot.pdf', bbox_inches='tight')

#Plot for SQrules VS SQrules Type comparison
for metric, name in all_metrics:
    fig_dims = (12, 6)
    fig, ax = plt.subplots(figsize=fig_dims, dpi=600)
    sns.set_style('white')
    y_range = np.arange(0, 1.1, .1)
    if name == 'MCC':
        y_range = np.arange(round(min(rule_vs_type.MCC), 1), 1.1, .1)
        # print('True')
    boxprops = dict(linestyle='-', linewidth=.7)
    medianprops = dict(linestyle='-', linewidth=.7)
    meanlineprops = dict(linestyle=':', linewidth=.7, color='black')
    flierprops = dict(marker='.', markersize=1)
    whiskerprops = dict(linestyle='-', linewidth=.7)
    capprops = dict(linestyle='-', linewidth=.7)
    sns.boxplot(y=name, x='Model',
                data=rule_vs_type,
                hue='Features',
                meanline=True,
                showmeans=True,
                width=0.5, palette='bright', boxprops=boxprops, medianprops=medianprops, flierprops=flierprops, whiskerprops=whiskerprops, capprops=capprops, meanprops=meanlineprops)
    plt.plot([], [], '-', linewidth=1, color='Black', label='median')
    plt.plot([], [], ':', linewidth=1, color='black', label='mean')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.set_yticks(y_range)
    ax.legend(bbox_to_anchor=(0, 1.005, 1, 0.005), loc=3, ncol=5, mode="expand", borderaxespad=0., fontsize="medium")

    target_file = 'Plots/Comparison/RulesVSType/'
    make_sure_folder_exists(target_file)
    fig.savefig(target_file + name + '_boxpot.pdf', bbox_inches='tight')