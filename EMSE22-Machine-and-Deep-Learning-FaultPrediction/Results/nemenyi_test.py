import pandas as pd
import numpy as np

import scikit_posthocs as sp
import matplotlib.pyplot as plt
import pickle
import os

def make_sure_folder_exists(path):
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        os.makedirs(folder)

models = ['RandomForest',
          'GradientBoost',
          'XGBoost',
          'FCN',
          'ResNet']


EXPERIMENTS = ['M', 'PRODUCT', 'PROCESS', 'PRODUCT-PROCESS', 'M-PRODUCT', 'M-PROCESS', 'M-PRODUCT-PROCESS']
EXPERIMENTS_LABELS = ['SQ', 'Kamei [49]','Rahman [99]', 'Kamei [49] + Rahman [99]','SQ + Rahman [99]','SQ + Kamei [49]','SQ + Kamei [49] + Rahman [99]']
accuracies = ['AUC', 'F1', 'Precision', 'MCC', 'Recall', 'FPR', 'FNR', 'TNR']

data = pd.read_csv("./Review/Table/Oversampling_True/result_dataframe.csv")
data.drop(['Oversampling', 'Anomalies', 'TP'], axis=1, inplace=True)

classifiers_results = []
for i in models:
    a = data[data.Model.isin([i])]
    by_subset = []
    for j in EXPERIMENTS:
        b = a[a.Label.isin([j])]
        b.drop(['Fold', 'Label', 'Model'], axis=1, inplace=True)
        b = b.add_prefix(j + '_')
        b.reset_index(inplace=True, drop=True)
        by_subset.append(b)
    results_by_subset = pd.concat(by_subset, axis=1)
    classifiers_results.append((i, results_by_subset))

subsets_results = []
for i in EXPERIMENTS:
    a = data[data.Label.isin([i])]
    by_subset = []
    for j in models:
        b = a[a.Model.isin([j])]
        b.drop(['Fold', 'Label', 'Model'], axis=1, inplace=True)
        b = b.add_prefix(j + '_')
        b.reset_index(inplace=True, drop=True)
        by_subset.append(b)
    results_by_subset = pd.concat(by_subset, axis=1)
    subsets_results.append((i,results_by_subset))

ranking_metric = []
ranking_models = []
for i in accuracies:
    accuracy_metric = "_%s" %i

    for metric, results in subsets_results:
        colX = [c for c in results.columns if accuracy_metric in c]
        X = results[colX]
        nemenyi = sp.posthoc_nemenyi_friedman(X)
        ranking_metric.append((i, metric, nemenyi))

        fig, ax = plt.subplots()
        im = ax.imshow(nemenyi, cmap='viridis')
        cbar = ax.figure.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(len(models)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(models)
        ax.set_yticklabels(models)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_title("Nemenyi %s, %s" %(i, metric))
        fig.tight_layout()

        target_file = './Risultati_review/Review/Plots/Heatmaps_nemenyi_metrics/'
        make_sure_folder_exists(target_file)
        plt.savefig(target_file+'%s_%s.png'%(i, metric))
        plt.close()

    for model, results in classifiers_results:
        colX = [c for c in results.columns if "_%s" %i in c]
        X = results[colX]
        nemenyi = sp.posthoc_nemenyi_friedman(X)
        ranking_models.append((i, model,nemenyi))

        fig, ax = plt.subplots()
        im = ax.imshow(nemenyi, cmap='viridis')
        cbar = ax.figure.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(len(EXPERIMENTS_LABELS)))
        ax.set_yticks(np.arange(len(EXPERIMENTS_LABELS)))
        ax.set_xticklabels(EXPERIMENTS_LABELS)
        ax.set_yticklabels(EXPERIMENTS_LABELS)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_title("Nemenyi %s, %s" %(i, model))
        fig.tight_layout()

        target_file = './Risultati_review/Review/Plots/Heatmaps_nemenyi_models/'
        make_sure_folder_exists(target_file)
        plt.savefig(target_file+'%s_%s.png'%(i, model))
        plt.close()

with open('./Risultati_review/Review/metrics_nemenyi.pkl', 'wb') as f:
    pickle.dump(ranking_metric, f)
with open('./Risultati_review/Review/models_nemenyi.pkl', 'wb') as f:
    pickle.dump(ranking_models, f)
