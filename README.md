# A Machine and Deep Learning analysis among SonarQube rules, Product, and Process Metrics for Faults Prediction

This repository is the official implementation of the paper *A Machine and Deep Learning analysis among SonarQube rules, Product, and Process Metrics for Faults Prediction* (https://link.springer.com/article/10.1007/s10664-022-10164-z). 

## Requirements ##

All the package required to run this analysis can be found in the `environment.yml` file. To install it it is needed a conda installation, and it is necessary to run the following command in the terminal: `conda env create -f environment.yml`. The environment has been tested on a Ubuntu 20.04 system.

This `README.md` file contains a description of how scripts used to run the analysis.
The two main files for running the analysis are the `analysis_ML.py` and the `analysis_DL.py` files.  

This use all the files contained in the `Code/` folder to run the full analsysis.
Specifically, the file in this folder are the following:
1. `Settings.py` contains all the settings for both the machine learning and the deep learning analysis.
2. `FeatureSelection.py` contains all the functions needed to run the initial feature selection to check for multicollinearity between the variables and remove them. This script uses the Variance Inflation Factor to check for multicollinearity.
3. `PreProc.py` contains the funcition needed to preprocess the data both for the Machine learning analysis and the Deep Learning one. Particularly, this function run the feature selection on the specific input variables chosen, and rearrange the data in a suitable format based on the type of analysis considered.
4. `Model_DL.py` contains the two Deep Learning models used for this work: ResNet18 and Fully Convolutional Network.
5. `Classification_crossValidation.py` contains the full validation pipeline for the machine learning models, including the classifiers definitions and the LOGO validation method used. 
6. `Train.py` similarly to the previous script, this file contains the full training pipeline for the deep learning models.
7. `MainClassicML.py` and `MainDeep.py` contain the functions that define the dataset to use based on the variable used, and defines the last settings before actually running the analysis pipelines.

## How to run the analysis ##

In order to run the Machine Learning analysis, it is enough to run from terminal the command: `python analysis_ML.py`.

In order to run the Deep Learning Analysis, it is necessary to run from terminal the command: `python analysis_DL.py <model> <Feature>`, where `<model>` is either `fcn` or `resenet`, and `<Feature>` is one of the following:
`['M', 'PRODUCT', 'PROCESS', 'PRODUCT-PROCESS', 'M-PRODUCT', 'M-PROCESS', 'M-PRODUCT-PROCESS', 'squid']`

### Result Analysis

1. Run `python plots_analysis.py` to build the box plots for each performance metric, comparing the classifiers between them and between a metric subset and the other by using the data in directories `Folds_Information/` and `Metrics_folds/`. It also generates creates the folder `Table`, which contains the .csv with the summary for all models for all subset of metrics. The box plots are saved in `Plots/`, divided in:
  - `Squid/`, containing the box plots comparing the ML classifiers for the analysis of the SonarQube Rules.
  - `Comparison/`, containing the box plots comparing the ML classifiers between them, for the different subsets of metrics.
2. Run `python nemenyi_test.py` to execute the Nemenyi test for all the results. This will generate the heatmaps and save them in two folders:
  - The folder `Heatmaps_nemenyi_models/`, containing the heathmaps comparing the performance of each ML classifier, based on the accuracy metrics, for the different metrics subsets.
  - The folder `Heatmaps_nemenyi_metrics/`, containing the heathmaps comparing the performance of the ML classifiers against one another for each accuracy metric, for each subset of metric used.
