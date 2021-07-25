# Fault Prediction based on Software Metrics and SonarQube Rules. Machine or Deep Learning?

This `README.md` file contains a description of the results and how to obtain them.

The full code for replicating the full study will be licensed and released upon acceptance of the work.


### Result Analysis

1. Run `python plots_analysis.py` to build the box plots for each performance metric, comparing the classifiers between them and between a metric subset and the other by using the data in directories `Folds_Information/` and `Metrics_folds/`. It also generates creates the folder `Table`, which contains the .csv with the summary for all models for all subset of metrics. The box plots are saved in `Plots/`, divided in:
  - `Squid/`, containing the box plots comparing the ML classifiers for the analysis of the SonarQube Rules.
  - `Comparison/`, containing the box plots comparing the ML classifiers between them, for the different subsets of metrics.
2. Run `python nemenyi_test.py` to execute the Nemenyi test for all the results. This will generate the heatmaps and save them in two folders:
  - The folder `Heatmaps_nemenyi_models/`, containing the heathmaps comparing the performance of each ML classifier, based on the accuracy metrics, for the different metrics subsets.
  - The folder `Heatmaps_nemenyi_metrics/`, containing the heathmaps comparing the performance of the ML classifiers against one another for each accuracy metric, for each subset of metric used.