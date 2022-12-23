import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

def make_sure_folder_exists(path):
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        os.makedirs(folder)

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

def feature_selection(data, colX, subset):
    X = data[colX]
    print('Calculating VIF')
    vif1 = calc_vif(X)
    a=vif1.VIF.max()

    counter = 1
    while a > 5:
        print('Iteration %s' %counter)
        maximum_a = vif1.loc[vif1['VIF'] == vif1['VIF'].max()]
        vif1 = vif1.loc[vif1['variables'] != maximum_a.iloc[0,0]]
        vif1 = calc_vif(X[vif1.variables.tolist()])
        a = vif1.VIF.max()
        # print(a)
        counter+=1


    X = data[vif1.variables.tolist()]

    target_file = './Review/Data/'
    make_sure_folder_exists(target_file)
    X.to_csv(target_file + 'dataframe_X_%s.csv' % subset, index=False)

    return X