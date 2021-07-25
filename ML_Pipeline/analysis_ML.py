import time

import Code.Settings as s

from Code.MainClassicML import ClassicMLModel


#######################################################################################################################
start = time.time()
for i in s.FEATURE:
#i = s.FEATURE
    print('-------------')
    print('Analysing %s features' %i)
    print('-------------')
    mean_fpr, mean_tpr, mean_auc, std_auc, tprs, importances_random, P, scores, column_names = ClassicMLModel(i, s.OVERSAMPLING)


end = time.time()
print('Total time: %s' % (end - start))

#######################################################################################################################