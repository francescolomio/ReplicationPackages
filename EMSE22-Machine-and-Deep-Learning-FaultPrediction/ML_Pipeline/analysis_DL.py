import time

import Code.Settings as s

from Code.MainDeep import DeepLearningModel
from Code.Train import trainCycle


#######################################################################################################################
start = time.time()
# for i in s.FEATURE:
print('-------------')
print('Analysing %s features' %s.FEATURE_DL)
print('-------------')
X, targets, model, class_weight, groups, colX = DeepLearningModel()
mean_fpr, mean_tpr, mean_auc, std_auc, tprs, importances_random, P, scores, column_names = trainCycle(X,
                                                                                        targets, model, class_weight, groups, s.OVERSAMPLING, s.FEATURE_DL, colX)

end = time.time()
print('Total time: %s' % (end - start))

#######################################################################################################################