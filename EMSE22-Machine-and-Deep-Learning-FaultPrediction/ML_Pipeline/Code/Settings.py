#######################################################################################################################
"""
Main Settings
"""

ANALYSIS = 'DL' #either ML or DL
VALIDATION = 'LOGO' #ONLY LOGO available at the time

#FEATURE = ['M', 'PRODUCT', 'PROCESS', 'PRODUCT-PROCESS', 'M-PRODUCT', 'M-PROCESS', 'M-PRODUCT-PROCESS', 'squid']
FEATURE = ['squid-M', 'squid-PRODUCT', 'squid-PROCESS', 'squid-PRODUCT-PROCESS', 'squid-M-PRODUCT', 'squid-M-PROCESS', 'squid-M-PRODUCT-PROCESS', 'squid']
OVERSAMPLING = 'True' #if True use SMOTE over-sampling technique

WINDOW_LEN = 10
#######################################################################################################################

"""
Main Settings for Classic Machine Learning
"""
PATH = './Review/Dataset/ProductProcess.csv'
TARGET = 'inducing'

"""
Main Settings for Deep Learning
"""
import sys

MODEL = str(sys.argv[1]) #'fcn' #either 'resnet' or 'fcn'
FEATURE_DL = str(sys.argv[2]) #'M'


trainedmodels_folder = './trained_models/%s/%s' %(MODEL, FEATURE_DL)
history_folder = './history_trained_models/%s/%s' %(MODEL, FEATURE_DL)



nb_epochs = 50
NB_CLASSES = 2
MINI_BATCH_SIZE = 64
#######################################################################################################################
"""
Plots
"""
xlim = [-0.05, 1.05]
ylim = [-0.05, 1.05]
#######################################################################################################################
