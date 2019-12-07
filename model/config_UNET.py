from model.UNET import *


#model configs:

#training params
N_EPOCH = 1000
LR = 0.003

# early stopping settings
MIN_DELTA = 0.0001 # thresold of improvement
PATIENCE = 20 # wait for 10 epoches for emprovement
BATCH_SIZE = 64
N_FOLD = 5 #number of folds for cross-validation
VERBOSE = 500 # print score every n batches


#input and output sizes of the model
INPUT_SIZE = 1104
OUT_SIZE = 1104

MODEL_PATH = './data/weights/'
MODEL_NAME = MODEL_PATH + 'UNET_model'

#dictionary of hyperparameters
HYPERPARAM = dict()
#global dropout rate
HYPERPARAM['dropout'] = 0.2
#number of filers for the model
HYPERPARAM['init_power'] = 5
#size of kernel of input channels
HYPERPARAM['kernel_size'] = 4
