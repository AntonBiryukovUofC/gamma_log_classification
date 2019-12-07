


from model.VGG import *

#model configs:

#training params
N_EPOCH = 1000
LR = 0.1

# early stopping settings
MIN_DELTA = 0.0001 # thresold of improvement
PATIENCE = 1 # wait for 10 epoches for emprovement
BATCH_SIZE = 512
N_FOLD = 5 #number of folds for cross-validation
VERBOSE = 500 # print score every n batches


#input and output sizes of the model
INPUT_SIZE = 1104
OUT_SIZE = 1104

MODEL_PATH = './model/'
MODEL_NAME = MODEL_PATH + 'VGG_model'

#dictionary of hyperparameters
HYPERPARAM = dict()

#global dropout rate
HYPERPARAM['Drop_rate'] = 0.25

#number of filers for the model
HYPERPARAM['n_filt_1'] = 32
HYPERPARAM['n_filt_2'] = 64
HYPERPARAM['n_filt_3'] = 128

#size of kernel of input channels
HYPERPARAM['kern_size_1'] = 4
HYPERPARAM['kern_size_2'] = 4
HYPERPARAM['kern_size_3'] = 4

