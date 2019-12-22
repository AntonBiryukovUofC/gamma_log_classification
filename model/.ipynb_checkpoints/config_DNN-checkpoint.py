


from model.DNN import *

#model configs:

#training params
N_EPOCH = 1000
LR = 0.001
LR_CUCLES = 5

# early stopping settings
MIN_DELTA = 0.001 # thresold of improvement
PATIENCE = 10 # wait for 10 epoches for emprovement
BATCH_SIZE = 4
N_FOLD = 2 #number of folds for cross-validation
VERBOSE = 500 # print score every n batches


#input and output sizes of the model
INPUT_SIZE = 1104
OUT_SIZE = 1104


MODEL_PATH = './model/'
MODEL_NAME = MODEL_PATH + 'DNN_model.h5'

#dictionary of hyperparameters
HYPERPARAM = dict()


HYPERPARAM['N_1'] = 1104
HYPERPARAM['N_2'] = 550
HYPERPARAM['N_3'] = 1104
HYPERPARAM ['kern_size'] = 4