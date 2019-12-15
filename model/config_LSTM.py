from model.LSTM import *


#model configs:

#training params
N_EPOCH = 1000
LR = 0.007

# early stopping settings
MIN_DELTA = 0.0001 # thresold of improvement
PATIENCE = 30 # wait for 10 epoches for emprovement
BATCH_SIZE = 32
N_FOLD = 5 #number of folds for cross-validation
VERBOSE = 500 # print score every n batches


#input and output sizes of the model
INPUT_SIZE = 1104
OUT_SIZE = 1104
SCALE = 2

MODEL_PATH = './data/weights/'
MODEL_NAME = MODEL_PATH + 'LSTM_model'
os.makedirs(MODEL_PATH,exist_ok=True)

#dictionary of hyperparameters
HYPERPARAM = dict()
#global dropout rate
HYPERPARAM['dropout'] = 0.2
#number of filers for the model


