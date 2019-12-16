from keras.optimizers import *
np.random.seed(42)
tf.set_random_seed(42)
#names:
DATA_PATH = './data/raw/'
TRAIN_NAME = 'train_cax.csv'
TEST_NAME = 'test_cax.csv'

TARGET = 'label'
DROPLIST = []

# imodel settings
from model.config_LSTM import *



PIC_FOLDER = './data/pictures/'
STACKING_FOLDER = './data/stacking/'
SUBMIT_FOLDER = './data/result/'
DEBUG_FOLDER = './data/debug/'

for f in [PIC_FOLDER,STACKING_FOLDER,SUBMIT_FOLDER,DEBUG_FOLDER]:
    os.makedirs(f,exist_ok=True)


import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)