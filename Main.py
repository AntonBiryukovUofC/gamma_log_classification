#import modules
from DataGenerator import *
from Pipeline import *
from config import *
import sys

start_fold = int(sys.argv[1])


CV = Pipeline(DL_model,start_fold)

CV.train()
print('________________________________________')
print('\n Model accuracy: ',1,'\n')
print('________________________________________')

""" 
os.system('dvc add ./data')
os.system('git add .')
os.system(f"git commit -m 'model , accuracy{accuracy}' ")
"""



"""
for i in features:

    X_train = GetData.X_train[:,i,:].copy()
    X_test = GetData.X_test[:, i, :].copy()

    X_train = np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    GetData.X_train = np.delete(GetData.X_train, i, 1)  #remove one of channels
    GetData.X_test = np.delete(GetData.X_test, i, 1)  # remove one of channels

    # save result in the array
    score_new = CV_loop(GetData,noise=True)  #count the score again

    # check the score to remove/keep the channel
    if score <= score_new:
        print('Score improved')
        print('Current score:', score)
        print('New score:', score_new)
        score = score_new
        gc.collect()
    else:
        GetData.X_train = np.append(GetData.X_train, X_train, 1)
        GetData.X_test = np.append(GetData.X_test, X_test, 1)
        print('Score did not improved')
        print('Current score:', score)
        print('New score:', score_new)
        gc.collect()


print('________________________________________')
print('\n AUC_ROC score on test set with noise and SBD: ',score,'\n')
print('________________________________________')
"""