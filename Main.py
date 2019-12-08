#import modules
from DataGenerator import *
from Pipeline import *
from config import *
import sys

start_fold = int(sys.argv[1])

GetData = DataGenerator()

CV = Pipeline(GetData, DL_model,start_fold)

score = CV.train()
print('________________________________________')
print('\n Model accuracy: ',score,'\n')
print('________________________________________')

""" 
os.system('dvc add ./data')
os.system('git add .')
os.system(f"git commit -m 'model , accuracy{accuracy}' ")
"""



"""
for i in range((GetData.X_train.shape[2]-1), 0, -1):

    print('\n')
    print('Checking the channel number ',i)
    print('\n')

    X_train = GetData.X_train.copy()
    X_test = GetData.X_test.copy()

    #X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
    #X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

    GetData.X_train = np.delete(GetData.X_train, i, 2)  #remove one of channels
    GetData.X_test = np.delete(GetData.X_test, i, 2)  # remove one of channels

    # save result in the array
    CV = Pipeline(GetData, DL_model, start_fold)
    score_new = CV.train()

    # check the score to remove/keep the channel
    if score <= score_new:
        print('Score improved')
        print('Current score:', score)
        print('New score:', score_new)
        score = score_new
        gc.collect()
    else:
        GetData.X_train = X_train #np.append(GetData.X_train, X_train, 2)
        GetData.X_test = X_test #np.append(GetData.X_test, X_test, 2)
        print('Score did not improved')
        print('Current score:', score)
        print('New score:', score_new)
        gc.collect()


print('__________________________________________________________________________')
print('\n')
print('RUN FINAL TRAINING')
print('\n')
print('__________________________________________________________________________')



# save result in the array
CV = Pipeline(GetData, DL_model, start_fold)
score = CV.train()


print('________________________________________')
print('\n Accuracy with SBD: ',score,'\n')
print('________________________________________')

np.save('./data/processed/test.csv',GetData.X_test)
np.save('./data/processed/train.csv',GetData.X_train)
np.save('./data/processed/y_train.csv',GetData.y_train)
"""
