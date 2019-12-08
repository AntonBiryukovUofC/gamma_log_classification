# imports
import pandas as pd
import numpy as np

#visualization
import matplotlib.pyplot as plt




def make_plot(df,target,i):

    fig = plt.figure(figsize=(25,5))



    index = list(np.where(target[i,:] == 0)[0])
    plt.scatter(index,df[i,index], c="blue", marker="o")

    index = list(np.where(target[i,:] == 1)[0])
    plt.scatter(index,df[i,index], c="orange", marker="o")

    index = list(np.where(target[i,:] == 2)[0])
    plt.scatter(index,df[i,index], c="red", marker="o")

    index = list(np.where(target[i,:] == 3)[0])
    plt.scatter(index,df[i,index], c="navy", marker="o")

    index = list(np.where(target[i,:] == 4)[0])
    plt.scatter(index,df[i,index], c="green", marker="o")

    plt.legend(['0','1','2','3','4'])
    plt.show()

    return 0



def convert(df):
    
    well_id = df['well_id'].unique()
    vector = np.zeros((well_id.shape[0],1100))
    
    vector_labels = np.zeros((well_id.shape[0],1100))
    
    for ind,i in enumerate(list(well_id)):
            vector[ind] = df[df['well_id'] == i]['GR'].values
            vector_labels[ind] = df[df['well_id'] == i]['label'].values
    return vector,vector_labels