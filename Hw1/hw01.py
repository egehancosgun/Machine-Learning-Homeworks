#DASC 521 
#Homework 1

import numpy as np
import pandas as pd

#Importing Datasets
x_load=np.genfromtxt("hw01_images.csv",delimiter=",")
y_load=np.genfromtxt("hw01_labels.csv",delimiter=",")

#Division into Training and Test Datasets
x_train=x_load[0:200,:]
y_train=y_load[0:200]
x_test=x_load[200:,:]
y_test=y_load[200:]

#Estimating Parameters
class_count= len(np.unique(y_train))

means = np.transpose(np.array([np.mean(x_train[y_train== (c + 1)],axis=0) for c in range(class_count)]))
print(means[:,0])
print(means[:,1])
deviations = np.transpose(np.array([np.sqrt(np.mean((x_train[y_train== (c + 1)] - np.transpose(means)[c])**2,axis=0)) for c in range(class_count)]))
print(deviations[:,0])
print(deviations[:,1])
priors=[np.mean(y_train == (c + 1)) for c in range(class_count)]
print(priors)
#Model Construction
liste=[]
def score_calculator(x,m,s,p):
    score=np.sum(-np.log(s)-(x-m)**2/s**2)+np.log(p)
    return score

def predictor(x):
    y_predicted=np.ones((200,1))
    for i in range(x.shape[0]):
        if score_calculator(x,means[i,1],deviations[i,1],priors[1])-score_calculator(x,means[i,0],deviations[i,0],priors[0])>0:
            y_predicted[i]+=1
    return y_predicted

y_hat_ontest=predictor(x_test)
y_hat_ontrain=predictor(x_train)

#Confusion Matrix

def comparison(x,y):
    
    correct_1_1=0
    correct_2_2=0
    false_1_2=0
    false_2_1=0

    for i in range(y.shape[0]):
        if x[i]==y[i]:
            if y[i]==1:
                correct_1_1+=1
            else:
                correct_2_2+=1
        else:
            if y[i]==1:
                false_2_1+=1
            else:
                false_1_2+=1         
    return np.array([[correct_1_1,false_1_2],[false_2_1,correct_2_2]])


confusion_matrix_1=pd.DataFrame(comparison(y_test,y_hat_ontest),columns=[1,2],index=[1,2])
print(confusion_matrix_1)
confusion_matrix_2=pd.DataFrame(comparison(y_train,y_hat_ontrain),columns=[1,2],index=[1,2])
print(confusion_matrix_2)
                





    