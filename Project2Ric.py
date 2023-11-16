from matplotlib import pyplot as plt
import sklearn.linear_model as lm

from toolbox_02450 import rlr_validate, train_neural_net, draw_neural_net

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
import pandas as pd
from sklearn import model_selection
import torch

from FunProj2Ric import AnnModel, baslineModel, kFoldANN


df = pd.read_csv("water_potability.csv")

# Extract attributes names
columns_names = df.columns

data = df.dropna(how='any')
data = data.values

# Feature transformation mean=0 standard deviation=1
n = len(data)
data = data - np.ones((n,1)) * np.mean(data,axis=0)
data = data * (1 / np.std(data,0))

y = data[:,0]
X = data[:, 1:]
# X = X[:len(X)//3, :]         #used to reduce sizing while test coding
# y = y[:len(y)//3]            #used to reduce sizing while test coding   
N, M = X.shape

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+ columns_names
M = M+1
   
    
#### riccardo regression part B ####

#two layer cross validation 
 
#defining outer loop
 
# K-fold crossvalidation
K_out = 5                   # number of folds outer loop
K_in = 5                    # number of folds inner loop
CV = model_selection.KFold(K_out, shuffle=True)   

#general varaiables for the 3 methods
start_hidden_units=1
gen_error_ANN = []
n_h_units_folds = []
# Values of lamda
lambdas = np.power(10., range(-5, 9))

print(X)
print(y)


# baslineModel(X, y, K_in,)  

model = lm.LinearRegression(fit_intercept=True) #no nead of fit interceptor if you add the column of 1 to your X
model = model.fit(X,y)
print("marioioio:", model.coef_)



for (k, (train_index, test_index)) in enumerate(CV.split(X,y.squeeze())): 
    #begin of the outer loop
    
    X_train_out = X[train_index,:]
    y_train_out = y[train_index]
    X_test_out = X[test_index,:]
    y_test_out = y[test_index]
    
    
    ## ANN inner loop use my function
    # best_model, model_errror, n_h_units = kFoldANN(X_train_out, y_train_out, K_in, start_hidden_units)
    # print("model errror and model:", model_errror, best_model, n_h_units)
    # n_h_units_folds.append(n_h_units)
    
    # AnnModel(X_train_out, y_train_out, X_test_out, y_test_out, n_h_units)

    
    
    
    # linear inner loop 
    # opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train_out, y_train_out, lambdas, K_in)
    # print("best lambda and test error", opt_lambda, opt_val_err, test_err_vs_lambda) # opt_val_err e' il piu basso degli error all'interno di test_err_vs_lambda 


    # baseline inner loop 
    # baslineModel(X_train_out, y_train_out, K_in,)  
