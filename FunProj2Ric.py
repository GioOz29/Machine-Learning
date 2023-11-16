from matplotlib import pyplot as plt
from toolbox_02450 import rlr_validate, train_neural_net, draw_neural_net
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
import pandas as pd
from sklearn import model_selection
import torch


###
# training the ANN and testing different hidden layers values with cross validation
#K = number of kfold crossvalidations
#start_hidden_units = than at every kfold round it gets increaased
def kFoldANN ( X, y, K, start_hidden_units):

    # C = 2  #kind of lambda but for th ANN
    # X = X[:len(X)//3, :]
    # y = y[:len(y)//3]
    N, M = X.shape 
         
    # Parameters for neural network classifier
    # n_hidden_units = 5      # number of hidden units
    start_hidden_units = 1      # number of hidden units
    n_replicates = 1        # number of networks trained in each k-fold
    max_iter = 10000      #decreased to improve testing time

    # K-fold crossvalidation
    # K = 5                   # only 5 folds to speed up this example
    CV = model_selection.KFold(K, shuffle=True)


    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

    # print('Training model of type:\n\n{}\n'.format(str(model())))
    errors = [] # make a list for storing generalizaition error in each loop
    mean_errors_fold = [] # make a list for storing mean error in each loop
    model_fold = [] #used to store every genesated model so that we can pick the best after comparison
    lowest_mean_error = 0  # used for chosing best model 
    best_model = 0         # used for chosing best model 
    k_best_model = 0       # used for chosing best model 
    for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
        print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
        
        # Extract training and test set for current CV fold, convert to tensors
        X_train = torch.Tensor(X[train_index,:])
        y_train = torch.Tensor(y[train_index])
        y_train = y_train.squeeze()
        X_test = torch.Tensor(X[test_index,:])
        y_test = torch.Tensor(y[test_index])
        y_test = y_test.squeeze()
        
        print(y_train.shape)
        print(y_test.shape)
        
        # Define the model
        model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, start_hidden_units+k), #M features to n_hidden_units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(start_hidden_units+k, 1), # n_hidden_units to 1 output neuron
                        # no final tranfer function, i.e. "linear output"
                        )

        # Train the net on training data
        net, final_loss, learning_curve = train_neural_net(model,
                                                        loss_fn,
                                                        X=X_train,
                                                        y=y_train,
                                                        n_replicates=n_replicates,
                                                        max_iter=max_iter)
        
        print('\n\tBest loss: {}\n'.format(final_loss))
        
        # Determine estimated class labels for test set
        y_test_est = np.empty(y_train.shape)
        y_test_est = net(X_test)
        
        # Determine errors and errors
        se = (y_test_est.float()-y_test.float())**2 # squared error
        mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
        mean_error = sum(mse) / len(mse)
        errors.append(mse) # store error rate for current CV fold 
        
        if(lowest_mean_error != 0):
            if(mean_error < lowest_mean_error):
                lowest_mean_error = mean_error
                best_model = model
                k_best_model = k
        else:
            lowest_mean_error = mean_error
            best_model = model     
        

    # Calculate mean of each array in errors
    for i, error_array in enumerate(errors):
        mean_error = sum(error_array) / len(error_array)
        mean_errors_fold.append(mean_error)
        print(f"Mean error for fold {i+1}: {mean_error}")
    
    return(best_model, lowest_mean_error, k_best_model+1)    


def AnnModel(X_train, y_train, X_test, y_test, hidden_units):
    N, M = X_train.shape 
    # Parameters for neural network classifier
    n_replicates = 1        # number of networks trained in each k-fold
    max_iter = 5000      #decreased to improve testing time

    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

    # print('Training model of type:\n\n{}\n'.format(str(model())))
    errors = [] # make a list for storing generalizaition error in each loop
    mean_errors_fold = [] # make a list for storing mean error in each loop
    model_fold = [] #used to store every genesated model so that we can pick the best after comparison
    lowest_mean_error = 0  # used for chosing best model 
    best_model = 0         # used for chosing best model 
    k_best_model = 0       # used for chosing best model 

    # Extract training and test set for current CV fold, convert to tensors
    X_torch_train = torch.Tensor(X_train)
    y_torch_train = torch.Tensor(y_train)
    X_torch_test = torch.Tensor(X_test)
    y_torch_test = torch.Tensor(y_test)
    
    print(y_torch_train.shape)
    print(y_torch_test.shape)
    
    # Define the model
    model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, hidden_units), #M features to n_hidden_units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(hidden_units,1), # n_hidden_units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )

    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                    loss_fn,
                                                    X=X_torch_train,
                                                    y=y_torch_train,
                                                    n_replicates=n_replicates,
                                                    max_iter=max_iter)
    
    print('\n\tBest loss: {}\n'.format(final_loss))
    
    # Determine estimated class labels for test set
    y_test_est = np.empty(y_torch_train.shape)
    y_test_est = net(X_torch_test)
    
    # Determine errors and errors
    se = (y_test_est.float()-y_torch_test.float())**2 # squared error
    mse = (sum(se).type(torch.float)/len(y_torch_test)).data.numpy() #mean
    mean_error = sum(mse) / len(mse)
    errors.append(mse) # store error rate for current CV fold 
       
        

    # Calculate mean of each array in errors
    for i, error_array in enumerate(errors):
        mean_error = sum(error_array) / len(error_array)
        mean_errors_fold.append(mean_error)
        print(f"Mean error for fold {i+1}: {mean_error}")
    
    return(best_model, lowest_mean_error)    
 
     
     
     
     
     
     
 
def baslineModel(X, y, K):
    N, M = X.shape 
    CV = model_selection.KFold(K, shuffle=True)

    for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
        print()
        print()
        print()
        print()
        print("inner round N", k)
        y_train = y[train_index]
        y_test =  y[test_index]

        # baseline = y_train.sum()/len(y)
        # print(baseline)
        baseline = y_train.mean()
        print("y train mean and hence bseline:", y.mean())
        
        # subtract baseline to all the values in test and evaluate the mean squared error
        ###tests to remove ###
        print("ytest")
        print((y_test)) 
        
        print("ytest- baseline")
        print((y_test- baseline))
        
    
        ###tests to remove ###
      
        se = (y_test-baseline)**2
        mse = ( sum(se)/len(y_test))
        print("mse baseline", mse)
    return