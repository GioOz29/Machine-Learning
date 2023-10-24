import pandas as pd
import numpy as np
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.linalg import svd
from sklearn import model_selection
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
from toolbox_02450 import rlr_validate

path_name = "./water_potability.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(path_name)

# Display the column names
column_names = df.columns

# Remove rows with missing values in any column
df_filtered = df.dropna(how='any')

# Select 100 random rows from df_filtered
# dataset = df_filtered.sample(n=100, random_state=42).reset_index(drop=True)
dataset = df_filtered

# Step 1: Count the occurrences of 0s and 1s
count_0 = (dataset.iloc[:, -1] == 0).sum()
count_1 = (dataset.iloc[:, -1] == 1).sum()

# Step 2: Create a Pie Chart
labels = ['Not Potable', 'Potable']
sizes = [count_0, count_1]
colors = ['red', 'aqua']
explode = (0.1, 0)  # Explode the '0' slice for emphasis

plt.figure(figsize=(6, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Display the pie chart
# plt.title('Distribution of Potablity')
# plt.show()

# Extract features and classification
X_c = dataset[['ph', 'Organic_carbon']].values  # Select only 'ph' and 'Turbidity' columns
Y_c = dataset['Potability'].values

# Define valid values for i and j based on your dataset
i = 0  # Index for 'ph'
j = 1  # Index for 'Turbidity'

# Define colors for 'Not Potable' and 'Potable'
colors = ['red' if label == 0 else 'aqua' for label in Y_c]

# Create a scatter plot with custom colors for each class
# plt.title('Water Potability')
# plt.scatter(x=X_c[:, i], y=X_c[:, j], c=colors, s=50, alpha=0.5)
# plt.xlabel('pH')
# plt.ylabel('Organic_carbon')
# plt.show()

# Control if the PH values are in range, so if the dataset has correct values
# ph = dataset.iloc[:, 0]
# is_in_range = (ph >= 0) & (ph <= 14)
# To see which rows meet the condition:
# rows_in_range = dataset[is_in_range]

# To count how many rows meet the condition:
# ph_range = is_in_range.sum()

# To print the rows that meet the condition:
# print(rows_in_range)

# To print the count of rows that meet the condition:
# print("Number of rows in the range [0, 14]:", ph_range)

# Step 1: Extract all columns except the last one
distribution = dataset.iloc[:, :-1]

# Step 2: Plot histograms and test for normality
# for column_name, column_data in distribution.items():
#     plt.figure(figsize=(6, 4))
#     plt.hist(column_data, bins=20, edgecolor='k')
#     plt.title(f'Distribution of {column_name}')
#     plt.xlabel(column_name)
#     plt.ylabel('Frequency')

#     # Step 3: Test for normality using the Shapiro-Wilk test
#     _, p_value = stats.shapiro(column_data)
#     alpha = 0.05  # Significance level
#     is_normal = p_value > alpha

#     if is_normal:
#         print(f"{column_name} is normally distributed")
#     else:
#         print(f"{column_name} is not normally distributed")

#     plt.show()

# Calculate the correlation matrix
correlation_matrix = dataset.corr()

# Create a heatmap to visualize the correlation matrix
# plt.figure()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
# plt.title('Correlation Heatmap')
# plt.show()


# Step 1: Data Preparation (Optional, but recommended)
# Standardize the data if necessary (PCA is sensitive to scale)
X = dataset
classLabels = X.iloc[:, -1] # Array with all with last column label FOR EVERY ROW usato dopo
X = X.values
N,M = X.shape

""" Subtract mean value from data and devide for variance """
N=len(X)
Xm = X - np.ones((N,1))*X.mean(axis=0)
Xm = Xm*(1/np.std(Xm,0))

""" PCA by computing SVD of Xm"""
U,S,Vh = svd(Xm,full_matrices=False)

"""Compute variance explained by principal components"""
rho = (S*S) / (S*S).sum() # We are looking at the variance per each value (subset of 1 component)

# print(rho)


threshold = 0.9

""" Plot variance explained"""
# plt.figure(figsize=(10, 8))
# plt.plot(range(1,len(rho)+1),rho,'x-')
# plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
# plt.plot([1,len(rho)],[threshold, threshold],'k--')
# plt.title('Variance explained by principal components');
# plt.xlabel('Principal component');
# plt.ylabel('Variance explained');
# plt.legend(['Individual','Cumulative','Threshold'])
# plt.grid()
# plt.show()


"""projecting the centeere data on the principal components
   scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
   of the vector V. So, for us to obtain the correct V, we transpose:"""
V = Vh.T

Z = Xm @ V  # Projection of X into the subset (in reality V has the same dimention of Xm)

# Indices of the principal components to be plotted
i = 0
j = 1

"""i dont like too much this part becaus im missing something this code is not working""" # ex 2.1.4 instead of the scatter of the column
# 1 and 2 of our dataset i want the scatter of the column pca1 and pca2 of our rotated dataset

# Plot PCA of the data
classNames = sorted(set(classLabels)) # Array with all list of labels (no duplicates)
classDict = dict(zip(classNames, range(len(classNames))))
arrayOfPotabilities = np.asarray([classDict[value] for value in classLabels]) # Array with all labels in number format
NumberOfClasse = len(classNames) # Number of classes
f = figure()
#title('Scatter on the first 2 PCA components')
#Z = array(Z)
# for c in range(NumberOfClasse):
#     # select indices belonging to class c:
#     class_mask = arrayOfPotabilities==c
#     plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
# legend(classNames)
# xlabel('PC{0}'.format(i+1))
# ylabel('PC{0}'.format(j+1))

# Output result to screen
#show()

column_names = column_names[:]
# how the Pc are composed by the original dimensions
# pcs = [0,1,2]
# pcs = [0,1,2,3,4,5,6,7,8,9]
# pcs = [0,1,2]
# pcs = [3,4,5,6]
pcs = [7,8,9]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
# plt.figure(figsize=(15, 8))
# for i in pcs:    
#     plt.bar(r+i*bw, V[:,i], width=bw)
# plt.xticks(r+bw, column_names)
# plt.xticks(rotation=45)
# plt.xlabel('Attributes')
# plt.ylabel('Component coefficients')
# plt.legend(legendStrs)
# plt.grid()
# plt.title('PCA Component Coefficients')
# plt.show()


### Scatter plot with pc1 pc2 with the points in red if not potable aqua if potable



## To use Xm for the feature transformation to have 0 mean and standard deviation of 1
# Xm is already standardized before


# Xm = X - np.ones((N,1))*X.mean(axis=0)
# Xm = Xm*(1/np.std(Xm,0))
y = Xm[:, 0]
Xr = Xm[:, 1:]
N, M = X.shape

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)

# Values of lambda
lambdas = np.power(10., range(-5, 9))

# Initialize variables
# T = lens(lambdas)
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))

k=0
for train_index, test_index in CV.split(Xm,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10    
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    #m = lm.LinearRegression().fit(X_train, y_train)
    #Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    #Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Display the results for the last cross-validation fold
    if k == K-1:
        figure(k, figsize=(12,8))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        #legend(attributeNames[1:], loc='best')
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
