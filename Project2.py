import pandas as pd
import numpy as np
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold, cross_val_score
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)


path_name = "./water_potability.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(path_name)

# Remove rows with missing values in any column
df_filtered = df.dropna(how='any')

X = df_filtered

X = X.values
N,M = X.shape

# Subctract mean value from data and devide for variance
# To use Xm for the feature transformation to have 0 mean and standard deviation of
N=len(X)
Xm = X - np.ones((N,1))*X.mean(axis=0)
Xm = Xm*(1/np.std(Xm,0))

y = Xm[:, 0]
Xr = Xm[:, 1:]
results = []
lambda_values = [1, 10, 50, 100, 1000, 1100]

for lambda_val in lambda_values:
    model = Ridge(alpha=lambda_val)  # Create a Ridge regression model with the current λ value
    #model = Lasso(alpha=lambda_val) 
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)  # 10-fold cross-validation
    scores = cross_val_score(model, Xr, y, cv=kfold, scoring='neg_mean_squared_error')  # gen error
    mean_error = -scores.mean()  # Calculate the mean error (make it positive)
    results.append(mean_error)

plt.plot(lambda_values, results, marker='o')
plt.xlabel('Lambda (Regularization Strength)')
plt.ylabel('Generalization Error')
plt.xscale('log')  # Since λ values are typically on a logarithmic scale
plt.title('Generalization Error vs. Lambda')
plt.grid(True)
plt.show()