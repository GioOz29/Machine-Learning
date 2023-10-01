import pandas as pd
import numpy as np
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.linalg import svd

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
plt.title('Distribution of Potablity')
plt.show()

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
for column_name, column_data in distribution.items():
    plt.figure(figsize=(6, 4))
    plt.hist(column_data, bins=20, edgecolor='k')
    plt.title(f'Distribution of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')

    # Step 3: Test for normality using the Shapiro-Wilk test
    _, p_value = stats.shapiro(column_data)
    alpha = 0.05  # Significance level
    is_normal = p_value > alpha

    if is_normal:
        print(f"{column_name} is normally distributed")
    else:
        print(f"{column_name} is not normally distributed")

    plt.show()

# Calculate the correlation matrix
correlation_matrix = dataset.corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


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

print(rho)


threshold = 0.9

""" Plot variance explained"""
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()


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
title('Scatter on the first 2 PCA components')
#Z = array(Z)
for c in range(NumberOfClasse):
    # select indices belonging to class c:
    class_mask = arrayOfPotabilities==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()

column_names = column_names[:]
# how the Pc are composed by the original dimensions
# pcs = [0,1,2]
pcs = [0,1,2,3,4,5,6,7]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, column_names)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('NanoNose: PCA Component Coefficients')
plt.show()