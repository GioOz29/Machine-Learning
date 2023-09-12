import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
from scipy.linalg import svd
import seaborn as sns

# Set the path of the raw data
path_name = "./water_potability.csv"

# Load data file (Water Potability)
df = pd.read_csv(path_name)

# Show the content of the dataframe
print("Show content of the dataframe")
print(df)

# Miss data analysis check for NA observations
print("Count of NA observations (0 means no NA in variable)")
print(df.isnull().sum())

# Outlier detection
import warnings
warnings.filterwarnings("ignore")
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df["ph"])
plt.subplot(1,2,2)
sns.distplot(df["Hardness"])
plt.show()

# Histogram of attributes
print("Histogram of attributes")
print(df.hist())