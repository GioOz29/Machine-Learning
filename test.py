from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
import pandas as pd
from sklearn import model_selection

# Load file
df = pd.read_csv("Machine-Learning/water_potability.csv")

# Extract attributes names
attribute_names = df.columns

# Extract class labels
class_labels = df["Potability"]

# Create a Dictionary for Class Labels
class_labels_counts = df["Potability"].value_counts()

# Convert the pandas Series to a dictionary
class_labels_dict = class_labels_counts.to_dict()

print(class_labels)