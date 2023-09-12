import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path_name = "/Users/giovanniorzalesi/Desktop/Machine Learning/water_potability.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(path_name)

# Display the column names
column_names = df.columns

# Remove rows with missing values in any column
df_filtered = df.dropna(how='any')

# # Select 100 random rows from df_filtered
# dataset = df_filtered.sample(n=100, random_state=42).reset_index(drop=True)

# # Step 1: Count the occurrences of 0s and 1s
# count_0 = (dataset.iloc[:, -1] == 0).sum()
# count_1 = (dataset.iloc[:, -1] == 1).sum()

# # Step 2: Create a Pie Chart
# labels = ['Not Potable', 'Potable']
# sizes = [count_0, count_1]
# colors = ['red', 'green']
# explode = (0.1, 0)  # Explode the '0' slice for emphasis

# plt.figure(figsize=(6, 6))
# plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
# plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# # Display the pie chart
# plt.title('Distribution of Potablity')
# plt.show()

# ph = dataset.iloc[:, 0]
# is_in_range = (ph >= 0) & (ph <= 14)
# # To see which rows meet the condition:
# rows_in_range = dataset[is_in_range]

# # To count how many rows meet the condition:
# ph_range = is_in_range.sum()

# # To print the rows that meet the condition:
# print(rows_in_range)

# # To print the count of rows that meet the condition:
# print("Number of rows in the range [0, 14]:", ph_range)

# # Step 1: Extract all columns except the last one
# distribution = dataset.iloc[:, :-1]

# # Step 2: Plot histograms and test for normality
# for column_name, column_data in distribution.items():
#     plt.figure(figsize=(6, 4))
#     plt.hist(column_data, bins=20, edgecolor='k')
#     plt.title(f'Distribution of {column_name}')
#     plt.xlabel(column_name)
#     plt.ylabel('Frequency')

#     # Step 3: Test for normality using the Shapiro-Wilk test
#     _, p_value = shapiro(column_data)
#     alpha = 0.05  # Significance level
#     is_normal = p_value > alpha

#     if is_normal:
#         print(f"{column_name} is normally distributed)")
#     else:
#         print(f"{column_name} is not normally distributed)")

#     plt.show()

# # Calculate the correlation matrix
# correlation_matrix = dataset.corr()

# # Create a heatmap to visualize the correlation matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
# plt.title('Correlation Heatmap')
# plt.show()


# Step 1: Data Preparation (Optional, but recommended)
# Standardize the data if necessary (PCA is sensitive to scale)
