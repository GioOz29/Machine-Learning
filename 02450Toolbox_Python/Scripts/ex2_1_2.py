# exercise 2.1.2

# Imports the numpy and xlrd package, then runs the ex2_1_1 code
from ex2_1_1 import *
# (requires data structures from ex. 2.1.1)

from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show

# Data attributes to be plotted
i = 0
j = 1

##
# Make a simple plot of the i'th attribute against the j'th attribute
# Notice that X is of matrix type (but it will also work with a numpy array)
# X = np.array(X) #Try to uncomment this line

#iterate over i and j to plot all combinations of attributes
def plotAll():
    for i in range(0, 4):
        for j in range(0, 4):
            if i != j:
                f = figure()
                title('NanoNose data' + ' ' + str(i) + ' ' + str(j))
                plot(X[:, i], X[:, j], 'o')
                xlabel(attributeNames[i])
                ylabel(attributeNames[j])
                show()

plotAll()
# plot(X[:, i], X[:, j], 'o')

# %%
# Make another more fancy plot that includes legend, class labels, 
# attribute names, and a title.
# f = figure()
# title('NanoNose data')

# for c in range(x):
#     # select indices belonging to class c:
#     class_mask = y==c
#     plot(X[class_mask,i], X[class_mask,j], 'o',alpha=.3)

# legend(classNames)
# xlabel(attributeNames[i])
# ylabel(attributeNames[j])

# Output result to screen
# show()
print('Ran Exercise 2.1.2')