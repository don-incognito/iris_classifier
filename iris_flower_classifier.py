# By Julio Daniel, 4 September 2020
# I reproduced this experiment from https://machinelearningmastery.com/machine-learning-in-python-step-by-step/


        # Check version of libraries

# Python version
import sys
print("Python: {}".format(sys.version))
# Scipy version
import scipy
print("Scipy: {}".format(scipy.__version__))
# numpy version
import numpy
print("Numpy: {}".format(numpy.__version__))
# matplotlib version
import matplotlib
print("Matplotlib: {}".format(matplotlib.__version__))
# pandas version
import pandas
print("Pandas: {}".format(pandas.__version__))
# sklearn version
import sklearn
print("Sklearn: {}".format(sklearn.__version__))

        # Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

        # Load dataset
from sklearn.utils import validation

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-lenght', 'petal-width', 'class']
dataset = read_csv(url, names=names)
print(dataset)

        # Dimensions of dataset
# We can get a quick idea of how many instances (rows)
# and how many attributes (columns) the data contains with the shape property
print(dataset.shape)

        # Peek at the Data
# It is a good idea to actually eyeball your data
print(dataset.head(20))

        # Statistical summary
# Now we can take a look at a summary of each attribute
# This includes the count, mean, the min and max values as well as some percentiles

        # Description
print(dataset.describe())

        # Class Distribution
# Let's now take a look at the number of instances (rows) that belong to each class.
# We can view this as an absolute count
print(dataset.groupby('class').size())

        # Complete example
# For reference, we can tie all of the previous elements together into
# a single scipt

# summarize the data
from pandas import read_csv
# load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-lenght', 'petal-width', 'class']
dataset = read_csv(url, names=names)
# shape
print(dataset.shape)
# head
print(dataset.head(20))
# description
print(dataset.describe())
# class distribution
print(dataset.groupby('class').size())

        # Data Visualization
# We now have a basic idea about the data. We need to extend that with some visualizations
# Looking at two types of plots
#     Univariate to understand each attributte
#     Multivariate to understand relationships between attributes

#     Univariate plots
# Given numeric input values, can create box and whisker plots of each

#      Box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# We can also create a histogram of each input variable
# to get an idea of the distribution

dataset.hist()
pyplot.show()

        # Multivariate plots
# First, let's look at scatter plots of all attributes. This can be helpful
# to spot structured relationships between input variables

        # scatter plot matrix
scatter_matrix(dataset)
pyplot.show()

        # Complete exmaple
# visualize the data
# from pandas import read_csv
# from pandas.plotting import scatter_matrix
# from matplotlib import pyplot
# # load dataset
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# dataset = read_csv(url, names=names)
# box and whisker plot
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()
# histograms
# dataset.hist()
# pyplot.show()
# scatter plot matrix
# scatter_matrix(dataset)
# pyplot.show()

        # Split out the validation dataset
array = dataset.values
x = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(x, y, test_size=0.2, random_state=1)
# You now have training data in the X,Y_train for preparing models
# You now have X,Y_validation sets that we can use later



        # Build Models
# We get an idea from the plots some of the classes
# are partially linearly separable in some dimensions
# so we are expecting generally good results

        # Here we test 6 different algorithms

    # Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    # print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


        # Compare algorithms
# A useful way to compare the samples of results for each algorithm
# is to create a box and whisker plot for each distribution and compare the distributions
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()


        # Make Predictions
# We can fit the model on the entire training dataset and
# Make predictions on the validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

        # Evaluate Predictions
# We can evaluate the predictions by
# comparing them to the expected results in the validation set,
# then calculate classification accuracy, as well as a confusion
# matrix and a classification report

print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))