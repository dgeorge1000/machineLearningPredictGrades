'''
This code takes in a ".csv" file of grades throughout the semester and finds an accurate linear regression of the data.
The hopes is that through running the model several thousands times that it would learn about how various factors
affect scores for exam 2. Some of these would be semester, current gpa, difficulty of course, credits, and test 1 score
Although it would be much better to provide a lot more data and more test scores, the model had a best accuracy of
over 80% and when analyzing data that wasn't an outlier it typically was within 5 points of the actual score
'''

# all the imports needed, pandas for loading the model, numpy for editing the dataset, sklearn for handling the model
# pickle for saving the model, and matplotlib for plotting the data
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import pickle
import matplotlib.pyplot as pyplot
from matplotlib import style

# read in our data set
data = pd.read_csv("gradestrain.csv", sep=";")

# find the number of lines in the data, where a break line separates training data vs prediction data
num_lines = 0
for lines in open("gradestrain.csv") :
    num_lines += 1
    # when the dataset reaches the break line, we know the rest of the data is for prediction
    if len(lines.strip()) == 0 :
        break

# get "num_lines" to the right size for the separation of data
num_lines = num_lines - 2

# parse through what we actually want from the dataset(everything but the class name)
data = data[["semester", "difficulty(1-4)", "gpa", "credits", "test1", "test2"]]

# get the label(or the information we want the model to learn to predict)
predict = "test2"


# ATTRIBUTES: returns to us a new dataset without "test2", and remove the test data sections
x_edit = np.array(data.drop([predict], 1))
x = x_edit[:num_lines]

# LABELS: returns to us only "test2" so the model can have the actual values to learn, and remove test data sections
y_edit = np.array(data[predict])
y = y_edit[:num_lines]

# set variables to hold the test data, one without the test2, and one with only test2 to compare with the model
data_predict_test = x_edit[num_lines:]
data_actual_test = y_edit[num_lines:]

# here we begin the model by creating variables to find the best accuracy model and the amount of runs we want
best = 0
runs = 5000
# trials = 100
# test_outcome = runs - trials
# total = 0
# total_outcome = 0
for i in range(runs) :
    # divides the data into random sections that changes for each run based on "test_size"
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.25)

    # creates the model looking for an accurate linear regression
    linear = linear_model.LinearRegression()

    # will fit the data to a best fit line
    linear.fit(x_train, y_train)

    # accuracy of the linear regression, and print the value
    acc = linear.score(x_test, y_test)
    print(acc)

    # ignore comments, was trying to compute various accuracy measures
    # total = total + acc
    # if(i == test_outcome) :
    #     total_outcome = total_outcome + acc

    # update best accuracy measurement
    if (acc > best):
        best = acc

    # save a pickle file of the model
    with open("studentmodel.pickle", "wb") as f:
        pickle.dump(linear, f)

# once again ignore these comments
# print("Total: ", total / runs)
# print("Total outcome: ", total_outcome / trials)

# print the best accuracy model from all the runs
print("\nBest: ", best)

# opens the saved model, so we can comment out the entire loop so we dont need to keep training the model
# use ''' (best = 0 ..... pickle.dump(linear, f) ''' to comment out the training portion
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# takes in an array of arrays and make lots of predictions on the test data which we have not trained the model on yet
predictions = linear.predict(data_predict_test)

# print the prediction and the actual value
print("Predictions: ")
for x in range(len(predictions)):
    print(predictions[x], data_predict_test[x], data_actual_test[x])

# plot the data to see a graph of test2 against various labels
words = ["semester", "difficulty(1-4)", "gpa", "credits"]
for i in words:
    t1 = "test1"
    t2 = "test2"
    style.use("ggplot")
    pyplot.scatter(data[i], data[t2]-data[t1])
    pyplot.xlabel(i)
    pyplot.ylabel("Difference between Test 1 and Test 2")
    pyplot.show()