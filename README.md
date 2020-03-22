# machineLearningPredictGrades

ABSTRACT:

Python machine learning code that takes in various parameters of a gradebook to predict the next exam. In the end, although not having enough data allowed for outliers (grade changes of 20+) to heavily influence the model, it was able to predict normal data within 5 points and was a great learning tool as an introduction to machine learning and eventually neural networks.

EXPLANATION:

The code is written in PyCharm that is interpreted with anaconda. The dataset must be take in 7 parameters seperated by ";" with each input on a new line. To seperate the training data and eventual prediction data, insert a break line in the data. The code takes into account the break line to divide the data so the user can apply the model to see what it would predict about future scores. One important aspect to note is, for data the user wants the model to predict, a test2 must still be provided so the user should insert a score they anticipate they will recieve for that value.

Once the data completes the runs it saved the model into a pickle file so that the model doesn't have to be trained everytime. Inset comment block on the entire for loop and the two variables on top of the loop. Python comments on PyCharm is ''' followed by a ''' to close the block.

Finally, the code uses matplotlib to display all the labels against test2 to see how test2 is influenced by the factors such as semester, gpa, difficulty, etc.

VALUES:

Best accuracy run:  88.04178035302945%

Test:

Prediction: 90.32193165303778      Data: Semester=6, Difficulty=4/4, gpa=4.0, credits=3      Actual: 93.0

Prediction: 104.25035114912606     Data: Semester=6, Difficulty=3/4, gpa=4.0, credits=3      Actual: 87.0

EVALUATION:

Not providing enough data is a major issue for the model. To create a proper linear regresssion there needs to be a lot of data so the model can omit outliers, however, in the data used their are several outliers that clearly are influencing the predictions. On the predicted dataset, the model predicts from a test1 of 100 that test2 would be 104.25, when the actual value was 87. That being said when looking at the graphs they do demonstrate accurate evaulations in that as values like the gpa, and semester goes up the test2-test1 difference is greater, and while values like difficulty goes up the test2-test1 difference is less.

Overall, this project is a create introduction to machine learning in python and it's a create way to apply machine learning to real world data, even as basic as college course grades.
