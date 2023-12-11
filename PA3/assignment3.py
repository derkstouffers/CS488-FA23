# Deric Shaffer
# CS488 - Assignment 3
# Due Date - Sept. 21st, 2023

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Default-of-Credit-Card-Clients.csv")

# get rid of the names of the columns row
data = data.drop(0)

# Q1. Create a training set that contains 80% of the labeled data and export it to a .csv file called training.csv. 
#   Create a test set that contains the remaining 20% and export it to a .csv file called testing.csv

# get rid of the unnamed column
x = data.drop(columns=[data.columns[0], 'Y'], axis=1)
y = data['Y']

q1_training, q1_test, y_training, y_test = train_test_split(x, y, test_size=.20, shuffle=False)

q1_training.to_csv('training.csv', index=False)
q1_test.to_csv('testing.csv', index=False)

# Q2. Using entropy as the impurity measure for splitting criterion, fit decision trees of different maximum depths to the training set. 
#   Submit the plot showing their respective training and test accuracies when applied to the training and test sets. 
#   What do you find?
depths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]

q2_train_accuracies = []
q2_test_accuracies = []

for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, criterion='entropy')
    clf.fit(q1_training, y_training)
    
    # predict on training and test sets
    train_prediction = clf.predict(q1_training)
    test_prediction = clf.predict(q1_test)

    # calculate accuracies for training and test sets
    q2_train_accuracies.append(accuracy_score(train_prediction, y_training))
    q2_test_accuracies.append(accuracy_score(test_prediction, y_test))

plt.figure(figsize=(10,6))

plt.plot(depths, q2_train_accuracies, 'ro-', depths, q2_test_accuracies, 'bo-')

plt.legend(['Training Accuracy','Test Accuracy'])

plt.xlabel('Max Depth of Decision Tree')
plt.ylabel('Accuracy')
plt.title('Decision Tree Accuracy vs. Max Depth')

plt.grid(True)
plt.show()

# Q3. train a k-nearest neighbor classifier and measure performance on both the training set and the test set,
#   Submit plots showing the trend as k varies from 1 to 25 for each of the two distances,
#   focusing on both training set and test set accuracies. What do you find?
k = [1, 5, 10, 15, 20, 25]
distances = ['euclidean', 'cosine']

q3_training_accuracies = {'euclidean':[], 'cosine':[]}
q3_test_accuracies = {'euclidean':[], 'cosine':[]}

for val in k:
    for d in distances:
        knn = KNeighborsClassifier(n_neighbors=val, metric=d)
        knn.fit(q1_training, y_training)
        
        # predict on training and test sets
        train_prediction = clf.predict(q1_training)
        test_prediction = clf.predict(q1_test)

        # calculate accuracies for training and test sets
        q3_training_accuracies[d].append(accuracy_score(train_prediction, y_training))
        q3_test_accuracies[d].append(accuracy_score(test_prediction, y_test))

# euclidean distance plot
plt.plot(k, q3_training_accuracies['euclidean'], 'ro-', k, q3_test_accuracies['euclidean'],'bo-')
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('Euclidean Distance')

plt.grid(True)
plt.show()

# cosine distance plot
plt.plot(k, q3_training_accuracies['cosine'], 'ro-', k, q3_test_accuracies['cosine'],'bo-')
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('Cosine Distance')

plt.grid(True)
plt.show()

# Q4. Can you show a way to improve testing accuracy from Q3?
scale = StandardScaler()
scale.fit(q1_training)

q4_training = scale.transform(q1_training)
q4_test = scale.transform(q1_test)

knn_e = KNeighborsClassifier(n_neighbors=25, metric='euclidean')
knn_c = KNeighborsClassifier(n_neighbors=25, metric='cosine')

knn_e.fit(q4_training, y_training)
knn_c.fit(q4_training, y_training)

print('Accuracy for (Euclidean) scaled testing is %.2f' % (accuracy_score(y_test, knn_e.predict(q1_test))))
print('Accuracy for (Cosine) scaled testing is %.2f' % (accuracy_score(y_test, knn_c.predict(q1_test))))