# Deric Shaffer
# CS488 - Assignment 4
# Due Date - Oct. 8th, 2023

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

crop_data = pd.read_csv('Crop_Recommendation.csv')

# Q1
x = crop_data.drop('label', axis = 1)
y = crop_data['label']

q1x_train, q1x_test, q1y_train, q1y_test = train_test_split(x, y, test_size = 0.2, random_state = 19)

avg_scores = []

for k in range(1,51):
    knn = KNeighborsClassifier(n_neighbors=k)
    kf = KFold(n_splits = 5)

    # calculate average accuracy score for each k
    scores = cross_val_score(knn, q1x_train, q1y_train, cv = kf, scoring = 'accuracy')
    avg_scores.append(scores.mean())

# determine best k value
best_k = range(1, 51)[np.argmax(avg_scores)]

# report k-NN accuracy on test set
knn_test = KNeighborsClassifier(n_neighbors=best_k)
knn_test.fit(q1x_test, q1y_test)

k_predict = knn_test.predict(q1x_test)

# Q2

# convert label to binary (rice vs. non-rice)
crop_data['label'] = crop_data['label'].apply(lambda x: 'rice' if x == 'rice' else 'non-rice')

x = crop_data.drop('label', axis = 1)
y = crop_data['label']

q2x_train, q2x_test, q2y_train, q2y_test = train_test_split(x, y, test_size = 0.2, random_state = 19)

lr = LogisticRegression(max_iter = 1000)
lr.fit(q2x_train, q2y_train)

# calculate probability scores for 'rice'
prob_score = lr.predict_proba(q2x_test)[:, 1]

# compute ROC curve
fpr, tpr, thresholds = roc_curve(q2y_test, prob_score, pos_label = 'rice')

# plot
plt.figure(figsize = (8, 6))
plt.plot(fpr, tpr, color='blue', lw = 2, label = 'ROC Curve (Rice)')
plt.plot([0, 1], [0, 1], color = 'gray', linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Rice vs. Non-Rice Classification')
plt.legend(loc="lower right")
plt.show()

# Q3
with open('SMSSpamCollection', 'r') as file:
    lines = file.readlines()

data = [line.strip().split('\t') for line in lines]

sms_data = pd.DataFrame(data, columns = ['label', 'message'])

q3x_train, q3x_test = train_test_split(sms_data, test_size = 0.2, random_state = 19)

tfidf_vector = TfidfVectorizer()

q3_train_tfidf = tfidf_vector.fit_transform(q3x_train['message'])
q3_test_tfidf = tfidf_vector.transform(q3x_test['message'])

train_label = q3x_train['label']
test_label = q3x_test['label']

# svc
svm = SVC()
svm.fit(q3_train_tfidf, train_label)

svm_predict = svm.predict(q3_test_tfidf)

svm_accuracy = accuracy_score(test_label, svm_predict)
svm_confusion = confusion_matrix(test_label, svm_predict)

# random forest
rf = RandomForestClassifier()
rf.fit(q3_train_tfidf, train_label)

rf_predict = rf.predict(q3_test_tfidf)

rf_accuracy = accuracy_score(test_label, rf_predict)
rf_confusion = confusion_matrix(test_label, rf_predict)

# non-graph outputs
with open('output.txt', 'w') as file:
    # Q1
    file.write('Q1\n')
    file.write('--------------------\n')

    file.write('Best K: ' + str(best_k) + '\n')
    file.write('Accuracy: ' + str(accuracy_score(k_predict, q1y_test)) + '\n\n')

    # Q3
    file.write('Q3\n')
    file.write('--------------------\n')

    file.write('SVM Accuracy: ' + str(svm_accuracy) + '\n')
    file.write('SVM Confusion Matrix: ' + str(svm_confusion) + '\n')

    file.write('Random Forest Accuracy: ' + str(rf_accuracy) + '\n')
    file.write('SVM Accuracy: ' + str(rf_confusion) + '\n')
