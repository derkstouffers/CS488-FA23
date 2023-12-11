import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mat73
import time

from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# import data
data = mat73.loadmat('smtp.mat')

# Q1

# extract features and labels from dataset
X = data['X']
Y = data['y']

# shuffle data
X, Y = shuffle(X, Y, random_state = 19)

# standardize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# fit the model
start1_time = time.time()
lof = LocalOutlierFactor(n_neighbors = 10, metric = 'euclidean')
lof_scores = lof.fit_predict(X)
end1_time = time.time()

# calculate processing time
processing1_time = end1_time - start1_time

# calculate ROC curve
fpr_lof, tpr_lof, _ = roc_curve(y_true = Y, y_score = lof_scores)
roc_auc_lof = auc(fpr_lof, tpr_lof)

# plot ROC curve
plt.figure(figsize = (8,8))
plt.plot(fpr_lof, tpr_lof, color = 'red', lw = 2, label = 'ROC Curve (area = {:.2f})'.format(roc_auc_lof))
plt.plot([0,1], [0,1], color = 'blue', lw = 2, linestyle = '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Local Outlier Factor')
plt.legend(loc = 'lower right')
plt.show()



# Q2

# fit the model
start2_time = time.time()
iso_f = IsolationForest(n_estimators = 100, contamination = 'auto', max_samples = 256, random_state = 19)
iso_f.fit(X)
iso_f_scores= -iso_f.decision_function(X)
end2_time = time.time()

# calculate processing time
processing2_time = end2_time - start2_time

# calculate ROC curve
fpr_iso, tpr_iso, _ = roc_curve(y_true = Y, y_score = iso_f_scores)
roc_auc_iso = auc(fpr_iso, tpr_iso)

# plot ROC curve
plt.figure(figsize = (8,8))
plt.plot(fpr_iso, tpr_iso, color = 'red', lw = 2, label = 'ROC Curve (area = {:.2f})'.format(roc_auc_iso))
plt.plot([0,1], [0,1], color = 'blue', lw = 2, linestyle = '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Isolation Forest')
plt.legend(loc = 'lower right')
plt.show()



# Q3

# determine accuracy of both
with open('output.txt', 'w') as file:
    file.write(f'AUC (Local Outlier Factor) = {roc_auc_lof}\n')
    file.write(f'AUC (Isolation Forest) = {roc_auc_iso}\n\n')

    file.write(f'Processing Time (Local Outlier Factor) = {processing1_time} seconds\n')
    file.write(f'Processing Time (Isolation Forest) = {processing2_time} seconds\n\n')

    # which method is more accurate
    if roc_auc_lof > roc_auc_iso:
        file.write('Local Outlier Factor is more accurate\n')
    elif roc_auc_lof < roc_auc_iso:
        file.write('Isolation Forest is more accurate\n')
    else:
        file.write('Both methods are equally accurate\n')
    
    # which is faster
    if processing1_time > processing2_time:
        file.write('\nLocal Outlier Factor is faster')
    elif processing1_time < processing2_time:
        file.write('\nIsolation Forest is faster')
    else:
        file.write('\nBoth methods are equally fast')
    