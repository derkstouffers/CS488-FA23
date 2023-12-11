# Deric Shaffer
# CS488 - Assignment 2
# Due Date - Sept. 10th, 2023

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations
from sklearn.decomposition import PCA

data = pd.read_csv("Default-of-Credit-Card-Clients.csv")

with open('output.txt', 'w') as file:
    # 1a. for each BILL_AMTX (BILL_AMT1-BILL_AMT6), calculate its average, std_dev, min, max val
    for i in range(1,7):
        bill_amt = f'BILL_AMT{i}'
        
        file.write('Average for ' + bill_amt + ' = ' + str(np.mean(data[bill_amt])) + '\n')
        file.write('Standard Deviation for '  + bill_amt + ' = ' + str(np.std(data[bill_amt])) + '\n')
        file.write('Minimum Value for ' + bill_amt + ' = ' + str(np.min(data[bill_amt])) + '\n')
        file.write('Maximum Value for ' + bill_amt + ' = ' + str(np.max(data[bill_amt])) + '\n\n')

    # 1b. Compute covariance and correlation between the attribute pairs
    bill_amt_cols = [f'BILL_AMT{i}' for i in range(1,7)]

    file.write('Covariance:\n')
    file.write(data[bill_amt_cols].cov().to_string() + '\n\n')

    file.write('Correlation:\n')
    file.write(data[bill_amt_cols].corr().to_string() + '\n\n')

    # 1c. Display the histogram for each quantitative attributes by discretizing 
    # into 5 separate bins and counting the frequency for each bin
    quant = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                            'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

    figs, axes = plt.subplots(4,5, figsize=(15,10))

    axes = axes.flatten()

    for i, attribute in enumerate(quant):
        axes[i].hist(data[attribute], bins=5, edgecolor='black')
        axes[i].set_title(f'Historgram of {attribute}')
        axes[i].set_xlabel(attribute)
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # 1d. Display a boxplot to show the distribution of values for each of those attributes. 
    #   Which attribute has outliers?
    figs, axes = plt.subplots(4,5, figsize=(15,10))

    axes = axes.flatten()

    for i, attribute in enumerate(quant):
        axes[i].boxplot(data[attribute])
        axes[i].set_title(f'Boxplot of {attribute}')
        axes[i].set_xlabel(attribute)

    plt.tight_layout()
    plt.show()

    # It looks like all of the quantitative attributes have outliers based on the boxplots

    # 1e. BILL_AMT1-BILL_AMT4, for each pair, use a scatterplot to visualize their joint distribution.
    #   Based on the plot, what are possible correlations that you can observe
    bill_amts = [ 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4']

    figs, axes = plt.subplots(2, 3, figsize=(15,10))

    axes = axes.flatten()

    for i, (attr1, attr2) in enumerate(combinations(bill_amts, 2)):
        axes[i].scatter(data[attr1], data[attr2], alpha=0.5)
        axes[i].set_title(f'Scatterplot of {attr1} vs. {attr2}')
        axes[i].set_xlabel(attr1)
        axes[i].set_ylabel(attr2)

    plt.tight_layout()
    plt.show()

    # Correlations: positive pattern, when one BILL_AMT increases, so does the other

    # 2a. Standardize LIMIT_BAL, AGE, and BILL_AMT columns & PAY_AMT columns
    standardize = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

    avgs = data[standardize].mean()
    stds = data[standardize].std()

    for column in standardize:
        data[column] = (data[column] - avgs[column]) / stds[column]

    # standardized = average pretty close to 0, std_dev pretty close to 1
    for column in standardize:
        avg = data[column].mean()
        std = data[column].std()
        file.write(f'Column: {column}, Average: {avg}, Standard Deviation: {std}\n')

    # 2b. Create a data sample of size 1000 which is randomly selected w/o replacement from original data
    random_2b = data.sample(n=1000, replace=False)
    random_2b.to_csv('2b_random_sample.csv', index=False)

    # 2c. Create a data sample of size 1000 which is randomly selected w/o replacement from original data
    #   such that the default labels are represented equally (ex. 500 instances of default = 1 and 500 of default = 0)
    #   Use parallel coordinates to visualize this data sample based on SEX --> PAY_6
    default_0 = data[data['default payment next month'] == 0].sample(n=500,replace=False)
    default_1 = data[data['default payment next month'] == 1].sample(n=500,replace=False)
    both = pd.concat([default_0, default_1], ignore_index=True)

    attributes = ["SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

    plt.figure(figsize=(10,6))

    pd.plotting.parallel_coordinates(both, 'default payment next month', color=['blue','red'], cols=attributes)
    plt.title('Parallel Coordinates Plot')
    plt.xlabel('Attributes')
    plt.ylabel('Attribute Values')
    plt.legend(title='Default Payment', loc='upper right', labels=['0', '1'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # 2d. Generate random sample of size 1000 from original data, use Principal component analysis to reduce
    #   the number of attributes by projecting the data to a lower-dimensional space. WE WANT 23 -> 2 for scatterplot
    random_2d = data.sample(n=1000, replace=False)
    pca = PCA(n_components=2)
    pca.fit(random_2d)

    proj = pca.transform(random_2d)
    proj = pd.DataFrame(proj, columns=['pc1','pc2'])

    plt.figure(figsize=(10,6))
    plt.scatter(proj['pc1'], proj['pc2'], alpha=0.5)
    plt.title('PCA Scatterplot')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)
    plt.show()
