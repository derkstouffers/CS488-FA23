# Deric Shaffer
# CS488 - Assignment 6
# Due Date - Nov. 13th, 2023

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

data = pd.read_csv("Online Retail.csv")
data.dropna()

# create pivot table
p_table = (data.groupby(['InvoiceNo', 'StockCode'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))

# convert to quantities to binary values (1 = purchased, 0 = else)
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

table_sets = p_table.applymap(encode_units)

# perform apriori algorithm to fidn frequent item sets
freq_sets = apriori(table_sets, min_support = 0.02, use_colnames = True)

# generate association rules w/ confidence
assc = association_rules(freq_sets, metric = 'confidence', min_threshold = 0.5)

# sort the rules by confidence in descending order
assc.sort_values(by = 'confidence', ascending = False)

# get top 10
with open('output.txt', 'w') as file:
    file.write(str(assc.head(10)))