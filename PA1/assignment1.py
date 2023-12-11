# Deric Shaffer
# CS488 - Assignment 1
# Due Date - Sept. 1st, 2023

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Default-of-Credit-Card-Clients.csv")

with open('output.txt', 'w') as file:
    # Q1. Read in the Default-of-Credit-Card-Clients.csv dataset, print the data in the first two rows
    file.write("First 2 Rows\n")
    file.write("-----------------------------------\n")

    # first 2 rows that aren't the column names
    for index, row in data.iloc[0:2].iterrows():
        file.write(str(row) + "\n")



    # Q2. Print the names and data types of all the columns
    file.write("\n\nColumn Name & Data Types\n")
    file.write("-----------------------------------\n")

    for col_name in data.columns:
        file.write(f"Column: '{col_name}'\t Data Type: {data[col_name].dtype}\n")



    # Q3. Calculate and print the number of rows and columns that this dataset contains. We will not count the first row because it contains the column names
    file.write("\n\nNumber of Rows & Columns\n")
    file.write("-----------------------------------\n")

    file.write("Number of Rows = " + str(data.shape[0]))
    file.write("\nNumber of Columns = " + str(data.shape[1]))



    # Q4. Calculate and print the distinct values of the column “EDUCATION”
    file.write("\n\nDistinct Values in the \"EDUCATION\" Column\n")
    file.write("-----------------------------------\n")

    file.write(str(data['EDUCATION'].unique()))



    # Q5. Calculate and print how many people have “default payment” = 1 and how many people have “default payment” = 0
    file.write("\n\nPeople with Default Payment = 1 & Default Payment = 0\n")
    file.write("-----------------------------------\n")

    # counters
    zeros = 0
    ones = 0

    # run through default payment column
    for i in data['default payment next month']:
        if i == 0:
            zeros += 1
        elif i == 1:
            ones += 1

    file.write("Default Payment = 0: " + str(zeros))
    file.write("\nDefault Payment = 1: " + str(ones))



    # Q6. Calculate and print how many people who are married and have “default payment” = 1
    file.write("\n\nPeople who are Married & Default Payment = 1\n")
    file.write("-----------------------------------\n")

    # married = 1, 2 = single, 3 = other
    counter = 0

    for index, row in data.iterrows():
        if row['MARRIAGE'] == 1 and row['default payment next month'] == 1:
            counter += 1
    
    file.write(str(counter))



    # Q7. Calculate and print how many people whose age is greater than 30 and have “default payment” = 1
    file.write("\n\nAge > 30 & Default Payemnt = 1\n")
    file.write("-----------------------------------\n")

    # counter
    thirty = 0

    for index, row in data.iterrows():
        if row['AGE'] > 30 and row['default payment next month'] == 1:
            thirty += 1

    file.write(str(thirty))



    # Q8. Calculate the average value of the “LIMIT_BAL” column when gender is male, and when gender is female
    file.write("\n\nAverage value for LIMIT_BAL for Male & Female\n")
    file.write("-----------------------------------\n")

    # 1 = male, 2 = female 
    # number of males & total balance for males
    male = 0
    male_bal = 0

    # number of females & total balance for females
    female = 0
    female_bal = 0

    for index, row in data.iterrows():
        if row['SEX'] == 1:
            male += 1
            male_bal += row['LIMIT_BAL']
        elif row['SEX'] == 2:
            female += 1
            female_bal += row['LIMIT_BAL']

    file.write("Male Average = " + str(male_bal / male))
    file.write("\nFemale Average = " + str(female_bal / female))



    # Q9. Plot a histogram for the column “default payment” when age is less than or equal to 30. Plot a histogram for the column “default payment” when age is greater than 30

    # Age <= 30
    under_30 = data[data['AGE'] <= 30]

    plt.hist(under_30['default payment next month'], bins=5, edgecolor='black')
    plt.title('Default Payment Histogram (Age <= 30)')
    plt.xlabel('Default Payment')
    plt.ylabel('Frequency')
    plt.show()

    # Age > 30
    over_30 = data[data['AGE'] > 30]

    plt.hist(over_30['default payment next month'], bins=5, edgecolor='black')
    plt.title('Default Payment Histogram (Age > 30)')
    plt.xlabel('Default Payment')
    plt.ylabel('Frequency')
    plt.show()

    # Q10. Draw a scatter plot with the data of the “AGE” column and the “LIMIT_BAL” column. The x axis represents the “AGE” column and the y axis represents the “LIMIT_BAL” column
    plt.scatter(data['AGE'], data['LIMIT_BAL'])
    plt.title('Scatter Plot of Age vs. Limit Balance')
    plt.xlabel('Age')
    plt.ylabel('Limit Balance')
    plt.show()
