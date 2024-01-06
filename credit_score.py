# Import Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import random


# Specifying Datatypes

dtypes = {
    'ID': str,
    'Customer_ID': str,
    'Month': str,
    'Name': str,
    'Age': str,
    'SSN': str,
    'Occupation': str,
    'Annual_Income': str,
    'Monthly_Inhand_Salary': float,
    'Num_Bank_Accounts': int,
    'Num_Credit_Card': int,
    'Interest_Rate': int,
    'Num_of_Loan': str,
    'Type_of_Loan': str,
    'Delay_from_due_date': int,
    'Num_of_Delayed_Payment': str,
    'Changed_Credit_Limit': str,
    'Num_Credit_Inquiries': float,
    'Credit_Mix': str,
    'Outstanding_Debt': str,
    'Credit_Utilization_Ratio': float,
    'Credit_History_Age': str,
    'Payment_of_Min_Amount': str,
    'Total_EMI_per_month': float,
    'Amount_invested_monthly': str,
    'Payment_Behaviour': str,
    'Monthly_Balance': str,
    'Credit_Score': str
}


df = pd.read_csv('train.csv', dtype=dtypes)




# --- Data Cleaning ------


df.drop(['ID', 'Customer_ID', 'SSN', 'Name', 'Type_of_Loan', ], axis='columns')


# Label Encoding

month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
score_order = ['Poor', 'Standard', 'Good']

oe_month = OrdinalEncoder(categories=[month_order])
oe_score = OrdinalEncoder(categories=[score_order])
le_occupation = LabelEncoder()

df['Month_n'] = oe_month.fit_transform(df[['Month']]).astype(int)
df['Occupation_n'] = le_occupation.fit_transform(df['Occupation']).astype(int)
df['Credit_Score_n'] = oe_score.fit_transform(df[['Credit_Score']]).astype(int)

#   Original Occupations: ['Accountant' 'Architect' 'Developer' 'Doctor' 'Engineer' 'Entrepreneur'
#    'Journalist' 'Lawyer' 'Manager' 'Mechanic' 'Media_Manager' 'Musician'
#    'Scientist' 'Teacher' 'Writer' '_______']
#
#   Encoded Values: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]


# Converting strings to numeric values

df_numeric_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
        'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
       'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
       'Num_Credit_Inquiries', 'Outstanding_Debt',
       'Credit_Utilization_Ratio', 'Total_EMI_per_month',
       'Amount_invested_monthly', 'Monthly_Balance']

df[df_numeric_cols] = df[df_numeric_cols].apply(pd.to_numeric, errors='coerce')


# Fill NaN values with the mean for each column

df[df_numeric_cols] = df[df_numeric_cols].fillna(df[df_numeric_cols].mean())


# Filling Label Encoded values

df['Month_n'] = df['Month_n'].fillna(random.randint(0,11))
df['Occupation_n'] = df['Occupation_n'].fillna(random.randint(0,15))


# Removing entries that do not have a credit score

df = df.dropna(subset=['Credit_Score'])






# -------Data Visualization------



# OCCUPATION/CREDIT SCORE

# Define colors for each credit score
colors = {'Poor': 'red', 'Standard': 'yellow', 'Good': 'green'}

# Group by occupation and credit score, then count occurrences
grouped = df.groupby(['Occupation', 'Credit_Score']).size().unstack(fill_value=0)

# Plotting
fig, ax = plt.subplots()

# Iterate over credit scores and plot stacked bars
bottom = None
for credit_score, color in colors.items():
    ax.bar(grouped.index, grouped[credit_score], bottom=bottom, label=credit_score, color=color)
    if bottom is None:
        bottom = grouped[credit_score]
    else:
        bottom += grouped[credit_score]

# Add labels and legend
ax.set_xlabel('Occupation')
ax.set_ylabel('Count')
ax.set_title('Credit Scores by Occupation')
ax.legend(title='Credit Score')

plt.xticks(rotation='vertical')
plt.savefig('occupations.png')




# ANNUAL INCOME/CREDIT SCORE

plt.clf()
plt.figure()

# Scatter plot
plt.scatter(df['Annual_Income'], df['Credit_Score_n'], c='blue', alpha=0.5)

# Adding labels
plt.xlabel('Annual Income')
plt.ylabel('Credit Score (Ordinal Encoded)')
plt.title('Scatter Plot of Annual Income and Credit Score')

plt.savefig('annual_income.png')




# MONTHLY INHAND/CREDIT SCORE

plt.figure(figsize=(10, 6))
plt.hist2d(df['Monthly_Inhand_Salary'], df['Credit_Score_n'], bins=(10, 5), cmap=plt.cm.Blues)
plt.colorbar(label='Frequency')

# Adding labels
plt.xlabel('Monthly Inhand Salary')
plt.ylabel('Credit Score (Ordinal Encoded)')
plt.title('Heatmap of Monthly Inhand Salary and Credit Score')

# Save the plot to a file
plt.savefig('monthly_inhand.png')




# -------Training Model----------
feature_columns = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
        'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
       'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
       'Num_Credit_Inquiries', 'Outstanding_Debt',
       'Credit_Utilization_Ratio', 'Total_EMI_per_month',
       'Amount_invested_monthly', 'Monthly_Balance', ]

# Comparing models

rf = RandomForestClassifier(
    n_estimators=60,
)

print(cross_val_score(rf, df[feature_columns], df['Credit_Score_n']))
#print(cross_val_score(SVC(), df[feature_columns], df['Credit_Score_n']))

rf.fit(df[feature_columns], df['Credit_Score_n'])



# Save the model to a file
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(rf, file)

