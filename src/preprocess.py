import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess():
    path="data\pima-indians-diabetes-database.csv"
    data = pd.read_csv(path)

    columns = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    data[columns] = data[columns].replace(0 , np.nan) 
    for column in columns:
        temp = data.groupby('Outcome')[column].median()
        data.loc[(data['Outcome'] == 0) & (data[column].isna()) , column ] = temp[0]
        data.loc[(data['Outcome'] == 1) & (data[column].isna()) , column ] = temp[1]

    numerical_columns = data.columns.to_list()[:-1]
    for col in numerical_columns:
        Q1 = data[col].quantile(0.25)  # First quartile (25%)
        Q3 = data[col].quantile(0.75)  # Third quartile (75%)
        IQR = Q3 - Q1  # Interquartile range
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        data = data[(data[col] > lower_bound) & (data[col] < upper_bound)]

    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    return train_test_split(X, y, test_size = 0.25, random_state = 44)
