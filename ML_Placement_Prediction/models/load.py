# import pandas as pd
# import numpy as np
#
# def load_data():
#     df = pd.read_csv('placement_predict_50k Dataset.csv')
#     df.rename(
#         columns={"target" : "placement_status"},
#         inplace = True
#     )
#     return df

import pandas as pd

import os


DATASET_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(__file__)
    ),
    "placement_predict_50k Dataset.csv"
)


def load_data(file_path=None):

    if file_path is None:

        file_path = DATASET_PATH

    df = pd.read_csv(
        file_path
    )

    return df


def summarize(df):

    return {
        "shape": df.shape,

        "columns": df.columns.tolist(),

        "missing_values":
            df.isnull().sum().to_dict(),

        "duplicates":
            df.duplicated().sum(),

        "data_types":
            df.dtypes.astype(str).to_dict(),

        "numeric_columns":
            df.select_dtypes(
                include="number"
            ).columns.tolist(),

        "categorical_columns":
            df.select_dtypes(
                include="object"
            ).columns.tolist(),

        "unique_values":
            df.nunique().to_dict()
    }

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Create Sample Data (Study Hours vs. Exam Score)
data = {
    'StudyHours': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    'ExamScore': [35, 45, 50, 60, 65, 75, 80, 90, 92, 98]
}
df = pd.DataFrame(data)

# 2. Define Features (X) and Target (y)
X = df[['StudyHours']]
y = df['ExamScore']

# 3. Split into Training and Testing Sets
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Fit the Built-in Linear Regression Model
model = LinearRegression()
model.fit(train_X, train_y)

# 5. Make Predictions on Test Data
y_pred = model.predict(test_X)

# 6. Evaluate Performance
mse = mean_squared_error(test_y, y_pred)
rsq = r2_score(test_y, y_pred)

print("--- Built-in Linear Regression Results ---")
print("Intercept (theta0):", model.intercept_)
print("Slope (theta1):", model.coef_[0])
print("MSE:", mse)
print("RSQ:", rsq)





import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Create Sample Data (Study Hours vs. Pass/Fail Status)
# PassStatus: 0 = Fail, 1 = Pass
data = {
    'StudyHours': [1.0, 1.5, 2.0, 3.5, 4.0, 5.5, 6.0, 7.5, 8.0, 9.0],
    'PassStatus': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# 2. Define Features (X) and Target (y)
X = df[['StudyHours']]
y = df['PassStatus']

# 3. Split into Training and Testing Sets
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Initialize and Fit the Built-in Logistic Regression Model
model = LogisticRegression()
model.fit(train_X, train_y)

# 5. Make Predictions on Test Data (predicts 0 or 1)
y_pred = model.predict(test_X)

# 6. Evaluate Performance using Accuracy
accuracy = accuracy_score(test_y, y_pred)

print("--- Logistic Regression Results ---")
print("Intercept (theta0):", model.intercept_[0])
print("Coefficient / Slope (theta1):", model.coef_[0][0])
print("Predictions on Test Set:", y_pred)
print("Actual Test Values:", test_y.values)
print("Accuracy:", accuracy)