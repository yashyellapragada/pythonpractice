import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

data = pd.read_csv('Values.csv')
citation_columns = ['numbers_of_citations_in_2017', 'numbers_of_citations_in_2018', 
                    'numbers_of_citations_in_2019', 'numbers_of_citations_in_2020', 
                    'numbers_of_citations_in_2021', 'numbers_of_citations_in_2022']
target_column = 'numbers_of_citations_in_2023'

# Split the data 
train_data, test_data = train_test_split(data[citation_columns + [target_column]], 
                                         test_size=0.2, random_state=42)

# Standardize the features for linear regression
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data[citation_columns])
test_scaled = scaler.transform(test_data[citation_columns])

# Linear Regression to predict 2023 citations
linear_reg = LinearRegression()
linear_reg.fit(train_scaled, train_data[target_column])

# Predictions for 2023 citations on training and test sets
train_preds_lr = linear_reg.predict(train_scaled)
test_preds_lr = linear_reg.predict(test_scaled)

# Calculating MSE and MAE for training and test sets
train_mse_lr = mean_squared_error(train_data[target_column], train_preds_lr)
test_mse_lr = mean_squared_error(test_data[target_column], test_preds_lr)
train_mae_lr = mean_absolute_error(train_data[target_column], train_preds_lr)
test_mae_lr = mean_absolute_error(test_data[target_column], test_preds_lr)

# Output MSE and MAE results
print("Linear Regression Results")
print(f"Training MSE: {train_mse_lr}, Test MSE: {test_mse_lr}")
print(f"Training MAE: {train_mae_lr}, Test MAE: {test_mae_lr}")

# Calculate citation ratio for 2023/2022 and classify into Low, Medium, or High categories
def classify_ratio(row):
    ratio = row['numbers_of_citations_in_2023'] / row['numbers_of_citations_in_2022']
    if ratio < 1.05:
        return 0  # Low
    elif 1.05 <= ratio <= 1.15:
        return 1  # Medium
    else:
        return 2  # High

# Add category to data
train_data['category'] = train_data.apply(classify_ratio, axis=1)
test_data['category'] = test_data.apply(classify_ratio, axis=1)

# Logistic Regression for classification
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(train_scaled, train_data['category'])

# Predictions for category classification on training and test sets
train_preds_log = log_reg.predict(train_scaled)
test_preds_log = log_reg.predict(test_scaled)

# Calculate accuracy for logistic regression
train_accuracy_log = accuracy_score(train_data['category'], train_preds_log)
test_accuracy_log = accuracy_score(test_data['category'], test_preds_log)

# Output logistic regression accuracy
print("\nLogistic Regression Results")
print(f"Training Accuracy: {train_accuracy_log}")
print(f"Test Accuracy: {test_accuracy_log}")
