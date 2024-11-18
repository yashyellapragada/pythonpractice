import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

file_path = 'Values.csv'
data = pd.read_csv(file_path)
citation_columns = ['numbers_of_citations_in_2017', 'numbers_of_citations_in_2018', 
                    'numbers_of_citations_in_2019', 'numbers_of_citations_in_2020', 
                    'numbers_of_citations_in_2021', 'numbers_of_citations_in_2022']
target_column = 'numbers_of_citations_in_2023'

# Spliting data
train_data, test_data = train_test_split(data[citation_columns + [target_column]], 
                                         test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(train_data[citation_columns])
X_test = scaler.transform(test_data[citation_columns])
y_train = train_data[target_column].values
y_test = test_data[target_column].values

# neural network model (6-3-1 architecture)
model = Sequential([
    Dense(3, input_dim=6, activation='relu', kernel_regularizer=l2(0.01)),  # Hidden layer with L2 regularization
    Dropout(0.2),  # Dropout layer to reduce overfitting
    Dense(1)  # Output layer with 1 neuron for regression (no activation function)
])

# Adam optimizer and a reduced learning rate
optimizer = Adam(learning_rate=0.001)  # Lower learning rate for stable convergence
model.compile(optimizer=optimizer, loss=MeanSquaredError())

# Training the model with early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=500, validation_split=0.2, 
                    batch_size=16, callbacks=[early_stopping], verbose=1)

# Evaluating model performance on training and test data
train_mse_nn = model.evaluate(X_train, y_train, verbose=0)
test_mse_nn = model.evaluate(X_test, y_test, verbose=0)

print("\nNeural Network Results")
print(f"Training MSE: {train_mse_nn}")
print(f"Test MSE: {test_mse_nn}")

# Comparison with linear regression (from HW4) - these are values from linear regression
linear_reg_train_mse = 20893.19 
linear_reg_test_mse = 58524.20  

print("\nComparison with Linear Regression")
print(f"Linear Regression Training MSE: {linear_reg_train_mse}")
print(f"Linear Regression Test MSE: {linear_reg_test_mse}")
print(f"Neural Network Training MSE Improvement: {linear_reg_train_mse - train_mse_nn}")
print(f"Neural Network Test MSE Improvement: {linear_reg_test_mse - test_mse_nn}")
