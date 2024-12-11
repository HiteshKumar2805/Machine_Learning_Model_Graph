from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv')

# Data separation
y = df['logS']
x = df.drop('logS', axis=1)

# Data split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Linear Regression Model
lr = LinearRegression()
lr.fit(x_train, y_train)
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)

# Evaluate Linear Regression
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)
lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

# Create Results DataFrame for Linear Regression
lr_results = pd.DataFrame({
    'Method': ['Linear Regression'],
    'Training MSE': [lr_train_mse],
    'Training R2': [lr_train_r2],
    'Test MSE': [lr_test_mse],
    'Test R2': [lr_test_r2]
})

# Random Forest Model
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)
y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)

# Evaluate Random Forest
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)
rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

# Create Results DataFrame for Random Forest
rf_results = pd.DataFrame({
    'Method': ['Random Forest'],
    'Training MSE': [rf_train_mse],
    'Training R2': [rf_train_r2],
    'Test MSE': [rf_test_mse],
    'Test R2': [rf_test_r2]
})

# Combine Results
df_models = pd.concat([lr_results, rf_results], axis=0).reset_index(drop=True)

print(df_models)

# Scatter Plot for Linear Regression
plt.figure(figsize=(5, 5))
plt.scatter(y_train, y_lr_train_pred, alpha=0.3, label="Predicted vs Actual")
z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)
plt.plot(y_train, p(y_train), color='#F8766D', label="Best Fit Line")
plt.xlabel('Experimental logS')
plt.ylabel('Predicted logS')
plt.legend()
plt.show()
