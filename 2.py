# sales_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# -------------------- Load Dataset --------------------
# Replace with your actual sales data CSV
data = pd.read_csv('file_report.csv')  # Should include 'Date', 'Revenue', etc.

# -------------------- Preprocess --------------------
# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Sort by date
data = data.sort_values('Date')

# Extract time features
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month

# -------------------- Exploratory Data Analysis --------------------
plt.figure(figsize=(10, 5))
sns.lineplot(x='Date', y='Revenue', data=data)
plt.title('Sales Revenue Over Time')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.tight_layout()
plt.show()

# Monthly Revenue Trend
monthly = data.groupby(['Year', 'Month'])['Revenue'].sum().reset_index()
monthly['Year-Month'] = pd.to_datetime(monthly[['Year', 'Month']].assign(DAY=1))

plt.figure(figsize=(10, 5))
sns.lineplot(x='Year-Month', y='Revenue', data=monthly)
plt.title('Monthly Sales Revenue Trend')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.tight_layout()
plt.show()

# -------------------- Predict Future Revenue --------------------
# Use month number as feature (simple regression)
monthly['Month_Num'] = np.arange(len(monthly))

X = monthly[['Month_Num']]
y = monthly['Revenue']

# Split into training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")

# -------------------- Plot Predictions --------------------
plt.figure(figsize=(10, 5))
plt.plot(monthly['Month_Num'], monthly['Revenue'], label='Actual Revenue')
plt.plot(X_test['Month_Num'], y_pred, label='Predicted Revenue', linestyle='--')
plt.title('Revenue Prediction (Linear Regression)')
plt.xlabel('Time (Months)')
plt.ylabel('Revenue')
plt.legend()
plt.tight_layout()
plt.show()




