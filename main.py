# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Set Seaborn style for visualizations
sns.set(style="darkgrid")

# Load and Prepare Data
data = pd.read_excel('./flood_susceptibility_data_gurugram.xlsx')
data['Date'] = pd.to_datetime(data['Date'])

# Separate features and target variable
X = data.drop(columns=["Date", "runoff (mm)"])
y = data["runoff (mm)"]

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Define Prediction Function
def predict_runoff(year, month, day):
    date = pd.Timestamp(year=year, month=month, day=day)
    
    # Check if the selected date has historical feature data
    if date in data['Date'].values:
        input_features = data.loc[data['Date'] == date].drop(columns=["Date", "runoff (mm)"])
        scaled_features = scaler.transform(input_features)
        predicted_runoff = model.predict(scaled_features)
        
        print(f"Predicted Runoff on {date.strftime('%Y-%m-%d')}: {predicted_runoff[0]:.2f} mm")
        
        # Visualization for the predicted date
        plt.figure(figsize=(8, 4))
        plt.bar(["Predicted Runoff"], [predicted_runoff[0]], color="teal")
        plt.title(f"Predicted Runoff for {date.strftime('%Y-%m-%d')}")
        plt.ylabel("Runoff (mm)")
        plt.show()
        
        # Flood Susceptibility Map Simulation (Illustrative)
        plt.figure(figsize=(10, 6))
        susceptibility_map = np.random.rand(10, 10) * predicted_runoff[0]  # Randomly simulates susceptibility values
        plt.imshow(susceptibility_map, cmap='Blues', interpolation='nearest')
        plt.colorbar(label="Flood Susceptibility Index")
        plt.title(f"Flood Susceptibility Map for {date.strftime('%Y-%m-%d')}")
        plt.show()
    else:
        print("No historical data available for this date. Please select a different date.")

# Prompt user for date input
print("Enter the year, month, and day for which to predict runoff (only July, August, or September).")
year = int(input("Year (2000-2024): "))
month = int(input("Month (7, 8, or 9): "))
day = int(input("Day (1-31): "))

# Call the prediction function
predict_runoff(year, month, day)

# Additional visualizations
# True vs Predicted Runoff Values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='teal', edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("True Runoff Values (mm)", fontsize=14)
plt.ylabel("Predicted Runoff Values (mm)", fontsize=14)
plt.title("True vs Predicted Runoff Values", fontsize=16, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Prediction Error Histogram
plt.figure(figsize=(8, 6))
errors = y_test - y_pred
sns.histplot(errors, kde=True, color="crimson", bins=30)
plt.title("Prediction Error Distribution", fontsize=16, fontweight='bold')
plt.xlabel("Prediction Error (mm)", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.show()

# Feature Importances (optional, if you want to analyze feature importance)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=feature_names[indices], palette="viridis")
plt.title("Feature Importance for Runoff Prediction", fontsize=16, fontweight='bold')
plt.xlabel("Importance", fontsize=14)
plt.ylabel("Features", fontsize=14)
plt.show()
