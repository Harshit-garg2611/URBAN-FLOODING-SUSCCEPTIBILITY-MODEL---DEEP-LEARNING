import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# Load Data
data = pd.read_excel('./flood_susceptibility_data_gurugram.xlsx')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Preprocess Data
data['runoff (mm)'].fillna(method='ffill', inplace=True)
if data['runoff (mm)'].isnull().any():
    data['runoff (mm)'].fillna(data['runoff (mm)'].mean(), inplace=True)

X = data.drop(columns=["runoff (mm)"])
y = data["runoff (mm)"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"Model Evaluation Metrics:\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nR^2 Score: {r2:.2f}")

# Define Prediction Function
def predict_future_runoff(year, month, day):
    # Set future date and validate input range
    date = pd.Timestamp(year=year, month=month, day=day)
    days_ahead = (date - data.index[-1]).days
    
    if days_ahead <= 0:
        print("The date should be in the future.")
        return
    if days_ahead > 365 * 100:
        print("The date is too far in the future for reliable prediction. Try within the next 100 years.")
        return

    # Forecast Features with Variability
    forecasted_features = {}
    for feature in X.columns:
        try:
            forecast_model = ARIMA(data[feature], order=(1, 1, 1))
            forecast_fit = forecast_model.fit()
            forecast = forecast_fit.forecast(steps=days_ahead)
            forecasted_value = forecast.iloc[-1]
            forecasted_features[feature] = forecasted_value
            
            print(f"Forecast for {feature} ({days_ahead} days ahead): {forecasted_value}")  # Debug output
            
        except Exception as e:
            print(f"Error forecasting {feature}: {e}")
            return

    # Prepare Data for Prediction
    forecasted_df = pd.DataFrame([forecasted_features])
    forecasted_scaled = scaler.transform(forecasted_df)
    predicted_runoff = model.predict(forecasted_scaled)
    print(f"Predicted Runoff on {date.strftime('%Y-%m-%d')}: {predicted_runoff[0]:.2f} mm")
    
    # Visualizations
    plt.figure(figsize=(20, 15))

    # Runoff Prediction
    plt.subplot(3, 3, 1)
    plt.bar(["Predicted Runoff"], [predicted_runoff[0]], color="teal")
    plt.title(f"Predicted Runoff for {date.strftime('%Y-%m-%d')}")
    plt.ylabel("Runoff (mm)")

    # Rainfall Simulation
    plt.subplot(3, 3, 2)
    plt.bar(["Predicted Rainfall"], [forecasted_features.get('precipitation (mm/day)', 0)], color="blue")
    plt.title("Predicted Rainfall")
    plt.ylabel("Rainfall (mm/day)")

    # Flood Susceptibility Map
    plt.subplot(3, 3, 4)
    susceptibility_levels = ['Low', 'Moderate', 'High', 'Very High']
    susceptibility_risks = [1, 2, 3, 4]
    predicted_risk = np.clip(int(predicted_runoff[0] // 10), 0, 3)
    plt.bar(susceptibility_levels, susceptibility_risks, color=['lightgreen', 'yellow', 'orange', 'red'])
    plt.axhline(y=predicted_risk + 1, color='blue', linestyle='--')
    plt.title("Flood Susceptibility Level")
    plt.ylabel("Risk Level")

    # True vs. Predicted Runoff
    plt.subplot(3, 3, 5)
    plt.plot(y_test.index, y_test, label="True Runoff", color="green")
    plt.plot(y_test.index, y_pred, label="Predicted Runoff", color="red", linestyle="--")
    plt.title("True vs Predicted Runoff")
    plt.xlabel("Date")
    plt.ylabel("Runoff (mm)")
    plt.legend()

    # Prediction Error Distribution
    plt.subplot(3, 3, 6)
    errors = y_test - y_pred
    sns.histplot(errors, bins=20, kde=True, color="purple")
    plt.title("Prediction Error Distribution")
    plt.xlabel("Prediction Error (mm)")
    plt.ylabel("Frequency")

    # Feature Importance
    plt.subplot(3, 3, 7)
    importances = model.feature_importances_
    feature_names = X.columns
    sns.barplot(x=importances, y=feature_names, palette="viridis")
    plt.title("Feature Importances")
    plt.xlabel("Importance")

    plt.tight_layout()
    plt.show()

# User Input for Future Date
print("Enter the future year, month, and day for runoff prediction (only months 7-9 are allowed).")
try:
    year = int(input("Year (e.g., 2030): "))
    
    # Month input with restriction
    month = int(input("Month (7-9): "))
    while month < 7 or month > 9:
        print("Invalid month. Please enter a month between 7 and 9.")
        month = int(input("Month (7-9): "))
    
    day = int(input("Day (1-31): "))
    
    # Call prediction function
    predict_future_runoff(year, month, day)

except ValueError:
    print("Invalid input. Please enter numeric values for the year, month, and day.")
