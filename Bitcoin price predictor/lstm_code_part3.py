import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import joblib

data_file_name_train = "bitcoin_price_processed_train"
data_file_name_validate = "bitcoin_price_processed_validate"
data_file_name_test = "bitcoin_price_processed_test"
data_file_ext = "csv"

data_train_df = pd.read_csv(data_file_name_train + "." + data_file_ext)
data_validate_df = pd.read_csv(data_file_name_validate + "." + data_file_ext)
data_test_df = pd.read_csv(data_file_name_test + "." + data_file_ext)


features = ["Open", "High", "Low", "Close", "Volume"]
data_train_scaled = data_train_df[features].values
data_validate_scaled = data_validate_df[features].values
data_test_scaled = data_test_df[features].values

data_train_df["Date"] = pd.to_datetime(data_train_df["Date"])
data_validate_df["Date"] = pd.to_datetime(data_validate_df["Date"])
data_test_df["Date"] = pd.to_datetime(data_test_df["Date"])
data_train_dates = data_train_df["Date"]
data_validate_dates = data_validate_df["Date"]
data_test_dates = data_test_df["Date"]

def construct_lstm_data(data, sequence_size, target_attr_idx):
    data_X, data_y = [], []
    for i in range(sequence_size, len(data)):
        data_X.append(data[i-sequence_size:i, :])
        data_y.append(data[i, target_attr_idx])
    return np.array(data_X), np.array(data_y)

sequence_size = 30
X_train, y_train = construct_lstm_data(data_train_scaled, sequence_size, 0)
data_all_scaled = np.concatenate([data_train_scaled, data_validate_scaled, data_test_scaled], axis=0)

train_size = len(data_train_scaled)
validate_size = len(data_validate_scaled)
test_size = len(data_test_scaled)

X_validate, y_validate = construct_lstm_data(
    data_all_scaled[train_size-sequence_size:train_size+validate_size, :], sequence_size, 0)
X_test, y_test = construct_lstm_data(
    data_all_scaled[-(test_size+sequence_size):, :], sequence_size, 0)

regressor = Sequential([
    Input(shape=(sequence_size, 5)),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])

regressor.compile(optimizer="adam", loss="mean_squared_error")

model_name = "bitcoin_price_lstm.model.keras"

best_model_checkpoint_callback = ModelCheckpoint(
    model_name, 
    monitor="val_loss", 
    save_best_only=True, 
    mode="min", 
    verbose=1)

history = regressor.fit(
    x=X_train, 
    y=y_train, 
    validation_data=(X_validate, y_validate), 
    epochs=100, 
    batch_size=16, 
    callbacks=[best_model_checkpoint_callback]
)

plt.figure(figsize=(18, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("LSTM Model Performance")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

best_model = load_model(model_name)

y_test_predict = best_model.predict(X_test)

scaler_model_name = "bitcoin_price_scaler"
scaler_model_ext = "gz"
sc = joblib.load(scaler_model_name + "." + scaler_model_ext)

def inverse_transform(data, scaler, feature_count):
    return scaler.inverse_transform(
        np.concatenate((data, np.ones((len(data), feature_count - 1))), axis=1))[:, 0]

feature_count = len(features)

y_test_inv = inverse_transform(y_test.reshape(-1, 1), sc, feature_count)

y_test_predict_inv = inverse_transform(y_test_predict, sc, feature_count)

plt.figure(figsize=(18, 5))

plt.plot(data_test_dates, y_test_inv, label="Testing Data", color="green")
plt.plot(data_test_dates, y_test_predict_inv, label="Testing Predictions", linewidth=1, color="limegreen")

plt.title("Bitcoin Price Predictions With LSTM")
plt.xlabel("Time")
plt.ylabel("Price (USD)")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
plt.legend()
plt.grid(color="lightgray")
plt.show()

recent_samples = 50
plt.figure(figsize=(18, 6))


plt.plot(data_test_dates, y_test_inv, label="Testing Data", color="green", linewidth=4)
plt.plot(data_test_dates, y_test_predict_inv, label="Testing Predictions", linewidth=2, color="limegreen")

plt.title("Bitcoin Price Predictions (Last 50 Days)")
plt.xlabel("Time")
plt.ylabel("Price (USD)")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.legend()
plt.grid(color="lightgray")
plt.show()
result_df = pd.DataFrame({
    'Date': data_test_dates,
    'Actual Price': y_test_inv,
    'Predicted Price': y_test_predict_inv.flatten()
})

result_df.to_csv('bitcoin_price_predictions.csv', index=False)
mse = mean_squared_error(y_test_inv, y_test_predict_inv)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_inv, y_test_predict_inv)

print(f"Final Model Error:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")