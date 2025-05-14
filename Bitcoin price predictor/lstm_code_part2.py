import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
import matplotlib.dates as mdates

pd.set_option("display.width", 120)

data_file_name = "updated_target_file"
data_file_ext = "csv"
data = pd.read_csv(data_file_name + "." + data_file_ext)
data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

features = ["Open", "High", "Low", "Close", "Volume"]

data_train, data_temp = train_test_split(data, test_size=0.2, shuffle=False, random_state=42)
data_validate, data_test = train_test_split(data_temp, test_size=0.5, shuffle=False, random_state=42)

data_train_dates = data_train["Date"]
data_validate_dates = data_validate["Date"]
data_test_dates = data_test["Date"]

print(f"Training Set: {data_train.shape}")
print(f"Validation Set: {data_validate.shape}")
print(f"Testing Set: {data_test.shape}")

plt.figure(figsize=(18, 6))
plt.plot(data_train_dates, data_train["Open"], color="cornflowerblue")
plt.plot(data_validate_dates, data_validate["Open"], color="orange")
plt.plot(data_test_dates, data_test["Open"], color="green")
plt.legend(["Train Data", "Validation Data", "Test Data"])
plt.title("Data Split for Bitcoin Price")
plt.xlabel("Samples Over Time")
plt.ylabel("Price (USD)")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
plt.grid()
plt.show()

sc = MinMaxScaler(feature_range=(0, 1))
data_train_scaled = sc.fit_transform(data_train[features])
data_validate_scaled = sc.transform(data_validate[features])
data_test_scaled = sc.transform(data_test[features])

scaler_model_name = "bitcoin_price_scaler"
scaler_model_ext = "gz"
joblib.dump(sc, scaler_model_name + "." + scaler_model_ext)

data_train_scaled_final = pd.DataFrame(data_train_scaled, columns=features)
data_train_scaled_final["Date"] = data_train_dates.values

data_validate_scaled_final = pd.DataFrame(data_validate_scaled, columns=features)
data_validate_scaled_final["Date"] = data_validate_dates.values

data_test_scaled_final = pd.DataFrame(data_test_scaled, columns=features)
data_test_scaled_final["Date"] = data_test_dates.values

data_file_name_train = "bitcoin_price_processed_train"
data_file_name_validate = "bitcoin_price_processed_validate"
data_file_name_test = "bitcoin_price_processed_test"
data_file_ext = "csv"
data_train_scaled_final.to_csv(data_file_name_train + "." + data_file_ext, index=None)
data_validate_scaled_final.to_csv(data_file_name_validate + "." + data_file_ext, index=None)
data_test_scaled_final.to_csv(data_file_name_test + "." + data_file_ext, index=None)