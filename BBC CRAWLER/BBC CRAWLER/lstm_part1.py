import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib
import matplotlib.dates as mdates
pd.set_option("display.width", 120)
data_file_name = "dataset"
data_file_ext = "csv"
data = pd.read_csv(data_file_name + "." + data_file_ext)
data.shape
data.head()
data.dtypes
data["Date"] = pd.to_datetime(data["Date"], format="%d-%m-%Y")



data.dtypes
features = ["average_sentiment_score","Open","Close","High","Volume","Low"]
target = "Average Scaled Polarity"
train_end_date = pd.to_datetime("2017-06-15")
validate_start_date = pd.to_datetime("2017-06-16")
validate_end_date = pd.to_datetime("2017-07-29")
test_start_date = pd.to_datetime("2017-07-30")
test_end_date = pd.to_datetime("2017-08-27")

data_train = data[data["Date"] <= train_end_date][features]
data_train_dates = data[data["Date"] <= train_end_date]["Date"]
data_validate = data[(data["Date"] >= validate_start_date) & (data["Date"] <= validate_end_date)][features]
data_validate_dates = data[(data["Date"] >= validate_start_date) & (data["Date"] <= validate_end_date)]["Date"]
data_test = data[(data["Date"] >= test_start_date) & (data["Date"] <= test_end_date)][features]
data_test_dates = data[(data["Date"] >= test_start_date) & (data["Date"] <= test_end_date)]["Date"]
print(f"Training Set: {data_train.shape}")
print(f"Validation Set: {data_validate.shape}")
print(f"Testing Set: {data_test.shape}")
print("Training Dataset:")
print(data_train.head())
print("Validation Dataset:")
print(data_validate.head())
print("Testing Dataset:")
print(data_test.head())
plt.figure(figsize=(18,6))
plt.plot(data_train_dates, data_train["Open"], color="cornflowerblue")
plt.plot(data_validate_dates, data_validate["Open"], color="orange")
plt.plot(data_test_dates, data_test["Open"], color="green")
plt.legend(["Train Data", "Validation Data", "Test Data"])
plt.title("Data Split for Google Stock Price")
plt.xlabel("Samples Over Time")
plt.ylabel("Price (USD)")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
plt.grid()
data[features].describe()
sc = MinMaxScaler(feature_range=(0,1))

data_train_scaled = sc.fit_transform(data_train)

data_validate_scaled = sc.transform(data_validate)
data_test_scaled = sc.transform(data_test)
scaler_model_name = "bitcoin_price_scaler"
scaler_model_ext = "gz"

joblib.dump(sc, scaler_model_name + "." + scaler_model_ext)
data_train_scaled_final = pd.DataFrame(data_train_scaled, columns=features, index=None)
data_train_scaled_final["Date"] = data_train_dates.values

data_validate_scaled_final = pd.DataFrame(data_validate_scaled, columns=features, index=None)
data_validate_scaled_final["Date"] = data_validate_dates.values

data_test_scaled_final = pd.DataFrame(data_test_scaled, columns=features, index=None)
data_test_scaled_final["Date"] = data_test_dates.values
data_file_name_train = "dataset_processed_train"
data_file_name_validate = "dataset_processed_validate"
data_file_name_test = "dataset_processed_test"
data_file_ext = "csv"
data_train_scaled_final.to_csv(data_file_name_train + "." + data_file_ext, index=None)
data_validate_scaled_final.to_csv(data_file_name_validate + "." + data_file_ext, index=None)
data_test_scaled_final.to_csv(data_file_name_test + "." + data_file_ext, index=None)