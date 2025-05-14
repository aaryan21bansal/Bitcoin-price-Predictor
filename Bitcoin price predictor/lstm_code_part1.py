import pandas as pd
import matplotlib.pyplot as plt

data_file_location = "BTC-USD"
data_file_name = "BTC-USD"
data_file_ext = "csv"

data = pd.read_csv(data_file_name + "." + data_file_ext)

data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

start_date = pd.to_datetime("01-08-2017", format="%d-%m-%Y")
end_date = pd.to_datetime("11-09-2022", format="%d-%m-%Y")

filtered_data = data[(data["Date"] >= start_date) & (data["Date"] <= end_date)]

filtered_data.shape

plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
plt.plot(filtered_data["Date"], filtered_data["Open"])
plt.xlabel("Time")
plt.ylabel("Open Price (USD)")
plt.title("Bitcoin Open Price (2017 - 11-09-2022)")
plt.grid()

plt.subplot(1,2,2)
plt.plot(filtered_data["Date"], filtered_data["Close"])
plt.xlabel("Time")
plt.ylabel("Close Price (USD)")
plt.title("Bitcoin Close Price (2017 - 11-09-2022)")
plt.grid()

plt.suptitle("Bitcoin Stock Over Time (2017 to 11-09-2022)")
plt.show()

data_file_name = "Filtered_BTC_2017_to_23_09_2022"
data_file_ext = "csv"
filtered_data.to_csv(data_file_name + "." + data_file_ext, index=None)
