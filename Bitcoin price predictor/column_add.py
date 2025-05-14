import pandas as pd

source_file = "daily_sentiment_averages.csv"
target_file = "Filtered_BTC_2017_to_23_09_2022.csv"

source_data = pd.read_csv(source_file)
target_data = pd.read_csv(target_file)
column_to_add = source_data["average_sentiment_score"]
target_data["average_sentiment_score"] = column_to_add

target_data.to_csv("updated_target_file.csv", index=False)

print("Column added and file saved as 'updated_target_file.csv'")
