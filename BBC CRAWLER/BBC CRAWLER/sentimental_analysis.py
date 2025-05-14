import os
import math
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2Model
import pandas as pd
from tqdm import tqdm

input_directory = 'news_articles/'
output_directory = 'sentimental_data/'

os.makedirs(output_directory, exist_ok=True)

# Prepare constants
MAX_LENGTH = 50
PAD_TOKEN = "<|pad|>"
EOS_TOKEN = "<|endoftext|>"

tokenizer = GPT2Tokenizer.from_pretrained(
    "gpt2",
    pad_token=PAD_TOKEN,
    eos_token=EOS_TOKEN,
    max_length=MAX_LENGTH,
    is_split_into_words=True
)

model = TFGPT2Model.from_pretrained(
    "gpt2",
    use_cache=False,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)

model.resize_token_embeddings(len(tokenizer))

def analyze_sentiment(text, tokenizer, model):
    text = str(text) + EOS_TOKEN
    inputs = tokenizer(
        text, 
        return_tensors='tf', 
        max_length=MAX_LENGTH, 
        truncation=True, 
        pad_to_max_length=True, 
        add_special_tokens=True
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    model.training = False
    outputs = model(input_ids, attention_mask=attention_mask)
    last_hidden_state = tf.reduce_mean(outputs.last_hidden_state, axis=1)
    logits_layer = tf.keras.layers.Dense(2, activation='softmax')
    logits = logits_layer(last_hidden_state).numpy()

    sentiment_score = logits[0][1]  
    sentiment_label = "Positive" if sentiment_score >= 0.5 else "Negative"
    return sentiment_label, sentiment_score

daily_averages = []

start_date = pd.Timestamp('2017-08-01')
end_date = pd.Timestamp('2024-08-01')

for current_date in pd.date_range(start=start_date, end=end_date):
    file_name = f'output_{current_date.strftime("%Y-%m-%d")}.csv'
    input_file_path = os.path.join(input_directory, file_name)

    if not os.path.exists(input_file_path):
        print(f"File not found: {input_file_path}, skipping.")
        continue

    df = pd.read_csv(input_file_path)
    assert 'article' in df.columns, f"File {file_name} must contain an 'article' column."

    results = []
    sentiment_scores = []  
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {file_name}"):
        article_sentiment, article_score = analyze_sentiment(row['article'], tokenizer, model)
        sentiment_scores.append(article_score)
        results.append({
            "article": row['article'],
            "article_sentiment": article_sentiment,
            "article_score": article_score
        })

    output_file_name = f'sentiment_results_{current_date.strftime("%Y-%m-%d")}.csv'
    output_file_path = os.path.join(output_directory, output_file_name)
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_file_path, index=False)

    average_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    if average_score>0.5 :
        average_sentiment = "positive"
    else :
        average_sentiment = "negative"
    daily_averages.append({
        "date": current_date.strftime("%Y-%m-%d"),
        "Sentiment" : average_sentiment,
        "average_sentiment_score": average_score
    })

    print(f"Sentiment analysis for {file_name} completed. Results saved to {output_file_path}")

daily_averages_df = pd.DataFrame(daily_averages)
daily_averages_output_path = os.path.join(output_directory, 'daily_sentiment_averages.csv')
daily_averages_df.to_csv(daily_averages_output_path, index=False)

print(f"Daily sentiment averages saved to {daily_averages_output_path}")
