import requests
from bs4 import BeautifulSoup
from datetime import date, timedelta
import time
import sys
import csv
import os
from urllib.parse import urljoin  

sys.stdout.reconfigure(encoding='utf-8')

start_date = date(2022, 9, 21)
end_date = date.today()

current_date = start_date

output_folder = 'news_articles'

os.makedirs(output_folder, exist_ok=True)

keywords = {
    "stock", "stocks", "share", "equity", "market", "trade", "trading", "investment",
    "investing", "investor", "portfolio", "bull market", "bear market", "NASDAQ", "Dow Jones", 
    "S&P 500", "financial", "stock price", "market analysis", "stock trading", "market trends", 
    "capital markets", "asset management", "dividends", "IPO", "bonds", "commodity", "financial markets", 
    "corporate earnings", "valuation", "stock forecast", "technical analysis", "fundamental analysis", 
    "risk management", "hedging", "investor sentiment", "economic indicators", "forecasts", 
    "market volatility", "stock price prediction", "bullish", "bearish", "price movement", "financial news", 
    "stock options", "trading volume", "liquidity", "capital gain", "blue-chip stocks", "growth stocks", 
    "dividend yield", "economic growth", "financial report", "market capitalization", "index fund", "ETF", 
    "corporate news", "M&A", "mergers", "acquisitions", "stock performance", "market sentiment"
}

processed_headlines = set()

base_url = "http://dracos.co.uk/made/bbc-news-archive/"

while current_date <= end_date:
    year = current_date.year
    month = current_date.month
    day = current_date.day

    url = f"{base_url}{year}/{month:02}/{day:02}/"
    print(url)

    try:
        response = requests.get(url)

        if response.status_code == 200:
            print(f"Successfully fetched data for {year}-{month:02}-{day:02}!")
            html_content = response.text

            soup = BeautifulSoup(html_content, "html.parser")

            headlines = soup.find_all("a")
            headlines_data = ['headline', 'article']

            file_path = os.path.join(output_folder, f"output_{year}-{month:02}-{day:02}.csv")

            with open(file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(headlines_data)
                for headline in headlines:
                    article_url = headline.get('href')
                    if not article_url:
                        continue 

                    headline_text = headline.get_text(strip=True)
                    if headline_text in processed_headlines:
                        continue
                    processed_headlines.add(headline_text)

                    absolute_url = urljoin(url, article_url)

                    try:
                        article_response = requests.get(absolute_url)
                        if article_response.status_code == 200:
                            article_soup = BeautifulSoup(article_response.text, 'html.parser')
                            article_text = " ".join([p.get_text(strip=True) for p in article_soup.find_all('p')])

                            if any(keyword in article_text.lower() for keyword in keywords):
                                writer.writerow([headline_text, article_text])
                                print([headline_text, article_text])
                        else:
                            print(f"Failed to fetch article: {absolute_url} (Status code: {article_response.status_code})")
                    except Exception as e:
                        print(f"Error fetching article from {absolute_url}: {e}")

        elif response.status_code == 404:
            print(f"No data found for {year}-{month:02}-{day:02}.")
        else:
            print(f"Failed to fetch data. HTTP Status code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Error occurred while fetching data: {e}")

    current_date += timedelta(days=1)
    time.sleep(1)
