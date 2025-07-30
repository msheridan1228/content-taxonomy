import urllib.request as urllib2
import json, time
import pandas as pd
from urllib import parse, response
import re
import tensorflow as tf
import tensorflow_text as text
import requests
from hierarchical_content_taxonomy.text_cleaning import clean_html, first_n_words, num_words

class WordPressScraper:
  def __init__(self, urls: list[str]):
      self.urls = urls
      self.required_columns = ['text', 'title', 'url', 'date', 'id']
      self.min_text_length = 20

  def get_wordpress_data(self, filename):
      data = self.fetch_from_wordpress()
      if len(data) == 0:
          print("No data to save.")
          return pd.DataFrame()
      data = self.clean_wordpress_data(data)
      if not data.empty:
          data.to_csv(filename, index=False)
      else:
          print("No data to save after cleaning.")
      return data

  def clean_wordpress_data(self, data):
      if data.empty:
          print("Data is empty. Cannot clean.")
          return pd.DataFrame()
      
      self.check_required_columns(data)

      data['site'] = data['url'].map(self.get_site)
      data['title'] = data['title'].map(clean_html)
      data = self.remove_dev_urls(data)
      if data.empty:
          print("No data after removing dev URLs.")
          return pd.DataFrame()
      
      data['year_published'] = data['date'].map(self.pull_year)
      recent_data = data.apply(lambda x: self.is_2020s_publish(x['year_published'], x['title']), axis=1)
      data = data[recent_data]

      data.dropna(inplace=True)
      data.drop_duplicates(subset=['title'], inplace=True)
      data.drop_duplicates(subset=['url'], inplace=True)

      data['text'] = data['text'].map(clean_html)
      data['text'] = data['text'].map(first_n_words)
      data['text_length'] = data['text'].map(num_words)

      data = data[~(data['text']=='')]
      data = data[data['text_length'] > self.min_text_length]
      data.drop_duplicates(subset=['text'], inplace=True)
      data.reset_index(drop=True, inplace=True)

      if data.empty:
          print("No data after cleaning.")
          return pd.DataFrame()

      return data
  
  def fetch_from_wordpress(self, filename='cleaned_wordpress_data.csv'):
      all_data = []
      for url in self.urls:
          base_api = f"{url.rstrip('/')}/wp-json/wp/v2/posts"
          df = self.get_all_articles(base_api)
          if not df.empty:
              all_data.append(df)

      final_df = pd.concat(all_data, ignore_index=True)
      final_df['text'] = final_df['content'].apply(lambda x: x['rendered'])
      final_df['title'] = final_df['title'].apply(lambda x: x['rendered'])
      final_df['url'] = final_df['link']
      final_df = final_df[self.required_columns]
      final_df.to_csv(filename, index=False)
      return final_df
  
  def remove_dev_urls(self, data):
      if 'site' not in data.columns:
          print("No 'site' column found in data.")
          return data
      bad_string_list = ['.pantheon.', '.lndo', '.local.', 'test.', 'sonic.', '.pantheonsite.', 'localhost', ':8000', '.staging.', 'development', 'non-prod']
      for bad_string in bad_string_list:
          data = data[~data['site'].str.contains(bad_string)]
      return data

    
  def get_all_articles(self, base_url, per_page=100):
        all_posts = []
        page = 1

        while True:
            paged_url = f"{base_url}?per_page={per_page}&page={page}"
            try:
                response = requests.get(paged_url, headers={"Accept": "application/json"})
                if response.status_code != 200:
                    print(f"Failed to fetch page {page}: {response.status_code}")
                    break

                posts = response.json()
                if not posts:
                    break

                all_posts.extend(posts)

                if len(posts) < per_page:
                    break  # Last page

                page += 1
                if page ==10:  # Limit to 10 pages for testing
                    break
                print(f"Fetched {len(posts)} posts from page {page}")

            except Exception as e:
                print(f"Error fetching page {page}: {e}")
                break

        return pd.DataFrame(all_posts)


  
  def check_required_columns(self, data):
      for column in self.required_columns:
          if column not in data.columns:
              raise ValueError(f"Required column '{column}' is missing from the DataFrame.")

  def get_site(self, url):
      parsed = parse.urlsplit(url)
      return parsed.netloc
  
  def pull_year(self, date):
    year = date[:4]
    return int(year)

  def is_2010s_year_in_title(self, title):
    return re.match(r'.*([2][0][1][1-9])', str(title))

  def is_1900s_year_in_title(self, title):
    return re.match(r'.*([1][9][0-9][0-9])', str(title))

  def is_2020s_publish(self, published_year, title):
    if published_year < 2020:
      return False
    elif self.is_2010s_year_in_title(title):
      return False
    elif self.is_1900s_year_in_title(title):
      return False
    else:
      return True
