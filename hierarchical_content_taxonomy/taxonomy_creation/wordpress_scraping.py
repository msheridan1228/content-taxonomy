import urllib.request as urllib2
import json, time
import pandas as pd
from urllib import parse
import re

from text_cleaning import clean_html, first_n_words, num_words, embed

class WordPressScraper:
  def __init__(self, urls, embedder=embed):
      self.urls = urls
      self.embedder = embedder

  def save_clean_wordpress_data(self, filename):
      data = self.get_article_data()
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
      data['site'] = data['url'].map(self.get_site)
      data['title'] = data['title'].map(clean_html)
      data['year_published'] = data['date'].map(self.pull_year)
      recent_data = data.map(lambda x: self.is_2020s_publish(x['year_published'], x['title']), axis=1)
      data = data[recent_data]
      data['text'] = data['text'].map(clean_html)
      data['text'] = data['text'].map(first_n_words)
      data['text_length'] = data['text'].map(num_words)
      data['title_plus_text'] = data['title'] + '. ' + data['text']
      data['text_embedding'] = self.embedder(data['title_plus_text']).numpy().tolist()
      data.dropna(inplace=True)
      data.drop_duplicates(subset=['text'], inplace=True)
      return data

  def get_article_data(self):
    all_of_it = self.get_from_wordpress(self.urls)
    data = self.wp_to_df(all_of_it)
    return data

  def get_from_wordpress(self):
    all_of_it = []
    page = "page="
    per_page = "&per_page="
    number_per_page = str(100)
    for url in self.urls:
      num_articles = int(urllib2.urlopen(url).getheader('X-Wp-Total'))
      print('total number of articles in:', url, num_articles)
      iterations = 1 + int(num_articles/int(number_per_page))
      print('total number of iterations to get data:', iterations)
      for i in range(iterations):
          while True:
              try:
                  content = urllib2.urlopen(url+page+str(i+1)+per_page+number_per_page)
                  if content.status == 200:
                    parsed_json = json.loads(content.read().decode('utf-8'))
                    all_of_it.append(parsed_json)
                    time.sleep(.1)
                  else:
                    print('request error',str(i))
              except Exception as e:
                  print(e)
                  print('try again', str(i))
                  time.sleep(5)
                  continue
              break
    return all_of_it

  def wp_to_df(all_of_it):
    data = pd.DataFrame()
    n_list = sum([len(all_of_it[i]) for i in range(len(all_of_it))])
    index_ = 0
    for o,k in enumerate(all_of_it):
        for i,j in enumerate(all_of_it[o]): # extract content from WP
            data.loc[index_,'id'] = j['id']
            try:
              data.loc[index_,'date'] = pd.to_datetime(j['date_gmt'].replace('T',' '))
            except:
              data.loc[index_,'date'] = ['1999-01-01']
            data.loc[index_,'slug'] = j['slug']
            data.loc[index_,'url'] = j['link']
            data.loc[index_,'title'] = j['title']['rendered']
            data.loc[index_,'text'] = j['content']['rendered']
            index_+=1
    return data

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
