from bs4 import BeautifulSoup
import re
import tensorflow_hub as hub
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TitleTextConcatenator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return (X['title'] + ' ' + X['text']).values.reshape(-1, 1)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, module_url="https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"):
        self.module_url = module_url
        self.embedder = hub.load(module_url)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X is expected to be a 2D array of strings
        return self.embedder([X])
    
    def fit_transform(self, X, y = None):
       return super().fit_transform(X, y)

def clean_article_text(text, n=1000):
  text = clean_html(text)
  text = first_n_words(text, n)
  return text

class CleanHtmlTransformer( BaseEstimator, TransformerMixin ):   
  def __init__ (self, cleaner_function=clean_article_text):
      self.cleaner_function = cleaner_function

  def fit( self, X, y = None ):
      return self 

  def transform( self, X, y = None ):
      # X comes as a 2D array from TitleTextConcatenator with shape (n_samples, 1)
      if hasattr(X, 'shape') and len(X.shape) == 2:
          # Flatten to 1D, apply cleaning function to each element, then reshape back
          flat_X = X.flatten()
          cleaned = np.array([self.cleaner_function(text) for text in flat_X])
          return cleaned.reshape(-1, 1)
      elif hasattr(X, 'apply'):
          # If it's a pandas Series
          clean_text = X.apply(self.cleaner_function)
          return clean_text.values.reshape(-1, 1)
      else:
          # Single string
          clean_text = self.cleaner_function(X)
          return np.array([clean_text]).reshape(-1, 1)
  
  def fit_transform(self, X, y=None):
      return self.transform(X, y)


def clean_html(text):

  # Handle bytes by converting to string
  if isinstance(text, bytes):
    text = text.decode('utf-8', errors='ignore')
  
  # Ensure text is a string
  text = str(text)
  
  if text is None:
    return " "
  
  text = re.sub('>', '> ',text)
  text = re.sub('<', ' <',text)
  soup = BeautifulSoup(text, "html.parser")
  text = soup.get_text(separator=" ", strip=True)

  # Now extract all the readable text contained in <div data-description="TEXT_HERE"> tags
  for description_tag in soup.find_all("div", attrs={"data-description": True}):
    description_with_html = description_tag["data-description"]
    soup_description = BeautifulSoup(description_with_html, "html.parser")
    text += " " + soup_description.get_text(separator=" ", strip=True)
  for header in soup.find_all("div", attrs={"data-hed": True}):
    header_with_html = header["data-hed"]
    soup_header = BeautifulSoup(header_with_html, "html.parser")
    text += " " + soup_header.get_text(separator=" ", strip=True)
    
  text = text.replace('\n','').replace('\r','').replace('\\', '')
  return text

def num_words(text):
  text = text.split()
  return len(text)

def first_n_words(text, n=1000):
  text = text.split()[:n]
  text = " ".join(text)
  return text

def is_min_text(len_text, min_len=20):
  return len_text > min_len


