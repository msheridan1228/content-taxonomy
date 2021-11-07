# Databricks notebook source
def clean_html(text):
  """
      text: a string
      return: modified initial string
  """
  text = re.sub('>', '> ',str(text))
  text = re.sub('<', ' <',text)
  text = BeautifulSoup(text, "html.parser") # HTML decoding
  text = text.text
  return text

def clean_html_top_1000(text):
  """
      text: a string
      return: modified initial string
  """
  text = re.sub('>', '> ',text)
  text = re.sub('<', ' <',text)
  text = BeautifulSoup(text, "html.parser") # HTML decoding
  text = text.text
  text = text.split()[:1000] #take only first 1000 words
  text = " ".join(text)
  return text

def num_words(text):
  text = text.split()
  return len(text)

def first_1k_words(text):
  text = text.split()[:1000]
  text = " ".join(text)
  return text

def first_n_words(text,  n):
  text = text.split()[:n]
  text = " ".join(text)
  return text

def is_full_text(len_text):
  return len_text > 20

# COMMAND ----------

module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
embed = hub.load(module_url)