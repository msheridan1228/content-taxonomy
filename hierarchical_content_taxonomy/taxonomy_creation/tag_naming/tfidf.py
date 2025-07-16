#### Old code compilation from 2021. This code is likely not functional as is. WIP to be turned into an extension of the TagNamer base class
from nltk.stem.porter import *
from collections import Counter 
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
ENGLISH_STOP_WORDS = set(stopwords.words('english'))
# COMMAND ----------

def tokenize(text):
  text = text.lower()
  text_words = re.findall(r'\w+', text) 
  text_nostop = [i for i in text_words if not i in ENGLISH_STOP_WORDS]
  text_nostop = [i for i in text_nostop if len(i)>1]
  return text_nostop

def filter_insignificant_pos(word_list,  
                         tag_suffixes =['CD', 'DT', 'CC', 'PRP$', 'EX' 'PRP', 'TO', 'WDT' , 'WP', 'WP$', 'WRB']):     
    good = [] 
  
    pos_dic = nltk.pos_tag(word_list)
          
   # for suffix in tag_suffixes: 
    for key, value in pos_dic:
          if (value in tag_suffixes) == False:
            good.append(key)
    return good 
  
def filter_insignificant_words(word_list,  
                         insignificant_words =['know', 'need', 'new', 'com', 'thing', 'best', 'do', 'I', 'is', 'cause', 'effect', 'vs', 'way', 'come', 'report', 'say', 'amp']):     
    good = [] 
    for word in word_list:
          if (word in insignificant_words) == False:
            good.append(word)
    return good 
  
def filter_num_words(word_list):     
    good = [] 
    for word in word_list:
          if any(char.isdigit() for char in word) == False:
            good.append(word)
    return good 

def lemmatize_word_list(word_list):
  lemmatizer = WordNetLemmatizer()
  lem = map(lemmatizer.lemmatize, word_list)
  return lem

def clean_title_text(document):
    word_list = tokenize(document)
    word_list = filter_insignificant_pos(word_list)
    word_list = lemmatize_word_list(word_list)
    word_list = filter_insignificant_words(word_list)
    word_list = filter_num_words(word_list)
    document = ' '.join(word_list)
    return document

def clean_title_word_list(top_n):
      most_occur = ' '.join(top_n)
      most_occur_token = tokenize(most_occur)
      most_occur_token = list(dict.fromkeys(most_occur_token)) #remove dupes while retaining order
      most_occur = '_'.join(most_occur_token)
      most_occur = str(most_occur).replace('__', '_')
      most_occur = str(most_occur).replace('--', '')
      return most_occur

def return_valid_cluster_title(total_cluster_list, sorted_words, n):
    top_n = sorted_words[:n]
    if len(top_n) > 0:
      most_occur = clean_title_word_list(top_n)
      if most_occur in total_cluster_list:
        print("duplicates for : " +  most_occur)
        most_occur = return_valid_cluster_title(total_cluster_list, sorted_words, n+1)
    else:
      most_occur = 'none'
    return most_occur

def pull_top_tfidf_words(tfidf, vectorizer, n):
  top_words_list = []
  for tag in np.arange(0, tfidf.shape[0]):
    tfidf_sorting = np.argsort(tfidf[tag].toarray()).flatten()[::-1]
    feature_array = np.array(vectorizer.get_feature_names())
    sorted_words = feature_array[tfidf_sorting]
    most_occur = return_valid_cluster_title(top_words_list, sorted_words, n)
    top_words_list.append(most_occur)
  return top_words_list



