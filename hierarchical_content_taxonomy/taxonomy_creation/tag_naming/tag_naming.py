#### Old code compilation (from 2021! Do not use this code) To be broken up and refactored into multiple classes in this folder

# COMMAND ----------

import wikipedia
import pandas as pd
import numpy as np
import sklearn.feature_extraction.text
import tensorflow as tf
import tensorflow_hub as hub
import nltk
import spacy   
import string
import scipy
import matplotlib.pyplot as plt
import re 
import scipy.cluster.hierarchy as shc
import cloudpickle
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering 
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.stem.porter import *
from collections import Counter 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup

nltk.download('averaged_perceptron_tagger')
# %matplotlib inline

# COMMAND ----------

#print(numpy.__version__)

# COMMAND ----------

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Prepare data**
# MAGIC For this example, we use the popular 20 Newsgroups dataset which contains roughly 18000 newsgroups posts on 20 topics.

# COMMAND ----------

import pandas as pd
import cloudpickle

# COMMAND ----------

tag_meta = pd.read_csv("/dbfs/FileStore/msheridan/data/tag_clusters_metadata_3.csv")

# COMMAND ----------

pickled_taxonomy_path = '/dbfs/FileStore/msheridan/rv-content-use-emb-clusters-ward-3.pkl'
with open(pickled_taxonomy_path, 'rb') as f:
  docs_df = cloudpickle.load(f)
docs_df = pd.DataFrame(docs_df)

# COMMAND ----------

docs_df.head()



# MAGIC %md
# MAGIC ## **Hierarchical Prediction exploration**

# COMMAND ----------

docs_df.head()

# COMMAND ----------

docs_df[docs_df['topic_level_3_cluster']==47].head(10)

# COMMAND ----------

docs_df[docs_df['topic_level_3_cluster']==47]['topic_level_3'].value_counts()

# COMMAND ----------

docs_df[docs_df['topic_level_3_cluster']==47]['topic_level_4'].value_counts()

# COMMAND ----------

docs_df[docs_df['topic_level_4_cluster']==27].head(10)

# COMMAND ----------

docs_df[docs_df['topic_level_4_cluster']==3]['topic_level_4'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC # **Mapping clusters to tag names**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Getting top 10 seed words

# COMMAND ----------

topic_clusters = docs_df.groupby(['topic_level_4_cluster'])['title'].apply(lambda x: '. '.join(x)).reset_index()
topic_clusters.head()  

# COMMAND ----------

def tokenize(text):
  text = text.lower()
  text_words = re.findall(r'\w+', text) 
  text_nostop = [i for i in text_words if not i in ENGLISH_STOP_WORDS]
  text_nostop = [i for i in text_nostop if len(i)>1]
  return text_nostop

# COMMAND ----------

topic_clusters['title_token'] = topic_clusters['title'].map(tokenize)
topic_clusters.head()

# COMMAND ----------

def filter_insignificant_pos(word_list,  
                         tag_suffixes =['CD', 'DT', 'CC', 'PRP$', 'EX' 'PRP', 'TO', 'WDT' , 'WP', 'WP$', 'WRB']):     
    good = [] 
  
    pos_dic = nltk.pos_tag(word_list)
          
   # for suffix in tag_suffixes: 
   #     print(suffix)
    for key, value in pos_dic:
        #  print(key, value)
        #  print(value in tag_suffixes)
          if (value in tag_suffixes) == False:
            good.append(key)
         #   print('appended')
    return good 

# COMMAND ----------

def filter_insignificant_words(word_list,  
                         insignificant_words =['know', 'need', 'new', 'com', 'things', 'thing', 'best', 'do','does', 'I', 'is']):     
    good = [] 
    for word in word_list:
        #  print(key, value)
        #  print(value in tag_suffixes)
          if (word in insignificant_words) == False:
            good.append(word)
         #   print('appended')
    return good 

# COMMAND ----------

good_test = filter_insignificant_pos(['wow', 'your', 'dog','is','the','best', 'dog', 'I','know','20'])
print(good_test)
good_test = filter_insignificant_words(good_test)
print(good_test)

# COMMAND ----------

def top_10_words(word_list):
  word_list = filter_insignificant_pos(word_list)
  word_list = filter_insignificant_words(word_list)
  counter = Counter(word_list)
  most_occur = counter.most_common(10)
  most_occur_zip = list(zip(*most_occur))
  return list(most_occur_zip[0])

# COMMAND ----------

test_list = topic_clusters['title_token'][35]

# COMMAND ----------

top_test = top_10_words(test_list)
top_test

# COMMAND ----------

topic_clusters.sample(10)

# COMMAND ----------

topic_clusters['top_words'] = topic_clusters['title_token'].map(top_10_words)
topic_clusters.head(10)

# COMMAND ----------




# COMMAND ----------

# MAGIC %md # Finding Closest matching word using pretrained bert embedding model
# MAGIC (use GPU cluster)

# # COMMAND ----------

# !pip install bert-embedding 
# !pip install mxnet-cu112
# !pip install torch
# !pip install torchvision
# !pip install pycuda

# # COMMAND ----------

# torch.cuda.is_available()

# # COMMAND ----------

# !pip ldconfig /usr/local/cuda/lib64

# COMMAND ----------

import torch
import mxnet as mx

# COMMAND ----------

from bert_embedding import BertEmbedding
from tqdm.auto import tqdm, trange

# COMMAND ----------

test = topic_clusters['title'][0]
print(test)

# COMMAND ----------

sentences = bert_abstract.split('\n')
result = bert(sentences)
toks, embs = result[0]
print(toks)
print(len(toks), len(embs))
print(embs[0][:10])

# COMMAND ----------

# MAGIC %md ## Process a corpus

# COMMAND ----------

# !wget http://pcai056.informatik.uni-leipzig.de/downloads/corpora/eng-com_web-public_2018_10K.tar.gz
# !tar -xzvf eng-com_web-public_2018_10K.tar.gz

# COMMAND ----------

with open('eng-com_web-public_2018_10K/eng-com_web-public_2018_10K-sentences.txt', 'r') as f:
    lines = f.readlines()

# COMMAND ----------

print(lines[0])
all_sentences = [l.split('\t')[1] for l in lines]

# COMMAND ----------

# MAGIC %## Create a search index

# COMMAND ----------


from sklearn.neighbors import KDTree
import numpy as np


class ContextNeighborStorage:
    def __init__(self, sentences, model):
        self.sentences = sentences
        self.model = model

    def process_sentences(self):
        result = self.model(self.sentences)

        self.sentence_ids = []
        self.token_ids = []
        self.all_tokens = []
        all_embeddings = []
        for i, (toks, embs) in enumerate(tqdm(result)):
            for j, (tok, emb) in enumerate(zip(toks, embs)):
                self.sentence_ids.append(i)
                self.token_ids.append(j)
                self.all_tokens.append(tok)
                all_embeddings.append(emb)
        all_embeddings = np.stack(all_embeddings)
        # we normalize embeddings, so that euclidian distance is equivalent to cosine distance
        self.normed_embeddings = (all_embeddings.T / (all_embeddings**2).sum(axis=1) ** 0.5).T

    def build_search_index(self):
        # this takes some time
        self.indexer = KDTree(self.normed_embeddings)

    def query(self, query_sent, query_word, k=10, filter_same_word=False):
        toks, embs = self.model([query_sent])[0]

        found = False
        for tok, emb in zip(toks, embs):
            if tok == query_word:
                found = True
                break
        if not found:
            raise ValueError('The query word {} is not a single token in sentence {}'.format(query_word, toks))
        emb = emb / sum(emb**2)**0.5

        if filter_same_word:
            initial_k = max(k, 100)
        else:
            initial_k = k
        di, idx = self.indexer.query(emb.reshape(1, -1), k=initial_k)
        distances = []
        neighbors = []
        contexts = []
        for i, index in enumerate(idx.ravel()):
            token = self.all_tokens[index]
            if filter_same_word and (query_word in token or token in query_word):
                continue
            distances.append(di.ravel()[i])
            neighbors.append(token)
            contexts.append(self.sentences[self.sentence_ids[index]])
            if len(distances) == k:
                break
        return distances, neighbors, contexts

# COMMAND ----------

storage = ContextNeighborStorage(sentences=all_sentences, model=bert)
storage.process_sentences()
storage.build_search_index()

# COMMAND ----------

# MAGIC %md ## Query homonymous words

# COMMAND ----------

distances, neighbors, contexts = storage.query(
     query_sent='It is an investment bank.', query_word='bank', k=5, filter_same_word=True)

# COMMAND ----------

# MAGIC %md #Ngram Testing

# COMMAND ----------

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

# COMMAND ----------

docs_df['topic_level_4'].value_counts().sample(10)

# COMMAND ----------

tag_corpus_dic = {}
for tag in docs_df['topic_level_4'].unique():
  tag_data = docs_df[docs_df['topic_level_4']==tag]
  tag_corpus = tag_data['cleaned_text']
  tag_corpus = tag_corpus.append(tag_data['title'])
  tag_corpus_string = tag_corpus.str.cat(sep=' ')
  tag_corpus_dic[tag] = tag_corpus_string

# COMMAND ----------

topic_4_corpus = pd.DataFrame.from_dict(tag_corpus_dic, orient='index', columns=['corpus']).reset_index().rename(columns={'index':'tag_name'})
tag_meta_4 = tag_meta[tag_meta['tag_level']=='topic_level_4']
tag_meta_4 = tag_meta_4.merge(topic_4_corpus, how='left', on='tag_name')
tag_meta_4.head()

# COMMAND ----------

testing = docs_df[docs_df['topic_level_4']=='diet_food_eat_avoid']
test_corpus = testing['cleaned_text']
test_corpus = test_corpus.append(testing['title'])
print(test_corpus)
testing.head()

# COMMAND ----------

testing = docs_df[docs_df['topic_level_3']=='food_eat_avoid_diet']
test_corpus = testing['cleaned_text']
test_corpus = test_corpus.append(testing['title'])
print(test_corpus)
testing.head()

# COMMAND ----------

vectorizer = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(2,2), max_df = 0.95, min_df=5)
#vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(2,2), min_df=5)

response = vectorizer.fit_transform(test_corpus)

# COMMAND ----------

feature_array = np.array(vectorizer.get_feature_names())
tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]

n = 30
ngram_testing = feature_array[tfidf_sorting][:n]
print(ngram_testing)

# COMMAND ----------

vectorizer_1 = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1,1), min_df=5)
response_1 = vectorizer_1.fit_transform(test_corpus)
feature_array_1 = np.array(vectorizer_1.get_feature_names())
tfidf_sorting_1 = np.argsort(response_1.toarray()).flatten()[::-1]

n = 20
top_n = feature_array_1[tfidf_sorting_1][:n]
print(top_n)

# COMMAND ----------

tag_pivot[tag_pivot['topic_level_4']=='diet_food_eat_avoid']

# COMMAND ----------

tag_pivot = pd.pivot_table(docs_df[['topic_level_1', 'topic_level_2', 'topic_level_3', "topic_level_4", 'url']], index=['topic_level_1', 'topic_level_2', 'topic_level_3', 'topic_level_4'], aggfunc=pd.Series.nunique)
tag_pivot = tag_pivot.sort_values(['topic_level_1', 'topic_level_2', 'topic_level_3', 'url'], ascending=False).reset_index()
tag_pivot.head(50)

# COMMAND ----------

# MAGIC %md #Hardmath

# COMMAND ----------

def pull_top_vec_words(tfidf, vectorizer, n = 4):
  top_words_list = []
  for tag in np.arange(0, tfidf.shape[0]):
    tfidf_sorting = np.argsort(tfidf[tag].toarray()).flatten()[::-1]
    feature_array = np.array(vectorizer.get_feature_names())
    top_n = feature_array[tfidf_sorting[:n]]
    if len(top_n) >0:
      most_occur = ', '.join(top_n)
    else:
      most_occur = 'none'
    top_words_list.append(most_occur)
  return top_words_list

# COMMAND ----------

tag_meta_4['cv_ngrams'] = pull_top_vec_words(response, vectorizer, 20)

# COMMAND ----------

tag_meta.head()

# COMMAND ----------

# all_docs_D = tag_meta['corpus']
# for topic_k in tag_meta_4['tag_name'].unique():
#   siblings = []
#   topic_k_docs = []
#   parent = 'tag'
#   parent_hierarchy_docs = []
#   topic_k_hierarchy_docs = []
#   ref_collection = parent_hierarchy_docs - topic_k_docs
#   possible_terms = []
#   term_score_dic = {}
#   for term_c in possible_terms:
#     docs_containing_term_c = 1
#     idf_d_c = log(|D| / docs_containing_term_c + 1)
#     sum_term_count = 1
#     S1 = idf_d_c * sum_term_count
#     top_freq_term_in_sibs = 1
    
#     num_term_in_parent = 1
#     num_term_in_topic_k_hierarchy_docs = 1
#     num_term_in_parent_hierarchy_docs  = 1
    
#     for topic_j in sub hierarchy:
#       path_length_l = topic_k_loc - topic_j_loc
#       num_term_c_in_doc_dk = 1
    
#     term_score_dic[term] = term_score