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

# COMMAND ----------

# MAGIC %md
# MAGIC # **Using Wiki to generate topic names**

# COMMAND ----------

#test wiki api
wikipedia.search("covid",results =10)

# COMMAND ----------

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed = hub.load(module_url)

# COMMAND ----------

#generates a list of candidate titles given a list of seed words
def generateCandidateList(wordList, nlp):
    # get the top 10 article title results from wikipedia
    titleList = []
    for thisWord in wordList:
        neighbors_10 = wikipedia.search(thisWord,results =10)
        titleList.append(neighbors_10)

    titleList=np.array(titleList).ravel()

    #take the titles and use the spacy model to turn parse them into noun chunks, then de-dupe
    noun_chunk_list = []
    for candidate in titleList:
        doc = nlp(str(candidate))
        for chunk in doc.noun_chunks:
            noun_chunk_list.append(chunk.text)

    noun_chunk_list = np.unique(noun_chunk_list).ravel()

    #combine original title list and noun chunk list
    #noun_chunk_and_title_list = np.concatenate((noun_chunk_list,titleList),axis= None) 
    #or actually lets not add in the original titles
    noun_chunk_and_title_list = noun_chunk_list


    #extract all bigrams from noun chunk list and original title list
    ngram_noun_chunk_and_title_list = []
    for thischunk in noun_chunk_and_title_list:
        vect = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(1,2))
        try:
            vect.fit([thischunk])
            for x in vect.get_feature_names():
                ngram_noun_chunk_and_title_list.append(x)
        except:
            continue


    #take all of the above and create a large candidate list        
    large_candidate_list = np.concatenate((ngram_noun_chunk_and_title_list, noun_chunk_list),axis=0)
    large_candidate_list =[str.lower(x) for x in large_candidate_list]
    large_candidate_list= np.unique(large_candidate_list)
    
    #should we de-dupe more intelligently?
    
    #the papers also have additional filters
    # only use candidates that have a wiki page of their own
    # use RACO similarity as a filtering mechanism this is based on a deprecated wiki API
    
    return large_candidate_list

# COMMAND ----------

#load up a spacy model for grammar parsing and noun chunking
nlp = spacy.load("en_core_web_sm")

# COMMAND ----------

def compute_semantic_matches(this_candidate_list, this_query_list):
    #first, need to turn the seed list and all possible candidates into semantic vectors
    searchString =str(this_query_list).translate(str.maketrans('', '', string.punctuation))
    searchVector = embed([searchString])
    vect_list =np.array(embed(this_candidate_list))

    #then, compute distance of each candidate from the searchstring vector
    distList = []
    for thisVector in vect_list: 
        #euclidian distance
        #thisDist = np.linalg.norm(searchVector-thisVector)
        #cosine distance
        #thisDist = scipy.spatial.distance.cosine(searchVector,thisVector)
        #dot product
        thisDist = np.dot(searchVector,thisVector)[0]
        #print(thisDist)
        distList.append(thisDist)

    semanticDistanceList = pd.DataFrame([this_candidate_list,distList]).T
    semanticDistanceList.columns = ['word', 'semantic_dist']
    semanticDistanceList.sort_values(by = 'semantic_dist', ascending=False, inplace= True)
    
    return semanticDistanceList

# COMMAND ----------

def do_fuzzy_matching(this_candidate_list,this_query_list ):
    searchString =str(this_query_list).translate(str.maketrans('', '', string.punctuation))
    fuzzMatchSetDF = pd.DataFrame(process.extract(searchString, this_candidate_list, limit = len(this_candidate_list), scorer = fuzz.token_set_ratio), columns = ['word', 'setMatch'])
    fuzzMatchSortDF = pd.DataFrame(process.extract(searchString, this_candidate_list, limit = len(this_candidate_list), scorer = fuzz.token_sort_ratio), columns = ['word', 'sortMatch'])
    combined_lex = pd.merge(left =fuzzMatchSetDF, right = fuzzMatchSortDF)
    
    return combined_lex

# should also implement some other metods of lexical similarity
# jaccard similarity, levenshtein distance, q-gram distance, etc.

# COMMAND ----------

def combine_semantic_lexical_dfs(thisSemanticMatchDF, thisLexicalMatchDF):
    combined_all = pd.merge(thisSemanticMatchDF,thisLexicalMatchDF)
    #scoring is totally arbitrary
    combined_all["score"] = (0.1*combined_all["setMatch"]+ 0.1*combined_all["sortMatch"])*(combined_all["semantic_dist"]*100)
    return combined_all.sort_values(by = "score", ascending=False)[0:20]

# COMMAND ----------

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

# MAGIC %md
# MAGIC ## Mapping to wiki search

# COMMAND ----------

def return_top_wiki_match(thisSemanticMatchDF, thisLexicalMatchDF):
    combined_all = pd.merge(thisSemanticMatchDF,thisLexicalMatchDF)
    #scoring is totally arbitrary
    combined_all["score"] = (0.1*combined_all["setMatch"]+ 0.1*combined_all["sortMatch"])*(combined_all["semantic_dist"]*100)
    return combined_all['word', 'score'].sort_values(by = "score", ascending=False)[0]

# COMMAND ----------

topic_seed_word_lists = topic_clusters['top_words']

# COMMAND ----------

all_candidate_title_lists_topic =[]
for x in tqdm(topic_seed_word_lists):
    all_candidate_title_lists_topic.append(generateCandidateList(x, nlp))

# COMMAND ----------

semanticMatchDF_list_topic = []
for cl, ql in zip(all_candidate_title_lists_topic, topic_seed_word_lists):
    semanticMatchDF_list_topic.append(compute_semantic_matches(cl,ql))

# COMMAND ----------

lexicalMatchDF_list_topic = []
for cl, ql in zip(all_candidate_title_lists_topic, topic_seed_word_lists):
    lexicalMatchDF_list_topic.append(do_fuzzy_matching(cl,ql))

# COMMAND ----------

combinedDF_list_topic = []
for s,l in zip(semanticMatchDF_list_topic, lexicalMatchDF_list_topic):
    combinedDF_list_topic.append(combine_semantic_lexical_dfs(s,l))

# COMMAND ----------

for i in np.arange(0,len(topic_clusters)):
    print(topic_seed_word_lists[i])
    print(combinedDF_list_topic[i].head(10))
    print("\n\n")

# COMMAND ----------

word = "nurse"
try:
        word_set = wn.synsets(word)[0]
        print(word_set.hypernyms())
        hn_list = sorted([lemma.name() for synset in word_set.hypernyms() for lemma in synset.lemmas()])[:5]
        print(hn_list)
        print(type(hn_list))
except: pass

# COMMAND ----------

test_word = ["nurse"]
word_embedding = embed(word)
test_list = ['programs', 'online', 'nursing', 'nurse', 'rn', 'bsn', 'practitioner', 'degree', 'master', 'msn']
test_sentence = [' '.join(test_list)]
print(test_sentence)
sentence_embedding = embed(test_sentence)
print(len(sentence_embedding[0]))
print(len(word_embedding[0]))
messages  = ["nurse", "programs online nursing nurse rn bsn practitioner degree master msn"]

# COMMAND ----------

distance = scipy.spatial.distance.cdist([word_embedding[0]],  [sentence_embedding[0]], "cosine")[0]

# COMMAND ----------

def embedding_similarity(messages_):
  message_embeddings_ = embed(messages_)
  distance = scipy.spatial.distance.cdist([message_embeddings_[0]],  [message_embeddings_[1]], "cosine")[0]
  print("Similarity score =  {}".format(1-distance))

# COMMAND ----------

embedding_similarity(messages)

# COMMAND ----------

# MAGIC %md
# MAGIC # Trying Wordnet

# COMMAND ----------

# !pip install wordnet

# COMMAND ----------

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

# COMMAND ----------

word = 'dog'
try:
        word_set = wn.synsets(word)[0]
        print(word_set.hypernyms())
        hn_list = sorted([lemma.name() for synset in word_set.hypernyms() for lemma in synset.lemmas()])[:5]
        print(hn_list)
        print(type(hn_list))
except: pass
for synset in wn.synsets(word):
    print("\tLemma: {}".format(synset.name()))
    print("\tDefinition: {}".format(synset.definition()))
    print("\tExample: {}".format(synset.examples()))

# COMMAND ----------

def generate_hypernym_set(seed_word_list):
  hypernym_set = []
  for word in seed_word_list:
      try:
        word_set = wn.synsets(word)[0]
        hn_list = sorted([lemma.name() for synset in word_set.hypernyms() for lemma in synset.lemmas()])[:5]
     #   print(hn_list)
     #   print(type(hn_list))
        if len(hn_list) > 0:
          hypernym_set.extend(hn_list)
      except: pass
  return list(set(hypernym_set))


# COMMAND ----------

all_candidate_hypernym_lists =[]

for i in np.arange(0,len(topic_clusters)):
    print("cluster words:")
    print(topic_seed_word_lists[i])
    for seed_word_list in topic_seed_word_lists:
   #   hypernym_set = generate_hypernym_set(seed_word_list)
      all_candidate_hypernym_lists.append(generate_hypernym_set(seed_word_list))
    print("hypernyms")
    print(all_candidate_hypernym_lists[i])
    print("\n\n")

#for seed_word_list in tqdm(topic_seed_word_lists):


# COMMAND ----------

semanticMatchDF_list_topic = []
for cl, ql in zip(all_candidate_hypernym_lists, topic_seed_word_lists):
    semanticMatchDF_list_topic.append(compute_semantic_matches(cl,ql))

# COMMAND ----------

lexicalMatchDF_list_topic = []
for cl, ql in zip(all_candidate_hypernym_lists, topic_seed_word_lists):
    lexicalMatchDF_list_topic.append(do_fuzzy_matching(cl,ql))

# COMMAND ----------

combinedDF_list_topic = []
for s,l in zip(semanticMatchDF_list_topic, lexicalMatchDF_list_topic):
    combinedDF_list_topic.append(combine_semantic_lexical_dfs(s,l))

# COMMAND ----------

len(combinedDF_list_topic)

# COMMAND ----------

for i in np.arange(0,len(topic_clusters)):
    print(topic_seed_word_lists[i])
    print(combinedDF_list_topic[i].head(10))
    print("\n\n")

# COMMAND ----------

# MAGIC %md # Summarization with one-two words
# MAGIC we want abstractive text summarization

# COMMAND ----------

# Importing requirements
# !pip install --upgraade transformers==4.6.1
# !pip install seq2seq_trainer
# !wget https://raw.githubusercontent.com/huggingface/transformers/master/examples/seq2seq/seq2seq_trainer.py
# !pip install rouge_score
# !pip install pytorch_lightning==0.7.5
# !pip install sentencepiece
# !pip install --upgrade torch==1.8.1
import transformers
from transformers import RobertaTokenizerFast
from transformers import EncoderDecoderModel
import seq2seq_trainer
from seq2seq_trainer import Seq2SeqTrainer
from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing import Optional
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration

# COMMAND ----------

import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

# COMMAND ----------

tag_meta.head()

# COMMAND ----------

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')
type(tokenizer)

# COMMAND ----------

topic_clusters = docs_df.groupby(["topic_level_4"])['title'].apply(lambda x: '. '.join(x)).reset_index()
topic_clusters.head()

# COMMAND ----------

#text = docs_df['title'][1000]
#text = tag_meta['top_articles'][5]
text = topic_clusters['title'][200]
text = ' '.join(text.split())[:1000]
print(len(text))
print(text)

# COMMAND ----------

text = "summarize:" + text
text

# COMMAND ----------

tokenized_text = tokenizer.encode(text, return_tensors="pt").to(device)

# COMMAND ----------

# summmarize 
summary_ids = model.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=1,
                                    max_length=2,
                                    early_stopping=True)

# COMMAND ----------

output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# COMMAND ----------

print("original text: \n", text)
print ("\n\nSummarized text: \n",output)


# COMMAND ----------



# COMMAND ----------

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

#import torch
src_text = [
    text]
#model_name = 'google/pegasus-xsum'
model_name = "facebook/m2m100_418M"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#tokenizer = PegasusTokenizer.from_pretrained(model_name)
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(device)
#model = PegasusForConditionalGeneration.from_pretrained(model_name, from_tf=True).to(device)
tokenizer.src_lang = "en"
batch = tokenizer(src_text, truncation=True, padding='longest', return_tensors="pt", forced_bos_token_id=tokenizer.get_lang_id("en")).to(device)
translated = model.generate(**batch)
tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
#assert tgt_text[0] == "California's largest electricity provider has turned off power to hundreds of thousands of customers."

# COMMAND ----------

print(tgt_text[0])

# COMMAND ----------

from transformers import pipeline
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
summarizer = pipeline("summarization", model="google/pegasus-xsum", tokenizer=PegasusTokenizer.from_pretrained("google/pegasus-xsum"))

# COMMAND ----------

summarizer(text, min_length=1, max_length=20)

# COMMAND ----------

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import SummarizationPipeline

# COMMAND ----------

model_name = 'lincoln/mbart-mlsum-automatic-summarization'

loaded_tokenizer = AutoTokenizer.from_pretrained(model_name)
loaded_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, from_tf=True)

nlp = SummarizationPipeline(model=loaded_model, tokenizer=loaded_tokenizer)

# COMMAND ----------

nlp(text)

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