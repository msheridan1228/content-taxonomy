import scipy.cluster.hierarchy as shc
from numpy.core import multiarray
import fastcluster as fc
from scipy.cluster.hierarchy import ward, fcluster
from sklearn.cluster import AgglomerativeClustering 
from nltk.stem.porter import *
from collections import Counter 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from nltk.stem import WordNetLemmatizer

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# COMMAND ----------

# MAGIC %md
# MAGIC # **Prepare data** 

# COMMAND ----------

docs_df = pd.read_csv('/users/msheridan/documents/marthasblog-wp-scrape.csv')
docs_df = docs_df[['site', 'title', 'url', 'text']]
docs_df.head()

# COMMAND ----------

print(docs_df.shape)
docs_df.dropna(inplace=True)
print(docs_df.shape)
docs_df = docs_df[~(docs_df['text']=='')]
print(docs_df.shape)
docs_df = docs_df.drop_duplicates(subset=['url'])
print(docs_df.shape)
docs_df = docs_df.drop_duplicates(subset=['text'])
print(docs_df.shape)

# COMMAND ----------

# Remove dev urls
bad_string_list = ['.pantheon.', '.lndo', '.local.', 'test.', 'sonic.', '.pantheonsite.', 'localhost', ':8000', '.staging.', 'qa.bankrate.com', 'dev.allconnect.com', 'development']
for bad_string in bad_string_list:
  docs_df = docs_df[~docs_df.site.str.contains(bad_string)]
  print(len(docs_df))

# COMMAND ----------

# MAGIC %run ./utils/model-training-utils

# COMMAND ----------

docs_df['cleaned_text_full'] = docs_df['text'].map(clean_html)
docs_df['cleaned_text'] = docs_df['cleaned_text_full'].map(first_1k_words)
docs_df['cleaned_title'] = docs_df['title'].map(clean_html)
docs_df['cleaned_text_and_title'] = docs_df['cleaned_title'] + '. ' + docs_df['cleaned_text']
docs_df['text_length'] = docs_df['cleaned_text_full'].map(num_words)


# COMMAND ----------

embeddings = embed(docs_df['cleaned_text_and_title']) 
embeddings_ls = [l.numpy() for l in embeddings]
docs_df['text_embedding'] =  embeddings_ls

# COMMAND ----------

print(docs_df['text_embedding'][0].shape)
docs_df['text_embedding'][0]

# COMMAND ----------

# MAGIC %md
# MAGIC # **Trying hierarchical clustering**

# COMMAND ----------

# MAGIC %md ##Determining optimal cluster sizes

# COMMAND ----------

Z1 = fc.linkage_vector(embeddings, method='ward', metric='euclidean')

# COMMAND ----------

# Determine optimal cluster sizes
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(Z1)

# COMMAND ----------

K = (np.round(np.logspace(.2, 2.85, num = 25), 0)).astype(int)
K_uni = np.unique(K)
print(K_uni)

# COMMAND ----------

# Determine optimal cluster sizes
distortions = []
model_dic = {}
cluster_dic = {}
for k in K_uni:
    print(k)
    model = AgglomerativeClustering(n_clusters = k, affinity = 'euclidean', linkage ='ward')
    model_dic.update({k:model})
    clusters = model.fit_predict(embeddings)
    cluster_dic.update({k:model})
    distortions.append(metrics.silhouette_score(embeddings, clusters, metric='euclidean'))

plt.figure(figsize=(12,8))
plt.plot(K_uni, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Avg Silhouette Width')
plt.xscale('log')
plt.title('The Silhouette method showing the optimal k')
# zip joins x and y coordinates in pairs
for x,y in zip(K_uni,distortions):

    label = "{}".format(x)

    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.show()

# COMMAND ----------

# MAGIC %md ##Creating Clusters

# COMMAND ----------

topic_level_1_clusters = fcluster(Z1,50, "distance")
max(topic_level_1_clusters)

# COMMAND ----------

topic_level_2_clusters = fcluster(Z1,30, "distance")
max(topic_level_2_clusters)

# COMMAND ----------

topic_level_3_clusters = fcluster(Z1,12, "distance")
max(topic_level_3_clusters)

# COMMAND ----------

topic_level_4_clusters = fcluster(Z1,5, "distance")
max(topic_level_4_clusters)

# COMMAND ----------

docs_df['topic_level_1_cluster'] = topic_level_1_clusters
docs_df['topic_level_2_cluster'] = topic_level_2_clusters
docs_df['topic_level_3_cluster'] = topic_level_3_clusters
docs_df['topic_level_4_cluster'] = topic_level_4_clusters

# COMMAND ----------

docs_df[docs_df['topic_level_4_cluster']==9].head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC # Pulling rough topic 'names'

# COMMAND ----------

# MAGIC  %run ./utils/tfidf-utils

# COMMAND ----------

key_list = ['topic_level_1_cluster', 'topic_level_2_cluster', 'topic_level_3_cluster', 'topic_level_4_cluster']
for key in key_list:
  print(key)
  topic_clusters = docs_df.groupby([key])['title'].apply(lambda x: ' '.join(x)).reset_index()
  topic_clusters['title_token'] = topic_clusters['title'].map(clean_title_text)
  if key in ['topic_level_4_cluster', 'topic_level_3_cluster']:
      vectorizer = TfidfVectorizer(min_df=0.0001,max_df=.98, stop_words='english',ngram_range=(2,2),  analyzer='word', encoding='latin-1')
  else:   
      #altering min/max for topic level 4 for better results
      vectorizer = TfidfVectorizer(min_df=0.01,max_df=.97, stop_words='english',ngram_range=(2,2),  analyzer='word', encoding='latin-1')  
  tfidf = vectorizer.fit_transform(topic_clusters['title_token'])
  topic_clusters[str(key).replace('_cluster', '')] = pull_top_tfidf_words(tfidf, vectorizer, 3)
  topic_clusters = topic_clusters[[key, str(key).replace('_cluster', '')]]
  docs_df = docs_df.merge(topic_clusters, how="left", on=key)

# COMMAND ----------

docs_df.sample(10)

# COMMAND ----------

def check_for_dupes(docs_df, tag):
  num_names = docs_df[tag].nunique()
  num_clusters = docs_df[str(tag)+'_cluster'].nunique()
  assert num_names == num_clusters, f'Duplicate cluster names at {tag}. {num_names} tag names, and {num_clusters} clusters'

tag_list = ['topic_level_1', 'topic_level_2', 'topic_level_3', 'topic_level_4']
for tag in tag_list:
  check_for_dupes(docs_df, tag)

# COMMAND ----------

export_columns = ['url', 'title', 'site', 'topic_level_1','topic_level_2','topic_level_3','topic_level_4', 'topic_level_1_cluster','topic_level_2_cluster', 'topic_level_3_cluster','topic_level_4_cluster']
docs_df.to_csv('/users/msheridan/documents/marthasblog-clusters.csv')

# COMMAND ----------

# MAGIC %md  #Pickling

# COMMAND ----------

filepath = '/users/msheridan/documents/marthasblog-clusters.pkl'
data_dict = docs_df.loc[:, docs_df.columns].to_dict(orient='records')
with open(filepath, "wb") as f:
  cloudpickle.dump(data_dict, f)

